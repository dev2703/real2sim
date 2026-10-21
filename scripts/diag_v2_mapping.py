"""R0: read-only v2 mapping/reset audit. Never clamp references or invent valid starts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from real2sim.data.raw import RawDataset, sha256_file
from real2sim.geometry.so3 import delta_action
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics


def gripper_scale_audit(env, calibration_path, observed_percent):
    """Encoder-derived relative-angle hypothesis; closure/ratio/sign remain unverified.

    This compares geometry at identical arm coordinates, without physics, target clipping,
    or changing the scene. Tip-sphere separation is a reproducible CAD diagnostic, not a
    measurement of the real jaw gap or the minimum distance between entire jaw surfaces.
    """
    calibration_path = Path(calibration_path)
    calibration = json.loads(calibration_path.read_text())
    gripper = calibration["gripper"]
    low, high = float(gripper["range_min"]), float(gripper["range_max"])
    if not (0 <= low < high <= 4095) or gripper["drive_mode"] != 0:
        raise ValueError("this SO-101 diagnostic requires ordered 12-bit endpoints and drive_mode=0")
    scale = (high - low) * (2 * np.pi / 4095) / 100
    zero = float(env.mapping.gripper_real_to_sim_rad(0))
    candidate = replace(env.mapping, gripper_affine_rad=(zero, scale))
    data = mujoco.MjData(env.model)
    tips = [env.model.geom(name).id for name in ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1")]
    if any(env.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE for g in tips):
        raise ValueError("expected the pinned jaw tip spheres")

    def gap(q):
        data.qpos[env.qadr] = 0.
        data.qpos[env.qadr[5]] = q
        mujoco.mj_kinematics(env.model, data)
        return float((np.linalg.norm(data.geom_xpos[tips[0]] - data.geom_xpos[tips[1]])
                      - sum(env.model.geom_size[g, 0] for g in tips)) * 1000)

    observed = np.asarray(observed_percent, float)
    rows = []
    for percent in [0., 15., 30., 45., 60., float(observed.max())]:
        current = float(env.mapping.gripper_real_to_sim_rad(percent))
        proposed = float(candidate.gripper_real_to_sim_rad(percent))
        rows.append({"percent": percent, "current_native_rad": current,
                     "candidate_native_rad": proposed,
                     "angular_difference_deg": float(np.rad2deg(proposed - current)),
                     "current_tip_sphere_separation_mm": gap(current),
                     "candidate_tip_sphere_separation_mm": gap(proposed),
                     "candidate_within_native_limits": bool(env.limits[5, 0] <= proposed <= env.limits[5, 1])})
    native_observed = candidate.gripper_real_to_sim_rad(observed)
    native_percent_limits = candidate.gripper_sim_to_real_percent(env.limits[5])
    return {
        "kind": "ENCODER_SCALE_CANDIDATE_NOT_ADOPTED",
        "calibration_path": str(calibration_path), "calibration_sha256": sha256_file(calibration_path),
        "reported_tick_endpoints": [low, high], "angle_denominator_ticks": 4095,
        "candidate_percent_to_joint": {"offset_rad": zero, "scale_rad_per_percent": scale},
        "relative_motor_rotation_deg_per_100_percent": float(np.rad2deg(scale) * 100),
        "current_joint_rotation_deg_per_100_percent": float(np.rad2deg(
            env.mapping.gripper_real_to_sim_rad(100) - env.mapping.gripper_real_to_sim_rad(0))),
        "candidate_native_limit_percent_interval": sorted(native_percent_limits.tolist()),
        "observed_percent_interval": [float(observed.min()), float(observed.max())],
        "observed_rows_outside_native_limits_with_candidate": int(np.sum(
            (native_observed < env.limits[5, 0]) | (native_observed > env.limits[5, 1]))),
        "ordinary_openings": rows, "tip_spheres": [env.model.geom(g).name for g in tips],
        "assumptions_not_verified": ["positive motor-to-jaw sign", "one-to-one motor-to-jaw ratio",
                                     "existing closure zero", "physical jaw shape matches CAD"],
        "interpretation": "Tip-sphere separation is not real jaw clearance. No physical fit, limit change, clipping, or configuration promotion was performed.",
    }


def contact_scan(env, kin, q, candidates):
    """Scan in-limit recorded poses only; table/self geometry, no object-registration claim."""
    base = env.model.body("base").id
    robot_bodies = set()
    for body in range(1, env.model.nbody):
        ancestor = body
        while ancestor and ancestor != base:
            ancestor = int(env.model.body_parentid[ancestor])
        if ancestor == base:
            robot_bodies.add(body)
    rows, pairs = [], Counter()
    for index in candidates:
        kin.forward(q[index])
        table, self_depth = 0., 0.
        pose_pairs = set()
        for c in kin.data.contact:
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(env.model.geom_bodyid[g1]), int(env.model.geom_bodyid[g2])
            names = {env.model.body(b).name for b in (b1, b2)}
            depth = max(0., float(-c.dist * 1000))
            if env.table_geom in (g1, g2) and ({b1, b2} & robot_bodies):
                table = max(table, depth)
            elif b1 in robot_bodies and b2 in robot_bodies and not names <= {"gripper", "moving_jaw_so101_v1"}:
                self_depth = max(self_depth, depth)
                if depth > kin.config["penetration_tolerance_m"] * 1000:
                    pose_pairs.add(":".join(sorted(names)))
        pairs.update(pose_pairs)
        rows.append({"source_frame": int(index) * 2, "table_penetration_mm": table,
                     "self_penetration_mm": self_depth})
    tolerance = kin.config["penetration_tolerance_m"] * 1000
    return {
        "in_limit_poses_scanned": len(rows), "out_of_limit_poses_not_scanned": len(q) - len(rows),
        "ik_penetration_tolerance_mm": tolerance,
        "table_overlap_poses": sum(r["table_penetration_mm"] > .001 for r in rows),
        "table_over_ik_tolerance_poses": sum(r["table_penetration_mm"] > tolerance for r in rows),
        "self_over_ik_tolerance_poses": sum(r["self_penetration_mm"] > tolerance for r in rows),
        "max_table_penetration_mm": max((r["table_penetration_mm"] for r in rows), default=0.),
        "max_self_penetration_mm": max((r["self_penetration_mm"] for r in rows), default=0.),
        "self_collision_pairs_by_pose": dict(pairs),
        "worst_table_frames": sorted(rows, key=lambda r: r["table_penetration_mm"], reverse=True)[:5],
        "worst_self_frames": sorted(rows, key=lambda r: r["self_penetration_mm"], reverse=True)[:5],
    }


def audit(root: Path, config: Path, *, full_contact_scan=False, gripper_calibration=None):
    ds = RawDataset(root)
    rows = []
    global_min, global_max = np.full(6, np.inf), np.full(6, -np.inf)
    with So101PickPlaceEnv(config, render_images=False) as env:
        kin = NativeKinematics(env.model, env.config)
        for episode in ds.episode_indices:
            g = ds.episode_rows(episode)
            states = ds.stack(g, "observation.state")[::2]
            q = np.array([sim_from_real(env.mapping, s) for s in states])
            global_min, global_max = np.minimum(global_min, q.min(0)), np.maximum(global_max, q.max(0))
            outside = (q < env.limits[:, 0]) | (q > env.limits[:, 1])
            candidates = np.flatnonzero(~outside.any(axis=1))
            first_valid = None
            for index in candidates:
                _, info = env.reset(
                    seed=0, scenario={"joint_positions": q[index], "ball_xy": [0.5, 0.3]}
                )
                pen = info["privileged_metrics"]["robot_table_penetration_m"]
                if pen <= 1e-6:
                    first_valid = int(index)
                    break
            reconstruction = []
            if first_valid is not None:
                for index in sorted(
                    set([first_valid, *candidates[:: max(1, len(candidates) // 8)].tolist()])
                ):
                    target = kin.forward(q[index])
                    # A different bounded seed prevents a trivial identity reconstruction.
                    seed = q[index].copy()
                    seed[:5] += np.where(seed[:5] < env.limits[:5].mean(axis=1), 0.01, -0.01)
                    result = kin.solve(target, seed, gripper=float(q[index, 5]))
                    reconstructed = kin.forward(result.qpos)
                    error = delta_action(reconstructed, target)
                    reconstruction.append(
                        {
                            "source_frame": int(index) * 2,
                            "accepted": result.accepted,
                            "position_error_mm": float(np.linalg.norm(error[:3]) * 1000),
                            "rotation_error_rad": float(np.linalg.norm(error[3:])),
                            "reason": result.reason,
                        }
                    )
            rows.append(
                {
                    "episode_index": episode,
                    "sampled_rows": len(q),
                    "start_out_of_limits": bool(outside[0].any()),
                    "start_outside_by_joint": outside[0].tolist(),
                    "outside_fraction_by_joint": outside.mean(axis=0).tolist(),
                    "first_nonpenetrating_in_limit_source_frame": None
                    if first_valid is None
                    else first_valid * 2,
                    "candidate_is_approved_task_start": False,
                    "fk_ik_reconstruction": reconstruction,
                    **({"full_contact_scan": contact_scan(env, kin, q, candidates)} if full_contact_scan else {}),
                }
            )
        _, original = env.reset(seed=0)
        configured_real = sim_from_real(env.mapping, env.config["initial_qpos_real_deg"])
        default_clipping = configured_real - env.initial_qpos
        task_path = Path(env.config["task_config"])
        return {
            "diagnostic": "R0 provisional mapping, not physical calibration",
            "script_sha256": sha256_file(Path(__file__)),
            "raw_root": str(root),
            "raw_files_sha256": {
                str(p.relative_to(root)): sha256_file(p)
                for directory in ("data", "meta")
                for p in sorted((root / directory).rglob("*"))
                if p.is_file()
            },
            "config_sha256": sha256_file(config),
            "task_sha256": sha256_file(task_path),
            **({"gripper_scale": gripper_scale_audit(
                env, gripper_calibration, ds.stack(ds.data, "observation.state")[:, 5])}
               if gripper_calibration is not None else {}),
            "source_hz": ds.fps,
            "sample_stride": 2,
            "nominal_policy_hz": ds.fps / 2,
            "physical_cadence_verified": False,
            "joint_names": env.config["joint_names"],
            "original_start": {
                "native_difference_from_requested": default_clipping.tolist(),
                "metrics": original["privileged_metrics"]["start_pose"],
            },
            "episodes": rows,
            "starts_outside_limits": sum(r["start_out_of_limits"] for r in rows),
            "range_only_offset_constraints": {
                "not_a_calibration": True,
                "native_recorded_min_rad": global_min.tolist(),
                "native_recorded_max_rad": global_max.tolist(),
                "constant_offset_lower_rad": (env.limits[:, 0] - global_min).tolist(),
                "constant_offset_upper_rad": (env.limits[:, 1] - global_max).tolist(),
                "constant_offset_possible_by_joint": (
                    env.limits[:, 0] - global_min <= env.limits[:, 1] - global_max).tolist(),
                "meaning": "Necessary range-only constraints, not independently verified offsets. No mapping, limits or geometry changed.",
            },
            "warning": "A later valid pose is only a candidate. Phase annotations must establish that the full grasp-and-place sequence remains.",
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v2",
    )
    p.add_argument("--config", type=Path, default=Path("configs/sim_so101.yaml"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--full-contact-scan", action="store_true")
    p.add_argument("--gripper-calibration", type=Path,
                   help="optional unchanged follower JSON; audit a relative-angle hypothesis only")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    result = audit(a.root, a.config, full_contact_scan=a.full_contact_scan,
                   gripper_calibration=a.gripper_calibration)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("starts_outside_limits", "source_hz", "nominal_policy_hz")}
        )
    )


if __name__ == "__main__":
    main()
