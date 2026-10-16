"""Bounded SO-101 recorded-control sensitivity; never a learned-policy result.

All episodes share each candidate's XY offset, gripper scaling and contact settings.
The ball remains dynamic and supported by the unchanged table. Native position/velocity
limits, actuator parameters, collision geometry and success thresholds remain unchanged.
The measured follower opening is compared with the leader command without assuming that
the loaded opening is an actuator command. Existing reviewed validation episodes are
diagnostic development cases, not an untouched test set.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np
import yaml

from real2sim.evaluation.metrics import TrialTracker
from real2sim.geometry.so3 import delta_action
from real2sim.policy.paired import file_sha256
from real2sim.sim.env import GRIPPER_BODIES, So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics

DEFAULT_CONFIG = Path("results/metrics/dinov2_act_pilot_20261015/configs/sim_so101.yaml")
DEFAULT_REPLAY = Path("results/audit/geometry_control_20261016/g0_complete/report.json")


def joint_target(state_next, leader_gripper, mode, scale=1.0):
    """Scale a diagnostic command around empty closure; never change the raw state."""
    target = np.array(state_next, dtype=float, copy=True)
    if mode == "leader":
        target[5] = leader_gripper * 100.0
    elif mode != "follower":
        raise ValueError(mode)
    target[5] = 1.2 + scale * (target[5] - 1.2)
    return target


@contextmanager
def contact_monitor(env, *, capture=False):
    """Read every solved contact before integration, without changing solver state."""
    original = mujoco.mj_step
    dt = float(env.model.opt.timestep)
    result = {"contact_substeps": 0, "bilateral_substeps": 0,
              "longest_bilateral_s": 0.0, "longest_bilateral_lift_s": 0.0,
              "bilateral_lift_substeps": 0, "max_bilateral_height_m": 0.0,
              "max_ball_height_m": 0.0, "max_contact_penetration_m": 0.0,
              "max_min_jaw_normal_force_n": 0.0, "first_contact_s": None,
              "first_contact_geom": None, "first_bilateral_s": None,
              "bilateral_slip_mean_mps": None, "by_geom": {}, "trace": []}
    bilateral_run = lift_run = 0
    slip_sum = 0.0
    slip_intervals = 0
    previous_local = None
    force = np.zeros(6)

    def step(model, data):
        nonlocal bilateral_run, lift_run, slip_sum, slip_intervals, previous_local
        touching = set()
        by_jaw = [0.0, 0.0]
        names = []
        for index, contact in enumerate(data.contact[:data.ncon]):
            pair = (int(contact.geom1), int(contact.geom2))
            if env.ball_geom not in pair:
                continue
            other = pair[1] if pair[0] == env.ball_geom else pair[0]
            if other == env.table_geom:
                continue
            touching.add(other)
            name = model.geom(other).name or f"geom_{other}"
            names.append(name)
            result["by_geom"][name] = result["by_geom"].get(name, 0) + 1
            result["max_contact_penetration_m"] = max(
                result["max_contact_penetration_m"], max(0.0, -float(contact.dist)))
            mujoco.mj_contactForce(model, data, index, force)
            for j, body in enumerate(GRIPPER_BODIES):
                if other in env.gripper_geoms[body]:
                    by_jaw[j] += max(0.0, float(force[0]))
        two_jaw = all(bool(touching & env.gripper_geoms[body]) for body in GRIPPER_BODIES)
        height = float(data.xpos[env.ball_body, 2] - env.table_top - env.task["ball"]["radius"])
        result["max_ball_height_m"] = max(result["max_ball_height_m"], height)
        if touching:
            result["contact_substeps"] += 1
            if result["first_contact_s"] is None:
                result["first_contact_s"] = float(data.time)
                result["first_contact_geom"] = names
        bilateral_run = bilateral_run + 1 if two_jaw else 0
        lift_run = lift_run + 1 if two_jaw and height > 0.03 else 0
        result["longest_bilateral_s"] = max(result["longest_bilateral_s"], bilateral_run * dt)
        result["longest_bilateral_lift_s"] = max(result["longest_bilateral_lift_s"], lift_run * dt)
        if two_jaw:
            result["bilateral_substeps"] += 1
            result["bilateral_lift_substeps"] += int(height > 0.03)
            result["max_bilateral_height_m"] = max(result["max_bilateral_height_m"], height)
            result["max_min_jaw_normal_force_n"] = max(result["max_min_jaw_normal_force_n"], min(by_jaw))
            if result["first_bilateral_s"] is None:
                result["first_bilateral_s"] = float(data.time)
            rotation = data.site_xmat[env.site_id].reshape(3, 3)
            local = rotation.T @ (data.xpos[env.ball_body] - data.site_xpos[env.site_id])
            if previous_local is not None:
                slip_sum += float(np.linalg.norm(local - previous_local) / dt)
                slip_intervals += 1
            previous_local = local
        else:
            previous_local = None
        if capture and touching:
            result["trace"].append({"time_s": float(data.time), "geoms": names,
                                    "bilateral": two_jaw, "normal_force_n": by_jaw,
                                    "height_m": height})
        return original(model, data)

    mujoco.mj_step = step
    try:
        yield result
    finally:
        mujoco.mj_step = original
        if slip_intervals:
            result["bilateral_slip_mean_mps"] = slip_sum / slip_intervals


def replay(config, joint, cart, episode, metadata, review, candidate, mode, capture=False):
    indices = np.flatnonzero(joint["episode_index"] == episode)
    with So101PickPlaceEnv(config, render_images=False) as env:
        env.max_episode_steps = len(indices) - 1
        q0 = np.clip(sim_from_real(env.mapping, joint["state"][indices[0]]), *env.limits.T)
        xy = np.array(metadata["ball_xy"]) + np.array(candidate["xy_offset_mm"]) / 1000.0
        scenario = {"joint_positions": q0, "ball_xy": xy, "bowl_xy": metadata["bowl_xy"]}
        _, initial = env.reset(seed=0, scenario=scenario)
        initial = initial["privileged_metrics"]
        bad_contacts = [c for c in initial["contacts"] if "ball_geom" in (c["geom1"], c["geom2"])
                        and "table_geom" not in (c["geom1"], c["geom2"])]
        result = {"episode": episode, "mode": mode, "candidate": candidate,
                  "initial_ball_xy": xy.tolist(), "invalid_initial_ball_contacts": bad_contacts,
                  "initial_robot_table_penetration_m": initial["robot_table_penetration_m"],
                  "initial_penetration_tolerance_m": float(env.config["ik"]["penetration_tolerance_m"]),
                  "initial_physical_valid": not bad_contacts and initial["robot_table_penetration_m"] <= env.config["ik"]["penetration_tolerance_m"]}
        if bad_contacts:
            return dict(result, valid=False)
        tracker = TrialTracker(int(env.task["episode"]["success_hold_s"] * env.control_hz),
                               .03, env.table_top, env.task["ball"]["radius"])
        kin = NativeKinematics(env.model, env.config)
        pose_errors, arm_errors, displacements, rows = [], [], [], []
        with contact_monitor(env, capture=capture) as contacts:
            for tick, (current, nxt) in enumerate(zip(indices[:-1], indices[1:], strict=True)):
                target = joint_target(joint["state"][nxt], cart["action"][current, 6], mode,
                                      candidate["gripper_scale"])
                _, _, _, _, info = env.step_real_joint_targets(target)
                m = info["privileged_metrics"]
                tracker.update(m)
                reference = kin.forward(sim_from_real(env.mapping, joint["state"][nxt]))
                error = delta_action(env.ee_pose(), reference)
                pose_errors.append(float(np.linalg.norm(error[:3])))
                arm_errors.append(float(np.linalg.norm(env.real_state()[:5] - target[:5])))
                timestamp = float(joint["timestamp"][nxt])
                if timestamp < review["closure"][0]:
                    displacements.append(float(np.linalg.norm(np.array(m["ball_xyz_task"][:2]) - xy)))
                if capture:
                    rows.append({"tick": tick, "time_s": timestamp, "target": target.tolist(),
                                 "state": env.real_state().tolist(), "pose_error_m": pose_errors[-1],
                                 "ball_xyz_task": m["ball_xyz_task"], "held": m["ball_held"],
                                 "lifted": m["lifted"], "in_bowl": m["ball_in_bowl"],
                                 "joint_rate_limited": info["control"]["rate_limited"],
                                 "joint_limit_clipped": info["control"]["limit_clipped"]})
        return dict(result, valid=result["initial_physical_valid"], execution_completed=True,
                    duration_s=float(joint["timestamp"][indices[-1]]),
                    success=tracker.success, held=tracker.held_ever, lifted=tracker.lifted,
                    min_ee_ball_distance_m=tracker.min_ee_ball_distance_m,
                    pose_error_mean_m=float(np.mean(pose_errors)),
                    pose_error_p95_m=float(np.quantile(pose_errors, .95)),
                    arm_joint_error_mean_deg_l2=float(np.mean(arm_errors)),
                    ball_displacement_before_closure_m=max(displacements, default=0.0),
                    contact=contacts, trace=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-report", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--modes", choices=["follower", "leader"], nargs="+", default=["follower", "leader"])
    parser.add_argument("--xy-grid-mm", type=float, nargs="+", default=[0.0])
    parser.add_argument("--offset-mm", type=float, nargs=2)
    parser.add_argument("--gripper-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--friction", type=float, nargs="+", default=[0.6])
    parser.add_argument("--damping", type=float, nargs="+", default=[1.0])
    parser.add_argument("--base-offset-mm", type=float, nargs="+")
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    xy_values = [args.offset_mm] if args.offset_mm is not None else list(itertools.product(args.xy_grid_mm, repeat=2))
    if any(abs(value) > 30 for xy in xy_values for value in xy):
        parser.error("Exploratory XY offsets are bounded to +/-30 mm per axis")
    if any(not .8 <= scale <= 1.2 for scale in args.gripper_scales):
        parser.error("Diagnostic gripper scales must be in [0.8, 1.2]")
    if not set(args.friction) <= {.3, .6, 1.0} or not set(args.damping) <= {.7, 1.0, 1.3}:
        parser.error("Use only the existing diagnostic friction/damping ranges")
    if args.base_offset_mm and any(not 0 <= value <= 30 for value in args.base_offset_mm):
        parser.error("Base/support sensitivity is bounded to [0, 30] mm")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "executed_script.txt").write_text(Path(__file__).read_text())
    manifest = json.loads(Path(".cache/derived_joint/manifest.json").read_text())
    cart_manifest = json.loads(Path(".cache/derived_cartesian/manifest.json").read_text())
    for name, m in (("derived_joint", manifest), ("derived_cartesian", cart_manifest)):
        if file_sha256(Path(".cache") / name / "arrays.npz") != m["files_sha256"]["arrays.npz"]:
            raise ValueError("Source checksum mismatch")
    joint = dict(np.load(".cache/derived_joint/arrays.npz", allow_pickle=False))
    cart = dict(np.load(".cache/derived_cartesian/arrays.npz", allow_pickle=False))
    for key in ("episode_index", "timestamp", "frame_index"):
        np.testing.assert_array_equal(joint[key], cart[key])
    reference = json.loads(args.reference_report.read_text())
    base = yaml.safe_load(args.config.read_text())
    task = yaml.safe_load(Path(base["task_config"]).read_text())
    heights = args.base_offset_mm or [task["robot_so101"]["base_z_offset"] * 1000]
    report = {"kind": "EXPLORATORY_RECORDED_CONTROL_GEOMETRY_CONTACT_NOT_POLICY_OR_CALIBRATION",
              "script_sha256": file_sha256(Path(__file__)), "config_sha256": file_sha256(args.config),
              "source_hashes": {"joint": manifest["files_sha256"]["arrays.npz"],
                                "cartesian": cart_manifest["files_sha256"]["arrays.npz"]},
              "reference_report": str(args.reference_report),
              "protocol": {"common_parameters": True, "ball_dynamic": True,
                           "safety_limits_unchanged": True, "collision_geometry_unchanged": True,
                           "split": "Reviewed historical validation used for diagnosis; not untouched evaluation",
                           "gripper_scale": "Hypothesis only, scaled about 1.2% empty closure; existing model limits still clip commands",
                           "support": "Fixed table/ball radius and world camera; base height candidates adjust task-relative camera Z consistently; original 1mm ball settle gap",
                           "force": "Read solved mj_contactForce before each physics step",
                           "lift": "Official unchanged score plus independent bilateral contact >30mm above rest"},
              "runs": []}
    for friction, damping, height in itertools.product(args.friction, args.damping, heights):
        cfg = copy.deepcopy(base)
        cfg["contacts"]["friction"][0] = friction
        cfg["contacts"]["solref"][1] = damping
        changed_task = copy.deepcopy(task)
        delta = height / 1000.0 - task["robot_so101"]["base_z_offset"]
        changed_task["robot_so101"]["base_z_offset"] = height / 1000.0
        for key in ("approx_position_task", "look_at_task"):
            changed_task["camera_iphone"][key][2] -= delta
        task_path = args.out / f"task_b{height}.yaml"
        task_path.write_text(yaml.safe_dump(changed_task, sort_keys=False))
        cfg["task_config"] = str(task_path.resolve())
        config = args.out / f"sim_f{friction}_d{damping}_b{height}.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        for xy, scale, mode, ep in itertools.product(xy_values, args.gripper_scales, args.modes, args.episodes):
            candidate = {"xy_offset_mm": list(xy), "gripper_scale": scale,
                         "friction": friction, "damping": damping, "base_offset_mm": height}
            run = replay(config, joint, cart, ep, reference["episodes"][str(ep)],
                         reference["review"][str(ep)], candidate, mode, args.capture)
            report["runs"].append(run)
            (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"episode": ep, "mode": mode, **candidate,
                              "valid": run["valid"], "lifted": run.get("lifted"),
                              "success": run.get("success"),
                              "bilateral_s": run.get("contact", {}).get("longest_bilateral_s"),
                              "bilateral_lift_s": run.get("contact", {}).get("longest_bilateral_lift_s")}), flush=True)


if __name__ == "__main__":
    main()
