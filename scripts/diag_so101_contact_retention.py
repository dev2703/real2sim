"""Read-only jaw diagnostics and matched fixed-pose holds under one candidate geometry.

The 20 s hold begins after five consecutive 10 Hz samples with active bilateral
jaw contacts and ball height >30 mm. No per-episode offset, gain, friction or
command fit is permitted. The hold freezes the last bounded native target,
including its gripper command; it is a mechanism probe, not a placement trial.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from run_so101_experiments import atomic_json, hash_file

from real2sim.evaluation.metrics import TrialTracker
from real2sim.geometry.so3 import delta_action
from real2sim.sim.env import GRIPPER_BODIES, So101PickPlaceEnv, real_from_sim, sim_from_real
from real2sim.sim.kinematics import NativeKinematics


def ball_contact_frame(contact, ball_geom):
    """Return the normal/force frame pointing onto the ball for either geom order."""
    if ball_geom not in (int(contact.geom1), int(contact.geom2)):
        raise ValueError("contact does not contain ball")
    return np.asarray(contact.frame).reshape(3, 3).T * (1 if contact.geom2 == ball_geom else -1)


def hold_ready(rows, count=5):
    return len(rows) >= count and all(
        r["bilateral"] and r["height_m"] > .03 for r in rows[-count:])


def tip_gap(env, data):
    tips = [env.model.geom(f"{name}_jaw_sph_tip1").id for name in ("fixed", "moving")]
    return float(np.linalg.norm(data.geom_xpos[tips[1]] - data.geom_xpos[tips[0]])
                 - env.model.geom_size[tips, 0].sum())


def snapshot(env):
    """Sample after env.step's existing mj_forward; do not alter simulation state.

    MuJoCo contact.frame points geom1 -> geom2. mj_contactForce is in that
    frame. Contact point slip is (J_ball - J_jaw) qvel projected tangentially.
    https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjcontact
    https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce
    """
    model, data = env.model, env.data
    rotation = data.site_xmat[env.site_id].reshape(3, 3)
    origin, ball = data.site_xpos[env.site_id], data.xpos[env.ball_body]
    contacts = []
    force = np.zeros(6)
    jac_ball, jac_other = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for index, contact in enumerate(data.contact[:data.ncon]):
        geoms = (int(contact.geom1), int(contact.geom2))
        if env.ball_geom not in geoms or contact.efc_address < 0:
            continue
        other = geoms[1] if geoms[0] == env.ball_geom else geoms[0]
        if other == env.table_geom:
            continue
        body = int(model.geom_bodyid[other])
        body_name = model.body(body).name
        frame = ball_contact_frame(contact, env.ball_geom)
        normal = frame[:, 0]
        mujoco.mj_contactForce(model, data, index, force)
        mujoco.mj_jac(model, data, jac_ball, None, contact.pos, env.ball_body)
        mujoco.mj_jac(model, data, jac_other, None, contact.pos, body)
        relative = (jac_ball - jac_other) @ data.qvel
        tangent = relative - np.dot(relative, normal) * normal
        fn = max(0., float(force[0]))
        utilization = (float(np.linalg.norm(force[1:3] / np.maximum(contact.friction[:2], 1e-12)) / fn)
                       if fn > 1e-8 else None)
        contacts.append({"geom": model.geom(other).name or f"geom_{other}",
            "body": body_name, "jaw": body_name if body_name in GRIPPER_BODIES else None,
            "position_tool_m": (rotation.T @ (contact.pos - origin)).tolist(),
            "normal_onto_ball_tool": (rotation.T @ normal).tolist(),
            "force_onto_ball_world_n": (frame @ force[:3]).tolist(),
            "normal_force_n": fn, "tangential_force_n": float(np.linalg.norm(force[1:3])),
            "sliding_friction_utilization": utilization,
            "tangential_surface_slip_mps": float(np.linalg.norm(tangent)),
            "penetration_m": max(0., -float(contact.dist))})
    jaws = {}
    for name in GRIPPER_BODIES:
        members = [c for c in contacts if c["jaw"] == name]
        total = sum(c["normal_force_n"] for c in members)
        def weighted(key, members=members, total=total):
            return sum((np.asarray(c[key]) * c["normal_force_n"] for c in members), np.zeros(3)) / total
        mean_normal = weighted("normal_onto_ball_tool") if total > 1e-8 else None
        jaws[name] = {"count": len(members), "normal_force_n": total,
            "contact_center_tool_m": weighted("position_tool_m").tolist() if total > 1e-8 else None,
            "mean_normal_onto_ball_tool": mean_normal.tolist() if mean_normal is not None else None}
    normals = [jaws[n]["mean_normal_onto_ball_tool"] for n in GRIPPER_BODIES]
    opposition = None
    if all(n is not None and np.linalg.norm(n) > 1e-8 for n in normals):
        a, b = (np.asarray(n) / np.linalg.norm(n) for n in normals)
        opposition = float(np.rad2deg(np.arccos(np.clip(a @ b, -1., 1.))))
    tip_vectors = {}
    for side, body_name in zip(("fixed", "moving"), GRIPPER_BODIES, strict=True):
        body_id = model.body(body_name).id
        tip_id = model.geom(f"{side}_jaw_sph_tip1").id
        tip_vectors[side] = (data.xmat[body_id].reshape(3, 3).T @
                            (ball - data.geom_xpos[tip_id])).tolist()
    return {"simulation_time_s": float(data.time), "sim_tip_surface_gap_m": tip_gap(env, data),
        "ball_from_tip_in_jaw_m": tip_vectors, "ball_tool_m": (rotation.T @ (ball - origin)).tolist(),
        "gravity_direction_tool": (rotation.T @ np.array([0., 0., -1.])).tolist(),
        "height_m": float(ball[2] - env.table_top - env.task["ball"]["radius"]),
        "bilateral": all(jaws[n]["count"] > 0 for n in GRIPPER_BODIES),
        "jaw_normal_opposition_deg": opposition, "jaws": jaws, "contacts": contacts}


def replay(config, joint, cart, indices, metadata, offset, mode, hold_seconds=20.):
    if mode not in ("recorded_motion", "fixed_target_hold"):
        raise ValueError(mode)
    with So101PickPlaceEnv(config, render_images=False) as env:
        hold_ticks = round(hold_seconds * env.control_hz)
        env.max_episode_steps = len(indices) + hold_ticks
        kin = NativeKinematics(env.model, env.config)
        qrefs = np.array([sim_from_real(env.mapping, q) for q in joint["state"][indices]])
        poses, reference_gaps = [], []
        for q in qrefs:
            poses.append(kin.forward(q))
            reference_gaps.append(tip_gap(env, kin.data))
        poses = np.array(poses)
        scenario = {"joint_positions": np.clip(qrefs[0], *env.limits.T),
            "ball_xy": np.asarray(metadata["ball_xy"]) + np.asarray(offset) / 1000,
            "bowl_xy": metadata["bowl_xy"]}
        _, info = env.reset(seed=0, scenario=scenario)
        initial = info["privileged_metrics"]
        bad = [c for c in initial["contacts"] if "ball_geom" in (c["geom1"], c["geom2"])
               and "table_geom" not in (c["geom1"], c["geom2"])]
        if bad or initial["robot_table_penetration_m"] > env.config["ik"]["penetration_tolerance_m"]:
            raise ValueError("invalid initial contact")
        tracker = TrialTracker(int(env.task["episode"]["success_hold_s"] * env.control_hz),
                               .03, env.table_top, env.task["ball"]["radius"])
        rows, hold_start, fixed_target, fixed_pose = [], None, None, None
        for k in range(len(indices) - 1 + hold_ticks):
            holding = fixed_target is not None
            if holding:
                if k - hold_start >= hold_ticks:
                    break
                _, _, _, _, info = env.step(fixed_target)
                reference = fixed_pose
                source_time = None
                observed = None
            else:
                if k >= len(indices) - 1:
                    break
                current, nxt = indices[k:k + 2]
                target = joint["state"][nxt].copy()
                target[5] = float(cart["action"][current, 6]) * 100
                _, _, _, _, info = env.step_real_joint_targets(target)
                reference = poses[k + 1]
                source_time = float(joint["timestamp"][nxt])
                observed = float(joint["state"][nxt, 5])
            m, control = info["privileged_metrics"], info["control"]
            tracker.update(m)
            row = snapshot(env)
            error = delta_action(env.ee_pose(), reference)
            row.update(tick=k + 1, source_time_s=source_time, fixed_hold=holding,
                reference_pose_error_m=float(np.linalg.norm(error[:3])),
                recorded_gripper_observed_pct=observed,
                scene_predicted_tip_gap_from_recorded_opening_m=None if holding else reference_gaps[k + 1],
                requested_gripper_command_pct=float(real_from_sim(env.mapping, control["requested_radians"])[5]),
                applied_gripper_target_pct=float(real_from_sim(env.mapping, control["applied_radians"])[5]),
                sim_gripper_observed_pct=float(env.real_state()[5]),
                requested_native_target=control["requested_radians"],
                applied_native_target=control["applied_radians"],
                gripper_ctrl_native_rad=float(env.data.ctrl[env.actuator_ids[5]]),
                held=m["ball_held"], in_bowl=m["ball_in_bowl"], ball_xyz_task=m["ball_xyz_task"],
                joint_limit_clipped=control["limit_clipped"], joint_rate_limited=control["rate_limited"],
                robot_table_penetration_m=m["robot_table_penetration_m"])
            if rows:
                row["ball_center_motion_relative_to_tool_mps"] = float(
                    np.linalg.norm(np.asarray(row["ball_tool_m"]) - rows[-1]["ball_tool_m"]) * env.control_hz)
            else:
                row["ball_center_motion_relative_to_tool_mps"] = None
            rows.append(row)
            if mode == "fixed_target_hold" and not holding and hold_ready(rows):
                hold_start = k + 1
                fixed_target = np.asarray(control["applied_radians"]).copy()
                fixed_pose = kin.forward(fixed_target)
        losses = [b for a, b in zip(rows[:-1], rows[1:], strict=True) if a["held"] and not b["held"]]
        held_rows = [r for r in rows if r["fixed_hold"]]
        summary = {"mode": mode, "initial_physical_valid": True, "ticks": len(rows),
            "lifted": tracker.lifted, "placement_success": tracker.success if mode == "recorded_motion" else None,
            "hold_anchor_source_time_s": rows[hold_start - 1]["source_time_s"] if hold_start is not None else None,
            "hold_duration_s": len(held_rows) / env.control_hz,
            "hold_end_bilateral": held_rows[-1]["bilateral"] if held_rows else None,
            "hold_loss_elapsed_s": [r["simulation_time_s"] - rows[hold_start - 1]["simulation_time_s"]
                                    for r in losses if r["fixed_hold"]],
            "hold_ball_center_displacement_tool_m": float(np.linalg.norm(
                np.asarray(held_rows[-1]["ball_tool_m"]) - rows[hold_start - 1]["ball_tool_m"])) if held_rows else None,
            "hold_max_reference_pose_error_m": max((r["reference_pose_error_m"] for r in held_rows), default=None),
            "loss_of_hold_times_s": [r["simulation_time_s"] for r in losses],
            "max_robot_table_penetration_m": max(r["robot_table_penetration_m"] for r in rows),
            "limit_clipped_ticks": sum(any(r["joint_limit_clipped"]) for r in rows),
            "rate_limited_ticks": sum(any(r["joint_rate_limited"]) for r in rows)}
        if hold_start is not None:
            anchor = rows[hold_start - 1]
            bilateral = [r for r in held_rows if r["bilateral"]]
            final = bilateral[-1] if bilateral else anchor
            summary.update(
                hold_limit_clipped_ticks=sum(any(r["joint_limit_clipped"]) for r in held_rows),
                hold_rate_limited_ticks=sum(any(r["joint_rate_limited"]) for r in held_rows),
                hold_ball_displacement_tool_before_bilateral_loss_m=float(np.linalg.norm(
                    np.asarray(final["ball_tool_m"]) - anchor["ball_tool_m"])),
                hold_first_nonbilateral_elapsed_s=next((r["simulation_time_s"] - anchor["simulation_time_s"]
                    for r in held_rows if not r["bilateral"]), None),
                hold_anchor_contact=anchor,
                hold_last_bilateral_contact=final)
        if losses:
            first = losses[0]
            i = rows.index(first)
            summary["first_hold_loss_contact_window"] = rows[max(0, i - 10):i + 1]
        return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/so101_root_cause.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/so101_contact_retention_20261019"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    plan = yaml.safe_load(args.plan.read_text())["replay"]
    config = Path(plan["config"])
    paths = [Path(".cache/derived_joint/arrays.npz"), Path(".cache/derived_cartesian/arrays.npz")]
    arrays = []
    for p in paths:
        manifest = json.loads((p.parent / "manifest.json").read_text())
        assert hash_file(p) == manifest["files_sha256"]["arrays.npz"]
        arrays.append(dict(np.load(p, allow_pickle=False)))
    joint, cart = arrays
    for key in ("episode_index", "timestamp", "frame_index"):
        np.testing.assert_array_equal(joint[key], cart[key])
    cfg = yaml.safe_load(config.read_text())
    sources = paths + [args.plan, config, Path(cfg["task_config"]), Path(cfg["model"]),
        Path(plan["reference_report"]), Path(__file__), Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/base_env.py")]
    hashes = {str(p): hash_file(p) for p in sources}
    reference = json.loads(Path(plan["reference_report"]).read_text())
    report = {"kind": "JAW_CONTACT_RETENTION_DIAGNOSTIC_NOT_CALIBRATION_OR_POLICY", "runs": [],
        "source_sha256": hashes, "common_ball_offset_mm": plan["common_ball_offset_mm"],
        "protocol": {"physics_safety_geometry_unchanged": True, "ball_dynamic": True,
            "hold_rule": "Freeze last bounded native arm and gripper targets after five consecutive bilateral samples above30mm, for20s.",
            "sampling": "10Hz after existing env.step mj_forward; no solver mutation or substep event timing claim.",
            "friction_utilization": "Tangential sliding components only; not a full friction-cone or grasp-wrench certificate.",
            "limitations": "Outcome-selected exploratory geometry; unverified gripper command provenance; nominal source time; historical validation diagnostic only.",
            "physical_grasp_gate_passed": False}}
    args.out.mkdir(parents=True)
    (args.out / "executed_script.py").write_text(Path(__file__).read_text())
    for episode in plan["episodes"]:
        indices = np.flatnonzero(joint["episode_index"] == episode)
        baseline_rows = None
        for mode in ("recorded_motion", "fixed_target_hold"):
            result, rows = replay(config, joint, cart, indices, reference["episodes"][str(episode)],
                                  plan["common_ball_offset_mm"], mode)
            result["episode"] = episode
            if mode == "recorded_motion":
                baseline_rows = rows
                old_path = Path("outputs/so101_root_cause_20261017") / f"episode{episode}_joint_reference_trace.json"
                previous = json.loads(old_path.read_text())["rows"]
                assert len(previous) == len(rows)
                difference = max(float(np.max(np.abs(np.asarray(a["ball_xyz_task"]) - b["ball_xyz_task"])))
                                 for a, b in zip(previous, rows, strict=True))
                assert difference < 1e-9, difference
                result["existing_replay_max_ball_difference_m"] = difference
            else:
                prefix = [r for r in rows if not r["fixed_hold"]]
                difference = max(float(np.max(np.abs(np.asarray(a["ball_xyz_task"]) - b["ball_xyz_task"])))
                                 for a, b in zip(baseline_rows[:len(prefix)], prefix, strict=True))
                assert difference < 1e-12
                result["matched_prefix_max_ball_difference_m"] = difference
            report["runs"].append(result)
            atomic_json(args.out / f"episode{episode}_{mode}_trace.json", {"rows": rows})
            atomic_json(args.out / "report.json", report)
            print(json.dumps({k: v for k, v in result.items() if "contact" not in k}, allow_nan=False), flush=True)
    for path, expected in hashes.items():
        assert hash_file(Path(path)) == expected, path
    atomic_json(args.out / "COMPLETE.json", {"runs": len(report["runs"]), "policy_success_established": False})


if __name__ == "__main__":
    main()
