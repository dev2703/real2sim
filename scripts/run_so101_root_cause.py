"""Matched recorded-reference replay separating Cartesian execution from contact failure.

Simulation only: no training, hardware, threshold changes or automatic gate promotion.
"""

from __future__ import annotations

import argparse
import copy
import json
import signal
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from diag_geometry_contact_sweep import contact_monitor
from diag_recorded_grasp_control import describe, jaw, phase
from run_so101_experiments import atomic_json, hash_file, now, queue_lock, read_json

from real2sim.evaluation.metrics import TrialTracker
from real2sim.geometry.so3 import delta_action
from real2sim.policy.adapters import CartesianActionAdapter
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics

MODES = ("joint_reference", "cartesian_delta", "cartesian_reference_feedback",
         "cartesian_persistent_target")


def reference_for_scene(env, states):
    """Recompute the diagnostic reference consistently; never rewrite training arrays."""
    kin = NativeKinematics(env.model, env.config)
    q = np.array([sim_from_real(env.mapping, state) for state in states])
    poses, gaps = [], []
    for positions in q:
        poses.append(kin.forward(positions))
        gaps.append(jaw(env, kin.data)[0])
    poses = np.array(poses)
    return q, poses, np.array(gaps), np.array([
        delta_action(a, b) for a, b in zip(poses[:-1], poses[1:], strict=True)])


def summarize_rows(rows):
    fields = ("reference_pose_error_m", "reference_rotation_error_rad", "movement_error_m",
              "bounded_request_error_m", "servo_error_m", "jaw_gap_error_m",
              "zero_motion_error_m")
    masks = {"all": np.ones(len(rows), dtype=bool),
             "active": np.array([r["active_reference"] for r in rows], dtype=bool)}
    masks["stationary"] = ~masks["active"]
    for value in ("approach", "closure", "lift_transport", "release", "after_release"):
        masks[value] = np.array([r["phase"] == value for r in rows])
    return {name: {field: describe([r[field] for r, use in zip(rows, mask, strict=True) if use])
                   for field in fields} for name, mask in masks.items()}


def run_trial(config, joint, cart, indices, metadata, review, mode, offset):
    if mode not in MODES:
        raise ValueError(mode)
    with So101PickPlaceEnv(config, render_images=False) as env:
        env.max_episode_steps = len(indices) - 1
        qrefs, poses, gaps, deltas = reference_for_scene(env, joint["state"][indices])
        scenario = {"joint_positions": np.clip(qrefs[0], *env.limits.T),
                    "ball_xy": np.asarray(metadata["ball_xy"]) + np.asarray(offset) / 1000,
                    "bowl_xy": metadata["bowl_xy"]}
        _, info = env.reset(seed=0, scenario=scenario)
        initial = info["privileged_metrics"]
        initial_contacts = [c for c in initial["contacts"]
            if "ball_geom" in (c["geom1"], c["geom2"]) and "table_geom" not in (c["geom1"], c["geom2"])]
        initial_valid = not initial_contacts and initial["robot_table_penetration_m"] <= env.config["ik"]["penetration_tolerance_m"]
        cfg = copy.deepcopy(env.config)
        cfg["ik"].update(position_convergence_m=1e-5, orientation_convergence_rad=1e-4)
        # Numerical convergence changes only. Acceptance/collision/velocity bounds stay identical.
        for key, value in env.config["ik"].items():
            if key not in ("position_convergence_m", "orientation_convergence_rad"):
                assert cfg["ik"][key] == value
        kin = NativeKinematics(env.model, cfg)
        adapter = CartesianActionAdapter(env, kin, target_reference=(
            "accepted_target" if mode == "cartesian_persistent_target" else "current_pose"))
        initial_reference_error = delta_action(env.ee_pose(), poses[0])
        tracker = TrialTracker(int(env.task["episode"]["success_hold_s"] * env.control_hz),
                               .03, env.table_top, env.task["ball"]["radius"])
        rows, losses = [], []
        previously_held = False
        with contact_monitor(env) as contact:
            for k, (current, nxt) in enumerate(zip(indices[:-1], indices[1:], strict=True)):
                before = env.ee_pose()
                requested = (delta_action(before, poses[k + 1])
                             if mode == "cartesian_reference_feedback" else deltas[k].copy())
                grip = float(cart["action"][current, 6])
                if mode == "joint_reference":
                    target = joint["state"][nxt].copy()
                    target[5] = grip * 100
                    _, _, _, _, info = env.step_real_joint_targets(target)
                    rejected, reason, iterations = False, "joint_control", 0
                else:
                    applied = adapter.apply(np.r_[requested, grip])
                    info, rejected = applied.info, applied.ik_rejected
                    result = adapter.residuals[-1]
                    reason, iterations = result["reason"], result["iterations"]
                after = env.ee_pose()
                achieved = delta_action(before, after)
                residual = delta_action(after, poses[k + 1])
                bounded = requested.copy()
                for section, limit in ((slice(0, 3), adapter.max_step_m), (slice(3, 6), adapter.max_step_rad)):
                    bounded[section] *= min(1., limit / max(np.linalg.norm(bounded[section]), 1e-12))
                bounded_action = bounded.copy()
                if mode != "joint_reference":
                    bounded = np.asarray(applied.detail["bounded_feedback"])
                metrics, control = info["privileged_metrics"], info["control"]
                tracker.update(metrics, ik_rejected=rejected)
                applied_pose = kin.forward(np.array(control["applied_radians"]))
                row = {"tick": k + 1, "source_tick": int(current),
                    "frame_index": int(joint["frame_index"][nxt]), "time_s": float(joint["timestamp"][nxt]),
                    "phase": phase(float(joint["timestamp"][nxt]), review),
                    "active_reference": bool(np.linalg.norm(deltas[k, :3]) > .001 or np.linalg.norm(deltas[k, 3:]) > .01),
                    "recorded_delta": deltas[k].tolist(), "requested_delta": requested.tolist(),
                    "bounded_delta_applied_by_cartesian_adapter": mode != "joint_reference",
                    "bounded_action_delta": bounded_action.tolist(),
                    "bounded_delta": bounded.tolist(), "achieved_delta": achieved.tolist(),
                    "reference_pose_error_m": float(np.linalg.norm(residual[:3])),
                    "reference_rotation_error_rad": float(np.linalg.norm(residual[3:])),
                    "movement_error_m": float(np.linalg.norm(achieved[:3] - deltas[k, :3])),
                    "bounded_request_error_m": float(np.linalg.norm(achieved[:3] - bounded[:3])),
                    "servo_error_m": float(np.linalg.norm(after[:3, 3] - applied_pose[:3, 3])),
                    "zero_motion_error_m": float(np.linalg.norm(deltas[k, :3])),
                    "jaw_gap_error_m": abs(jaw(env, env.data)[0] - gaps[k + 1]),
                    "recorded_gripper_observed_pct": float(joint["state"][nxt, 5]),
                    "recorded_gripper_command_pct": grip * 100,
                    "sim_gripper_observed_pct": float(env.real_state()[5]),
                    "ik_rejected": rejected, "ik_reason": reason, "ik_iterations": iterations,
                    "cartesian_control": applied.detail if mode != "joint_reference" else None,
                    "joint_limit_clipped": control["limit_clipped"], "joint_rate_limited": control["rate_limited"],
                    "held": metrics["ball_held"], "lifted": metrics["lifted"], "in_bowl": metrics["ball_in_bowl"],
                    "ball_xyz_task": metrics["ball_xyz_task"], "tool_xyz_task": metrics["tool_xyz_task"],
                    "robot_table_penetration_m": metrics["robot_table_penetration_m"]}
                if previously_held and not metrics["ball_held"]:
                    losses.append(row)
                previously_held = metrics["ball_held"]
                rows.append(row)
        limit_difference = abs(qrefs - np.clip(qrefs, *env.limits.T))
        summary = {"mode": mode, "ticks": len(rows), "initial_physical_valid": initial_valid,
            "initial_reference_pose_error_m": float(np.linalg.norm(initial_reference_error[:3])),
            "initial_reference_rotation_error_rad": float(np.linalg.norm(initial_reference_error[3:])),
            "target_reference": adapter.target_reference if mode != "joint_reference" else "recorded_joints",
            "initial_reference_outside_joint_limits": bool(np.any(limit_difference[0] > 1e-8)),
            "initial_robot_table_penetration_m": initial["robot_table_penetration_m"],
            "initial_non_table_ball_contacts": initial_contacts,
            "reference_outside_joint_limits_ticks": int(np.any(limit_difference > 1e-8, axis=1).sum()),
            "max_reference_limit_violation_rad": float(limit_difference.max()),
            "success": tracker.success, "lifted": tracker.lifted, "held": tracker.held_ever,
            "phase_failed": tracker.phase_failed(), "ik_reasons": dict(Counter(r["ik_reason"] for r in rows)),
            "limit_clipped_ticks": sum(any(r["joint_limit_clipped"]) for r in rows),
            "rate_limited_ticks": sum(any(r["joint_rate_limited"]) for r in rows),
            "accepted_no_motion_ticks": int(sum(not r["ik_rejected"] and np.linalg.norm(r["achieved_delta"]) < 1e-8
                                                 and r["active_reference"] for r in rows)),
            "strata": summarize_rows(rows), "contact": contact, "loss_of_hold_events": losses,
            "candidate_minus_old_translation_label_m": describe(np.linalg.norm(
                deltas[:, :3] - cart["action"][indices[:-1], :3], axis=1))}
        return summary, rows


def run(plan, out):
    replay = plan["replay"]
    joint_manifest = read_json(Path(".cache/derived_joint/manifest.json"))
    cart_manifest = read_json(Path(".cache/derived_cartesian/manifest.json"))
    paths = [Path(".cache/derived_joint/arrays.npz"), Path(".cache/derived_cartesian/arrays.npz")]
    for path, manifest in zip(paths, (joint_manifest, cart_manifest), strict=True):
        assert hash_file(path) == manifest["files_sha256"]["arrays.npz"]
    joint, cart = (dict(np.load(path, allow_pickle=False)) for path in paths)
    for key in ("episode_index", "frame_index", "timestamp"):
        np.testing.assert_array_equal(joint[key], cart[key])
    config = Path(replay["config"])
    config_data = yaml.safe_load(config.read_text())
    task = Path(config_data["task_config"])
    sources = paths + [config, task, Path(__file__), Path(replay["reference_report"]),
        Path("scripts/diag_geometry_contact_sweep.py"), Path("scripts/diag_recorded_grasp_control.py"),
        Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/kinematics.py"),
        Path("src/real2sim/policy/adapters.py"), Path("src/real2sim/evaluation/metrics.py")]
    hashes = {str(path): hash_file(path) for path in sources}
    atomic_json(out / "MANIFEST.json", {"plan": plan, "source_sha256": hashes,
        "contract": "Scene FK recomputed from recorded follower states for every route; same recorded gripper command in all routes. Old training data/normalizers/weights remain unchanged.",
        "caution": "Exploratory geometry and outcome-selected common offset, nominal source timing and unverified gripper provenance. Diagnostic results cannot validate calibration or learned policy."})
    (out / "executed_script.py").write_text(Path(__file__).read_text())
    reference = read_json(Path(replay["reference_report"]))
    report = {"kind": "MATCHED_REFERENCE_CONTROL_DIAGNOSTIC_NOT_POLICY", "runs": [],
              "physical_grasp_gate_passed": False, "panda_allowed": False}
    for ep in replay["episodes"]:
        assert ep in joint_manifest["split"]["val"]
        indices = np.flatnonzero(joint["episode_index"] == ep)
        for mode in replay["modes"]:
            atomic_json(out / "RUNNING.json", {"at": now(), "episode": ep, "mode": mode,
                                              "completed_runs": len(report["runs"])})
            summary, rows = run_trial(config, joint, cart, indices, reference["episodes"][str(ep)],
                                     reference["review"][str(ep)], mode, replay["common_ball_offset_mm"])
            summary["episode"] = ep
            report["runs"].append(summary)
            atomic_json(out / f"episode{ep}_{mode}_trace.json", {"rows": rows})
            atomic_json(out / "report.json", report)
            print(json.dumps({k: summary[k] for k in ("episode", "mode", "ticks", "initial_physical_valid",
                                                      "lifted", "success", "ik_reasons")}), flush=True)
    for path, expected in hashes.items():
        assert hash_file(Path(path)) == expected, f"Input changed during replay: {path}"
    report["complete"] = True
    report["note"] = "Compare each control route by phase, including initial reference mismatch and rejected targets. Persistent targets use only current pose at initialization and accepted action history, never future recorded FK. Common geometry is unvalidated; no gate opens automatically."
    atomic_json(out / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/so101_root_cause.yaml"))
    args = parser.parse_args()
    plan = yaml.safe_load(args.plan.read_text())
    out = Path(plan["out"])
    def interrupted(signum, _frame):
        raise RuntimeError(f"Interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    with queue_lock(Path(plan["previous_queue"])), queue_lock(out):
        if (out / "COMPLETE.json").exists():
            print("Targeted replay already complete; refusing duplicate work.")
            return
        if (out / "MANIFEST.json").exists():
            raise FileExistsError("Existing partial replay retained; use a reviewed new output path to retry")
        atomic_json(out / "RUNNING.json", {"at": now(), "stage": "checking_inputs"})
        try:
            run(plan, out)
            atomic_json(out / "COMPLETE.json", {"at": now(), "status": "diagnostic_complete",
                "runs": len(plan["replay"]["episodes"]) * len(plan["replay"]["modes"]),
                "grasp_success": "not_established", "panda_allowed": False})
        except BaseException as error:
            atomic_json(out / "FAILED.json", {"at": now(), "error": repr(error),
                "traceback": traceback.format_exc(), "panda_allowed": False})
            raise
        finally:
            (out / "RUNNING.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
