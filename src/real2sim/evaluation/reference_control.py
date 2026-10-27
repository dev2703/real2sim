"""Geometry-consistent recorded and stationary control diagnostics for SO-101.

These simulation diagnostics preserve source arrays and native controller limits. They
compare joint replay, Cartesian deltas and pose feedback without evaluating a learned policy.
Reusable controls live here independently of dated experiment drivers and queue machinery.
"""

from __future__ import annotations

import copy
from collections import Counter
from contextlib import contextmanager

import mujoco
import numpy as np

from real2sim.evaluation.metrics import TrialTracker
from real2sim.geometry.so3 import delta_action
from real2sim.policy.adapters import CartesianActionAdapter
from real2sim.sim.env import GRIPPER_BODIES, So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics

MODES = ("joint_reference", "cartesian_delta", "cartesian_reference_feedback",
         "cartesian_persistent_target")
STATIONARY_MODES = ("fixed_joint_target", "zero_delta_current_pose", "fixed_initial_pose_feedback",
                    "zero_delta_accepted_target")


def _describe(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return {'n': 0}
    return {'n': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.quantile(a, .95)), 'max': float(a.max())}


def _phase(time, review):
    if time < review['closure'][0]:
        return 'approach'
    for name, (start, stop) in review.items():
        if start <= time < stop:
            return name
    return 'after_release'


def _jaw(env, data):
    tips = [env.model.geom(f'{name}_jaw_sph_tip1').id for name in ('fixed', 'moving')]
    points = data.geom_xpos[tips].copy()
    gap = np.linalg.norm(points[1] - points[0]) - env.model.geom_size[tips, 0].sum()
    return float(gap), points


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


def reference_for_scene(env, states):
    """Recompute the diagnostic reference consistently; never rewrite training arrays."""
    kin = NativeKinematics(env.model, env.config)
    q = np.array([sim_from_real(env.mapping, state) for state in states])
    poses, gaps = [], []
    for positions in q:
        poses.append(kin.forward(positions))
        gaps.append(_jaw(env, kin.data)[0])
    poses = np.array(poses)
    return q, poses, np.array(gaps), np.array([
        delta_action(a, b) for a, b in zip(poses[:-1], poses[1:], strict=True)])


def _summarize_rows(rows):
    fields = ("reference_pose_error_m", "reference_rotation_error_rad", "movement_error_m",
              "bounded_request_error_m", "servo_error_m", "jaw_gap_error_m",
              "zero_motion_error_m")
    masks = {"all": np.ones(len(rows), dtype=bool),
             "active": np.array([r["active_reference"] for r in rows], dtype=bool)}
    masks["stationary"] = ~masks["active"]
    for value in ("approach", "closure", "lift_transport", "release", "after_release"):
        masks[value] = np.array([r["phase"] == value for r in rows])
    return {name: {field: _describe([r[field] for r, use in zip(rows, mask, strict=True) if use])
                   for field in fields} for name, mask in masks.items()}


def run_trial(config, joint, cart, indices, metadata, review, mode, offset):
    """Replay a shared recorded reference through one bounded control route."""
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
                    "phase": _phase(float(joint["timestamp"][nxt]), review),
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
                    "jaw_gap_error_m": abs(_jaw(env, env.data)[0] - gaps[k + 1]),
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
            "strata": _summarize_rows(rows), "contact": contact, "loss_of_hold_events": losses,
            "candidate_minus_old_translation_label_m": _describe(np.linalg.norm(
                deltas[:, :3] - cart["action"][indices[:-1], :3], axis=1))}
        return summary, rows


def probe(config, state, mode, ticks=200):
    """Measure unloaded stationary drift without changing native controller bounds."""
    if mode not in STATIONARY_MODES:
        raise ValueError(mode)
    with So101PickPlaceEnv(config, render_images=False) as env:
        q = sim_from_real(env.mapping, state)
        if np.any(q < env.limits[:, 0]) or np.any(q > env.limits[:, 1]):
            raise ValueError("Stationary hold requires an in-limit reference; do not clip it")
        env.max_episode_steps = ticks
        env.reset(seed=0, scenario={"joint_positions": q, "ball_xy": [.5, .3]})
        initial = env.info()["privileged_metrics"]
        if initial["robot_table_penetration_m"] > env.config["ik"]["penetration_tolerance_m"]:
            raise ValueError("Invalid stationary-hold start")
        initial_pose = env.ee_pose().copy()
        cfg = copy.deepcopy(env.config)
        cfg["ik"].update(position_convergence_m=1e-5, orientation_convergence_rad=1e-4)
        adapter = CartesianActionAdapter(env, NativeKinematics(env.model, cfg), target_reference=(
            "accepted_target" if mode == "zero_delta_accepted_target" else "current_pose"))
        rows = []
        for tick in range(ticks):
            before = env.ee_pose()
            if mode == "fixed_joint_target":
                _, _, _, _, info = env.step(q)
                rejected = False
            else:
                requested = (delta_action(before, initial_pose)
                             if mode == "fixed_initial_pose_feedback" else np.zeros(6))
                result = adapter.apply(np.r_[requested, state[5] / 100])
                info, rejected = result.info, result.ik_rejected
            error = delta_action(initial_pose, env.ee_pose())
            metrics = info["privileged_metrics"]
            rows.append({"tick": tick + 1, "translation_drift_m": float(np.linalg.norm(error[:3])),
                "rotation_drift_rad": float(np.linalg.norm(error[3:])), "ik_rejected": rejected,
                "native_target_drift_rad": float(np.max(abs(np.array(info["control"]["applied_radians"]) - q))),
                "joint_limit_clipped": info["control"]["limit_clipped"],
                "joint_rate_limited": info["control"]["rate_limited"],
                "robot_table_penetration_m": metrics["robot_table_penetration_m"],
                "ball_held": metrics["ball_held"],
                "ball_non_table_contact": bool(env._ball_contacts() - {env.table_geom})})
        return {"mode": mode, "ticks": ticks, "seconds": ticks / env.control_hz,
            "initial_reference": "Measured initial simulated pose, shared by every route; no future recorded targets",
            "end_drift_mm": rows[-1]["translation_drift_m"] * 1000,
            "max_drift_mm": max(r["translation_drift_m"] for r in rows) * 1000,
            "ik_rejections": sum(r["ik_rejected"] for r in rows),
            "ball_non_table_contact_ticks": sum(r["ball_non_table_contact"] for r in rows),
            "max_robot_table_penetration_m": max(r["robot_table_penetration_m"] for r in rows),
            "rows": rows}


def longest_valid_window(qpos, limits, min_ticks=30):
    """Return [start, stop) for the longest contiguous valid run; earliest wins a tie."""
    qpos, limits = np.asarray(qpos), np.asarray(limits)
    valid = (np.isfinite(qpos).all(axis=1)
             & (qpos >= limits[:, 0]).all(axis=1) & (qpos <= limits[:, 1]).all(axis=1))
    edges = np.diff(np.r_[False, valid, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts) == 0:
        raise ValueError("No in-limit window")
    selected = int(np.argmax(stops - starts))
    start, stop = int(starts[selected]), int(stops[selected])
    if stop - start - 1 < min_ticks:
        raise ValueError("No sufficiently long in-limit motion window")
    return start, stop

