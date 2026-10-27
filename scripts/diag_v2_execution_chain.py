"""Trace bounded policy requests through unchanged IK, native control and physics.

Recorded images supply one two-action chunk at each frozen training-data start.
This is a robot-response diagnostic, not a task rollout or a physical timing fit.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch

from real2sim.config import load_yaml
from real2sim.data.raw import RawDataset, sha256_file
from real2sim.evaluation.run import check_contract_timing_and_image
from real2sim.geometry.so3 import delta_action
from real2sim.policy.act import load_checkpoint
from real2sim.policy.adapters import CartesianActionAdapter, ObservationAdapter
from real2sim.serialization import write_json
from real2sim.sim.contact_solver import contact_solver_settings
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics

EPISODES = (11, 8, 23)
MODES = ("policy_prefix", "recorded_achieved_delta_reference", "hold")
LAGS = np.arange(-10, 21)


def stats(values):
    values = np.asarray(values, float)
    if not values.size:
        return {"n": 0}
    if not np.isfinite(values).all():
        raise ValueError("nonfinite summary values")
    return {"n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "min": float(values.min()), "max": float(values.max())}


def select_anchors(arrays, split, per_episode=12):
    if not 1 <= per_episode <= 12:
        raise ValueError("use one to twelve predeclared starts per episode")
    if (not set(EPISODES) <= set(split["train"])
            or set(EPISODES) & (set(split["val"]) | set(split["reserved"]))):
        raise ValueError("execution starts must be the three declared training episodes")
    selected = []
    for episode in EPISODES:
        ticks = np.flatnonzero(arrays["episode_index"] == episode)
        valid = []
        for tick in ticks:
            if tick < ticks[0] + 1 or tick + 2 > ticks[-1] or (tick - ticks[0]) % 2:
                continue
            if any(np.any(arrays.get(key, np.zeros(len(arrays["action"]), bool))[tick:tick + 2])
                   for key in ("label_is_extrapolated", "action_is_invalid", "observation_is_invalid")):
                continue
            valid.append(int(tick))
        offsets = np.unique(np.linspace(0, len(valid) - 1, min(per_episode, len(valid))).astype(int))
        selected.extend({"episode": episode, "tick": valid[i],
                         "source_frame": int(arrays["frame_index"][valid[i]])} for i in offsets)
    return selected


def native_window_status(window, limits):
    """Reject out-of-range observed/reference poses without modifying their values."""
    window, limits = np.asarray(window, float), np.asarray(limits, float)
    if window.shape != (3, 6) or limits.shape != (6, 2) or not np.isfinite(window).all():
        raise ValueError("three finite source joint vectors and six native ranges required")
    outside = (window < limits[:, 0]) | (window > limits[:, 1])
    if outside[0].any():
        return "observed_native_joint_limit"
    if outside[1:].any():
        return "reference_native_joint_limit"
    return None


def descriptive_lags(states, actions, lags=LAGS, threshold=.2):
    """One common causal-reference support for every lag; no compensating shift."""
    states, actions, lags = np.asarray(states, float), np.asarray(actions, float), np.asarray(lags, int)
    if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 6:
        raise ValueError("aligned six-joint recorded states/actions required")
    low, high = max(1, -int(lags.min())), len(states) - max(0, int(lags.max()))
    reference = np.arange(low, high)
    if not len(reference):
        raise ValueError("recording too short for the declared lag support")
    moving = ((abs(states[reference] - states[reference - 1]) > threshold)
              | (abs(actions[reference] - actions[reference - 1]) > threshold))
    counts = moving.sum(0)
    errors = []
    for lag in lags:
        difference = abs(actions[reference] - states[reference + lag])
        errors.append([float(difference[moving[:, j], j].mean()) if counts[j] else None for j in range(6)])
    best = []
    for j, count in enumerate(counts):
        if not count:
            best.append(None)
            continue
        index = int(np.argmin([row[j] for row in errors]))
        best.append({"lag_frames": int(lags[index]), "mae_in_joint_units": errors[index][j],
                     "at_search_boundary": index in (0, len(lags) - 1)})
    return {"reference_ticks": len(reference), "moving_components_per_joint": counts,
            "lag_frames": lags, "moving_mae_by_lag": errors, "best_lag_descriptive_only": best,
            "positive_lag_definition": "Recorded leader command at k is compared to follower readback at k+lag.",
            "interpretation": "Descriptive correlations combine response, pose offsets and operator motion. They do not identify sensor/acquisition latency or sent-servo command timing; no shift is adopted."}


def clock_audit(raw, arrays, selected, camera):
    output = []
    hashes = {}
    for episode in EPISODES:
        rows = raw.episode_rows(episode)
        states, actions = (raw.stack(rows, key) for key in ("observation.state", "action"))
        segment = raw.video_segment(camera, episode)
        hashes[str(segment.path)] = sha256_file(segment.path)
        positions, pts = [], []
        for position, timestamp, _ in raw.iter_frames(camera, episode):
            positions.append(position)
            pts.append(timestamp)
        pts, positions = np.asarray(pts), np.asarray(positions)
        source_clock = rows["timestamp"].to_numpy(float)
        frame_index = rows["frame_index"].to_numpy(int)
        ticks = np.flatnonzero(arrays["episode_index"] == episode)
        requested = [r for r in selected if r["episode"] == episode]
        decoded = {int(p): float(t) for p, t in zip(positions, pts, strict=True)}
        associations = [{"derived_tick": row["tick"], "source_frame": row["source_frame"],
                         "derived_timestamp_s": float(arrays["timestamp"][row["tick"]]),
                         "encoded_video_pts_relative_s": decoded[row["source_frame"]] - segment.from_ts}
                        for row in requested]
        count = min(len(pts), len(source_clock))
        output.append({"episode": episode, "raw_rows": len(rows), "decoded_video_frames": len(pts),
                       "nominal_raw_fps": raw.fps, "source_row_interval_s": stats(np.diff(source_clock)),
                       "encoded_pts_interval_s": stats(np.diff(pts)),
                       "derived_row_interval_s": stats(np.diff(arrays["timestamp"][ticks])),
                       "row_clock_minus_frame_index_over_nominal_fps_s": stats(source_clock - frame_index / raw.fps),
                       "encoded_pts_minus_row_clock_s": stats(pts[:count] - segment.from_ts - source_clock[:count]),
                       "selected_frame_associations": associations,
                       "leader_follower_descriptive_lag": descriptive_lags(states, actions)})
    return {"episodes": output, "video_sha256": hashes,
            "physical_clock_alignment_established": False,
            "physical_acquisition_intervals_available": False,
            "sent_or_applied_hardware_commands_available": False,
            "interpretation": "Container PTS, source frame numbers and nominal row timestamps are audited as recorded bookkeeping. Agreement does not establish when photons/readbacks were acquired or commands reached physical servos."}


def initialize(env, arrays, tick):
    q = sim_from_real(env.mapping, arrays["joint_state"][tick])
    env.reset(seed=0, scenario={"joint_positions": q})
    previous = sim_from_real(env.mapping, arrays["joint_state"][tick - 1])
    dt = float(arrays["timestamp"][tick] - arrays["timestamp"][tick - 1])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("causal source clock must advance")
    env.data.qvel[env.vadr] = (q - previous) / dt
    mujoco.mj_forward(env.model, env.data)
    return q, env.data.qvel[env.vadr].copy()


def dependency_ledger(sim_path, eval_path, checkpoint, data):
    sim = load_yaml(sim_path)
    files = [Path(__file__), sim_path, eval_path, data / "manifest.json", data / "stats.json",
             *(checkpoint / p for p in ("model.safetensors", "config.json", "stats.json", "run_info.json", "real2sim_contract.json")),
             *(Path(sim[key]) for key in ("model", "manifest", "task_config", "data_config")),
             *(Path("src/real2sim") / p for p in ("artifacts.py", "serialization.py", "policy/act.py", "policy/adapters.py", "policy/attention.py",
                "sim/env.py", "sim/base_env.py", "sim/kinematics.py", "sim/model.py", "geometry/so3.py", "geometry/so101_fk.py", "data/raw.py"))]
    return {str(p.resolve()): sha256_file(p) for p in files}


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads((args.data / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or sha256_file(args.data / name) != expected:
            raise ValueError(f"data payload differs: {name}")
    if manifest["action_representation"] != "cartesian" or len(manifest["state_names"]) != 13:
        raise ValueError("canonical Cartesian checkpoint/data required")
    arrays = dict(np.load(args.data / "arrays.npz", allow_pickle=False))
    selected = select_anchors(arrays, manifest["split"], args.per_episode)
    before = dependency_ledger(args.sim_config, args.eval_config, args.checkpoint, args.data)
    args.out.mkdir(parents=True)
    (args.out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    plan = {"schema": "bounded_execution_chain_plan_v1", "episodes": list(EPISODES),
            "starts": selected, "actions_per_start": 2, "modes": MODES, "source_sha256": before,
            "initialization": "Observed source joints; causal previous-readback finite-difference velocity. Previous native actuator command is reset to observed joints because physical applied-command history is unavailable.",
            "camera_input": "One recorded source image at each anchor, one chunk inference. Second action is queued; no simulator object state is fed to policy.",
            "reserved_episodes_used": [], "promotion_development_rollouts": False}
    write_json(args.out / "plan.json", plan)
    started = time.perf_counter()
    policy, norm, contract = load_checkpoint(args.checkpoint, "cpu")
    load_ms = (time.perf_counter() - started) * 1000
    if (contract["derived_files_sha256"] != manifest["files_sha256"]
            or norm.stats != json.loads((args.data / "stats.json").read_text())
            or contract["state_names"] != manifest["state_names"]
            or contract["camera_keys"] != manifest["camera_keys"]
            or policy.config.n_action_steps != 2):
        raise ValueError("checkpoint/source or execution contract differs")
    images = {key: np.load(args.data / path, mmap_mode="r") for key, path in manifest["image_files"].items()}
    config = load_yaml(args.eval_config)["cartesian"]
    report = {"schema": "bounded_execution_chain_v1", "plan_sha256": sha256_file(args.out / "plan.json"),
              "physical_calibration_accepted": False, "checkpoint_promoted": False,
              "reserved_episodes_used": [], "task_success_evaluated": False,
              "development_rollouts_run": False, "model_load_ms": load_ms,
              "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
              "cases": [], "excluded": [],
              "limitations": ["Two physics steps per independent source start are controller-response diagnostics, not complete-task development rollouts.",
                  "Recorded images and unmatched provisional simulated objects are used. No task, grasp or rendered appearance accuracy is established.",
                  "Causal initial joint velocity is estimated from sampled readbacks; original applied servo target/history and physical response are unknown.",
                  "All Cartesian metrics depend on provisional FK. Native/scene exclusions do not prove real-recording collisions.",
                  "Wall timing covers cached-observation processing, model inference and simulator control/physics; acquisition, transport, live rendering and hardware actuation are unmeasured.",
                  "The unchanged adapter may bound model actions, native gripper commands and soft-limit overshoot IK seeds; raw observed references are never clipped for admission."]}
    with So101PickPlaceEnv(args.sim_config, render_images=False) as env:
        check_contract_timing_and_image(contract, env)
        report["simulation_contract"] = {"native_limits_rad": env.limits, "control_hz": env.control_hz,
            "physics_hz": env.config["physics_hz"], "ik": env.config["ik"],
            "contact_solver": contact_solver_settings(env.model), "rendering_measured": False}
        for source in selected:
            tick = source["tick"]
            window = np.stack([sim_from_real(env.mapping, arrays["joint_state"][t]) for t in range(tick, tick + 3)])
            reason = native_window_status(window, env.limits)
            if reason:
                report["excluded"].append({**source, "reason": reason, "raw_native_joint_window": window})
                continue
            try:
                q, qvel = initialize(env, arrays, tick)
            except ValueError as exc:
                report["excluded"].append({**source, "reason": "strict_source_reset_rejected", "detail": str(exc)})
                continue
            observer = ObservationAdapter(contract, norm, device="cpu")
            class PreviousPose:
                def ee_pose(self, tick=tick):
                    pose = np.eye(4)
                    pose[:3, 3] = arrays["state"][tick - 1, :3]
                    return pose
            observer.advance_tick(PreviousPose())
            t0 = time.perf_counter()
            obs = env.observation()
            obs.update({key: np.asarray(image[tick]).transpose(2, 0, 1).astype(np.float32) / 255
                        for key, image in images.items()})
            t1 = time.perf_counter()
            batch = observer.batch(obs, env)
            t2 = time.perf_counter()
            actual = observer.state(env)
            if not np.allclose(actual, arrays["state"][tick], atol=1e-5, rtol=1e-5):
                raise ValueError("reconstructed causal source proprioception differs")
            with torch.inference_mode():
                actions = norm.unaction(policy.predict_action_chunk(batch)).cpu().numpy()[0, :2]
            t3 = time.perf_counter()
            if not np.isfinite(actions).all():
                raise ValueError("nonfinite predicted prefix")
            case = {**source, "source_timestamp_s": float(arrays["timestamp"][tick]),
                    "initial_q_rad": q, "causal_initial_qvel_rad_s": qvel,
                    "source_proprio_max_abs_error": float(np.max(np.abs(actual - arrays["state"][tick]))),
                    "cached_observation_ms": (t1 - t0) * 1000, "preprocessing_ms": (t2 - t1) * 1000,
                    "inference_ms": (t3 - t2) * 1000, "first_model_call": not report["cases"], "modes": {}}
            hold = np.zeros_like(actions)
            hold[:, 6] = arrays["state"][tick, -1]
            for mode, target_actions in zip(MODES, (actions, arrays["action"][tick:tick + 2], hold), strict=True):
                initialize(env, arrays, tick)
                ik = NativeKinematics(env.model, env.config)
                adapter = CartesianActionAdapter(env, ik, max_step_m=config["max_step_m"],
                    max_step_rad=config["max_step_rad"], target_reference=config["target_reference"])
                traced, calls = [], {}
                original_step, original_solve = env.step, ik.solve
                def measured_step(action, step=original_step, calls=calls):
                    began = time.perf_counter()
                    value = step(action)
                    calls["environment_step_ms"] = (time.perf_counter() - began) * 1000
                    return value
                def measured_solve(*a, solve=original_solve, calls=calls, **kw):
                    began = time.perf_counter()
                    value = solve(*a, **kw)
                    calls["ik_solve_ms"] = (time.perf_counter() - began) * 1000
                    calls["ik_result_native_q_rad"] = value.qpos.copy()
                    return value
                env.step, ik.solve = measured_step, measured_solve
                try:
                    for offset, action in enumerate(target_actions):
                        before_pose, before_q = env.ee_pose(), env.sim_state()
                        old_reference = adapter.accepted_target
                        began = time.perf_counter()
                        result = adapter.apply(action)
                        total_ms = (time.perf_counter() - began) * 1000
                        achieved_pose, after_q = env.ee_pose(), env.sim_state()
                        applied = np.asarray(result.info["control"]["applied_radians"])
                        command_pose = ik.forward(applied)
                        reference_q = window[offset + 1]
                        reference_pose = ik.forward(reference_q)
                        request = np.asarray(result.detail["bounded_target_pose"])
                        row = {"offset": offset, "source_tick": tick + offset,
                            "simulated_time_s": float(env.data.time), "raw_action": action.copy(),
                            "native_before_rad": before_q, "native_after_rad": after_q,
                            "native_ctrl_before_rad": (q if offset == 0 else traced[-1]["native_control"]["applied_radians"]),
                            "raw_reference_next_native_rad": reference_q,
                            "source_leader_gripper_command_percent": float(arrays["joint_action"][tick + offset, 5]),
                            "requested_gripper_percent": float(action[6] * 100),
                            "achieved_gripper_percent": float(env.real_state()[5]),
                            "previous_accepted_target_pose": old_reference,
                            "adapter": result.detail, "native_control": result.info["control"],
                            "ik_rejected": result.ik_rejected, "saturation": result.saturation,
                            "ik_seed_soft_limit_bound_required": bool(np.any(before_q < env.limits[:, 0]) or np.any(before_q > env.limits[:, 1])),
                            "achieved_tool_motion_m_rad": delta_action(before_pose, achieved_pose),
                            "applied_command_fk_motion_m_rad": delta_action(before_pose, command_pose),
                            "command_to_request_translation_error_mm": float(np.linalg.norm(command_pose[:3, 3] - request[:3, 3]) * 1000),
                            "achieved_to_request_translation_error_mm": float(np.linalg.norm(achieved_pose[:3, 3] - request[:3, 3]) * 1000),
                            "achieved_to_applied_command_translation_error_mm": float(np.linalg.norm(achieved_pose[:3, 3] - command_pose[:3, 3]) * 1000),
                            "achieved_to_recorded_next_translation_error_mm": float(np.linalg.norm(achieved_pose[:3, 3] - reference_pose[:3, 3]) * 1000),
                            "robot_table_penetration_m": result.info["privileged_metrics"]["robot_table_penetration_m"],
                            "timing_ms": {**calls, "adapter_and_step_total_ms": total_ms,
                                "controller_excluding_environment_step_ms": max(0., total_ms - calls["environment_step_ms"])}}
                        traced.append(row)
                finally:
                    env.step, ik.solve = original_step, original_solve
                case["modes"][mode] = traced
            report["cases"].append(case)
            write_json(args.out / "report.json", report)
            print(f"ep{source['episode']} frame{source['source_frame']}: "
                  f"policy rejections={sum(r['ik_rejected'] for r in case['modes']['policy_prefix'])}", flush=True)
    raw = RawDataset(Path(manifest["source"]["root"]))
    report["recording_clocks"] = clock_audit(raw, arrays, selected, contract["camera_keys"][0])
    report["summary"] = {"preselected_starts": len(selected), "executed_starts": len(report["cases"]),
        "excluded_starts": dict(Counter(r["reason"] for r in report["excluded"])),
        "warm_inference_ms": stats([c["inference_ms"] for c in report["cases"] if not c["first_model_call"]]),
        "cold_first_inference_ms": next((c["inference_ms"] for c in report["cases"] if c["first_model_call"]), None),
        "cached_observation_ms": stats([c["cached_observation_ms"] for c in report["cases"]]),
        "preprocessing_ms": stats([c["preprocessing_ms"] for c in report["cases"]]), "modes": {}}
    for mode in MODES:
        rows = [r for c in report["cases"] for r in c["modes"][mode]]
        report["summary"]["modes"][mode] = {"actions": len(rows), "ik_rejections": sum(r["ik_rejected"] for r in rows),
            "ik_reasons": dict(Counter(r["adapter"]["ik_reason"] for r in rows)),
            **{key: stats([r[key] for r in rows]) for key in (
                "command_to_request_translation_error_mm", "achieved_to_request_translation_error_mm",
                "achieved_to_applied_command_translation_error_mm", "achieved_to_recorded_next_translation_error_mm")},
            **{key: stats([r["timing_ms"][key] for r in rows]) for key in (
                "ik_solve_ms", "environment_step_ms", "adapter_and_step_total_ms")},
            "native_limit_clipped_actions": sum(any(r["native_control"]["limit_clipped"]) for r in rows),
            "native_rate_limited_actions": sum(any(r["native_control"]["rate_limited"]) for r in rows),
            "native_gripper_clipped_actions": sum(r["adapter"]["native_gripper_clipped"] for r in rows),
            "maximum_robot_table_penetration_m": max((r["robot_table_penetration_m"] for r in rows), default=0.)}
    report["remaining_evidence"] = {
        "physical_camera_acquisition_and_sent_command_clock": "unmeasured",
        "physical_loaded_hold_release": "unmeasured; see retained sensitivity evidence, not a physics validation",
        "source_calibration_applicability": "independent camera/base/joint/jaw verification pending",
        "small_training_gate": "not_passed; no promotion development/reserved suite is run by this diagnostic",
        "recovery": "correction data and phase-aware evaluation integration pending; no recovery training performed",
        "panda": "source-task completion and separate scripted target-control/gripper verification required before transfer claims"}
    report["contact_evidence_sha256"] = {str(p): sha256_file(p) for p in (
        Path("results/v2/loaded_hold_release_evidence_20261024.json"), Path("configs/v2/measurements.template.json"),
        Path("docs/v2_control_contract_comparison_20261024.md"))}
    after = dependency_ledger(args.sim_config, args.eval_config, args.checkpoint, args.data)
    if after != before:
        raise ValueError("consumed execution sources changed during the diagnostic")
    report["source_sha256_after"] = after
    write_json(args.out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/v2/three_episode_prefix2_20261025/training/checkpoints/002000"))
    parser.add_argument("--data", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone"))
    parser.add_argument("--sim-config", type=Path, default=Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("configs/v2/eval_so101.yaml"))
    parser.add_argument("--per-episode", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run(args)
    print(json.dumps(report["summary"], default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x, indent=2))
