"""Recorded-demonstration bridge and unchanged native HF ACT feedback screen.

Replay and paired-image probes are controller/appearance diagnostics. They do
not turn unmatched provisional simulator objects into real-episode task labels.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from real2sim.data.raw import RawDataset, sha256_file
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo
from real2sim.policy.follower_joint import check_joint_request
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics
from scripts.evaluate_hf_act_20261027 import SNAPSHOT, prepare_scene, saturate_gripper_only

EPISODES = (4, 7, 18, 32, 38, 40)
CAMERAS = ("observation.images.top", "observation.images.iphone")
REPLAY_MODES = ("recorded_leader_command", "recorded_next_readback_reference")


def save(path, value):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, default=convert, indent=2, allow_nan=False) + "\n")


def stats(values):
    values = np.asarray(values, float)
    if not values.size:
        return {"n": 0}
    if not np.isfinite(values).all():
        raise ValueError("nonfinite metric")
    return {"n": int(values.size), "mean": float(values.mean()),
            "p50": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def native_valid(q, limits):
    q = np.asarray(q, float)
    return bool(q.shape == (6,) and np.isfinite(q).all()
                and np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1]))


def first_strict_start(env, states):
    excluded = Counter()
    for frame, state in enumerate(states[:-60]):
        q = sim_from_real(env.mapping, state)
        if not native_valid(q, env.limits):
            excluded["recorded_native_limits"] += 1
            continue
        try:
            env.reset(seed=0, scenario={"joint_positions": q})
        except ValueError as exc:
            excluded[str(exc)] += 1
            continue
        return frame, dict(excluded)
    return None, dict(excluded)


def guarded_step(env, guard, requested):
    before = env.sim_state()
    guard.data.qpos[:] = env.data.qpos
    accepted, reason, checked = check_joint_request(
        guard, before, env.data.ctrl[env.actuator_ids], requested,
        env.max_velocity / env.control_hz, samples=17)
    _, _, _, truncated, info = env.step(requested if accepted else before)
    if any(info["control"]["limit_clipped"]):
        raise AssertionError("guarded command was silently clipped")
    return before, accepted, reason, checked, truncated, info


def replay(raw, scene, out):
    directory = out / "recorded_replay"
    directory.mkdir()
    report = {"task_success_evaluated": False, "episodes": [], "runs": [],
        "initialization": "First native-limit and strict table/support-compatible source pose; zero velocity and actuator targets equal source pose, because original servo history is unavailable.",
        "objects": "Seed0 provisional scene, unmatched to recorded episode; no task placement claim.",
        "reference_mode": "Next recorded follower readback is an explicit noncausal tracking reference, never a policy input or learned result."}
    with So101PickPlaceEnv(scene, render_images=False) as env:
        guard = NativeKinematics(env.model, env.config)
        for episode in EPISODES:
            rows = raw.episode_rows(episode)
            states, actions = (raw.stack(rows, key) for key in ("observation.state", "action"))
            start, excluded = first_strict_start(env, states)
            initial = sim_from_real(env.mapping, states[0])
            record = {"episode": episode, "source_length": len(rows),
                "full_episode_start_native_valid": native_valid(initial, env.limits),
                "source_initial_native_rad": initial, "chosen_suffix_start_frame": start,
                "chosen_suffix_start_time_s": None if start is None else start / raw.fps,
                "discarded_prefix_reasons": excluded}
            report["episodes"].append(record)
            print(json.dumps({"replay_episode": episode, "start": start}), flush=True)
            if start is None:
                continue
            for mode in REPLAY_MODES:
                trial = directory / f"episode_{episode:03d}_{mode}"
                trial.mkdir()
                env.reset(seed=0, scenario={"joint_positions": sim_from_real(env.mapping, states[start])})
                writer = CaptionedVideo(trial / "rollout.mp4", 25., (360, 640))
                reasons, reference_errors, applied_errors, saturations = Counter(), [], [], 0
                abort, ticks = None, 0
                try:
                    with (trial / "ticks.jsonl").open("w") as handle:
                        for frame in range(start, min(len(rows) - 1, start + env.max_episode_steps)):
                            raw_command = actions[frame] if mode == REPLAY_MODES[0] else states[frame + 1]
                            # Demonstration references remain unmodified, including raw gripper.
                            requested = sim_from_real(env.mapping, raw_command)
                            _, accepted, reason, checked, truncated, info = guarded_step(env, guard, requested)
                            reasons[reason] += 1
                            ticks += 1
                            current = env.real_state().astype(float)
                            reference_errors.append(abs(current - states[frame + 1]))
                            applied_errors.append(abs(env.sim_state() - np.asarray(info["control"]["applied_radians"])))
                            saturations += sum(info["control"]["limit_clipped"])
                            handle.write(json.dumps({"source_frame": frame, "raw_source_command": raw_command.tolist(),
                                "requested_native_rad": requested.tolist(), "accepted": accepted, "reason": reason,
                                "checked_applied_native_rad": checked.tolist(), "control": info["control"],
                                "readback_after_degree_percent": current.tolist(),
                                "recorded_next_readback_degree_percent": states[frame + 1].tolist()}) + "\n")
                            writer.write(cv2.cvtColor(env.render("iphone"), cv2.COLOR_RGB2BGR),
                                f"RECORDED REPLAY {mode}; unmatched provisional scene")
                            if truncated:
                                break
                except (ValueError, RuntimeError) as exc:
                    abort = f"{type(exc).__name__}: {exc}"
                finally:
                    writer.close()
                errors = np.asarray(reference_errors)
                applied = np.asarray(applied_errors)
                report["runs"].append({"episode": episode, "mode": mode, "start_frame": start,
                    "ticks": ticks, "abort": abort, "guard_reasons": dict(reasons),
                    "source_readback_mae_degree_percent": errors.mean(0) if len(errors) else None,
                    "source_readback_arm_abs_error_degrees": stats(errors[:, :5]) if len(errors) else {"n": 0},
                    "applied_to_achieved_mae_native_rad": applied.mean(0) if len(applied) else None,
                    "silently_clipped_components": saturations, "video": str(trial / "rollout.mp4"),
                    "trace": str(trial / "ticks.jsonl")})
                save(directory / "report.json", report)
    return report


def freeze(snapshot):
    model = NativeLeRobotACT(snapshot)
    model.policy.eval().requires_grad_(False)
    if any(m.training for m in model.policy.modules()):
        raise AssertionError("native ACT must be in evaluation mode")
    return model


def paired_probe(raw, scene, replay_report, snapshots, out):
    directory = out / "paired_images"
    directory.mkdir()
    report = {"task_success_evaluated": False, "candidate_pairs": 18, "excluded": [], "cases": [],
        "meaning": "Within each checkpoint/state, vary recorded vs rendered RGB only. Unmatched object layouts, provisional optics and rendering appearance are jointly varied, so output sensitivity is not a causal camera-calibration estimate.",
        "image_processing": "Decoded RGB1280x720 directly resized to640x360 with OpenCV INTER_LINEAR, then native processors exactly once; no letterbox.",
        "clock": "Source row frame position paired to container-decoded episode frame position; physical acquisition synchronization remains unverified."}
    anchors = []
    with So101PickPlaceEnv(scene, render_images=False) as env:
        for episode_record in replay_report["episodes"]:
            episode, start = episode_record["episode"], episode_record["chosen_suffix_start_frame"]
            if start is None:
                report["excluded"].append({"episode": episode, "reason": "no strict source start"})
                continue
            rows = raw.episode_rows(episode)
            states, actions = (raw.stack(rows, key) for key in ("observation.state", "action"))
            frames = [start + offset for offset in (0, 25, 50)]
            decoded = {camera: {} for camera in CAMERAS}
            for camera in CAMERAS:
                for frame, pts, image in raw.iter_frames(camera, episode):
                    if frame in frames:
                        decoded[camera][frame] = (cv2.resize(image.to_ndarray(format="rgb24"),
                            (640, 360), interpolation=cv2.INTER_LINEAR), pts)
                    if frame >= frames[-1]:
                        break
            for frame in frames:
                case = {"episode": episode, "frame": frame, "source_timestamp_s": float(rows.iloc[frame]["timestamp"])}
                q = sim_from_real(env.mapping, states[frame])
                if not native_valid(q, env.limits):
                    report["excluded"].append({**case, "reason": "source native limits"})
                    continue
                try:
                    env.reset(seed=0, scenario={"joint_positions": q})
                except ValueError as exc:
                    report["excluded"].append({**case, "reason": str(exc)})
                    continue
                if any(frame not in decoded[camera] for camera in CAMERAS):
                    report["excluded"].append({**case, "reason": "missing raw image"})
                    continue
                if not np.allclose(env.real_state(), states[frame], atol=1e-5, rtol=0):
                    raise AssertionError("paired proprioception differs")
                real = {camera: decoded[camera][frame][0] for camera in CAMERAS}
                simulated = {camera: env.render(env.camera_bindings[camera]) for camera in CAMERAS}
                contact = np.vstack([np.hstack([real[camera], simulated[camera]]) for camera in CAMERAS])
                cv2.imwrite(str(directory / f"episode_{episode:03d}_frame_{frame:04d}_real_left_sim_right.jpg"),
                            cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))
                anchors.append((case, states[frame], actions[frame:frame + 10], real, simulated))
    for name, snapshot in snapshots.items():
        model = freeze(snapshot)
        repeats = []
        for case, state, reference, real, simulated in anchors:
            predictions = {}
            for condition, images in (("recorded", real), ("rendered", simulated)):
                model.reset()
                predictions[condition] = model.predict_action_chunk(state, images)
            model.reset()
            repeated = model.predict_action_chunk(state, real)
            repeats.append(float(abs(repeated - predictions["recorded"]).max()))
            if not np.array_equal(repeated, predictions["recorded"]):
                raise AssertionError("unchanged input prediction is not deterministic")
            prefix_mae = {condition: abs(prediction[:len(reference)] - reference).mean(0)
                          for condition, prediction in predictions.items()}
            output_path = directory / f"{name}_episode_{case['episode']:03d}_frame_{case['frame']:04d}.npz"
            np.savez_compressed(output_path, state=state, recorded=predictions["recorded"],
                                rendered=predictions["rendered"], reference=reference)
            report["cases"].append({**case, "checkpoint": name, "weights_sha256": model.weights_sha256,
                "same_state_input_degree_percent": state, "executed_prefix_groundtruth_mae": prefix_mae,
                "recorded_to_rendered_prefix_action_change": abs(predictions["recorded"][:10] - predictions["rendered"][:10]).mean(0),
                "same_input_repeat_max_absolute_error": repeats[-1], "raw_predictions": str(output_path)})
        print(json.dumps({"paired_checkpoint": name, "pairs": len(anchors), "max_repeat_error": max(repeats, default=0.)}), flush=True)
        save(directory / "report.json", report)
    report["pairs_per_checkpoint"] = len(anchors)
    for name in snapshots:
        cases = [case for case in report["cases"] if case["checkpoint"] == name]
        report[name] = {"n": len(cases), "mean_prefix_mae_degree_percent": {
            condition: np.mean([case["executed_prefix_groundtruth_mae"][condition] for case in cases], axis=0)
            for condition in ("recorded", "rendered")},
            "mean_input_swap_action_change_degree_percent": np.mean([case["recorded_to_rendered_prefix_action_change"] for case in cases], axis=0)}
    save(directory / "report.json", report)
    return report


def live_screen(scene, snapshot, prefix, out):
    out.mkdir()
    model = freeze(snapshot)
    report = {"checkpoint": str(snapshot), "weights_sha256": model.weights_sha256,
              "execution_prefix": prefix, "saved_execution_prefix": model.config.n_action_steps,
              "seeds": [0], "physical_calibration_accepted": False,
              "no_training_or_corrective_IK": True, "policy_eval_mode": True,
              "parameters_require_grad": False, "trials": []}
    with So101PickPlaceEnv(scene, render_images=False) as env:
        env.reset(seed=0)
        model.reset()
        guard = NativeKinematics(env.model, env.config)
        tracker = TrialTracker(success_hold_ticks=math.ceil(env.control_hz), approach_distance_m=.03,
            table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
        reasons, queue, latencies = Counter(), [], []
        abort, saturated_count = None, 0
        writer = CaptionedVideo(out / "rollout.mp4", 25., (360, 640))
        started = time.perf_counter()
        try:
            with (out / "ticks.jsonl").open("w") as handle:
                for tick in range(env.max_episode_steps):
                    if not queue:
                        before_infer = time.perf_counter()
                        images = {camera: env.render(env.camera_bindings[camera]) for camera in CAMERAS}
                        queue = list(model.predict_action_chunk(env.real_state(), images)[:prefix])
                        latencies.append((time.perf_counter() - before_infer) * 1000)
                    raw_action = queue.pop(0)
                    requested, bounded = saturate_gripper_only(raw_action, env)
                    saturated = bool(requested[5] != bounded[5])
                    saturated_count += saturated
                    before, accepted, reason, checked, truncated, info = guarded_step(env, guard, bounded)
                    reasons[reason] += 1
                    done = tracker.update(info["privileged_metrics"], saturation=float(saturated), ik_rejected=not accepted)
                    handle.write(json.dumps({"tick": tick, "raw_model_degree_percent": raw_action.tolist(),
                        "requested_native_rad": requested.tolist(), "gripper_bounded_native_rad": bounded.tolist(),
                        "gripper_saturated": saturated, "guard_accepted": accepted, "guard_reason": reason,
                        "checked_applied_native_rad": checked.tolist(), "readback_before_native_rad": before.tolist(),
                        "readback_after_native_rad": env.sim_state().tolist(), "control": info["control"],
                        "scoring_only": info["privileged_metrics"]}) + "\n")
                    writer.write(cv2.cvtColor(env.render("iphone"), cv2.COLOR_RGB2BGR),
                        f"FROZEN REAL-DATA HF ACT; prefix {prefix}; provisional scene")
                    if tick % 100 == 0:
                        print(json.dumps({"live": str(out), "tick": tick, "guard": dict(reasons)}), flush=True)
                    if done or truncated:
                        break
        except (ValueError, RuntimeError) as exc:
            abort = f"{type(exc).__name__}: {exc}"
        finally:
            writer.close()
        row = tracker.row(experiment_id="REAL_REPLAY_HF_BRIDGE_20261027", controller_source="FROZEN_NATIVE_HF_ACT",
            robot="so101", seed=0, checkpoint_sha256=model.weights_sha256, checkpoint_step=None,
            mean_latency_ms=float(np.mean(latencies)) if latencies else None,
            video_path=str(out / "rollout.mp4"), abort=abort)
        report.update(summary=summarize([row]), trials=[{**row, "guard_reasons": dict(reasons),
            "gripper_saturations": saturated_count, "policy_calls": len(latencies)}],
            wall_seconds=time.perf_counter() - started, inference_ms=stats(latencies),
            timing_scope="Simulation pauses while synchronous host inference runs; no real-time hardware claim.",
            unchanged_weight_file=sha256_file(snapshot / "model.safetensors") == model.weights_sha256)
    save(out / "report.json", report)
    print(json.dumps({"live_complete": str(out), "summary": report["summary"]}), flush=True)
    return report


def run(out, local_root):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    raw = RawDataset(local_root)
    if raw.fps != 25.:
        raise ValueError("requires25Hz source dataset")
    snapshots = {"root": SNAPSHOT, "latest": SNAPSHOT / "latest"}
    for snapshot in snapshots.values():
        experiment = json.loads((snapshot / "experiment.json").read_text())
        if list(EPISODES) != experiment["val_episodes"]:
            raise ValueError("bridge episodes differ from the immutable checkpoint validation split")
    scene_dir = out / "scene"
    scene_dir.mkdir()
    scene = prepare_scene(scene_dir)
    dependencies = [Path(__file__), Path("scripts/evaluate_hf_act_20261027.py"),
        Path("src/real2sim/policy/native_lerobot_act.py"), Path("src/real2sim/policy/follower_joint.py"),
        Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/kinematics.py"),
        Path("assets/menagerie/robotstudio_so101/so101.xml"),
        Path("configs/v2/sim_so101.waypoint_20261026.yaml"), Path("configs/v2/task.scalar_measured_20261026.yaml"),
        *scene_dir.glob("*.yaml"), local_root / "meta/info.json", *local_root.glob("data/**/*.parquet"),
        *local_root.glob("meta/episodes/**/*.parquet")]
    for snapshot in snapshots.values():
        dependencies += list(snapshot.glob("*.json")) + list(snapshot.glob("*.safetensors"))
    dependencies += [raw.video_segment(camera, episode).path for episode in EPISODES for camera in CAMERAS]
    hashes = {str(p): sha256_file(p) for p in dict.fromkeys(dependencies)}
    (out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    plan = {"schema": "real_replay_hf_bridge_plan_20261027_v1", "episodes": EPISODES,
        "replay_modes": REPLAY_MODES, "probe_anchors": "First strict-compatible source frame plus0/25/50 frames, excluded explicitly if incompatible.",
        "live_matrix": [{"checkpoint": name, "prefix": prefix, "seeds": [0]} for name in snapshots for prefix in (10, 1)],
        "native_limits_physics_guards_unchanged": True, "weights_frozen": True,
        "physical_calibration_accepted": False, "source_sha256": hashes,
        "registration_basis": "No independently accepted correction exists. Keep provisional mapping to avoid inventing a calibration.",
        "no_reserved_evaluation_claim": True}
    save(out / "plan.json", plan)
    report = {"status": "running", "plan_sha256": sha256_file(out / "plan.json"),
              "physical_calibration_accepted": False, "source_sha256_before": hashes}
    save(out / "report.json", report)
    report["recorded_replay"] = replay(raw, scene, out)
    save(out / "report.json", report)
    report["paired_images"] = paired_probe(raw, scene, report["recorded_replay"], snapshots, out)
    save(out / "report.json", report)
    report["live"] = {}
    for name, snapshot in snapshots.items():
        for prefix in (10, 1):
            key = f"{name}_prefix{prefix}"
            report["live"][key] = live_screen(scene, snapshot, prefix, out / key)
            save(out / "report.json", report)
    after = {name: sha256_file(Path(name)) for name in hashes}
    report.update(status="complete", source_sha256_after=after,
                  runtime_source_changes=[name for name in hashes if hashes[name] != after[name]])
    save(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local-root", type=Path,
        default=Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v2")
    args = parser.parse_args()
    run(args.out, args.local_root)
