"""Full learned-policy development rollouts in the frozen provisional scene.

This diagnostic preserves the checkpoint, mapping, registered cameras, and control
contract. It does not satisfy calibrated evaluate_v2 prerequisites. Ground truth
is used only for scoring, never to construct the policy input or its actions.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np
import torch

from real2sim.config import load_yaml
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.run import build_learned_controller
from real2sim.policy.paired import file_sha256
from real2sim.sim.contact_solver import contact_solver_settings
from real2sim.sim.env import So101PickPlaceEnv


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=jsonable, allow_nan=False) + "\n")


def snapshot(root):
    files = json.loads((root / "INPUTS.json").read_text())["files"]
    hashes = {p: file_sha256(Path(p)) for p in files}
    changed = [p for p, expected in files.items() if hashes[p] != expected]
    if changed:
        raise ValueError(f"frozen queue inputs changed: {changed}")
    return hashes


def picture(path, obs, camera_key):
    rgb = np.moveaxis(obs[camera_key], 0, -1)
    cv2.imwrite(str(path), cv2.cvtColor(np.rint(rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/v2/learning_screen_20261022/runs/L1_resnet_iphone/attempt-001/checkpoints/004000"))
    p.add_argument("--config", type=Path, default=Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml"))
    p.add_argument("--eval-config", type=Path, default=Path("configs/v2/eval_so101.yaml"))
    p.add_argument("--screen", type=Path, default=Path("outputs/v2/learning_screen_20261022"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--extension-seconds", type=float, default=None,
                   help="Explicit diagnostic timeout extension; frozen task remains unchanged")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    before = snapshot(args.screen)
    rows, trials = [], []
    with closing(So101PickPlaceEnv(args.config)) as env:
        ctrl, contract, meta = build_learned_controller(env, args.checkpoint, load_yaml(args.eval_config), "cpu")
        configured_timeout = float(env.task["episode"]["timeout_s"])
        default_ticks = env.max_episode_steps
        if args.extension_seconds is not None:
            if args.extension_seconds < configured_timeout:
                raise ValueError("extension must not shorten the configured full episode")
            env.max_episode_steps = int(math.ceil(args.extension_seconds * env.control_hz))
        eval_cfg = load_yaml(args.eval_config)
        source = sorted(Path("src/real2sim").rglob("*.py"))
        source_hashes = {str(path): file_sha256(path) for path in source}
        report = {
            "kind": "FULL_LEARNED_POLICY_PROVISIONAL_SCENE_DEVELOPMENT_DIAGNOSTIC",
            "limitations": "Unaccepted frozen geometry; iPhone-only policy; selected recorded validation reset episode 9 frame 100. Seeds vary ball location within the configured start region. No retraining, reserved episodes, hardware, checkpoint contract editing, or formal calibrated evaluation. Ground truth is scoring only.",
            "physical_calibration_accepted": False,
            "configured_timeout_s": configured_timeout,
            "configured_max_ticks": default_ticks,
            "diagnostic_timeout_extension_s": args.extension_seconds,
            "effective_max_ticks": env.max_episode_steps,
            "control_hz": env.control_hz,
            "physics_hz": env.config["physics_hz"],
            "strict_initial_state": env.config["strict_initial_state"],
            "checkpoint": meta, "contract": contract,
            "contact_solver": contact_solver_settings(env.model),
            "config": str(args.config), "config_sha256": file_sha256(args.config),
            "eval_config_sha256": file_sha256(args.eval_config),
            "task_sha256": file_sha256(Path(env.config["task_config"])),
            "script_sha256": file_sha256(Path(__file__)),
            "source_hashes_before": source_hashes,
            "frozen_inputs_before": before,
            "trials": trials,
        }
        for seed in args.seeds:
            trial_dir = args.out / f"seed_{seed:02d}"
            trial_dir.mkdir()
            obs, reset_info = env.reset(seed=seed)
            ctrl.reset()
            tracker = TrialTracker(
                success_hold_ticks=int(math.ceil(env.task["episode"]["success_hold_s"] * env.control_hz)),
                approach_distance_m=float(eval_cfg["phase"]["approach_distance_m"]),
                table_top_z=env.table_top, ball_radius_m=float(env.task["ball"]["radius"]))
            picture(trial_dir / "initial.jpg", obs, env.camera_keys[0])
            start = reset_info["privileged_metrics"]
            trajectory, reasons, abort = [], collections.Counter(), None
            checkpoint_at_task_timeout = None
            t0 = time.perf_counter()
            with (trial_dir / "ticks.jsonl").open("w") as stream:
                for tick in range(1, env.max_episode_steps + 1):
                    try:
                        result = ctrl.step(obs, env)
                    except (RuntimeError, ValueError) as exc:
                        abort = str(exc)
                        break
                    obs = result.obs
                    metrics = result.info["privileged_metrics"]
                    done = tracker.update(metrics, saturation=result.saturation, ik_rejected=result.ik_rejected)
                    reasons[result.detail.get("ik_reason", "none")] += 1
                    record = {
                        "tick": tick, "sim_time_s": tick / env.control_hz,
                        "detail": result.detail, "control": result.info["control"],
                        "q_native_rad": env.sim_state(),
                        "tool_xyz_task": metrics["tool_xyz_task"],
                        "ball_xyz_task": metrics["ball_xyz_task"],
                        "ee_ball_distance_m": metrics["ee_ball_distance_m"],
                        "ball_held": metrics["ball_held"], "lifted": metrics["lifted"],
                        "ball_in_bowl": metrics["ball_in_bowl"],
                        "robot_table_penetration_m": metrics["robot_table_penetration_m"],
                        "ik_rejected": result.ik_rejected, "saturation": result.saturation,
                    }
                    stream.write(json.dumps(record, default=jsonable, allow_nan=False) + "\n")
                    trajectory.append(record)
                    if tick == default_ticks:
                        checkpoint_at_task_timeout = {
                            "success": tracker.success, "pick_place_success": tracker.pick_place_success,
                            "lifted": tracker.lifted, "held_ever": tracker.held_ever,
                            "min_ee_ball_distance_m": tracker.min_ee_ball_distance_m,
                            "phase_failed": tracker.phase_failed(), "ik_rejections": tracker.ik_rejections,
                        }
                    if tick % 125 == 0:
                        print(f"seed={seed} tick={tick}/{env.max_episode_steps} distance={metrics['ee_ball_distance_m']:.3f} min={tracker.min_ee_ball_distance_m:.3f} held={tracker.held_ever} ik_rejections={tracker.ik_rejections}", flush=True)
                    if done or result.truncated:
                        break
            picture(trial_dir / "final.jpg", obs, env.camera_keys[0])
            lat = ctrl.latencies_ms[1:] or ctrl.latencies_ms
            row = tracker.row(
                experiment_id="DIAGNOSTIC_FULL_PROVISIONAL_LEARNED_POLICY", controller_source="LEARNED_ACT",
                robot="so101", seed=seed, checkpoint_sha256=meta["checkpoint_sha256"],
                checkpoint_step=meta["checkpoint_step"], mean_latency_ms=float(np.mean(lat)) if lat else None,
                video_path="", abort=abort)
            rows.append(row)
            final = env.info()["privileged_metrics"]
            trial = {
                **row, "wall_s": time.perf_counter() - t0,
                "simulated_seconds": tracker.ticks / env.control_hz,
                "held_ever": tracker.held_ever,
                "approached": tracker.min_ee_ball_distance_m <= tracker.approach_distance_m,
                "policy_calls": ctrl.n_policy_calls, "ik_reason_counts": dict(reasons),
                "task_timeout_snapshot": checkpoint_at_task_timeout,
                "initial_ball_xyz_task": start["ball_xyz_task"],
                "initial_tool_xyz_task": start["tool_xyz_task"],
                "initial_ee_ball_distance_m": start["ee_ball_distance_m"],
                "final_ee_ball_distance_m": final["ee_ball_distance_m"],
                "final_tool_xyz_task": final["tool_xyz_task"],
                "maximum_robot_table_penetration_m": max((x["robot_table_penetration_m"] for x in trajectory), default=0),
                "diagnostics": str(trial_dir / "ticks.jsonl"),
            }
            trials.append(trial)
            report["summary"] = summarize(rows)
            write_json(args.out / "report.json", report)
            print("TRIAL_RESULT " + json.dumps(trial, default=jsonable), flush=True)
        report["frozen_inputs_after"] = snapshot(args.screen)
        after = {path: file_sha256(Path(path)) for path in source_hashes}
        report["runtime_source_changes"] = [path for path in after if after[path] != source_hashes[path]]
        write_json(args.out / "report.json", report)


if __name__ == "__main__":
    main()
