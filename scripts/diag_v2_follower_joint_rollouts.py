"""Common-seed bounded rollout screen of explicitly guarded joint target policies."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from real2sim.config import load_yaml
from real2sim.data.follower_joint_targets import VARIANTS
from real2sim.data.raw import sha256_file
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.run import build_learned_controller
from real2sim.sim.contact_solver import contact_solver_settings
from real2sim.sim.env import So101PickPlaceEnv


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False) + "\n")


def run(pilot: Path, sim_path: Path, eval_path: Path, out: Path, seeds: list[int], max_ticks: int):
    if out.exists() or not 1 <= max_ticks <= 125 or not 1 <= len(set(seeds)) <= 3 or len(set(seeds)) != len(seeds):
        raise ValueError("fresh directory,1..125ticks and1..3distinct seeds required")
    for name in VARIANTS:
        if not (pilot / name / "training_result.json").is_file():
            raise FileNotFoundError(f"completed training required: {name}")
    torch.set_num_threads(2)
    out.mkdir(parents=True)
    eval_cfg = load_yaml(eval_path)
    eval_cfg["cartesian"]["target_reference"] = "current_pose"
    local_eval = out / "eval_joint.yaml"
    local_eval.write_text(yaml.safe_dump(eval_cfg, sort_keys=False))
    source_files = [*sorted(Path("src/real2sim").rglob("*.py")), Path(__file__), sim_path,
                    Path(load_yaml(sim_path)["task_config"]), local_eval]
    before = {str(path): sha256_file(path) for path in source_files}
    report = {"kind": "BOUNDED_GUARDED_JOINT_TARGET_DEVELOPMENT_SCREEN",
              "physical_calibration_accepted": False, "checkpoint_promoted": False,
              "reserved_episodes_used": [], "development_seeds": seeds,
              "requested_max_ticks": max_ticks, "source_sha256_before": before,
              "eval_config": str(local_eval), "candidates": {},
              "limitations": "Three common seeds, selected strict recorded episode9/frame100 reset, unchanged provisional scene.125ticks/10seconds screens approach only. Target/path probes use unchanged native/workspace/collision gates and17static samples, not continuous collision certification. Simulator object state is scoring only. No reserved evaluation or physical mapping acceptance."}
    initial = {}
    for name in reversed(VARIANTS):
        checkpoint = pilot / name / "checkpoints/best"
        candidate_dir = out / name
        candidate_dir.mkdir()
        rows = []
        with So101PickPlaceEnv(sim_path) as env:
            if not env.config.get("strict_initial_state") or not math.isclose(env.control_hz, 12.5):
                raise ValueError("strict recorded reset and declared12.5Hz clock required")
            controller, contract, metadata = build_learned_controller(env, checkpoint, eval_cfg, "cpu")
            if contract["joint_target_semantics"]["variant"] != name:
                raise ValueError("joint target semantics and candidate name differ")
            candidate = {"checkpoint": metadata, "contract": contract, "trials": [],
                         "native_joint_limits_rad": env.limits.tolist(),
                         "unchanged_ik_gate_config": env.config["ik"],
                         "contact_solver": contact_solver_settings(env.model),
                         "control_hz": env.control_hz, "physics_hz": env.config["physics_hz"]}
            report["candidates"][name] = candidate
            for seed in seeds:
                trial_dir = candidate_dir / f"seed_{seed:02d}"
                trial_dir.mkdir()
                obs, reset = env.reset(seed=seed)
                controller.reset()
                scoring = reset["privileged_metrics"]
                reset_state = {"q_native_rad": env.sim_state().tolist(),
                               "ball_task_scoring_only": scoring["ball_xyz_task"],
                               "tool_task_scoring_only": scoring["tool_xyz_task"]}
                if seed not in initial:
                    initial[seed] = reset_state
                elif initial[seed] != reset_state:
                    raise ValueError("matched candidates have different initial states")
                tracker = TrialTracker(success_hold_ticks=math.ceil(env.task["episode"]["success_hold_s"] * env.control_hz),
                                       approach_distance_m=eval_cfg["phase"]["approach_distance_m"],
                                       table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
                abort, reasons, durations, trace = None, Counter(), [], []
                with (trial_dir / "ticks.jsonl").open("w") as stream:
                    for tick in range(1, min(max_ticks, env.max_episode_steps) + 1):
                        started = time.perf_counter()
                        try:
                            result = controller.step(obs, env)
                        except (RuntimeError, ValueError) as exc:
                            abort = str(exc)
                            break
                        durations.append((time.perf_counter() - started) * 1000)
                        obs = result.obs
                        metrics = result.info["privileged_metrics"]
                        done = tracker.update(metrics, saturation=result.saturation, ik_rejected=result.ik_rejected)
                        reasons[result.detail.get("ik_reason", "none")] += 1
                        record = {"tick": tick, "sim_time_s": tick / env.control_hz,
                                  "detail": result.detail, "control": result.info["control"],
                                  "native_joint_state_rad": env.sim_state(),
                                  "ee_ball_distance_m_scoring_only": metrics["ee_ball_distance_m"],
                                  "tool_task_scoring_only": metrics["tool_xyz_task"],
                                  "ball_task_scoring_only": metrics["ball_xyz_task"],
                                  "robot_table_penetration_m_scoring_only": metrics["robot_table_penetration_m"],
                                  "held_scoring_only": metrics["ball_held"],
                                  "ik_rejected": result.ik_rejected}
                        stream.write(json.dumps(record, default=json_value, allow_nan=False) + "\n")
                        trace.append(record)
                        if done or result.truncated:
                            break
                latency = controller.latencies_ms[1:] or controller.latencies_ms
                row = tracker.row(experiment_id="BOUNDED_GUARDED_JOINT_TARGET_DEVELOPMENT_SCREEN",
                                  controller_source="LEARNED_ACT", robot="so101", seed=seed,
                                  checkpoint_sha256=metadata["checkpoint_sha256"], checkpoint_step=metadata["checkpoint_step"],
                                  mean_latency_ms=float(np.mean(latency)) if latency else None,
                                  video_path="", abort=abort)
                rows.append(row)
                trial = {**row, "initial_state": reset_state, "guard_reason_counts": dict(reasons),
                         "policy_calls": controller.n_policy_calls,
                         "max_robot_table_penetration_m": max((r["robot_table_penetration_m_scoring_only"] for r in trace), default=0),
                         "step_wall_ms_median": float(np.median(durations)) if durations else None,
                         "step_wall_ms_p95": float(np.quantile(durations, .95)) if durations else None,
                         "trace": str(trial_dir / "ticks.jsonl")}
                candidate["trials"].append(trial)
                write_json(trial_dir / "summary.json", trial)
                write_json(out / "report.json", report)
                print(json.dumps({"candidate": name, "seed": seed, "ticks": row["ticks"],
                                  "closest_distance_m": row["min_ee_ball_distance_m"],
                                  "rejections": row["ik_rejections"], "abort": abort}), flush=True)
            candidate["summary"] = summarize(rows)
    after = {str(path): sha256_file(path) for path in source_files}
    report.update(source_sha256_after=after, source_changed_during_run=[p for p in before if before[p] != after[p]],
                  matched_initial_states_verified=True)
    write_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, default=Path("outputs/v2/follower_joint_pilot_20261024"))
    p.add_argument("--sim", type=Path, default=Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml"))
    p.add_argument("--eval", type=Path, default=Path("configs/v2/eval_so101.yaml"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-ticks", type=int, default=125)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.pilot, args.sim, args.eval, args.out, args.seeds, args.max_ticks)
