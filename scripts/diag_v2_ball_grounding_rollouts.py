"""Bounded matched development approach screen for explicit ball-image inputs.

Uses each checkpoint's saved detector/observation/action contract through the
normal inference adapter. Ground truth enters scoring only. This ten-second
screen does not promote a checkpoint or consume reserved evaluation episodes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from real2sim.config import load_yaml
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.run import build_learned_controller
from real2sim.policy.paired import file_sha256
from real2sim.policy.visual_grounding import BallImageFeatures, ball_features, rgb_from_env
from real2sim.serialization import json_default as json_value
from real2sim.serialization import write_json
from real2sim.sim.contact_solver import contact_solver_settings
from real2sim.sim.env import So101PickPlaceEnv


def source_hashes(args):
    sim = load_yaml(args.sim_config)
    files = [*sorted(Path("src/real2sim").rglob("*.py")), Path(__file__), args.sim_config,
             args.eval_config, *(Path(sim[k]) for k in ("task_config", "data_config", "model", "manifest"))]
    return {str(p): file_sha256(p) for p in files}


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    if not 1 <= args.max_ticks <= 125:
        raise ValueError("this bounded approach screen permits one to 125 ticks")
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) > 3:
        raise ValueError("use at most three distinct matched development seeds")
    checkpoints = {"disabled_control": args.disabled_checkpoint, "ball_input": args.ball_checkpoint}
    contracts = {}
    for mode, checkpoint in checkpoints.items():
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"completed checkpoint missing: {checkpoint}")
        contract = json.loads((checkpoint / "real2sim_contract.json").read_text())
        spec = BallImageFeatures.from_dict(contract["visual_grounding"])
        if spec.enabled != (mode == "ball_input"):
            raise ValueError(f"{mode} checkpoint has wrong detector enabled state")
        contracts[mode] = contract
    for key in ("calibration_sha256", "camera_keys", "state_names", "action_names", "policy_hz",
                "split", "letterbox", "controller_semantics"):
        if contracts["disabled_control"].get(key) != contracts["ball_input"].get(key):
            raise ValueError(f"paired checkpoint contracts differ: {key}")
    args.out.mkdir(parents=True)
    (args.out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    before = source_hashes(args)
    report = {"kind": "BOUNDED_DEVELOPMENT_BALL_INPUT_APPROACH_SCREEN",
        "physical_calibration_accepted": False, "checkpoint_promoted": False,
        "reserved_episodes_used": [], "development_seeds": args.seeds,
        "requested_max_ticks": args.max_ticks,
        "source_sha256_before": before, "candidates": {},
        "limitations": "Single training seed and bounded ten-second development rollouts in the "
            "unchanged provisional simulator scene. Strict valid recorded episode-9 frame-100 reset; "
            "seeds vary only the original configured ball region. No reserved episodes, geometry "
            "acceptance, physical measurements or checkpoint promotion. Detector features come only "
            "from the current rendered image through the saved training/inference contract. Simulator "
            "ground truth is scoring only. Success flags cover this bounded screen, not full-timeout evaluation."}
    eval_cfg = load_yaml(args.eval_config)
    for mode, checkpoint in checkpoints.items():
        directory = args.out / mode
        directory.mkdir()
        with So101PickPlaceEnv(args.sim_config) as env:
            if not env.config.get("strict_initial_state", False):
                raise ValueError("bounded screen requires the existing strict reset")
            if not math.isclose(env.control_hz, 12.5):
                raise ValueError("125 ticks must remain the declared ten-second budget")
            ctrl, contract, meta = build_learned_controller(env, checkpoint, eval_cfg, "cpu")
            spec = BallImageFeatures.from_dict(contract["visual_grounding"])
            candidate = {"checkpoint": meta, "contract": contract, "trials": [],
                "contact_solver": contact_solver_settings(env.model),
                "native_joint_limits_rad": env.limits.tolist(),
                "original_ik_gates": env.config["ik"],
                "sim_config_sha256": file_sha256(args.sim_config),
                "effective_max_ticks": min(args.max_ticks, env.max_episode_steps),
                "control_hz": env.control_hz, "physics_hz": env.config["physics_hz"]}
            report["candidates"][mode] = candidate
            rows = []
            for seed in args.seeds:
                trial_dir = directory / f"seed_{seed:02d}"
                trial_dir.mkdir()
                obs, reset = env.reset(seed=seed)
                ctrl.reset()
                tracker = TrialTracker(success_hold_ticks=math.ceil(
                    env.task["episode"]["success_hold_s"] * env.control_hz),
                    approach_distance_m=eval_cfg["phase"]["approach_distance_m"],
                    table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
                initial = reset["privileged_metrics"]
                initial_q = env.sim_state().copy()
                trace, reasons, abort = [], Counter(), None
                with (trial_dir / "ticks.jsonl").open("w") as stream:
                    for tick in range(1, candidate["effective_max_ticks"] + 1):
                        detector = ball_features(rgb_from_env(obs[spec.camera_key], spec), spec)
                        try:
                            result = ctrl.step(obs, env)
                        except (RuntimeError, ValueError) as exc:
                            abort = str(exc)
                            break
                        obs = result.obs
                        metric = result.info["privileged_metrics"]
                        done = tracker.update(metric, saturation=result.saturation,
                                              ik_rejected=result.ik_rejected)
                        reasons[result.detail.get("ik_reason", "none")] += 1
                        record = {"tick": tick, "sim_time_s": tick / env.control_hz,
                            "observed_ball_image_features": detector,
                            "detail": result.detail, "control": result.info["control"],
                            "native_joint_state_rad": env.sim_state(),
                            "ee_ball_distance_m_scoring_only": metric["ee_ball_distance_m"],
                            "tool_xyz_task_scoring_only": metric["tool_xyz_task"],
                            "ball_xyz_task_scoring_only": metric["ball_xyz_task"],
                            "ball_held_scoring_only": metric["ball_held"],
                            "robot_table_penetration_m_scoring_only": metric["robot_table_penetration_m"],
                            "ik_rejected": result.ik_rejected}
                        stream.write(json.dumps(record, default=json_value, allow_nan=False) + "\n")
                        trace.append(record)
                        if done or result.truncated:
                            break
                lat = ctrl.latencies_ms[1:] or ctrl.latencies_ms
                row = tracker.row(experiment_id="DEVELOPMENT_BALL_INPUT_APPROACH_SCREEN",
                    controller_source="LEARNED_ACT", robot="so101", seed=seed,
                    checkpoint_sha256=meta["checkpoint_sha256"], checkpoint_step=meta["checkpoint_step"],
                    mean_latency_ms=float(np.mean(lat)) if lat else None, video_path="", abort=abort)
                rows.append(row)
                trial = {**row, "initial_ee_ball_distance_m_scoring_only": initial["ee_ball_distance_m"],
                    "initial_native_joint_state_rad": initial_q,
                    "initial_ball_xyz_task_scoring_only": initial["ball_xyz_task"],
                    "initial_tool_xyz_task_scoring_only": initial["tool_xyz_task"],
                    "policy_calls": ctrl.n_policy_calls, "ik_reasons": dict(reasons),
                    "ball_feature_visible_ticks": sum(bool(r["observed_ball_image_features"][3]) for r in trace),
                    "max_robot_table_penetration_m": max((r["robot_table_penetration_m_scoring_only"] for r in trace), default=0.),
                    "tick_trace": str(trial_dir / "ticks.jsonl")}
                candidate["trials"].append(trial)
                write_json(trial_dir / "summary.json", trial)
                write_json(args.out / "report.json", report)
                print(json.dumps({"mode": mode, "seed": seed, "ticks": row["ticks"],
                    "minimum_distance_m": row["min_ee_ball_distance_m"], "ik_rejections": row["ik_rejections"],
                    "phase_failed": row["phase_failed"], "abort": abort}), flush=True)
            candidate["summary"] = summarize(rows)
    after = source_hashes(args)
    report["source_sha256_after"] = after
    report["source_changed_during_run"] = [p for p in sorted(before.keys() | after.keys())
                                          if before.get(p) != after.get(p)]
    report["comparison_valid_for_checkpoint_promotion"] = False
    report["resolved_sim_config"] = load_yaml(args.sim_config)
    report["resolved_eval_config"] = eval_cfg
    write_json(args.out / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    pilot = Path("outputs/v2/ball_grounding_pilot_20261024")
    p.add_argument("--disabled-checkpoint", type=Path, default=pilot / "disabled_control/checkpoints/best")
    p.add_argument("--ball-checkpoint", type=Path, default=pilot / "ball_input/checkpoints/best")
    p.add_argument("--sim-config", type=Path, default=Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml"))
    p.add_argument("--eval-config", type=Path, default=Path("configs/v2/eval_so101.yaml"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-ticks", type=int, default=125)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
