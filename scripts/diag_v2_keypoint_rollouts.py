"""Bounded matched development rollouts of action-only/auxiliary keypoint heads."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from real2sim.config import load_yaml
from real2sim.data.raw import sha256_file
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.run import build_learned_controller
from real2sim.policy.act import load_checkpoint
from real2sim.sim.contact_solver import contact_solver_settings
from real2sim.sim.env import So101PickPlaceEnv


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=serialize, allow_nan=False) + "\n")


def dependencies(sim, args):
    source = ["config.py", "policy/act.py", "policy/adapters.py", "policy/dataset.py",
              "policy/keypoint_bottleneck.py", "policy/dinov2_act.py", "policy/flow_matching.py",
              "sim/env.py", "sim/model.py", "sim/kinematics.py", "sim/cameras.py",
              "sim/contact_solver.py", "geometry/so3.py", "geometry/so101_fk.py"]
    paths = [*(Path("src/real2sim") / p for p in source), Path(__file__), args.sim, args.eval,
             *(Path(sim[k]) for k in ("task_config", "data_config", "model", "manifest"))]
    return {str(path): sha256_file(path) for path in paths}


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    if not 1 <= args.ticks <= 125 or not args.seeds or len(set(args.seeds)) != len(args.seeds) or len(args.seeds) > 3:
        raise ValueError("bounded screen permits at most three distinct seeds and125 ticks each")
    sim, evaluation = load_yaml(args.sim), load_yaml(args.eval)
    before = dependencies(sim, args)
    args.out.mkdir(parents=True)
    report = {"kind": "BOUNDED_AUXILIARY_KEYPOINT_DEVELOPMENT_ROLLOUTS",
              "physical_calibration_accepted": False, "checkpoint_promoted": False,
              "reserved_episodes_used": [], "seeds": args.seeds, "ticks_per_trial": args.ticks,
              "source_sha256_before": before, "candidates": {},
              "limitations": "One training seed, small bottleneck heads, strict recorded episode9/frame100 reset in unchanged provisional scene. Ten-second approach screen, not full task evaluation. Ground truth is used for scoring only. No physical calibration, reserved evaluation or task promotion."}
    initial = {}
    for mode in ("action_only", "auxiliary_ball"):
        checkpoint = args.pilot / mode
        # Exercise the same central factory used by normal simulator inference.
        factory_policy, _, factory_contract = load_checkpoint(checkpoint, "cpu")
        if factory_policy.config.type != "keypoint_bottleneck_v1":
            raise ValueError("generic factory did not load keypoint policy")
        del factory_policy
        with So101PickPlaceEnv(args.sim) as env:
            if not env.config.get("strict_initial_state") or not math.isclose(env.control_hz, 12.5):
                raise ValueError("retain the strict declared reset and12.5Hz control")
            controller, contract, meta = build_learned_controller(env, checkpoint, evaluation, "cpu")
            if contract != factory_contract:
                raise ValueError("factory and controller contracts differ")
            candidate = {"checkpoint": meta, "contract_sha256": sha256_file(checkpoint / "real2sim_contract.json"),
                         "contact_solver": contact_solver_settings(env.model),
                         "native_limits_rad": env.limits.tolist(), "ik_gates": env.config["ik"], "trials": []}
            report["candidates"][mode] = candidate
            for seed in args.seeds:
                obs, info = env.reset(seed=seed)
                controller.reset()
                state = {"native": env.sim_state().tolist(),
                         "ball": info["privileged_metrics"]["ball_xyz_task"],
                         "tool": info["privileged_metrics"]["tool_xyz_task"]}
                if mode == "action_only":
                    initial[seed] = state
                elif state != initial[seed]:
                    raise ValueError("candidate reset states differ")
                batch = controller.obs_adapter.batch(obs, env)
                with torch.inference_mode():
                    _, uv, probability_logit, _ = controller.policy.predictions(batch)
                directory = args.out / mode / f"seed_{seed:02d}"
                directory.mkdir(parents=True)
                tracker = TrialTracker(success_hold_ticks=math.ceil(env.task["episode"]["success_hold_s"] * env.control_hz),
                                       approach_distance_m=evaluation["phase"]["approach_distance_m"],
                                       table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
                reasons, abort, max_penetration = Counter(), None, 0.
                with (directory / "ticks.jsonl").open("w") as stream:
                    for tick in range(1, args.ticks + 1):
                        try:
                            result = controller.step(obs, env)
                        except (ValueError, RuntimeError) as exc:
                            abort = str(exc)
                            break
                        obs = result.obs
                        metric = result.info["privileged_metrics"]
                        done = tracker.update(metric, saturation=result.saturation, ik_rejected=result.ik_rejected)
                        reasons[result.detail.get("ik_reason", "none")] += 1
                        max_penetration = max(max_penetration, metric["robot_table_penetration_m"])
                        trace = {"tick": tick, "detail": result.detail, "control": result.info["control"],
                                 "native_joints_rad": env.sim_state(), "ee_ball_distance_m_scoring": metric["ee_ball_distance_m"],
                                 "ball_xyz_scoring": metric["ball_xyz_task"], "tool_xyz_scoring": metric["tool_xyz_task"],
                                 "ball_held_scoring": metric["ball_held"], "ik_rejected": result.ik_rejected,
                                 "robot_table_penetration_m_scoring": metric["robot_table_penetration_m"]}
                        stream.write(json.dumps(trace, default=serialize, allow_nan=False) + "\n")
                        if done or result.truncated:
                            break
                latencies = controller.latencies_ms[1:] or controller.latencies_ms
                row = tracker.row(experiment_id="AUXILIARY_KEYPOINT_APPROACH_SCREEN", controller_source="LEARNED_KEYPOINT_HEAD",
                                  robot="so101", seed=seed, checkpoint_sha256=meta["checkpoint_sha256"], checkpoint_step=meta["checkpoint_step"],
                                  mean_latency_ms=float(np.mean(latencies)), video_path="", abort=abort)
                trial = {**row, "initial_state": state, "initial_predicted_ball_uv_fraction": uv[0].tolist(),
                         "initial_predicted_detector_event_probability": float(probability_logit[0].sigmoid()),
                         "ik_reasons": dict(reasons), "maximum_robot_table_penetration_m": max_penetration,
                         "policy_calls": controller.n_policy_calls, "inference_p95_ms": float(np.quantile(latencies, .95)),
                         "trace": str(directory / "ticks.jsonl")}
                candidate["trials"].append(trial)
                write_json(directory / "summary.json", trial)
                write_json(args.out / "report.json", report)
                print(json.dumps({"mode": mode, "seed": seed, "ticks": tracker.ticks,
                                  "minimum_distance_mm": tracker.min_ee_ball_distance_m * 1000,
                                  "ik": tracker.ik_rejections, "success": tracker.pick_place_success}), flush=True)
            candidate["summary"] = summarize(candidate["trials"])
    after = dependencies(sim, args)
    report.update(source_sha256_after=after, source_changed_during_run=[p for p in before if before[p] != after[p]],
                  matched_initial_states=True)
    write_json(args.out / "report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, default=Path("outputs/v2/aux_keypoint_pilot_20261024"))
    p.add_argument("--sim", type=Path, default=Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml"))
    p.add_argument("--eval", type=Path, default=Path("configs/v2/eval_so101.yaml"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--ticks", type=int, default=125)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args)
