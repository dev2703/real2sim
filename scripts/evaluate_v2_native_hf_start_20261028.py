"""Bounded native-HF ACT reset and causal joint-coordinate comparisons.

Uses the user's actual frozen real-data ACT, native saved processors and both
simulated camera inputs. Canonicalization is a declared software coordinate
adapter derived only from training initial states; it is not physical calibration.
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

from real2sim.artifacts import sha256_file
from real2sim.data.raw import RawDataset
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.policy.follower_joint import check_joint_request
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.serialization import json_default, write_json
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics
from scripts.evaluate_hf_act_20261027 import SNAPSHOT, prepare_scene, saturate_gripper_only

SCENE = Path("configs/v2/sim_so101.waypoint_20261026.yaml")
MEASURED = Path("configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml")
DATA = Path("~/.cache/huggingface/lerobot/local/so101_act_pickplace_v2").expanduser()


def save(path, value):
    write_json(path, value, atomic=True)


def training_start(snapshot, data):
    provenance = json.loads((snapshot / "experiment.json").read_text())
    raw = RawDataset(data)
    initial = np.asarray([raw.stack(raw.episode_rows(episode), "observation.state")[0]
                          for episode in provenance["train_episodes"]])
    return np.median(initial, axis=0), {
        "estimator": "componentwise median of first observed training frame only",
        "episodes": provenance["train_episodes"],
        "validation_episodes_not_used": provenance["val_episodes"],
        "mean_degree_percent": initial.mean(0), "std_degree_percent": initial.std(0),
        "metadata_sha256": sha256_file(data / "meta/info.json"),
    }


def run(out, snapshot=SNAPSHOT, *, seeds=(0, 1, 2, 3, 4),
        variants=("baseline", "start_closed", "canonical_arm_closed", "measured_closed")):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    native = NativeLeRobotACT(snapshot)
    native.policy.eval().requires_grad_(False)
    median, derivation = training_start(snapshot, DATA)
    dependencies = [Path(__file__), Path("scripts/evaluate_hf_act_20261027.py"),
        Path("src/real2sim/policy/native_lerobot_act.py"),
        Path("src/real2sim/policy/follower_joint.py"),
        Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/model.py"),
        Path("src/real2sim/sim/kinematics.py"), Path("src/real2sim/sim/cameras.py"),
        Path("src/real2sim/evaluation/metrics.py"), SCENE,
        *[snapshot / name for name in ("experiment.json", "config.json", "model.safetensors",
            "policy_preprocessor.json", "policy_postprocessor.json",
            "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors")]]
    if "measured_closed" in variants:
        dependencies.append(MEASURED)
    before = {str(path.resolve()): sha256_file(path) for path in dependencies}
    plan = {"schema": "native_hf_start_comparison_20261028_v1", "status": "running",
        "checkpoint": str(snapshot.resolve()), "checkpoint_sha256": native.weights_sha256,
        "frozen_HF_policy": True, "requires_grad": False,
        "source_training_initial_median_degree_percent": median,
        "source_initial_derivation": derivation, "seeds": list(seeds),
        "seed_scope": "Previously used development seeds only; no confirmation claim",
        "variants": list(variants), "execution_prefix": native.config.n_action_steps,
        "policy_hz": 25., "lighting_changed": False, "corrective_IK": False,
        "canonicalization": "Arm only: HF state = measured degree state minus fixed offset; native degree command = HF action plus same fixed offset. Offset = configured simulated start minus source-training median. Gripper retains native percentage meaning. This is a software coordinate hypothesis, not accepted physical mapping.",
        "physical_mapping_variant": "No new physical joint offsets used without independent evidence; measured_closed changes measured scene scalars only.",
        "source_sha256_before": before, "cases": {}}
    save(out / "plan.json", plan)
    (out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    for variant in variants:
        directory = out / variant
        directory.mkdir()
        scene = prepare_scene(directory, MEASURED if variant == "measured_closed" else SCENE)
        report = {"variant": variant, "status": "running", "trials": [],
            "checkpoint_sha256": native.weights_sha256, "lighting_changed": False,
            "physical_calibration_accepted": False}
        with So101PickPlaceEnv(scene, render_images=False) as env:
            guard = NativeKinematics(env.model, env.config)
            arm_offset = np.zeros(5)
            # Use the configured start, not pre-reset data or trial outcomes.
            configured_real = np.asarray(env.config["initial_qpos_real_deg"], float)
            if variant == "canonical_arm_closed":
                arm_offset = configured_real[:5] - median[:5]
            start = env.initial_qpos.copy()
            if variant != "baseline":
                start[5] = env.mapping.gripper_real_to_sim_rad(median[5])
            report.update(arm_canonical_offset_degrees=arm_offset,
                reset_native_joint_radians=start, source_closed_percent=float(median[5]),
                nominal_lighting=env.config.get("appearance", {}).get("lighting"),
                actual_task=env.task)
            save(directory / "report.json", report)
            for seed in seeds:
                trial = directory / f"seed_{seed:03d}"
                trial.mkdir()
                tracker = TrialTracker(success_hold_ticks=math.ceil(env.control_hz),
                    approach_distance_m=.03, table_top_z=env.table_top,
                    ball_radius_m=env.task["ball"]["radius"])
                queue, latency, reasons = [], [], Counter()
                abort, initial = None, None
                saturated = 0
                started = time.perf_counter()
                try:
                    _, initial = env.reset(seed=seed, scenario={"joint_positions": start})
                    native.reset()
                    for camera in native.camera_keys:
                        rgb = env.render(env.camera_bindings[camera])
                        cv2.imwrite(str(trial / f"initial_{camera.rsplit('.', 1)[-1]}.png"), rgb[:, :, ::-1])
                    with (trial / "ticks.jsonl").open("w") as stream:
                        for tick in range(env.max_episode_steps):
                            virtual = None
                            if not queue:
                                t0 = time.perf_counter()
                                virtual = env.real_state().copy()
                                virtual[:5] -= arm_offset
                                images = {key: env.render(env.camera_bindings[key]) for key in native.camera_keys}
                                queue = list(native.predict_action_chunk(virtual, images)[:native.config.n_action_steps])
                                latency.append((time.perf_counter() - t0) * 1000)
                            raw = queue.pop(0)
                            transformed = raw.copy()
                            transformed[:5] += arm_offset
                            requested, bounded = saturate_gripper_only(transformed, env)
                            saturated += int(requested[5] != bounded[5])
                            before_q = env.sim_state()
                            guard.data.qpos[:] = env.data.qpos
                            accepted, reason, checked = check_joint_request(guard, before_q,
                                env.data.ctrl[env.actuator_ids], bounded,
                                env.max_velocity / env.control_hz, samples=17)
                            _, _, _, truncated, info = env.step(bounded if accepted else before_q)
                            reasons[reason] += 1
                            pm = info["privileged_metrics"]
                            done = tracker.update(pm, saturation=float(requested[5] != bounded[5]),
                                ik_rejected=not accepted)
                            stream.write(json.dumps({"tick": tick, "HF_raw_degree_percent": raw,
                                "HF_input_degree_percent_on_inference": virtual,
                                "canonical_native_degree_percent": transformed,
                                "requested_native_rad": requested, "bounded_native_rad": bounded,
                                "guard_accepted": accepted, "guard_reason": reason,
                                "guard_checked_command": checked, "native_before": before_q,
                                "native_after": env.sim_state(), "control": info["control"],
                                "scoring_only": pm}, default=json_default, allow_nan=False) + "\n")
                            if done or truncated:
                                break
                except (ValueError, RuntimeError) as exc:
                    abort = f"{type(exc).__name__}: {exc}"
                row = tracker.row(experiment_id="NATIVE_HF_START_COMPARISON_20261028",
                    controller_source="FROZEN_NATIVE_HF_ACT" if variant != "canonical_arm_closed" else "FROZEN_NATIVE_HF_ACT_CAUSAL_JOINT_CANONICALIZATION",
                    robot="so101", seed=seed, checkpoint_sha256=native.weights_sha256,
                    checkpoint_step=None, mean_latency_ms=float(np.mean(latency)) if latency else None,
                    video_path="", abort=abort)
                row.update(guard_reasons=dict(reasons), policy_calls=len(latency),
                    gripper_saturations=saturated, wall_s=time.perf_counter() - started,
                    initial_scoring_only=initial, trace=str(trial / "ticks.jsonl"))
                report["trials"].append(row)
                report["summary"] = summarize(report["trials"])
                save(directory / "report.json", report)
                print(json.dumps({"variant": variant, "seed": seed, "success": row["pick_place_success"],
                    "lifted": row["lifted"], "distance_m": row["min_ee_ball_distance_m"],
                    "phase": row["pick_place_phase_failed"], "abort": abort, "guard": dict(reasons)}), flush=True)
        report["status"] = "complete"
        save(directory / "report.json", report)
        plan["cases"][variant] = {"report": str(directory / "report.json"), "summary": report["summary"]}
        save(out / "plan.json", plan)
    after = {path: sha256_file(path) for path in before}
    plan.update(status="complete", source_sha256_after=after,
        runtime_source_changes=[path for path in before if before[path] != after[path]],
        checkpoint_unchanged=native.weights_sha256 == sha256_file(snapshot / "model.safetensors"))
    save(out / "plan.json", plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--snapshot", default=SNAPSHOT, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--variants", nargs="+", choices=("baseline", "start_closed", "canonical_arm_closed", "measured_closed"),
                        default=["baseline", "start_closed", "canonical_arm_closed", "measured_closed"])
    args = parser.parse_args()
    run(args.out, args.snapshot, seeds=tuple(args.seeds), variants=tuple(args.variants))
