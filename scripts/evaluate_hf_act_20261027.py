"""Evaluate the user's pinned native LeRobot ACT on SO-101 feedback.

Preserves saved image sizes, camera order, normalizers and absolute joint action
meaning. Only gripper actuator saturation is applied before the existing static
command/path guard; arm requests are never clipped into validity. Simulator
camera registration remains provisional, particularly the nominal wrist optics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
import yaml

from real2sim.config import load_yaml, project_path
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo
from real2sim.policy.follower_joint import check_joint_request
from real2sim.policy.native_lerobot_act import NativeLeRobotACT, file_sha256
from real2sim.serialization import json_default as json_value
from real2sim.serialization import write_json as _write_json
from real2sim.sim.cameras import nominal_so101_wrist_camera
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics

REPO = "bottowork/so101-act-h256-b16"
REVISION = "e696c8a078935432e3501e2c28c1fb2598fbff7e"
SNAPSHOT = Path(f".cache/hf_models_20261027/models--bottowork--so101-act-h256-b16/snapshots/{REVISION}")


def write_json(path, value):
    _write_json(path, value, atomic=True)


def prepare_scene(out, source=Path("configs/v2/sim_so101.waypoint_20261026.yaml")):
    config = load_yaml(source)
    task = load_yaml(project_path(config["task_config"]))
    data = load_yaml(project_path(config["data_config"]))
    task["additional_cameras"] = [nominal_so101_wrist_camera(project_path(config["model"]))]
    data.update(camera_keys=["observation.images.top", "observation.images.iphone"],
                policy_hz=25., excluded_cameras=[])
    config.update(control_hz=25., render={"width": 640, "height": 360})
    config["task_config"] = str((out / "task.yaml").resolve())
    config["data_config"] = str((out / "data.yaml").resolve())
    data["task_config"] = config["task_config"]
    for name, payload in (("task.yaml", task), ("data.yaml", data), ("sim.yaml", config)):
        (out / name).write_text(yaml.safe_dump(payload, sort_keys=False))
    return out / "sim.yaml"


def saturate_gripper_only(raw, env):
    """Physical gripper bound; preserve unbounded arm requests for hard rejection."""
    requested = sim_from_real(env.mapping, np.asarray(raw, float))
    bounded = requested.copy()
    bounded[5] = np.clip(bounded[5], *env.limits[5])
    return requested, bounded


def run(snapshot, out, seeds, execute_steps=None, video=True):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    checkpoint = NativeLeRobotACT(snapshot)
    n = checkpoint.config.n_action_steps if execute_steps is None else execute_steps
    if not 1 <= n <= checkpoint.config.chunk_size:
        raise ValueError("execution prefix must fit saved chunk")
    scene = prepare_scene(out)
    files = [Path(__file__), Path("src/real2sim/artifacts.py"), Path("src/real2sim/serialization.py"),
        scene, out / "task.yaml", out / "data.yaml",
        Path("src/real2sim/policy/native_lerobot_act.py"),
        Path("src/real2sim/policy/follower_joint.py"),
        *[snapshot / name for name in ("config.json", "experiment.json", "model.safetensors",
            "policy_preprocessor.json", "policy_postprocessor.json",
            "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors")]]
    hashes = {str(path): file_sha256(path) for path in files}
    report = {"schema": "native_hf_act_20261027_v1", "status": "running",
        "repo_id": REPO, "revision": REVISION, "snapshot": str(snapshot),
        "weights_sha256": checkpoint.weights_sha256, "execution_steps": n,
        "checkpoint_execution_steps": checkpoint.config.n_action_steps,
        "image_shape_chw": [3, 360, 640], "camera_order": checkpoint.camera_keys,
        "state_and_action": "five arm joint degrees plus gripper percent; absolute servo commands",
        "policy_hz": 25., "fps_basis": "25Hz source v2 recording metadata; not separately pinned in model experiment.json",
        "image_processing": "native RGB640x360 /255, then saved native processor exactly once; no repository letterbox",
        "runtime_overrides": {"device": "cpu", "temporary_pretrained_backbone_initialization": None},
        "gripper_only_saturation": True, "arm_limits_collision_and_path_gates_unchanged": True,
        "physical_calibration_accepted": False, "wrist_camera": "pinned nominal CAD optical transform; not physically verified",
        "no_task_goal_or_corrective_ik": True, "source_sha256_before": hashes,
        "seeds": seeds, "trials": []}
    write_json(out / "report.json", report)
    rows = []
    with closing(So101PickPlaceEnv(scene, render_images=False)) as env:
        if env.camera_keys != checkpoint.camera_keys:
            raise ValueError("checkpoint camera ordering differs from the supplied scene")
        guard = NativeKinematics(env.model, env.config)
        for seed in seeds:
            directory = out / f"seed_{seed:03d}"
            directory.mkdir()
            env.reset(seed=seed)
            checkpoint.reset()
            tracker = TrialTracker(success_hold_ticks=math.ceil(env.control_hz), approach_distance_m=.03,
                table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
            queue, latencies, reasons, abort = [], [], {}, None
            saturations = 0
            first_image = None
            writer = CaptionedVideo(directory / "rollout.mp4", 25., (360, 640)) if video else None
            try:
                with (directory / "ticks.jsonl").open("w") as handle:
                    for tick in range(env.max_episode_steps):
                        before = env.sim_state()
                        infer_ms = 0.
                        if not queue:
                            started = time.perf_counter()
                            images = {key: env.render(env.camera_bindings[key]) for key in checkpoint.camera_keys}
                            if first_image is None:
                                first_image = images["observation.images.iphone"].copy()
                            raw = checkpoint.predict_action_chunk(env.real_state(), images)
                            queue = list(raw[:n])
                            infer_ms = (time.perf_counter() - started) * 1000
                            latencies.append(infer_ms)
                        raw_action = queue.pop(0)
                        requested, bounded = saturate_gripper_only(raw_action, env)
                        saturated = bool(requested[5] != bounded[5])
                        saturations += saturated
                        command_start = env.data.ctrl[env.actuator_ids].copy()
                        guard.data.qpos[:] = env.data.qpos
                        accepted, reason, applied = check_joint_request(guard, before, command_start,
                            bounded, env.max_velocity / env.control_hz, samples=17)
                        obs, _, _, truncated, info = env.step(bounded if accepted else before)
                        reasons[reason] = reasons.get(reason, 0) + 1
                        pm = info["privileged_metrics"]
                        done = tracker.update(pm, saturation=float(saturated), ik_rejected=not accepted)
                        record = {"tick": tick, "inference_ms": infer_ms,
                            "raw_model_degree_percent": raw_action, "requested_native_rad": requested,
                            "gripper_bounded_native_rad": bounded, "gripper_saturated": saturated,
                            "guard_accepted": accepted, "guard_reason": reason,
                            "checked_applied_native_rad": applied, "control": info["control"],
                            "readback_before_native_rad": before, "readback_after_native_rad": env.sim_state(),
                            "scoring_only": pm}
                        handle.write(json.dumps(record, default=json_value, allow_nan=False) + "\n")
                        if writer:
                            rgb = env.render("iphone")
                            writer.write(rgb[:, :, ::-1].copy(), "FROZEN HF ACT; saved normalization; joint command guard")
                        if done or truncated:
                            break
            except (ValueError, RuntimeError) as exc:
                abort = f"{type(exc).__name__}: {exc}"
            finally:
                if writer:
                    writer.close()
            row = tracker.row(experiment_id="HF_ACT_H256_B16_20261027", controller_source="FROZEN_NATIVE_HF_ACT",
                robot="so101", seed=seed, checkpoint_sha256=checkpoint.weights_sha256,
                checkpoint_step=None, mean_latency_ms=float(np.mean(latencies)) if latencies else None,
                video_path=str(directory / "rollout.mp4") if video else "", abort=abort)
            rows.append(row)
            report["trials"].append({**row, "guard_reasons": reasons, "gripper_saturations": saturations,
                "policy_calls": len(latencies), "policy_call_p95_ms": float(np.quantile(latencies, .95)) if latencies else None,
                "trace": str(directory / "ticks.jsonl")})
            report["summary"] = summarize(rows)
            write_json(out / "report.json", report)
            print(json.dumps({"seed": seed, "placed": row["pick_place_success"], "lifted": row["lifted"],
                "phase": row["phase_failed"], "distance_m": row["min_ee_ball_distance_m"], "guard": reasons}), flush=True)
    report["source_sha256_after"] = {name: file_sha256(name) for name in hashes}
    report["runtime_source_changes"] = [name for name in hashes if hashes[name] != report["source_sha256_after"][name]]
    report["status"] = "complete"
    write_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--execute-steps", type=int)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run(args.snapshot, args.out, args.seeds, args.execute_steps, video=not args.no_video)
