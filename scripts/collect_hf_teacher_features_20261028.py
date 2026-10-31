"""Cache frozen native HF decoder features during disclosed teacher replays.

SO-101 accepted teacher commands supply the common six-value targets. Panda
follows their FK grasp points through its unchanged IK adapter. Live rendered
two-camera images and measured/virtual joint states feed the actual HF model;
only whole successful replay episodes enter the train/validation feature packs.
This script does not train a model or count teacher success as learned success.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from real2sim.artifacts import sha256_file
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.panda.adapter import PandaIK
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.policy.follower_joint import check_joint_request
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.serialization import json_default, write_json
from real2sim.sim.kinematics import NativeKinematics

if __package__:
    from scripts import evaluate_hf_act_panda_20261028 as transfer
else:
    import evaluate_hf_act_panda_20261028 as transfer

TEACHER = Path("outputs/v2/matched_scene_act_20261026/data")


def save(path, payload):
    write_json(path, payload, atomic=True)


def resample_episode(arrays, episode):
    ids = np.flatnonzero(arrays["episode_index"] == episode)
    if not len(ids):
        raise ValueError(f"No teacher episode {episode}")
    raw = arrays["action"][ids]
    # Two25Hz intervals per source12.5Hz interval. The final source request is
    # held through the terminal interval; targets after the source are masked.
    phase = np.arange(2 * len(ids), dtype=float) / 2
    left = np.minimum(phase.astype(int), len(ids) - 1)
    right = np.minimum(left + 1, len(ids) - 1)
    fraction = phase - left
    return ((1 - fraction[:, None]) * raw[left] + fraction[:, None] * raw[right]).astype(np.float32)


def features_pack(records):
    names = ("features", "target_normalized", "valid", "raw_state", "robot_id", "episode_index", "tick")
    if not records:
        return {"features": np.empty((0, 50, 256), np.float32),
            "target_normalized": np.empty((0, 50, 6), np.float32), "valid": np.empty((0, 50), bool),
            "raw_state": np.empty((0, 6), np.float32),
            **{name: np.empty(0, np.int64) for name in ("robot_id", "episode_index", "tick")}}
    return {name: np.stack([row[name] for row in records]) for name in names}


def run(out, snapshot, *, train_seeds=(100, 101, 102, 103, 104), val_seeds=(200, 201), stride=10):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    native = NativeLeRobotACT(snapshot)
    native.policy.eval().requires_grad_(False)
    normalization = load_file(Path(snapshot) / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    mean = normalization["action.mean"].numpy()
    std = normalization["action.std"].numpy()
    source_scene, task = transfer.prepare_source(out, (0.,) * 5)
    source = transfer.retarget.VirtualSource(source_scene)
    try:
        panda_scene, wrist = transfer.prepare_panda(out, source, task, -.15, wrist_mount="external_hand")
        with np.load(TEACHER / "arrays.npz", allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in ("episode_index", "action")}
        provenance = json.loads((TEACHER / "manifest.json").read_text())
        if not set(train_seeds) <= set(provenance["split"]["train"]) or not set(val_seeds) <= set(provenance["split"]["val"]):
            raise ValueError("Feature collection must preserve teacher train/validation episodes")
        watched = [Path(__file__), Path(transfer.__file__), Path(transfer.retarget.__file__),
            Path("src/real2sim/policy/native_lerobot_act.py"), TEACHER / "arrays.npz", TEACHER / "manifest.json",
            Path(snapshot) / "model.safetensors", Path(snapshot) / "config.json"]
        before = {str(path.resolve()): sha256_file(path) for path in watched}
        report = {"schema": "native_hf_teacher_decoder_features_20261028_v1", "status": "collecting",
            "weights_sha256": native.weights_sha256, "source": str(TEACHER),
            "training_seeds": train_seeds, "validation_seeds": val_seeds, "anchor_stride": stride,
            "feature_source": "Actual frozen native HF action_head forward-prehook,50queriesx256features",
            "target_source": "Source successful SO-101 accepted absolute joint commands, linear12.5to25Hz resampling",
            "normalization": {"mean": mean, "std": std, "eps": 1e-8},
            "features": "float32[N,50,256]", "target_normalized": "float32[N,50,6]",
            "valid": "bool[N,50]", "robot_id": "0SO101,1Panda",
            "selection": "Every attempt scored; only whole successful teacher replays supply supervised anchors",
            "policy_inference_drives_actions": False, "task_truth_used_only_for_scoring_and_selection": True,
            "lighting": "Fixed unchanged nominal lighting", "panda_wrist": wrist,
            "source_sha256_before": before, "trials": [], "physical_calibration_accepted": False}
        (out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
        save(out / "report.json", report)
        cache, captured = {"train": [], "val": []}, {}

        def hook(_module, args):
            captured["features"] = args[0].detach().cpu().numpy()[0].copy()

        hook_handle = native.policy.model.action_head.register_forward_pre_hook(hook)
        try:
            with PandaPickPlaceEnv(panda_scene, render_images=False) as panda:
                mapped, opened = transfer.panda_start(panda, source, -.15)
                panda_ik = PandaIK(panda.model, panda.config, task_origin=panda.offset)
                so_guard = NativeKinematics(source.source.model, source.source.config)
                for robot_id, robot, env in ((0, "so101", source.source), (1, "panda", panda)):
                    for partition, seeds in (("train", train_seeds), ("val", val_seeds)):
                        for seed in seeds:
                            directory = out / f"{robot}_{seed}"
                            directory.mkdir()
                            targets = resample_episode(arrays, seed)
                            source.q = source.source.initial_qpos.copy()
                            source.last_target = source.q.copy()
                            env.reset(seed=seed, scenario={"joint_positions": mapped,
                                "gripper_opening": opened} if robot == "panda" else None)
                            native.reset()
                            tracker = TrialTracker(success_hold_ticks=math.ceil(25.), approach_distance_m=.03,
                                table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
                            rows, reasons, abort, started = [], {}, None, time.perf_counter()
                            with (directory / "ticks.jsonl").open("w") as stream:
                                try:
                                    for tick in range(env.max_episode_steps):
                                        action = targets[min(tick, len(targets) - 1)]
                                        if tick % stride == 0 and tick < len(targets):
                                            if robot == "so101":
                                                state = env.real_state()
                                            else:
                                                state, _ = source.source_state(env)
                                                state[5] = np.clip(env.gripper_opening() / env.max_opening, 0., 1.) * 90.
                                            images = {key: env.render(env.camera_bindings[key]) for key in native.camera_keys}
                                            if tick == 0:
                                                for key, image in images.items():
                                                    transfer.save_rgb(directory / (key.rsplit(".", 1)[-1] + "_initial.png"), image)
                                            native.predict_action_chunk(state, images)
                                            feature = captured.pop("features")
                                            if feature.shape != (50, 256):
                                                raise ValueError(f"Unexpected action_head features {feature.shape}")
                                            indices = tick + np.arange(50)
                                            raw_future = targets[np.minimum(indices, len(targets) - 1)]
                                            rows.append({"features": feature, "target_normalized": ((raw_future - mean) / (std + 1e-8)).astype(np.float32),
                                                "valid": indices < len(targets), "raw_state": state.astype(np.float32),
                                                "robot_id": robot_id, "episode_index": seed, "tick": tick})
                                        _, bounded = source.source_action(action)
                                        if robot == "so101":
                                            current = env.sim_state()
                                            so_guard.data.qpos[:] = env.data.qpos
                                            accepted, reason, _ = check_joint_request(so_guard, current,
                                                env.data.ctrl[env.actuator_ids], bounded,
                                                env.max_velocity / env.control_hz, samples=17)
                                            _, _, _, truncated, info = env.step(bounded if accepted else current)
                                        else:
                                            desired = transfer.retarget.target_from_source(source, bounded)
                                            observed = env.ee_pose()
                                            delta = desired[:3, 3] - observed[:3, 3]
                                            if np.linalg.norm(delta) > .024:
                                                desired[:3, 3] = observed[:3, 3] + delta * .024 / np.linalg.norm(delta)
                                            percent = float(source.mapping.gripper_sim_to_real_percent(bounded[5]))
                                            width = transfer.retarget.panda_opening_from_percent(percent, env.max_opening)
                                            solved = panda_ik.solve(desired, env.arm_qpos(), finger_opening=max(width, env.gripper_opening()))
                                            accepted, reason = solved.accepted, solved.reason
                                            command = np.r_[solved.qpos if accepted else env.arm_qpos(), width]
                                            _, _, _, truncated, info = env.step(command)
                                        reasons[reason] = reasons.get(reason, 0) + 1
                                        pm = info["privileged_metrics"]
                                        done = tracker.update(pm, ik_rejected=not accepted,
                                            saturation=float(np.mean(info["control"]["limit_clipped"])))
                                        stream.write(json.dumps({"tick": tick, "source_command_degree_percent": action,
                                            "accepted": accepted, "reason": reason, "readback": env.sim_state(),
                                            "scoring_only": {key: pm[key] for key in ("ball_xyz_task", "tool_xyz_task",
                                                "ball_held", "lifted", "ball_in_bowl", "ball_resting", "ee_ball_distance_m")}},
                                            default=json_default, allow_nan=False) + "\n")
                                        if done or truncated:
                                            break
                                except (ValueError, RuntimeError) as exc:
                                    abort = f"{type(exc).__name__}: {exc}"
                            outcome = tracker.row(experiment_id="HF_HEAD_TEACHER_FEATURES_20261028", controller_source="SO101_TEACHER_COMMAND_REPLAY",
                                robot=robot, seed=seed, checkpoint_sha256=None, checkpoint_step=None,
                                mean_latency_ms=None, video_path="", abort=abort)
                            included = bool(outcome["pick_place_success"])
                            np.savez_compressed(directory / "anchors.npz", **features_pack(rows))
                            if included:
                                cache[partition].extend(rows)
                            report["trials"].append({**outcome, "partition": partition, "included_in_supervision": included,
                                "anchors": len(rows), "guard_reasons": reasons, "wall_s": time.perf_counter() - started})
                            for name in cache:
                                np.savez_compressed(out / f"{name}.npz", **features_pack(cache[name]))
                            report["supervised_anchors"] = {name: len(items) for name, items in cache.items()}
                            report["summary"] = summarize(report["trials"])
                            save(out / "report.json", report)
                            print(json.dumps({"robot": robot, "seed": seed, "partition": partition,
                                "teacher_success": included, "anchors": len(rows), "phase": outcome["phase_failed"],
                                "abort": abort}), flush=True)
        finally:
            hook_handle.remove()
        after = {path: sha256_file(path) for path in before}
        report.update(status="complete", source_sha256_after=after,
            runtime_source_changes=[path for path in before if before[path] != after[path]],
            checkpoint_unchanged=native.weights_sha256 == sha256_file(Path(snapshot) / "model.safetensors"))
        save(out / "report.json", report)
        return report
    finally:
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=transfer.native.SNAPSHOT)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--val-seeds", type=int, nargs="+", default=[200, 201])
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    run(args.out, args.snapshot, train_seeds=args.train_seeds, val_seeds=args.val_seeds, stride=args.stride)
