"""Bounded CPU checkpoint inference on recorded validation inputs, without simulator claims."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from real2sim.config import load_yaml
from real2sim.geometry.so101_fk import JointMapping, So101FK
from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.paired import file_sha256
from real2sim.sim.cameras import camera_bindings


def select_anchors(dataset, per_episode):
    """Distinct two-action anchors spread across each validation episode."""
    if per_episode < 1:
        raise ValueError("per_episode must be positive")
    a = dataset.arrays
    chosen = []
    for episode in dataset.manifest["split"]["val"]:
        candidates = [i for i, tick in enumerate(dataset.indices)
                      if a["episode_index"][tick] == episode
                      and (tick - a["ep_start"][tick]) % 2 == 0
                      and not a["label_is_extrapolated"][tick]]
        offsets = np.unique(np.linspace(0, len(candidates) - 1,
                                       min(per_episode, len(candidates))).astype(int))
        chosen.extend(candidates[i] for i in offsets)
    return chosen


def simulator_preflight(task_path, sim_path, contract):
    """Explain why this inference smoke is offline, without inventing registration."""
    task, sim = load_yaml(task_path), load_yaml(sim_path)
    declared = set(camera_bindings(task, "so101"))
    fk = So101FK(mapping=JointMapping.from_task_config(task))
    initial = fk.mapping.arm_real_to_sim_rad(sim["initial_qpos_real_deg"])
    return {
        "configured_calibrated_task_exists": Path(sim["task_config"]).exists(),
        "missing_recorded_camera_mounts": sorted(set(contract["camera_keys"]) - declared),
        "configured_initial_arm_joint_limit_violations": int(np.sum(
            (initial < fk.joint_range[:, 0]) | (initial > fk.joint_range[:, 1]))),
        "physical_calibration_accepted": task.get("physical_calibration_accepted", False),
        "closed_loop_executed": False,
    }


def smoke(checkpoint, data, task_path, sim_path, per_episode):
    load_start = time.perf_counter()
    policy, norm, contract = load_checkpoint(checkpoint, "cpu")
    load_s = time.perf_counter() - load_start
    manifest = json.loads((data / "manifest.json").read_text())
    for key in ("camera_keys", "state_names", "action_names", "split", "policy_hz"):
        if manifest[key] != contract[key]:
            raise ValueError(f"checkpoint/data mismatch: {key}")
    if contract["derived_files_sha256"] != manifest["files_sha256"]:
        raise ValueError("checkpoint/data hashes differ")
    if file_sha256(data / "arrays.npz") != manifest["files_sha256"]["arrays.npz"]:
        raise ValueError("array payload differs from manifest")
    if file_sha256(task_path) != manifest["calibration_sha256"]:
        raise ValueError("task mapping differs from checkpoint")
    if manifest["action_representation"] != "cartesian" or policy.config.n_action_steps != 2:
        raise ValueError("this smoke requires two-action Cartesian prefixes")
    ds = DerivedTickDataset(data, manifest["split"]["val"], policy.config.chunk_size, norm)
    chosen = select_anchors(ds, per_episode)
    if set(manifest["split"]["val"]) & set(manifest["split"]["reserved"]):
        raise ValueError("validation/reserved split overlaps")
    predictions, swapped_predictions, labels, timings, selected = [], [], [], [], []
    full_shape = None
    # Keep two complete camera views paired. Shift to a different episode's observation.
    stride = len(chosen) // len(manifest["split"]["val"])
    for count, index in enumerate(chosen):
        start = time.perf_counter()
        batch = default_collate([ds[index]])
        obs = {key: value for key, value in batch.items() if key.startswith("observation.")}
        with torch.inference_mode():
            output = norm.unaction(policy.predict_action_chunk(obs)).cpu().numpy()
        timings.append((time.perf_counter() - start) * 1000)
        if not np.isfinite(output).all():
            raise ValueError("nonfinite policy output")
        if output.shape != (1, policy.config.chunk_size, 7):
            raise ValueError(f"unexpected output shape: {output.shape}")
        full_shape = list(output.shape)
        swapped = dict(obs)
        other = ds[chosen[(count + stride) % len(chosen)]]
        for key in manifest["camera_keys"]:
            swapped[key] = other[key][None]
        with torch.inference_mode():
            changed = norm.unaction(policy.predict_action_chunk(swapped)).cpu().numpy()
        if not np.isfinite(changed).all():
            raise ValueError("nonfinite swapped-image policy output")
        tick = int(batch["tick"][0])
        selected.append({"episode": int(batch["episode_index"][0]), "tick": tick,
                         "source_frame": int(ds.arrays["frame_index"][tick])})
        for offset in range(2):
            if not batch["action_is_pad"][0, offset]:
                predictions.append(output[0, offset])
                swapped_predictions.append(changed[0, offset])
                labels.append(ds.arrays["action"][tick + offset])
    pred, swapped, label = map(np.asarray, (predictions, swapped_predictions, labels))
    train_mean = norm.action_mean
    def score(p):
        return {
            "translation_mean_mm": float(np.linalg.norm(p[:, :3] - label[:, :3], axis=1).mean() * 1000),
            "rotation_vector_mae_rad": np.abs(p[:, 3:6] - label[:, 3:6]).mean(axis=0).tolist(),
            "gripper_mae_fraction": float(np.abs(p[:, -1] - label[:, -1]).mean()),
        }
    steady = np.asarray(timings[3:])
    report = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
        "checkpoint_metadata_sha256": {name: file_sha256(checkpoint / name) for name in (
            "config.json", "stats.json", "real2sim_contract.json", "run_info.json")},
        "manifest_sha256": file_sha256(data / "manifest.json"),
        "model_load_seconds": load_s, "device": "cpu", "torch_threads": torch.get_num_threads(),
        "camera_keys": contract["camera_keys"], "action_names": contract["action_names"],
        "units": "dx/dy/dz metres; d_rx/d_ry/d_rz rotation-vector radians; gripper fraction",
        "full_output_shape": full_shape, "executed_prefix_actions": len(pred),
        "selected_validation_anchors": selected, "reserved_episodes_used": [], "all_outputs_finite": True,
        "prediction": score(pred), "paired_images_from_other_episode": score(swapped),
        "training_mean_baseline": score(np.broadcast_to(train_mean, label.shape)),
        "zero_translation_baseline_mean_mm": float(np.linalg.norm(label[:, :3], axis=1).mean() * 1000),
        "image_swap_translation_change_mean_mm": float(np.linalg.norm(pred[:, :3] - swapped[:, :3], axis=1).mean() * 1000),
        "image_swap_gripper_change_mean_fraction": float(np.abs(pred[:, -1] - swapped[:, -1]).mean()),
        "action_translation_norm_max_m": float(np.linalg.norm(pred[:, :3], axis=1).max()),
        "action_rotation_norm_max_rad": float(np.linalg.norm(pred[:, 3:6], axis=1).max()),
        "gripper_range": [float(pred[:, -1].min()), float(pred[:, -1].max())],
        "latency": {"warmup_ms": timings[:3], "steady_calls": len(steady),
                    "median_ms": float(np.median(steady)), "p95_ms": float(np.quantile(steady, .95)),
                    "includes": "cached image access, normalization, vision/head forward, denormalization; batch size 1; image-swap calls excluded"},
        "simulator_preflight": simulator_preflight(task_path, sim_path, contract),
    }
    del policy, ds
    gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=Path("outputs/v2/learning_screen_20261022"))
    parser.add_argument("--jobs", nargs="+", default=["L1_resnet_iphone_top", "L7_clip_last2_lr3e-06"])
    parser.add_argument("--data", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone_top"))
    parser.add_argument("--task", type=Path, default=Path("configs/v2/task.exploratory_20261022.yaml"))
    parser.add_argument("--sim", type=Path, default=Path("configs/v2/sim_so101.yaml"))
    parser.add_argument("--per-episode", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    status = json.loads((args.screen / "STATUS.json").read_text())
    report = {"kind": "RECORDED_VALIDATION_CPU_INFERENCE_SMOKE", "learned_task_success": None,
              "physical_calibration_accepted": False, "script_sha256": file_sha256(Path(__file__)),
              "limitations": "Sparse teacher-forced inputs. Camera swaps measure sensitivity, not causal visual grounding or task success. No simulator rollout, retraining, hardware or reserved episodes.",
              "models": {}}
    for name in args.jobs:
        job = status["jobs"][name]
        if job["status"] != "completed":
            raise ValueError(f"job is not complete: {name}")
        report["models"][name] = smoke(Path(job["best_checkpoint"]), args.data, args.task,
                                       args.sim, args.per_episode)
        print(name, json.dumps(report["models"][name]["prediction"]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
