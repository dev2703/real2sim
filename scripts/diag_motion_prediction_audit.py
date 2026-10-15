"""Frozen real-image ACT audit: executed prefixes, motion strata and phase proxies.

This script does not alter labels, checkpoints, split or benchmark results. Phase
proxies describe recorded kinematics; they do not establish a successful grasp.
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.paired import file_sha256


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95))}


def metrics(pred, label, std):
    if not len(label):
        return {"n": 0}
    error = pred - label
    pn, ln = np.linalg.norm(pred[:, :3], axis=1), np.linalg.norm(label[:, :3], axis=1)
    pr, lr = np.linalg.norm(pred[:, 3:6], axis=1), np.linalg.norm(label[:, 3:6], axis=1)
    directional = (ln > 0.001) & (pn > 1e-8)
    cos = np.sum(pred[directional, :3] * label[directional, :3], axis=1) / (
        pn[directional] * ln[directional]
    )
    return {
        "n": len(label),
        "mae_normalized": float(np.abs(error / std).mean()),
        "mae_raw_per_dim": np.abs(error).mean(axis=0).tolist(),
        "rmse_raw_per_dim": np.sqrt(np.square(error).mean(axis=0)).tolist(),
        "translation_vector_error_m": describe(np.linalg.norm(error[:, :3], axis=1)),
        "rotation_vector_error_rad": describe(np.linalg.norm(error[:, 3:6], axis=1)),
        "label_translation_m": describe(ln), "pred_translation_m": describe(pn),
        "label_rotation_rad": describe(lr), "pred_rotation_rad": describe(pr),
        "mean_translation_magnitude_ratio": float(pn.mean() / ln.mean()) if ln.mean() else None,
        "pred_translation_below_3mm_fraction": float((pn <= 0.003).mean()),
        "pred_translation_below_0_1mm_fraction": float((pn <= 0.0001).mean()),
        "label_translation_below_3mm_fraction": float((ln <= 0.003).mean()),
        "translation_direction_cosine_mean_when_label_above_1mm": float(cos.mean()) if len(cos) else None,
        "direction_rows": int(directional.sum()),
        "pred_gripper_outside_unit_interval_fraction": float(((pred[:, 6] < 0) | (pred[:, 6] > 1)).mean()),
    }


def phase_masks(arrays, ticks, approach_ticks):
    label = arrays["action"][ticks]
    state = arrays["state"][ticks]
    dp = np.linalg.norm(label[:, :3], axis=1)
    dr = np.linalg.norm(label[:, 3:6], axis=1)
    active = (dp > 0.001) | (dr > 0.01)
    # Every scored row has a real successor; extrapolated labels are excluded.
    closing = arrays["state"][ticks + 1, -1] - state[:, -1] <= -0.01
    return {
        "all": np.ones(len(ticks), dtype=bool),
        "arm_active": active,
        "arm_stationary": ~active,
        "translation_above_3mm": dp > 0.003,
        "near_zero_arm": (dp <= 0.0001) & (dr <= 0.001),
        "reviewed_stationary_ball_approach_proxy": np.isin(ticks, approach_ticks),
        "observed_gripper_closure_proxy": closing,
        "upward_with_gripper_at_most_45pct_proxy": (label[:, 2] > 0.001) & (state[:, -1] <= 0.45),
    }


def predict(checkpoint, data, expected_ticks, device, batch_size):
    policy, norm, contract = load_checkpoint(checkpoint, device=device)
    dataset = DerivedTickDataset(data, contract["split"]["val"], policy.config.chunk_size, norm)
    for key in ("state_names", "action_names", "camera_keys", "label_semantics"):
        if contract[key] != dataset.manifest[key]:
            raise ValueError(f"checkpoint/data contract mismatch: {key}")
    if (contract["derived_files_sha256"] != dataset.manifest["files_sha256"]
            or contract["split"] != dataset.manifest["split"]):
        raise ValueError("checkpoint/data checksum or split mismatch")
    if policy.config.n_action_steps != 2:
        raise ValueError("this audit requires the frozen two-action execution cadence")
    arrays = dataset.arrays
    anchors = [i for i, t in enumerate(dataset.indices) if (t - arrays["ep_start"][t]) % 2 == 0]
    loader = DataLoader(Subset(dataset, anchors), batch_size=batch_size, num_workers=0, shuffle=False)
    raw_predictions, ticks, anchor_ticks, prefix_offsets = [], [], [], []
    policy.eval()
    with torch.inference_mode():
        for batch in loader:
            obs = {k: v.to(device) for k, v in batch.items() if k.startswith("observation.")}
            prediction = norm.unaction(policy.predict_action_chunk(obs)[:, :2]).cpu().numpy()
            valid = (~batch["action_is_pad"][:, :2]).numpy()
            offsets = np.broadcast_to(np.arange(2), valid.shape)
            anchors_grid = np.broadcast_to(batch["tick"].numpy()[:, None], valid.shape)
            ticks.extend((anchors_grid + offsets)[valid].tolist())
            anchor_ticks.extend(anchors_grid[valid].tolist())
            prefix_offsets.extend(offsets[valid].tolist())
            raw_predictions.append(prediction[valid])
    raw = np.concatenate(raw_predictions)
    ticks = np.asarray(ticks, dtype=np.int64)
    if not np.array_equal(ticks, expected_ticks) or len(np.unique(ticks)) != len(ticks):
        raise ValueError("non-overlapping execution prefixes did not cover each valid label exactly once")
    if not np.isfinite(raw).all():
        raise ValueError("non-finite policy output")
    result = {
        "raw": raw, "std": norm.action_std.copy(), "anchor_ticks": np.asarray(anchor_ticks),
        "prefix_offsets": np.asarray(prefix_offsets),
        "metadata": {"checkpoint": str(checkpoint), "sha256": file_sha256(checkpoint / "model.safetensors"),
                     "chunk_size": policy.config.chunk_size, "executed_prefix": 2,
                     "inference_observations": len(anchors)},
    }
    del policy, dataset, loader
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("outputs/act_cartesian/checkpoints/006000"))
    parser.add_argument("--pilot", type=Path, default=Path("outputs/act_cartesian_dinov2_pilot_20261015/checkpoints/001250"))
    parser.add_argument("--data", type=Path, default=Path(".cache/derived_cartesian"))
    parser.add_argument("--pairs", type=Path, default=Path(".cache/paired_cartesian_refit_20261015/val"))
    parser.add_argument("--out", type=Path, default=Path("results/audit/motion_prediction_20261015"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "executed_script.txt").write_text(Path(__file__).read_text())
    torch.set_num_threads(1)
    manifest = json.loads((args.data / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if file_sha256(args.data / name) != expected:
            raise ValueError(f"derived file checksum mismatch: {name}")
    paired_manifest = json.loads((args.pairs / "manifest.json").read_text())
    if (paired_manifest["derived_manifest"] != manifest or paired_manifest["split_name"] != "val"
            or file_sha256(args.pairs / "ticks.npy") != paired_manifest["sha256"]["ticks.npy"]):
        raise ValueError("reviewed approach ticks have incompatible provenance")
    approach_ticks = np.load(args.pairs / "ticks.npy", allow_pickle=False)
    arrays = dict(np.load(args.data / "arrays.npz", allow_pickle=False))
    selected = np.isin(arrays["episode_index"], manifest["split"]["val"])
    ticks = np.flatnonzero(selected & ~arrays["label_is_extrapolated"])
    label = arrays["action"][ticks]
    results = {}
    for name, checkpoint in (("parent", args.parent), ("pilot", args.pilot)):
        results[name] = predict(checkpoint, args.data, ticks, args.device, args.batch_size)
        print(f"Completed {name}: {len(ticks)} held-out actions", flush=True)
    if not np.array_equal(results["parent"]["std"], results["pilot"]["std"]):
        raise ValueError("checkpoint normalizers differ")
    anchors = results["parent"]["anchor_ticks"]
    if not np.array_equal(anchors, results["pilot"]["anchor_ticks"]):
        raise ValueError("checkpoint inference cadence differs")
    zero = np.zeros_like(label)
    zero[:, 6] = arrays["state"][anchors, -1]
    predictions = {name: result["raw"] for name, result in results.items()}
    predictions["zero_arm_current_gripper_hold"] = zero
    masks = phase_masks(arrays, ticks, approach_ticks)
    std = results["parent"]["std"]
    report = {
        "kind": "HELD_OUT_REAL_IMAGE_MOTION_STRATIFIED_OFFLINE_AUDIT",
        "created_at": datetime.now(UTC).isoformat(), "device": args.device,
        "source": manifest["source"], "split": manifest["split"],
        "derived_files_sha256": manifest["files_sha256"],
        "script_sha256": file_sha256(Path(__file__)),
        "action_names": manifest["action_names"],
        "label_semantics": manifest["label_semantics"],
        "checkpoints": {name: result["metadata"] for name, result in results.items()},
        "protocol": {
            "coverage": "All held-out valid actions scored exactly once: episode-local anchors 0,2,4,..., first two predicted actions; exclude extrapolated terminal labels and padding.",
            "observation": "Recorded real iPhone image and recorded 13-D state at each anchor; no simulated images or rollout feedback.",
            "zero_baseline": "Six zero arm deltas; absolute gripper target held at the observed gripper at the inference anchor for both prefix actions. No future gripper input.",
            "prediction_clipping": "Raw denormalized predictions, before execution-side delta/gripper clipping; identical to offline error convention.",
            "arm_active": "Recorded translation norm >1 mm OR rotation-vector norm >0.01 rad per 100 ms tick; stationary is the exact complement.",
            "near_zero_arm": "Recorded translation <=0.1 mm AND rotation <=0.001 rad per tick.",
            "reviewed_stationary_ball_approach_proxy": "Action tick present in the existing reviewed stationary-ball paired-image eligibility file. A phase proxy, not an independent grasp annotation.",
            "observed_gripper_closure_proxy": "Next recorded follower gripper minus current <=-0.01 (one percentage point) per tick; includes any closure anywhere in an episode.",
            "upward_with_gripper_at_most_45pct_proxy": "Recorded dz >1 mm and current follower gripper <=45%; describes upward relatively-closed-hand motion, not confirmed object lift.",
            "phase_overlap": "Phase proxies can overlap; only active/stationary partition the dataset.",
            "interpretation": "Single fixed thresholds are descriptive, not tuned to either model. No fitting, retraining, label edits, split changes or closed-loop success claims. Cartesian measurements inherit the existing provisional real-to-sim FK mapping. Temporal samples within an episode remain correlated.",
        },
        "valid_actions": len(ticks), "excluded_terminal_actions": int((selected & arrays["label_is_extrapolated"]).sum()),
        "strata": {group: {name: metrics(pred[mask], label[mask], std) for name, pred in predictions.items()}
                   for group, mask in masks.items()},
        "episodes": {str(ep): {
            "counts": {group: int((mask & (arrays["episode_index"][ticks] == ep)).sum()) for group, mask in masks.items()},
            "metrics": {name: metrics(pred[arrays["episode_index"][ticks] == ep], label[arrays["episode_index"][ticks] == ep], std)
                        for name, pred in predictions.items()},
        } for ep in manifest["split"]["val"]},
        "prefix_offsets": {str(offset): {name: metrics(pred[results["parent"]["prefix_offsets"] == offset],
                                                       label[results["parent"]["prefix_offsets"] == offset], std)
                                          for name, pred in predictions.items()} for offset in range(2)},
    }
    np.savez_compressed(args.out / "predictions.npz", tick=ticks, episode=arrays["episode_index"][ticks],
                        frame=arrays["frame_index"][ticks], timestamp=arrays["timestamp"][ticks],
                        anchor_tick=anchors, prefix_offset=results["parent"]["prefix_offsets"],
                        label=label, **predictions, **{f"mask_{k}": v for k, v in masks.items()})
    report["predictions_sha256"] = file_sha256(args.out / "predictions.npz")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Frozen ACT motion audit", "", "Held-out real images; every valid action scored once using the executed two-action prefix.", "",
             "| Recorded-motion stratum | Actions | Parent L1 | DINO L1 | Zero-arm/hold-gripper L1 | Parent / DINO / zero translation error (mm) |",
             "|---|---:|---:|---:|---:|---:|"]
    for group, values in report["strata"].items():
        if values["parent"]["n"]:
            error = " / ".join(f"{values[n]['translation_vector_error_m']['mean'] * 1000:.2f}" for n in predictions)
            lines.append(f"| {group} | {values['parent']['n']} | {values['parent']['mae_normalized']:.4f} | {values['pilot']['mae_normalized']:.4f} | {values['zero_arm_current_gripper_hold']['mae_normalized']:.4f} | {error} |")
    lines.extend(["", "L1 is normalized across seven action dimensions; translation error is the mean Euclidean error in raw metres, shown above in mm. These executed-prefix metrics differ from the existing overlapping full-20-action validation.", "",
                  "The zero baseline holds the gripper observed at inference for both actions; it does not access future states. Active means >1 mm translation or >0.01 rad rotation per 100 ms. Closure and upward-motion proxies do not prove contact or a successful lift. Full definitions, per-episode results, raw per-dimension errors and checkpoint hashes are in report.json; predictions.npz allows independent reanalysis.", ""])
    (args.out / "review.md").write_text("\n".join(lines))
    print(json.dumps({"out": str(args.out), "valid_actions": len(ticks), "stratum_counts": {k: int(v.sum()) for k, v in masks.items()}}), flush=True)


if __name__ == "__main__":
    main()
