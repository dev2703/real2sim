"""Unique executed-prefix error for all v2 action representations; no simulator ground truth inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.paired import file_sha256
from real2sim.policy.validation import action_metrics, executed_prefix_report


def metrics(pred, label, *, action_kind):
    return action_metrics(pred, label, action_kind=action_kind)


def audit(checkpoint, data, device="cpu", episodes=None, annotations=None):
    policy, norm, contract = load_checkpoint(checkpoint, device)
    manifest = json.loads((data / "manifest.json").read_text())
    for key in ("camera_keys", "state_names", "action_names", "split", "policy_hz"):
        if contract[key] != manifest[key]:
            raise ValueError(f"checkpoint/data mismatch: {key}")
    if contract["derived_files_sha256"] != manifest["files_sha256"]:
        raise ValueError("checkpoint data hashes differ")
    episodes = manifest["split"]["val"] if episodes is None else episodes
    if not set(episodes) <= set(manifest["split"]["train"] + manifest["split"]["val"]):
        raise ValueError("offline tuning diagnostics cannot consume reserved episodes")
    kind = manifest["action_representation"]
    ds = DerivedTickDataset(data, episodes, policy.config.chunk_size, norm)
    n = int(policy.config.n_action_steps)
    if n != 2:
        raise ValueError("v2 uses two executed actions per call")
    a = ds.arrays
    anchors = [i for i, t in enumerate(ds.indices) if (t - a["ep_start"][t]) % n == 0]
    ticks = []
    predictions = []
    anchor_ticks = []
    for batch in DataLoader(Subset(ds, anchors), batch_size=16, shuffle=False):
        obs = {k: v.to(device) for k, v in batch.items() if k.startswith("observation.")}
        kwargs = (
            {"sampling_seed": [int(t) for t in batch["tick"]]}
            if getattr(policy.config, "type", None) == "small_chunk_head"
            else {}
        )
        with torch.inference_mode():
            pred = norm.unaction(policy.predict_action_chunk(obs, **kwargs)[:, :n]).cpu().numpy()
        valid = ~batch["action_is_pad"][:, :n].numpy()
        grid = batch["tick"].numpy()[:, None] + np.arange(n)[None, :]
        ticks.extend(grid[valid].tolist())
        anchor_ticks.extend(np.broadcast_to(batch["tick"].numpy()[:, None], valid.shape)[valid].tolist())
        predictions.extend(pred[valid])
    ticks = np.asarray(ticks, dtype=int)
    pred = np.asarray(predictions)
    expected = np.flatnonzero(np.isin(a["episode_index"], episodes) & ~a["label_is_extrapolated"])
    if not np.array_equal(ticks, expected) or not np.isfinite(pred).all():
        raise ValueError("invalid prediction or nonunique executed-prefix coverage")
    label = a["action"][ticks]
    canonical = a.get("canonical_state", a["state"])
    active = np.linalg.norm(canonical[ticks, 9:12], axis=1) > 0.005
    groups = {"all": np.ones(len(ticks), bool), "active": active, "stationary": ~active}
    if annotations:
        rows = {r["episode_index"]: r for r in annotations["episodes"]}
        phase = np.full(len(ticks), "unannotated", dtype=object)
        for i, t in enumerate(ticks):
            boundaries = rows[int(a["episode_index"][t])]["phases_source_frame"]
            for name in ("approach", "grasp", "lift", "release"):
                start = boundaries.get(name)
                if start is not None and a["frame_index"][t] >= start:
                    phase[i] = name
        groups.update(
            {
                name: phase == name
                for name in ("approach", "grasp", "lift", "release", "unannotated")
            }
        )
    per_episode = {}
    for ep in episodes:
        mask = a["episode_index"][ticks] == ep
        per_episode[str(ep)] = metrics(pred[mask], label[mask], action_kind=kind)
        if kind == "cartesian":
            per_episode[str(ep)]["summed_delta_error_mm"] = float(
                np.linalg.norm((pred[mask, :3] - label[mask, :3]).sum(axis=0)) * 1000
            )
    return {
        "executed_prefix": executed_prefix_report(pred, ticks, anchor_ticks, a, manifest, norm.action_std, episodes),
        "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
        "action_representation": kind,
        "episodes": episodes,
        "valid_actions": len(ticks),
        "camera_keys": contract["camera_keys"],
        "scores": {
            name: metrics(pred[mask], label[mask], action_kind=kind)
            for name, mask in groups.items()
        },
        "per_episode": per_episode,
        "active_threshold_m_per_s": 0.005,
        "note": "Teacher-forced unique executed prefixes. Summed deltas are a drift proxy, not rollout success. Training episodes measure fitting only.",
    }


def profile(checkpoint, data, device, repeats=30):
    policy, norm, contract = load_checkpoint(checkpoint, device)
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["files_sha256"] != contract["derived_files_sha256"]:
        raise ValueError("profile data differs from checkpoint")
    ds = DerivedTickDataset(data, manifest["split"]["val"], policy.config.chunk_size, norm)
    timings = []
    for i in range(repeats + 3):
        start = time.perf_counter()
        batch = torch.utils.data.default_collate([ds[0]])
        obs = {k: v.to(device) for k, v in batch.items() if k.startswith("observation.")}
        with torch.inference_mode():
            output = norm.unaction(policy.predict_action_chunk(obs)).cpu().numpy()
        elapsed = (time.perf_counter() - start) * 1000
        if not np.isfinite(output).all():
            raise ValueError("nonfinite output")
        if i >= 3:
            timings.append(elapsed)
    return {
        "repeats": repeats,
        "camera_keys": contract["camera_keys"],
        "p95_ms": float(np.quantile(timings, 0.95)),
        "median_ms": float(np.median(timings)),
        "passes_120ms": bool(np.quantile(timings, 0.95) < 120),
        "includes": "cached-image access, normalization, tensor transfer, all vision/head computation, action denormalization and output transfer; excludes sensor acquisition/rendering",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, nargs="+")
    p.add_argument("--annotations", type=Path)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.out.exists():
        raise FileExistsError(a.out)
    report = (
        profile(a.checkpoint, a.data, a.device)
        if a.profile
        else audit(
            a.checkpoint,
            a.data,
            a.device,
            a.episodes,
            json.loads(a.annotations.read_text()) if a.annotations else None,
        )
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
