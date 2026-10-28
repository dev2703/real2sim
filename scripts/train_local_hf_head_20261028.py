"""Fit only a local residual head to cached frozen-HF decoder features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from real2sim.artifacts import sha256_file
from real2sim.policy.local_hf_head import ResidualActionHead
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.serialization import write_json

SNAPSHOT = Path(".cache/hf_models_20261027/models--bottowork--so101-act-h256-b16/snapshots/e696c8a078935432e3501e2c28c1fb2598fbff7e/latest")


def validate_feature_source(data, snapshot):
    """A decoder feature cache is specific to the exact frozen base weights."""
    provenance = json.loads((data / "report.json").read_text())
    if provenance.get("status") != "complete":
        raise ValueError("Feature collection must be complete before training")
    if provenance.get("checkpoint_unchanged") is not True or provenance.get("runtime_source_changes"):
        raise ValueError("Feature collection changed its checkpoint or runtime sources")
    if provenance.get("weights_sha256") != sha256_file(snapshot / "model.safetensors"):
        raise ValueError("Cached decoder features belong to a different HF checkpoint; recollect them")
    return provenance


def load_rows(path, prefix=10):
    with np.load(path, allow_pickle=False) as data:
        valid = data["valid"][:, :prefix].astype(bool)
        features = data["features"][:, :prefix][valid].astype(np.float32)
        targets = data["target_normalized"][:, :prefix][valid].astype(np.float32)
        robots = np.broadcast_to(data["robot_id"][:, None], valid.shape)[valid]
        episodes = np.broadcast_to(data["episode_index"][:, None], valid.shape)[valid]
    if not len(features) or not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Empty or non-finite cached supervision")
    return torch.from_numpy(features), torch.from_numpy(targets), robots, episodes


def evaluate(head, features, targets, robots):
    with torch.no_grad():
        errors = (head(features) - targets).abs().numpy()
    per_robot = {str(robot): errors[robots == robot].mean(0).tolist() for robot in np.unique(robots)}
    return {"mean_absolute_normalized_error": float(errors.mean()),
            "per_robot_joint_mae_normalized": per_robot}


def run(data, out, snapshot=SNAPSHOT, *, kind="mlp", steps=2000):
    validate_feature_source(data, snapshot)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    x, y, robots, episodes = load_rows(data / "train.npz")
    vx, vy, vrobots, vepisodes = load_rows(data / "val.npz")
    if set(episodes) & set(vepisodes):
        raise ValueError("Training and validation source episodes overlap")
    base = NativeLeRobotACT(snapshot)
    base.policy.eval().requires_grad_(False)
    head = ResidualActionHead(base.policy.model.action_head, x.mean(0), x.std(0), kind)
    optimizer = torch.optim.AdamW(head.residual.parameters(), lr=5e-4, weight_decay=1e-4)
    weights = np.array([1 / np.sum(robots == robot) for robot in robots], dtype=np.float64)
    sampling = torch.from_numpy(weights / weights.sum())
    source_hashes = {str(p): sha256_file(p) for p in (
        data / "train.npz", data / "val.npz", Path(__file__),
        Path("src/real2sim/policy/local_hf_head.py"), snapshot / "model.safetensors")}
    trainable = sum(p.numel() for p in head.residual.parameters())
    frozen = sum(p.numel() for p in base.policy.parameters())
    report = {"schema": "local_hf_residual_action_head_20261028_v1", "status": "training",
        "kind": kind, "base_checkpoint_sha256": base.weights_sha256, "base_hf_weights_frozen": True,
        "trainable_parameters": trainable, "original_frozen_parameters": frozen,
        "local_only": True, "hf_upload_performed": False, "lighting_variations": False,
        "supervised_prefix_steps": 10, "policy_hz": 25., "seed": 42,
        "data": str(data), "train_rows": len(x), "validation_rows": len(vx),
        "train_episodes": np.unique(episodes).tolist(), "val_episodes": np.unique(vepisodes).tolist(),
        "train_rows_by_robot": {str(r): int(np.sum(robots == r)) for r in np.unique(robots)},
        "selection": "Minimum held-out source-episode validation MAE; no rollout outcome selection",
        "source_sha256_before": source_hashes, "baseline_validation": evaluate(head, vx, vy, vrobots),
        "history": []}
    write_json(out / "report.json", report)
    best, best_step, started = float("inf"), None, time.perf_counter()
    for step in range(1, steps + 1):
        head.train()
        indices = torch.multinomial(sampling, min(256, len(x)), replacement=True)
        prediction = head(x[indices])
        loss = torch.nn.functional.smooth_l1_loss(prediction, y[indices], beta=.1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.residual.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == steps:
            head.eval()
            validation = evaluate(head, vx, vy, vrobots)
            metric = validation["mean_absolute_normalized_error"]
            entry = {"step": step, "training_loss": float(loss.detach()), **validation}
            report["history"].append(entry)
            if metric < best:
                best, best_step = metric, step
                torch.save(head.state_dict(), out / "head.pt")
            write_json(out / "report.json", report)
            print(json.dumps(entry), flush=True)
    after = {p: sha256_file(p) for p in source_hashes}
    if after != source_hashes:
        raise RuntimeError("Training inputs or frozen source changed")
    metadata = {"schema": report["schema"], "kind": kind,
        "base_checkpoint_sha256": base.weights_sha256, "head_sha256": sha256_file(out / "head.pt"),
        "best_step": best_step, "validation_mae_normalized": best,
        "supervised_prefix_steps": 10, "policy_hz": 25.,
        "trainable_parameters": trainable, "original_frozen_parameters": frozen,
        "local_only": True, "training_report": str(out / "report.json")}
    write_json(out / "head.json", metadata)
    report.update(status="complete", best_step=best_step, best_validation_mae_normalized=best,
                  source_sha256_after=after, runtime_source_changes=[],
                  training_wall_s=time.perf_counter() - started, head=metadata)
    write_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--kind", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    run(args.data, args.out, args.snapshot, kind=args.kind, steps=args.steps)
