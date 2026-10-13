"""Gate-G1 check: load a trained checkpoint, verify its data contract, predict on held-out episodes.

    uv run python -m real2sim.policy.check --checkpoint outputs/act_joint_so101/checkpoints/last \
        --data .cache/derived_joint --out results/metrics/act_joint_g1_check.json

This is an offline open-loop check on recorded validation episodes. It is NOT a closed-loop or
task-success result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from real2sim.policy.act import load_checkpoint, resolve_device
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.train import evaluate


def check(checkpoint: Path, data: Path, device: str = "auto", max_batches: int | None = None) -> dict:
    device = resolve_device(device)
    ck = checkpoint.resolve()
    policy, norm, contract = load_checkpoint(ck, device=device)
    manifest = json.loads((data / "manifest.json").read_text())
    problems = []
    for key in ("camera_keys", "state_names", "action_names", "label_semantics"):
        if contract[key] != manifest[key]:
            problems.append(f"{key} differs between checkpoint and data")
    if contract["split"]["sha256"] != manifest["split"]["sha256"]:
        problems.append("split hash differs")
    if contract.get("derived_files_sha256") != manifest.get("files_sha256"):
        problems.append("derived data files differ from the ones the checkpoint was trained on")
    if problems:
        raise ValueError("; ".join(problems))
    val = DerivedTickDataset(data, manifest["split"]["val"], policy.config.chunk_size, normalizer=norm)
    loader = DataLoader(val, batch_size=16, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    obs = {k: v.to(device) for k, v in batch.items() if k.startswith("observation.")}
    with torch.no_grad():
        chunk = policy.predict_action_chunk(obs)
    raw = norm.unaction(chunk).cpu().numpy()
    metrics = evaluate(policy, loader, device, np.asarray(norm.action_std, np.float64), max_batches, detailed=True)
    run_info = json.loads((ck / "run_info.json").read_text())
    with (ck / "model.safetensors").open("rb") as weights:
        checkpoint_sha256 = hashlib.file_digest(weights, "sha256").hexdigest()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_resolved": str(ck),
        "checkpoint_sha256": checkpoint_sha256,
        "step": run_info["step"],
        "device": device,
        "contract_ok": True,
        "camera_keys": contract["camera_keys"],
        "image_shape_chw": contract["image_shape_chw"],
        "chunk_shape": list(chunk.shape),
        "action_names": contract["action_names"],
        "first_chunk_first_action_raw": [round(float(x), 4) for x in raw[0, 0]],
        "all_finite": bool(np.isfinite(raw).all()),
        "val_episodes": manifest["split"]["val"],
        "val_open_loop_l1_normalised": metrics["l1"],
        "val_open_loop_l1_raw_per_dim": metrics["l1_raw_per_dim"],
        "val_open_loop_rmse_normalised": metrics["rmse"],
        "val_open_loop_rmse_raw_per_dim": metrics["rmse_raw_per_dim"],
        "val_open_loop_p95_abs_normalised": metrics["p95_abs"],
        "val_open_loop_p95_abs_raw_per_dim": metrics["p95_abs_raw_per_dim"],
        "valid_action_components": metrics["valid_action_components"],
        "reconstruction_loss": run_info.get("reconstruction_loss", {"kind": "l1"}),
        "val_samples": metrics["samples"],
        "dataset_revision": contract["source"]["revision"],
        "split_sha256": contract["split"]["sha256"],
        "note": "Offline open-loop prediction error on held-out recorded episodes; not closed-loop success. RMSE and absolute-error p95 pool valid chunk action components; raw-unit metrics are per dimension. Overlapping chunks are not independent samples.",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m real2sim.policy.check", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-batches", type=int, default=None)
    a = p.parse_args(argv)
    res = check(Path(a.checkpoint), Path(a.data), a.device, a.max_batches)
    text = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(text, file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
