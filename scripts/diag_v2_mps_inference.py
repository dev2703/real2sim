"""Bounded, synchronized batch-one MPS profiling on matched recorded inputs."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from diag_v2_inference_smoke import select_anchors
from torch.utils.data import default_collate

from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.paired import file_sha256


def profile(checkpoint: Path, data: Path, repeats: int) -> dict:
    policy, norm, contract = load_checkpoint(checkpoint, "mps")
    manifest = json.loads((data / "manifest.json").read_text())
    for key in ("camera_keys", "state_names", "action_names", "split", "policy_hz"):
        if manifest[key] != contract[key]:
            raise ValueError(f"checkpoint/data mismatch: {key}")
    if manifest["files_sha256"] != contract["derived_files_sha256"]:
        raise ValueError("checkpoint/data hashes differ")
    ds = DerivedTickDataset(data, manifest["split"]["val"], policy.config.chunk_size, norm)
    anchors = select_anchors(ds, 4)
    selected = []
    timings, warmup = [], []
    for call in range(repeats + 3):
        index = anchors[max(0, call - 3) % len(anchors)]
        torch.mps.synchronize()
        start = time.perf_counter()
        batch = default_collate([ds[index]])
        obs = {key: value.to("mps") for key, value in batch.items()
               if key.startswith("observation.")}
        with torch.inference_mode():
            output = norm.unaction(policy.predict_action_chunk(obs)).cpu().numpy()
        torch.mps.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if not np.isfinite(output).all():
            raise ValueError("nonfinite output")
        if call < 3:
            warmup.append(elapsed)
        else:
            timings.append(elapsed)
            selected.append({"tick": int(batch["tick"][0]),
                             "episode": int(batch["episode_index"][0])})
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
        "metadata_sha256": {name: file_sha256(checkpoint / name) for name in (
            "config.json", "stats.json", "real2sim_contract.json", "run_info.json")},
        "manifest_sha256": file_sha256(data / "manifest.json"),
        "camera_keys": contract["camera_keys"], "warmup_ms": warmup,
        "calls": repeats, "distinct_anchors": len(anchors),
        "selected_validation_anchors": selected,
        "all_outputs_finite": True, "timings_ms": timings,
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.quantile(timings, 0.95)),
        "passes_120ms": bool(np.quantile(timings, 0.95) < 120),
    }
    del policy, ds, batch, obs, output
    gc.collect()
    torch.mps.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=Path("outputs/v2/learning_screen_20261022"))
    parser.add_argument("--data", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone_top"))
    parser.add_argument("--jobs", nargs="+", default=["L1_resnet_iphone_top", "L7_clip_last2_lr3e-06"])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.repeats < 24:
        raise ValueError("use at least 24 calls to cover every selected anchor")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; this diagnostic does not substitute CPU")
    torch.set_num_threads(2)
    status = json.loads((args.screen / "STATUS.json").read_text())
    report = {
        "kind": "RECORDED_VALIDATION_MPS_INFERENCE_PROFILE", "device": "mps",
        "machine": platform.machine(), "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(), "batch_size": 1,
        "script_sha256": file_sha256(Path(__file__)),
        "synchronization": "torch.mps.synchronize immediately before and after each timed call",
        "includes": "cached image access, normalization, collation, input transfer, all vision/head computation, action denormalization, output transfer",
        "excludes": "model loading, sensor acquisition and simulator rendering",
        "reserved_episodes_used": [], "learned_task_success": None, "models": {},
    }
    for name in args.jobs:
        job = status["jobs"][name]
        if job["status"] != "completed":
            raise ValueError(f"job is not complete: {name}")
        report["models"][name] = profile(Path(job["best_checkpoint"]), args.data, args.repeats)
        print(name, report["models"][name]["median_ms"], report["models"][name]["p95_ms"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
