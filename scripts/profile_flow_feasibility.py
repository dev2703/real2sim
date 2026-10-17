"""Bounded local F0 smoke: 64 anchors/objective and full-vision inference timing.

This does not train a useful policy or modify the comparative experiment queue.
"""

import argparse
import json
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from run_so101_experiments import atomic_json, queue_lock
from torch.utils.data import DataLoader, Subset

from real2sim.policy.dataset import DerivedTickDataset, Normalizer
from real2sim.policy.flow_matching import (
    FEATURE_KEY,
    FlowHeadConfig,
    FlowHeadPolicy,
    build_frozen_backbone,
)
from real2sim.policy.paired import file_sha256


def synchronize():
    torch.mps.synchronize()


def run(out):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the actual laptop feasibility check")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    data = Path(".cache/experiments_20261016/canonical")
    manifest = json.loads((data / "manifest.json").read_text())
    norm = Normalizer.from_json(data / "stats.json")
    dataset = DerivedTickDataset(data, manifest["split"]["train"], 20, norm)
    valid_indices = np.flatnonzero(~dataset.arrays["label_is_extrapolated"][dataset.indices])
    indices = np.random.default_rng(0).choice(valid_indices, 64, replace=False).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=16, shuffle=False)
    model = yaml.safe_load(Path("outputs/experiments_20261016_stage1/configs/A1D.yaml").read_text())["model"]
    cfg = FlowHeadConfig(dinov2_config=model["dinov2_config"], dinov2_model_id=model["dinov2_model_id"],
                         dinov2_revision=model["dinov2_revision"], camera_key=manifest["camera_keys"][0])
    backbone = build_frozen_backbone(cfg, pretrained_path=model["dinov2_pretrained_path"]).to("mps")
    backbone.requires_grad_(False).eval()
    cached = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            features = backbone(batch[cfg.camera_key].to("mps"))["feature_map"].flatten(2).transpose(1, 2)
            if not torch.isfinite(features).all():
                raise FloatingPointError("nonfinite visual features")
            cfg.feature_tokens, cfg.feature_dim = features.shape[1:]
            cached.append({"observation.state": batch["observation.state"].to("mps"),
                           "action": batch["action"].to("mps"),
                           "action_is_pad": batch["action_is_pad"].to("mps"),
                           FEATURE_KEY: features.detach()})
    synchronize()
    report = {"kind": "F0_HARDWARE_SMOKE_NOT_POLICY_QUALITY", "device": "mps", "seed": 0,
              "config": asdict(cfg), "anchors_per_objective": 64, "optimizer_updates_per_objective": 1,
              "selected_ticks": dataset.indices[indices].tolist(), "feature_extraction_wall_s": time.perf_counter()-started,
              "dataset_manifest_sha256": file_sha256(data / "manifest.json"),
              "script_sha256": file_sha256(Path(__file__)), "objectives": {},
              "latency_includes": "CPU image/state normalization, device transfer, full frozen DINO, head sampling, action denormalization and CPU output transfer",
              "latency_excludes": "camera acquisition, robot observation read and controller execution",
              "cadence_ms": 200., "grasp_success": None, "panda_allowed": False}
    for objective in ("regression", "flow"):
        atomic_json(out / "RUNNING.json", {"stage": objective, "examples": 64})
        torch.manual_seed(0)
        policy = FlowHeadPolicy(replace(cfg, objective=objective), backbone=backbone).to("mps")
        parameters = list(policy.head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.05)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(73021)
        synchronize()
        start = time.perf_counter()
        loss_values = []
        for batch in cached:
            loss, _ = policy(batch, generator=generator)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite smoke training loss")
            (loss / 4).backward()
            loss_values.append(float(loss))
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        synchronize()
        training = {"wall_s": time.perf_counter()-start, "loss": float(np.mean(loss_values)),
                    "gradient_norm_before_clipping": float(gradient),
                    "trainable_parameters": sum(p.numel() for p in parameters)}
        if not all(torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError("nonfinite trained weights")
        policy.eval()
        del optimizer
        timings = {}
        for steps in ((4, 8) if objective == "flow" else (4,)):
            samples = []
            for i in range(33):
                tick = int(dataset.indices[indices[i % len(indices)]])
                synchronize()
                start = time.perf_counter()
                batch = {cfg.camera_key: torch.from_numpy(np.ascontiguousarray(norm.image(dataset.images[cfg.camera_key][tick])))[None].to("mps"),
                         "observation.state": torch.from_numpy(norm.state(dataset.arrays["state"][tick]))[None].to("mps")}
                prediction = norm.unaction(policy.predict_action_chunk(batch, sampling_seed=i, sampling_steps=steps)).cpu()
                synchronize()
                elapsed = (time.perf_counter()-start)*1000
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("nonfinite inference output")
                if i >= 3:
                    samples.append(elapsed)
            timings[str(steps) if objective == "flow" else "regression"] = {
                "repeats": len(samples), "warmups": 3, "median_ms": float(np.median(samples)),
                "p95_ms": float(np.quantile(samples, .95)), "max_ms": max(samples),
                "under_200ms_p95": bool(np.quantile(samples, .95) < 200)}
        report["objectives"][objective] = {"training_smoke": training, "inference": timings,
            "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "memory_note": "Snapshot allocation with shared cached features; not a measured peak."}
        print(json.dumps({objective: report["objectives"][objective]}), flush=True)
        del policy
    report["comparative_training_launched"] = False
    report["next_gate"] = "Collection timing/provenance and geometry remain unresolved. Hardware feasibility does not establish a policy advantage."
    atomic_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with queue_lock(Path("outputs/experiments_20261016_stage1")):
        args.out.mkdir(parents=True)
        try:
            run(args.out)
            atomic_json(args.out / "COMPLETE.json", {"status": "COMPLETE", "kind": "hardware_feasibility_only",
                                                      "grasp_success": None, "panda_allowed": False})
        except BaseException:
            atomic_json(args.out / "FAILED.json", {"status": "FAILED", "error": traceback.format_exc()})
            raise
        finally:
            (args.out / "RUNNING.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
