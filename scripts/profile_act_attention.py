"""Profile real ACT forwards/backwards; distinguish SDPA from actual Flash kernels."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml

from real2sim.policy.act import build_act_config, build_policy
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.losses import ReconstructionLoss, act_training_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batches", nargs="+", type=int, default=[16])
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data = Path(cfg["data"])
    manifest = json.loads((data / "manifest.json").read_text())
    ds = DerivedTickDataset(data, manifest["split"]["train"], 20)
    torch.set_num_threads(2)
    result = {"device": args.device, "torch": str(torch.__version__), "profiles": []}
    for batch_size in args.batches:
        for backend in ("legacy", "sdpa"):
            torch.manual_seed(0)
            model_cfg = dict(cfg["model"], attention_backend=backend)
            policy = build_policy(build_act_config(model_cfg, manifest, args.device)).to(args.device).train()
            batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=batch_size)))
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            sync = torch.cuda.synchronize if args.device == "cuda" else torch.mps.synchronize
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _iteration in range(3):
                policy.zero_grad(set_to_none=True)
                sync()
                started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    loss, _ = act_training_loss(policy, batch, ReconstructionLoss())
                loss.backward()
                sync()
                times.append(time.perf_counter() - started)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite profile loss")
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    act_training_loss(policy, batch, ReconstructionLoss())[0].backward()
                sync()
            kernels = sorted({event.key for event in profile.key_averages()
                              if "attention" in event.key or "scaled_dot" in event.key})
            memory = (torch.cuda.max_memory_allocated() if args.device == "cuda"
                      else torch.mps.driver_allocated_memory())
            result["profiles"].append({"backend_requested": backend, "batch_size": batch_size,
                                       "forward_backward_s": times[1:], "allocated_bytes": memory,
                                       "memory_metric": "cuda_peak_allocated" if args.device == "cuda" else "mps_driver_current",
                                       "attention_ops": kernels,
                                       "flash_observed": any("flash_attention" in name for name in kernels)})
            del policy, batch, profile
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
            else:
                torch.mps.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
