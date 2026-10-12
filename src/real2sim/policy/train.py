"""Train LeRobot ACT on a derived real-data directory.

    uv run python -m real2sim.policy.train --config configs/act_joint.yaml \
        --data .cache/derived_joint --out outputs/act_joint_so101 [--smoke-steps 2] [--resume DIR]

Writes ``<out>/checkpoints/<step>/`` (LeRobot ``save_pretrained`` + stats + data contract +
optimizer state), ``<out>/training_history.csv`` and ``<out>/resolved_config.yaml``. Validation
runs the policy in eval mode (no VAE posterior, i.e. inference conditions) on whole held-out
episodes and reports L1 in normalised units and in raw units per action dimension.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from lerobot.policies.act.modeling_act import ACTPolicy
from torch.utils.data import DataLoader

from real2sim.config import load_yaml
from real2sim.policy.act import build_act_config, resolve_device, save_checkpoint
from real2sim.policy.dataset import DerivedTickDataset

HISTORY_FIELDS = ["step", "split", "loss", "l1", "kld", "l1_raw_mean", "l1_raw_per_dim", "lr", "sec_per_step", "wall_s", "samples"]


def update_last_link(checkpoints: Path, target_name: str) -> None:
    """Point ``checkpoints/last`` at ``target_name`` atomically (no window without a ``last``)."""
    tmp = checkpoints / f".last.tmp.{os.getpid()}"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target_name)
    os.replace(tmp, checkpoints / "last")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch: dict, device: str) -> dict:
    # Blocking copies: an async host->MPS copy from an unpinned tensor that is freed right after
    # this call read reused memory and produced garbage/NaN losses (observed 2026-10-12).
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(policy: ACTPolicy, loader: DataLoader, device: str, action_std: np.ndarray, max_batches: int | None) -> dict:
    policy.eval()
    abs_sum = torch.zeros(len(action_std), dtype=torch.float64)
    count = 0.0
    n_samples = 0
    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        batch = to_device(batch, device)
        pred = policy.predict_action_chunk(batch)
        valid = (~batch["action_is_pad"]).unsqueeze(-1).float()
        err = (pred - batch["action"]).abs() * valid
        abs_sum += err.sum(dim=(0, 1)).cpu().double()
        count += valid.sum().item()
        n_samples += pred.shape[0]
    policy.train()
    per_dim_norm = (abs_sum / max(count, 1)).numpy()
    per_dim_raw = per_dim_norm * action_std
    return {
        "l1": float(per_dim_norm.mean()),
        "l1_raw_per_dim": [round(float(x), 4) for x in per_dim_raw],
        "l1_raw_mean": float(per_dim_raw.mean()),
        "samples": n_samples,
    }


def make_loader(ds, batch_size: int, shuffle: bool, workers: int, seed: int, drop_last: bool) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=drop_last,
        generator=g,
        persistent_workers=workers > 0,
        pin_memory=False,
    )


def infinite(loader: DataLoader):
    while True:
        yield from loader


def train(cfg: dict, data_dir: Path, out_dir: Path, smoke_steps: int | None = None, resume: Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    tcfg = cfg["train"]
    tcfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if smoke_steps is not None:
        tcfg.update(steps=smoke_steps, log_every=1, val_every=smoke_steps, save_every=smoke_steps, val_max_batches=2)
    device = resolve_device(tcfg.get("device", "auto"))
    seed_everything(int(tcfg["seed"]))

    manifest = json.loads((data_dir / "manifest.json").read_text())
    chunk = int(cfg["model"]["chunk_size"])
    train_ds = DerivedTickDataset(data_dir, manifest["split"]["train"], chunk)
    val_ds = DerivedTickDataset(data_dir, manifest["split"]["val"], chunk, normalizer=train_ds.normalizer)
    assert not set(manifest["split"]["train"]) & set(manifest["split"]["val"])

    act_cfg = build_act_config(cfg["model"], manifest, device)
    policy = ACTPolicy(act_cfg).to(device)
    policy.train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=act_cfg.optimizer_lr, weight_decay=act_cfg.optimizer_weight_decay)
    start_step = 0
    if resume is not None:
        policy.load_state_dict(ACTPolicy.from_pretrained(str(resume)).state_dict())
        state = torch.load(resume / "training_state.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])

    bs = int(tcfg["batch_size"])
    accum = int(tcfg.get("grad_accum", 1))
    workers = int(tcfg.get("num_workers", 0))
    train_loader = make_loader(train_ds, bs, True, workers, int(tcfg["seed"]), drop_last=True)
    val_loader = make_loader(val_ds, bs, False, workers, 0, drop_last=False)
    steps = int(tcfg["steps"])

    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "config": cfg,
        "device": device,
        "act_config": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(act_cfg).items()},
        "data_manifest": {k: manifest[k] for k in ("source", "split", "camera_keys", "label_semantics", "policy_hz", "files_sha256", "prepare_version")},
        "n_train_ticks": len(train_ds),
        "n_val_ticks": len(val_ds),
        "n_params": sum(p.numel() for p in policy.parameters()),
        "platform": {"python": platform.python_version(), "torch": str(torch.__version__), "machine": platform.machine(), "system": platform.system()},
        "smoke_steps": smoke_steps,
    }
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    hist_path = out_dir / "training_history.csv"
    kept: list[dict] = []
    if resume is not None and hist_path.exists():
        # Drop rows logged after the resumed checkpoint so steps are never duplicated.
        with hist_path.open(newline="") as f:
            kept = [r for r in csv.DictReader(f) if int(r["step"]) <= start_step]
    hist_f = hist_path.open("w", newline="")
    writer = csv.DictWriter(hist_f, fieldnames=HISTORY_FIELDS)
    writer.writeheader()
    writer.writerows(kept)

    action_std = np.asarray(train_ds.normalizer.action_std, np.float64)
    it = infinite(train_loader)
    t_start = time.time()
    t_window = time.time()
    run = {"loss": 0.0, "l1": 0.0, "kld": 0.0, "n": 0}
    last_val = None
    print(f"device={device} params={resolved['n_params'] / 1e6:.1f}M train_ticks={len(train_ds)} val_ticks={len(val_ds)} bs={bs}x{accum}", file=sys.stderr)
    for step in range(start_step + 1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            batch = to_device(next(it), device)
            loss, ld = policy.forward(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {ld}")
            (loss / accum).backward()
            run["loss"] += loss.item() / accum
            run["l1"] += ld["l1_loss"] / accum
            run["kld"] += ld.get("kld_loss", 0.0) / accum
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(tcfg["grad_clip_norm"]))
        if not math.isfinite(float(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        run["n"] += 1

        if step % int(tcfg["log_every"]) == 0 or step == steps:
            if device == "mps":
                torch.mps.synchronize()
            dt = (time.time() - t_window) / run["n"]
            row = {
                "step": step,
                "split": "train",
                "loss": round(run["loss"] / run["n"], 6),
                "l1": round(run["l1"] / run["n"], 6),
                "kld": round(run["kld"] / run["n"], 6),
                "lr": act_cfg.optimizer_lr,
                "sec_per_step": round(dt, 4),
                "wall_s": round(time.time() - t_start, 1),
                "samples": run["n"] * bs * accum,
            }
            writer.writerow(row)
            hist_f.flush()
            print(f"step {step} loss {row['loss']:.4f} l1 {row['l1']:.4f} kld {row['kld']:.4f} {dt:.3f}s/step grad_norm {float(grad_norm):.2f}", file=sys.stderr)
            run = {"loss": 0.0, "l1": 0.0, "kld": 0.0, "n": 0}
            t_window = time.time()

        if step % int(tcfg["val_every"]) == 0 or step == steps:
            last_val = evaluate(policy, val_loader, device, action_std, tcfg.get("val_max_batches"))
            writer.writerow({"step": step, "split": "val", "l1": round(last_val["l1"], 6), "l1_raw_mean": round(last_val["l1_raw_mean"], 4), "l1_raw_per_dim": json.dumps(last_val["l1_raw_per_dim"]), "wall_s": round(time.time() - t_start, 1), "samples": last_val["samples"]})
            hist_f.flush()
            print(f"  val step {step}: l1 {last_val['l1']:.4f} raw mean {last_val['l1_raw_mean']:.3f} per-dim {last_val['l1_raw_per_dim']}", file=sys.stderr)
            t_window = time.time()

        if step % int(tcfg["save_every"]) == 0 or step == steps:
            ck = out_dir / "checkpoints" / f"{step:06d}"
            save_checkpoint(ck, policy, data_dir / "stats.json", manifest, {"step": step, "name": cfg.get("name"), "device": device, "last_val": last_val, "wall_s": round(time.time() - t_start, 1)}, optimizer)
            update_last_link(out_dir / "checkpoints", ck.name)
            print(f"  saved {ck}", file=sys.stderr)
            t_window = time.time()
    hist_f.close()
    return {"steps": steps, "last_val": last_val, "out": str(out_dir), "device": device, "wall_s": round(time.time() - t_start, 1)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m real2sim.policy.train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data", default=None, help="derived dataset dir (default: config 'data')")
    p.add_argument("--out", default=None, help="output dir (default: outputs/<name>)")
    p.add_argument("--smoke-steps", type=int, default=None, help="run N steps with tiny validation and save")
    p.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_yaml(args.config)
    data = Path(args.data or cfg["data"])
    out = Path(args.out or f"outputs/{cfg['name']}")
    res = train(
        cfg,
        data,
        out,
        smoke_steps=args.smoke_steps,
        resume=Path(args.resume) if args.resume else None,
        overrides={"device": args.device, "batch_size": args.batch_size, "grad_accum": args.grad_accum, "steps": args.steps, "num_workers": args.num_workers},
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
