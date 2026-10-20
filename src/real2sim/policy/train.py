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
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from lerobot.policies.act.modeling_act import ACTPolicy
from torch.utils.data import DataLoader

from real2sim.config import load_yaml
from real2sim.policy.act import (
    build_act_config,
    build_policy,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.losses import ReconstructionLoss, act_training_loss
from real2sim.policy.paired import PairedViewDataset, file_sha256
from real2sim.policy.sampling import ResumableBatches, restore_rng, rng_state
from real2sim.policy.schedule import learning_rate

HISTORY_FIELDS = ["step", "split", "loss", "l1", "kld", "reconstruction", "weighted_kl", "trajectory", "grad_norm", "l1_raw_mean", "l1_raw_per_dim", "lr", "lr_backbone", "sec_per_step", "wall_s", "samples"]


def update_last_link(checkpoints: Path, target_name: str) -> None:
    """Point ``checkpoints/last`` at ``target_name`` atomically (no window without a ``last``)."""
    update_checkpoint_link(checkpoints, target_name, "last")


def update_checkpoint_link(checkpoints: Path, target_name: str, name: str) -> None:
    tmp = checkpoints / f".{name}.tmp.{os.getpid()}"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target_name)
    os.replace(tmp, checkpoints / name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prune_checkpoints(checkpoints: Path, created: set[str]) -> None:
    """Optional retention, limited to numbered checkpoint directories created by this train call."""
    keep = {(checkpoints / name).resolve().name for name in ("best", "last")}
    for name in sorted(created - keep):
        path = checkpoints / name
        if not (name.isdigit() and len(name) == 6) or path.is_symlink():
            raise ValueError(f"refusing to prune unexpected checkpoint: {path}")
        shutil.rmtree(path)
        created.remove(name)


def to_device(batch: dict, device: str) -> dict:
    # Blocking copies: an async host->MPS copy from an unpinned tensor that is freed right after
    # this call read reused memory and produced garbage/NaN losses (observed 2026-10-12).
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(policy: ACTPolicy, loader: DataLoader, device: str, action_std: np.ndarray, max_batches: int | None, detailed: bool = False) -> dict:
    policy.eval()
    abs_sum = torch.zeros(len(action_std), dtype=torch.float64)
    count = 0.0
    n_samples = 0
    errors = []
    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        batch = to_device(batch, device)
        pred = policy.predict_action_chunk(batch)
        if not torch.isfinite(pred).all():
            raise FloatingPointError("non-finite validation prediction")
        valid = (~batch["action_is_pad"]).unsqueeze(-1).float()
        err = (pred - batch["action"]).abs() * valid
        abs_sum += err.sum(dim=(0, 1)).cpu().double()
        count += valid.sum().item()
        n_samples += pred.shape[0]
        if detailed:
            errors.append((pred - batch["action"])[~batch["action_is_pad"]].detach().cpu().double().numpy())
    policy.train()
    per_dim_norm = (abs_sum / max(count, 1)).numpy()
    per_dim_raw = per_dim_norm * action_std
    result = {
        "l1": float(per_dim_norm.mean()),
        "l1_raw_per_dim": [round(float(x), 4) for x in per_dim_raw],
        "l1_raw_mean": float(per_dim_raw.mean()),
        "samples": n_samples,
    }
    if detailed:
        if not count:
            raise ValueError("no valid validation actions")
        error = np.concatenate(errors)
        if not np.isfinite(error).all():
            raise FloatingPointError("non-finite validation error")
        result.update(
            rmse=float(np.sqrt(np.mean(error ** 2))),
            rmse_raw_per_dim=(np.sqrt(np.mean(error ** 2, axis=0)) * action_std).tolist(),
            p95_abs=float(np.percentile(np.abs(error), 95)),
            p95_abs_raw_per_dim=(np.percentile(np.abs(error), 95, axis=0) * action_std).tolist(),
            valid_action_components=int(error.size),
        )
    return result


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
    loss_config = ReconstructionLoss(**cfg.get("reconstruction_loss", {}))
    tcfg = cfg["train"]
    tcfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    schedule_steps = int(tcfg.get("schedule_steps", tcfg["steps"]))
    if schedule_steps < int(tcfg["steps"]):
        raise ValueError("schedule_steps must cover all requested training steps")
    if smoke_steps is not None:
        tcfg.update(steps=smoke_steps, log_every=1, val_every=smoke_steps, save_every=smoke_steps, val_max_batches=2)
    device = resolve_device(tcfg.get("device", "auto"))
    precision = tcfg.get("precision", "float32")
    if precision not in {"float32", "bfloat16"} or (precision == "bfloat16" and device != "cuda"):
        raise ValueError("precision must be float32, or bfloat16 on CUDA")
    seed_everything(int(tcfg["seed"]))

    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest.get("contract_version"):
        for name, digest in manifest["files_sha256"].items():
            if Path(name).name != name or file_sha256(data_dir / name) != digest:
                raise ValueError(f"v2 data checksum mismatch: {name}")
    chunk = int(cfg["model"]["chunk_size"])
    train_episodes = tcfg.get("diagnostic_episodes", manifest["split"]["train"])
    tiny_fit = "diagnostic_episodes" in tcfg
    if tiny_fit:
        conditions = manifest["split"].get("conditions", {})
        if (len(train_episodes) != 3 or not set(train_episodes) <= set(manifest["split"]["train"])
                or {conditions.get(str(e)) for e in train_episodes} != {"baseline", "ball_shift", "bowl_shift"}):
            raise ValueError("tiny-fit diagnostics require three training episodes, one per condition")
        if (loss_config.trajectory_weight or cfg.get("augmentation") or cfg.get("paired_images") or cfg["model"].get("use_vae", True)
                or cfg["model"].get("dropout", .1) != 0 or cfg["model"].get("optimizer_weight_decay", .05) != 0
                or int(tcfg["steps"]) * int(tcfg["batch_size"]) * int(tcfg.get("grad_accum", 1)) > 32000):
            raise ValueError("tiny fit must be deterministic and unregularized, capped at 32000 anchors")
    train_ds = DerivedTickDataset(data_dir, train_episodes, chunk,
                                 augmentation=cfg.get("augmentation"))
    val_ds = DerivedTickDataset(data_dir, train_episodes if tiny_fit else manifest["split"]["val"], chunk, normalizer=train_ds.normalizer)
    assert not set(manifest["split"]["train"]) & set(manifest["split"]["val"])
    paired_provenance = None
    if cfg.get("paired_images"):
        pcfg = cfg["paired_images"]
        train_ds = PairedViewDataset(train_ds, pcfg["root"], pcfg["probability"])
        paired_provenance = {**pcfg, "manifest_sha256": file_sha256(Path(pcfg["root"]) / "manifest.json"),
                             "training_data_kind": "real_labels_with_mixed_real_and_sim_approach_images"}

    act_cfg = build_act_config(cfg["model"], manifest, device)
    policy = build_policy(act_cfg, initialize_backbone=resume is None).to(device)
    initial_provenance = None
    if cfg.get("initial_checkpoint"):
        if resume is not None:
            raise ValueError("initial_checkpoint and resume are mutually exclusive")
        initial = Path(cfg["initial_checkpoint"])
        parent, parent_norm, parent_contract = load_checkpoint(initial, device="cpu")
        for key in ("source", "split", "camera_keys", "state_names", "action_names", "policy_hz", "label_semantics"):
            if parent_contract[key] != manifest[key]:
                raise ValueError(f"initial checkpoint {key} differs from data")
        if parent_contract["derived_files_sha256"] != manifest["files_sha256"] or parent_norm.stats != train_ds.normalizer.stats:
            raise ValueError("initial checkpoint data hashes or normalization differ")
        transfer = None
        if cfg.get("initialization") == "nonvisual_act":
            from real2sim.policy.dinov2_act import warmstart_nonvisual

            if act_cfg.vision_backbone != "dinov2":
                raise ValueError("nonvisual initialization is restricted to the DINO pilot")
            transfer = warmstart_nonvisual(policy, parent)
        else:
            policy.load_state_dict(parent.state_dict(), strict=True)
        del parent
        initial_provenance = {"path": str(initial), "sha256": file_sha256(initial / "model.safetensors"),
                              "optimizer": "fresh; weights-only initialization", "transfer": transfer}
    policy.train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=act_cfg.optimizer_lr, weight_decay=act_cfg.optimizer_weight_decay)
    start_step = 0
    if resume is not None:
        previous_loss = json.loads((resume / "run_info.json").read_text()).get("reconstruction_loss", {})
        if ReconstructionLoss(**previous_loss) != loss_config:
            raise ValueError("cannot resume with a different reconstruction loss")
        resumed, resumed_norm, resumed_contract = load_checkpoint(resume, device="cpu")
        for key in ("source", "split", "camera_keys", "state_names", "action_names", "policy_hz", "label_semantics",
                    "contract_version", "controller_semantics", "calibration_sha256", "camera_transforms"):
            if resumed_contract.get(key) != manifest.get(key):
                raise ValueError(f"resume checkpoint {key} differs from data")
        if resumed_contract["derived_files_sha256"] != manifest["files_sha256"] or resumed_norm.stats != train_ds.normalizer.stats:
            raise ValueError("resume checkpoint data hashes or normalization differ")
        policy.load_state_dict(resumed.state_dict(), strict=True)
        del resumed
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        if cfg.get("scheduler") != json.loads((resume / "run_info.json").read_text()).get("scheduler"):
            raise ValueError("cannot resume with a different scheduler")
        if schedule_steps != json.loads((resume / "run_info.json").read_text()).get("scheduler_total_steps", schedule_steps):
            raise ValueError("cannot resume with a different scheduler horizon; declare schedule_steps before screening")

    schedule = cfg.get("scheduler")
    peaks = [act_cfg.optimizer_lr, act_cfg.optimizer_lr_backbone]
    if schedule:
        # Validate before beginning; absolute step determines LR on resume.
        learning_rate(1, schedule_steps, min(peaks), **schedule)

    bs = int(tcfg["batch_size"])
    accum = int(tcfg.get("grad_accum", 1))
    workers = int(tcfg.get("num_workers", 0))
    exact_resume = bool(tcfg.get("exact_resume", False))
    if exact_resume and workers:
        raise ValueError("exact_resume requires num_workers=0 (no prefetched stochastic samples)")
    train_loader = None if exact_resume else make_loader(train_ds, bs, True, workers, int(tcfg["seed"]), drop_last=True)
    val_loader = make_loader(val_ds, bs, False, workers, 0, drop_last=False)
    steps = int(tcfg["steps"])
    if len(train_ds) < bs or bs < 1 or accum < 1 or steps <= start_step:
        raise ValueError("training requires a full positive batch and steps beyond the resume step")
    patience = int(tcfg.get("early_stopping_patience", 0))
    min_steps = int(tcfg.get("early_stopping_min_steps", 0))
    min_delta = float(tcfg.get("early_stopping_min_delta", 0.0))
    if patience < 0 or min_steps < 0 or not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("invalid early stopping configuration")

    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "config": cfg,
        "device": device,
        "act_config": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(act_cfg).items()},
        "data_manifest": {k: manifest[k] for k in ("source", "split", "camera_keys", "label_semantics", "policy_hz", "files_sha256", "prepare_version")},
        "n_train_ticks": len(train_ds),
        "validation_scope": "tiny-fit training episodes (not generalization)" if tiny_fit else "validation episodes",
        "n_val_ticks": len(val_ds),
        "n_params": sum(p.numel() for p in policy.parameters()),
        "n_trainable_params": sum(p.numel() for p in policy.parameters() if p.requires_grad),
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
    it = ResumableBatches(train_ds, bs, int(tcfg["seed"])) if exact_resume else infinite(train_loader)
    if resume is not None and exact_resume:
        continuation = state.get("continuation")
        if not continuation or continuation["grad_accum"] != accum:
            raise ValueError("exact resume requires saved sampler/RNG and identical accumulation")
        if continuation["augmentation"] != cfg.get("augmentation") or continuation["paired_images"] != paired_provenance:
            raise ValueError("resume augmentation or paired image provenance differs")
        if continuation.get("train_episodes", manifest["split"]["train"]) != train_episodes:
            raise ValueError("resume sampled training episodes differ")
        it.load_state_dict(continuation["sampler"])
        restore_rng(continuation["rng"])
    t_start = time.time()
    t_window = time.time()
    run = {"loss": 0.0, "l1": 0.0, "kld": 0.0, "reconstruction": 0.0, "trajectory": 0.0, "grad_norm": 0.0, "n": 0}
    created_checkpoints: set[str] = set()
    last_val = None
    best_l1, patience_reference, bad_validations, best_step = math.inf, math.inf, 0, None
    if resume is not None:
        previous = json.loads((resume / "run_info.json").read_text())
        best_l1 = previous.get("best_val_l1", math.inf)
        best_step = previous.get("best_step")
        patience_reference = previous.get("patience_reference", best_l1)
        bad_validations = previous.get("bad_validations", 0)
    stopped_early = False
    print(f"device={device} params={resolved['n_params'] / 1e6:.1f}M train_ticks={len(train_ds)} val_ticks={len(val_ds)} bs={bs}x{accum}", file=sys.stderr)
    for step in range(start_step + 1, steps + 1):
        improved = False
        if schedule:
            for group, peak in zip(optimizer.param_groups, peaks, strict=True):
                group["lr"] = learning_rate(step, schedule_steps, peak, **schedule)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            batch = to_device(next(it), device)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()
            with autocast:
                loss, ld = act_training_loss(
                    policy, batch, loss_config,
                    action_mean=train_ds.normalizer.action_mean,
                    action_std=train_ds.normalizer.action_std,
                    action_representation=manifest.get("action_representation"),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {ld}")
            (loss / accum).backward()
            run["loss"] += loss.item() / accum
            run["l1"] += ld["l1_loss"] / accum
            run["kld"] += ld.get("kld_loss", 0.0) / accum
            run["reconstruction"] += ld["reconstruction_loss"] / accum
            run["trajectory"] += ld.get("trajectory_loss", 0.0) / accum
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(tcfg["grad_clip_norm"]))
        if not math.isfinite(float(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        run["grad_norm"] += float(grad_norm)
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
                "reconstruction": round(run["reconstruction"] / run["n"], 6),
                "trajectory": round(run["trajectory"] / run["n"], 6),
                "weighted_kl": round(act_cfg.kl_weight * run["kld"] / run["n"], 6),
                "grad_norm": round(run["grad_norm"] / run["n"], 6),
                "lr": optimizer.param_groups[0]["lr"],
                "lr_backbone": optimizer.param_groups[1]["lr"],
                "sec_per_step": round(dt, 4),
                "wall_s": round(time.time() - t_start, 1),
                "samples": run["n"] * bs * accum,
            }
            writer.writerow(row)
            hist_f.flush()
            print(f"step {step} loss {row['loss']:.4f} l1 {row['l1']:.4f} kld {row['kld']:.4f} {dt:.3f}s/step grad_norm {float(grad_norm):.2f} reconstruction[{loss_config.kind}] {row['reconstruction']:.4f} weighted_kl {row['weighted_kl']:.4f}", file=sys.stderr)
            run = {"loss": 0.0, "l1": 0.0, "kld": 0.0, "reconstruction": 0.0, "trajectory": 0.0, "grad_norm": 0.0, "n": 0}
            t_window = time.time()

        if step % int(tcfg["val_every"]) == 0 or step == steps:
            last_val = evaluate(policy, val_loader, device, action_std, tcfg.get("val_max_batches"))
            if not math.isfinite(last_val["l1"]):
                raise FloatingPointError("non-finite validation loss")
            improved = last_val["l1"] < best_l1
            if improved:
                best_l1, best_step = last_val["l1"], step
            if last_val["l1"] < patience_reference - min_delta:
                patience_reference, bad_validations = last_val["l1"], 0
            else:
                bad_validations += 1
            stopped_early = bool(patience and step >= min_steps and bad_validations >= patience)
            writer.writerow({"step": step, "split": "val", "l1": round(last_val["l1"], 6), "l1_raw_mean": round(last_val["l1_raw_mean"], 4), "l1_raw_per_dim": json.dumps(last_val["l1_raw_per_dim"]), "wall_s": round(time.time() - t_start, 1), "samples": last_val["samples"]})
            hist_f.flush()
            print(f"  val step {step}: l1 {last_val['l1']:.4f} raw mean {last_val['l1_raw_mean']:.3f} per-dim {last_val['l1_raw_per_dim']}", file=sys.stderr)
            t_window = time.time()

        if improved or stopped_early or step % int(tcfg["save_every"]) == 0 or step == steps:
            ck = out_dir / "checkpoints" / f"{step:06d}"
            save_checkpoint(ck, policy, data_dir / "stats.json", manifest, {
                "step": step, "seed": int(tcfg["seed"]), "name": cfg.get("name"), "device": device, "last_val": last_val,
                "best_val_l1": best_l1, "best_step": best_step,
                "patience_reference": patience_reference, "bad_validations": bad_validations,
                "augmentation": cfg.get("augmentation"),
                "diagnostic_episodes": train_episodes if tiny_fit else None,
                "paired_images": paired_provenance, "initial_checkpoint": initial_provenance,
                "reconstruction_loss": asdict(loss_config),
                "scheduler": schedule, "scheduler_total_steps": schedule_steps,
                "examples_seen": step * bs * accum,
                "wall_s": round(time.time() - t_start, 1)}, optimizer,
                training_state={"sampler": it.state_dict(), "rng": rng_state(), "grad_accum": accum,
                                "augmentation": cfg.get("augmentation"), "paired_images": paired_provenance,
                                "train_episodes": train_episodes}
                if exact_resume else None)
            update_last_link(out_dir / "checkpoints", ck.name)
            if improved:
                update_checkpoint_link(out_dir / "checkpoints", ck.name, "best")
            created_checkpoints.add(ck.name)
            if tcfg.get("keep_only_best_and_last", False):
                prune_checkpoints(out_dir / "checkpoints", created_checkpoints)
            print(f"  saved {ck}", file=sys.stderr)
            t_window = time.time()
        if stopped_early:
            print(f"Early stopping at {step}; best validation {best_l1:.6f} at {best_step}", file=sys.stderr)
            break
    hist_f.close()
    result = {"steps": step, "requested_steps": steps, "stopped_early": stopped_early,
              "reconstruction_loss": asdict(loss_config),
              "best_step": best_step, "best_val_l1": best_l1, "last_val": last_val,
              "paired_images": paired_provenance, "initial_checkpoint": initial_provenance,
              "out": str(out_dir), "device": device, "wall_s": round(time.time() - t_start, 1)}
    (out_dir / "training_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


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
