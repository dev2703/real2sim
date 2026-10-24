"""Bounded frozen-DINO flow/regression experiment with completion/failure flags.

Run cache once, then train the two objectives sequentially against the same
cache/split/budget. COMPLETE.json means the command finished, never grasp success.
Every command also writes FAILED.json on exceptions and exits nonzero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from real2sim.data.prepare import compute_stats
from real2sim.data.resample import chunk_indices
from real2sim.policy.act import resolve_device
from real2sim.policy.dataset import Normalizer
from real2sim.policy.flow_matching import (
    FEATURE_KEY,
    FlowHeadConfig,
    FlowHeadPolicy,
    build_frozen_backbone,
    load_flow_checkpoint,
    save_flow_checkpoint,
)
from real2sim.policy.paired import file_sha256
from real2sim.policy.sampling import ResumableBatches, restore_rng, rng_state
from real2sim.policy.schedule import learning_rate
from real2sim.policy.validation import VERSION as VALIDATION_VERSION
from real2sim.policy.validation import action_metrics, executed_prefix_report


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def synchronize(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


def prune_head_checkpoints(root: Path, best_examples: int, last_examples: int) -> list[str]:
    """Bound disk usage to best/last, only inside this trainer's numbered outputs."""
    keep = {f"{best_examples:09d}", f"{last_examples:09d}"}
    candidates = [p for p in root.iterdir() if p.name.isdigit() and len(p.name) == 9]
    if root.is_symlink() or any(p.is_symlink() or not p.is_dir() for p in candidates):
        raise ValueError("refusing to prune unexpected head checkpoint paths")
    removed = []
    for path in candidates:
        if path.name not in keep:
            shutil.rmtree(path)
            removed.append(path.name)
    return removed


def image_identity(manifest, arrays, stats):
    """Allow observation variants to reuse images without trusting changed states."""
    return {
        "images": {k: manifest["files_sha256"][v] for k, v in manifest["image_files"].items()},
        "shape": manifest["image_shape_hwc"], "camera_keys": manifest["camera_keys"],
        "image_normalization": stats["image_normalization"],
        "ticks_sha256": hashlib.sha256(np.stack([
            arrays[k] for k in ("episode_index", "frame_index", "row_index")
        ], axis=1).astype("<i8").tobytes()).hexdigest(),
    }


def build_cache(args):
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest.get("visual_grounding"):
        raise ValueError("visual_grounding requires ACT on-demand image features; frozen-feature caches/heads do not support this contract")
    arrays = dict(np.load(args.data / "arrays.npz", allow_pickle=False))
    norm = Normalizer.from_json(args.data / "stats.json")
    model = yaml.safe_load(args.vision_config.read_text())["model"]
    vision = model.get("vision_backbone", "dinov2")
    if vision not in ("dinov2", "clip") or model.get(f"{vision}_trainable_blocks", 0):
        raise ValueError("feature caches require an explicitly frozen DINO or CLIP encoder")
    architecture = model[f"{vision}_config"]
    config = FlowHeadConfig(
        state_dim=len(manifest["state_names"]), camera_key=manifest["camera_keys"][0],
        camera_keys=manifest["camera_keys"], feature_tokens=266 * len(manifest["camera_keys"]),
        chunk_size=getattr(args, "chunk_size", 20),
        vision_backbone=vision, feature_dim=architecture["hidden_size"],
        **{f"{vision}_{key}": model[f"{vision}_{key}"] for key in ("config", "model_id", "revision")},
    )
    device = resolve_device(args.device)
    backbone = build_frozen_backbone(config, pretrained_path=model[f"{vision}_pretrained_path"]).to(device)
    images = {}
    for key in config.camera_keys:
        image_name = manifest["image_files"][key]
        if file_sha256(args.data / image_name) != manifest["files_sha256"][image_name]:
            raise ValueError("derived images do not match their recorded hash")
        images[key] = np.load(args.data / image_name, mmap_mode="r", allow_pickle=False)
    ticks = np.arange(len(images[config.camera_keys[0]]), dtype=np.int64)
    if args.episodes:
        ticks = ticks[np.isin(arrays["episode_index"], args.episodes)]
    if not len(ticks):
        raise ValueError("no image ticks selected")
    np.save(args.out / "ticks.npy", ticks, allow_pickle=False)
    features = None
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(ticks), args.batch_size):
            views = []
            for key in config.camera_keys:
                image = np.stack([norm.image(images[key][t]) for t in ticks[start:start + args.batch_size]])
                value = backbone(torch.from_numpy(image).to(device))["feature_map"]
                views.append(value.flatten(2).transpose(1, 2).cpu().numpy().astype(np.float16))
            value = np.concatenate(views, axis=1)
            if not np.isfinite(value).all():
                raise FloatingPointError("nonfinite frozen DINO features")
            if features is None:
                config.feature_tokens, config.feature_dim = value.shape[1:]
                features = np.lib.format.open_memmap(args.out / "features.npy", mode="w+",
                                                     dtype=np.float16, shape=(len(ticks), *value.shape[1:]))
            features[start:start + len(value)] = value
            if start % (args.batch_size * 20) == 0:
                write_json(args.out / "progress.json", {"stage": "feature_cache", "ticks": start + len(value),
                           "total": len(ticks), "wall_s": time.perf_counter() - started})
    features.flush()
    save_file({k: v.detach().cpu().contiguous() for k, v in backbone.state_dict().items()},
              args.out / "backbone.safetensors")
    report = {"kind": f"frozen_{vision}_features", "config": asdict(config), "dtype": "float16",
              "augmentation": None, "image_identity": image_identity(manifest, arrays, norm.stats),
              "ticks": len(ticks), "shape": list(features.shape), "device": device,
              "wall_s": time.perf_counter() - started,
              "sha256": {name: file_sha256(args.out / name) for name in
                         ("features.npy", "ticks.npy", "backbone.safetensors")}}
    write_json(args.out / "manifest.json", report)
    return report


class CachedFeatureDataset(Dataset):
    """Episode-local actions; image features never augmented after caching."""

    def __init__(self, data: Path, cache: Path, episodes: list[int], normalizer: Normalizer,
                 chunk_size=20, *, executed_prefix=False):
        self.manifest = json.loads((data / "manifest.json").read_text())
        self.arrays = dict(np.load(data / "arrays.npz", allow_pickle=False))
        if np.any(self.arrays.get("action_is_invalid", False)):
            raise ValueError("cached heads do not support action_is_invalid filtered targets; use ACT with the shared valid-action mask")
        self.cache_manifest = json.loads((cache / "manifest.json").read_text())
        self.normalizer = normalizer
        self.chunk_size = chunk_size
        identity = image_identity(self.manifest, self.arrays, normalizer.stats)
        if identity != self.cache_manifest["image_identity"]:
            raise ValueError("cached features do not match image/tick/preprocessing provenance")
        if not (cache / "COMPLETE.json").is_file():
            raise ValueError("feature cache has no completion flag")
        self.features = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
        cached_ticks = np.load(cache / "ticks.npy", allow_pickle=False)
        if len(cached_ticks) != len(self.features) or len(np.unique(cached_ticks)) != len(cached_ticks):
            raise ValueError("invalid cache tick indexing")
        if list(self.features.shape) != self.cache_manifest["shape"]:
            raise ValueError("feature cache shape mismatch")
        self.feature_index = np.full(len(self.arrays["state"]), -1, np.int64)
        self.feature_index[cached_ticks] = np.arange(len(cached_ticks))
        selected = np.isin(self.arrays["episode_index"], episodes)
        selected &= ~self.arrays["label_is_extrapolated"]
        self.indices = np.flatnonzero(selected)
        if executed_prefix:
            self.indices = self.indices[(self.indices - self.arrays["ep_start"][self.indices]) % 2 == 0]
        if not len(self.indices) or np.any(self.feature_index[self.indices] < 0):
            raise ValueError("missing cached features for selected episode anchors")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        tick = int(self.indices[i])
        indices, pad = chunk_indices(self.arrays["ep_start"], self.arrays["ep_len"], tick, self.chunk_size)
        pad |= self.arrays["label_is_extrapolated"][indices]
        return {
            "observation.state": torch.tensor(self.normalizer.state(self.arrays["state"][tick]), dtype=torch.float32),
            FEATURE_KEY: torch.tensor(np.asarray(self.features[self.feature_index[tick]]), dtype=torch.float32),
            "action": torch.tensor(self.normalizer.action(self.arrays["action"][indices]), dtype=torch.float32),
            "action_is_pad": torch.from_numpy(pad), "tick": tick,
        }


def fit_normalizer(data: Path, train_episodes: list[int]):
    original = json.loads((data / "stats.json").read_text())
    arrays = dict(np.load(data / "arrays.npz", allow_pickle=False))
    selected = np.isin(arrays["episode_index"], train_episodes)
    if not selected.any():
        raise ValueError("normalizer has no selected training observations")
    stats = {"observation.state": compute_stats(arrays["state"][selected]),
             "action": compute_stats(arrays["action"][selected & ~arrays["label_is_extrapolated"]]),
             "image_normalization": original["image_normalization"],
             "computed_on": "inner training episodes only; extrapolated action placeholders excluded",
             "train_episodes": sorted(train_episodes)}
    return Normalizer(stats)


def resolve_split(manifest, path: Path | None):
    if manifest.get("contract_version") and path is None:
        return dict(manifest["split"])
    source_split = manifest.get("source_split", manifest["split"])
    training = sorted(source_split["train"])
    if path:
        split = json.loads(path.read_text())
        if "split" in split:
            split = split["split"]
        train, val = sorted(split["train"]), sorted(split["val"])
    elif "historical_val" in manifest["split"]:
        train, val = sorted(manifest["split"]["train"]), sorted(manifest["split"]["val"])
    else:
        val = training[4::5]
        train = sorted(set(training) - set(val))
    if not train or not val or set(train) & set(val):
        raise ValueError("inner training and validation episodes must be nonempty and disjoint")
    if not (set(train) | set(val)) <= set(training):
        raise ValueError("F0 selection may only use the original training episodes")
    if train == sorted(manifest["split"]["train"]) and val == sorted(manifest["split"]["val"]):
        # Preserve seed/hash/exclusion metadata for the common experiment audit.
        return dict(manifest["split"])
    return {"train": train, "val": val, "historical_val": source_split["val"]}


def metric(pred, label, std):
    if not len(pred):
        return {"n": 0}
    error = pred - label
    trans = np.linalg.norm(error[:, :3], axis=1)
    rotation = np.linalg.norm(error[:, 3:6], axis=1)
    magnitude = np.linalg.norm(pred[:, :3], axis=1)
    target_magnitude = np.linalg.norm(label[:, :3], axis=1)
    directional = (target_magnitude > .001) & (magnitude > 1e-8)
    cosine = (pred[directional, :3] * label[directional, :3]).sum(1) / (
        magnitude[directional] * target_magnitude[directional])
    return {**action_metrics(pred, label, action_kind="cartesian", std=std),
            "n": len(pred), "l1": float(np.abs(error / std).mean()),
            "mae_raw_per_dim": np.abs(error).mean(0).tolist(),
            "translation_mean_m": float(trans.mean()), "translation_p95_m": float(np.quantile(trans, .95)),
            "rotation_mean_rad": float(rotation.mean()), "rotation_p95_rad": float(np.quantile(rotation, .95)),
            "translation_amplitude_ratio": float(magnitude.mean() / target_magnitude.mean()) if target_magnitude.mean() else None,
            "translation_direction_cosine": float(cosine.mean()) if len(cosine) else None}


@torch.inference_mode()
def evaluate(policy, dataset, device, batch_size, seed, sampling_steps=4):
    policy.eval()
    predictions, ticks, anchors = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        obs = {k: v.to(device) for k, v in batch.items() if k.startswith("observation.")}
        # Stable one-noise draw per anchor, independent of batch size or model objective.
        seeds = [int(seed) * 1_000_003 + int(t) for t in batch["tick"]]
        pred = policy.predict_action_chunk(obs, sampling_seed=seeds, sampling_steps=sampling_steps)[:, :2]
        raw = dataset.normalizer.unaction(pred).cpu().numpy()
        valid = ~batch["action_is_pad"][:, :2].numpy()
        anchor = np.broadcast_to(batch["tick"].numpy()[:, None], valid.shape)
        ticks.extend((anchor + np.arange(2)[None, :])[valid].tolist())
        anchors.extend(anchor[valid].tolist())
        predictions.append(raw[valid])
    predictions = np.concatenate(predictions)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("nonfinite action predictions")
    ticks, anchors = np.asarray(ticks), np.asarray(anchors)
    a = dataset.arrays
    episodes = np.unique(a["episode_index"][dataset.indices])
    expected = np.flatnonzero(np.isin(a["episode_index"], episodes) & ~a["label_is_extrapolated"])
    if not np.array_equal(expected, ticks):
        raise ValueError("executed-prefix predictions did not cover each valid action exactly once")
    label = a["action"][ticks]
    canonical = a.get("canonical_state", a["state"])
    grip_index = -1 if "canonical_state" in a else dataset.manifest["state_names"].index("gripper_obs")
    zero = np.zeros_like(predictions)
    zero[:, 6] = canonical[anchors, grip_index]
    persistence = zero.copy()
    has_previous = anchors > a["ep_start"][anchors]
    persistence[has_previous, :6] = a["action"][anchors[has_previous] - 1, :6]
    active = (np.linalg.norm(label[:, :3], axis=1) > .001) | (np.linalg.norm(label[:, 3:6], axis=1) > .01)
    grip = canonical[ticks, grip_index]
    masks = {"all": np.ones(len(ticks), bool), "active": active, "stationary": ~active,
             "downward_open_approach_proxy": (label[:, 2] < -.001) & (grip > .45),
             "closure_proxy": canonical[ticks + 1, grip_index] - grip <= -.01,
             "upward_closed_lift_proxy": (label[:, 2] > .001) & (grip <= .45)}
    predictions_by_name = {"model": predictions, "zero_motion_hold_gripper": zero, "persistence": persistence}
    report = {
        "sampling_steps": sampling_steps if policy.config.objective == "flow" else 1,
        "sampling_seed": seed, "sample_selection": "single deterministic noise draw per anchor; no best-of-many",
        "coverage": "episode-local anchors 0,2,...; two executed actions; no extrapolation/padding/duplicates",
        "phase_caution": "Kinematic proxies overlap and are not independently verified grasp phase annotations.",
        "strata": {name: {p: metric(values[mask], label[mask], dataset.normalizer.action_std)
                            for p, values in predictions_by_name.items()} for name, mask in masks.items()},
        "episodes": {str(int(ep)): {
            p: {**metric(values[a["episode_index"][ticks] == ep], label[a["episode_index"][ticks] == ep],
                         dataset.normalizer.action_std),
                "integrated_translation_error_m": float(np.linalg.norm(
                    (values - label)[a["episode_index"][ticks] == ep, :3].sum(0)))}
            for p, values in predictions_by_name.items()} for ep in episodes},
    }
    common = executed_prefix_report(predictions, ticks, anchors, a, dataset.manifest,
                                    dataset.normalizer.action_std, episodes)
    report["executed_prefix"] = common
    report["offline_selection"] = common["offline_selection"]
    return report, {"tick": ticks, "anchor_tick": anchors, "label": label, **predictions_by_name}


def train_head(args):
    finalize_only = getattr(args, "finalize_only", False)
    if finalize_only and not getattr(args, "resume", None):
        raise ValueError("finalize-only requires a full-budget resume checkpoint")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest.get("visual_grounding"):
        raise ValueError("visual_grounding requires ACT on-demand image features; frozen-feature caches/heads do not support this contract")
    if manifest["action_names"] != ["dx", "dy", "dz", "d_rx", "d_ry", "d_rz", "gripper_target"]:
        raise ValueError("F0 requires the unchanged seven-dimensional Cartesian action contract")
    split = resolve_split(manifest, args.split_json)
    existing_normalizer = Normalizer.from_json(args.data / "stats.json")
    normalizer = (existing_normalizer if sorted(existing_normalizer.stats.get("train_episodes", [])) == split["train"]
                  else fit_normalizer(args.data, split["train"]))
    cache = json.loads((args.cache / "manifest.json").read_text())
    for name, expected in cache["sha256"].items():
        if file_sha256(args.cache / name) != expected:
            raise ValueError(f"feature cache checksum mismatch: {name}")
    config = FlowHeadConfig(**{**cache["config"], "objective": args.objective,
                               "state_dim": len(manifest["state_names"]), "sampling_seed": args.seed})
    train = CachedFeatureDataset(args.data, args.cache, split["train"], normalizer, config.chunk_size)
    val = CachedFeatureDataset(args.data, args.cache, split["val"], normalizer, config.chunk_size, executed_prefix=True)
    policy = FlowHeadPolicy(config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    noise_generator = torch.Generator().manual_seed(args.seed + 73_021)
    iterator = ResumableBatches(train, args.batch_size, args.seed)
    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    contract = {key: manifest[key] for key in ("camera_keys", "letterbox", "state_names", "action_names",
                "label_semantics", "policy_hz", "source", "prepare_version")}
    for key in ("observation_variant", "experiment_split", "source_manifest_sha256", "source_split",
                "canonical_state_names", "source_derived", "observation_semantics", "contract_version",
                "action_representation", "controller_semantics", "calibration_sha256", "camera_transforms", "transfer_shared_task_sha256"):
        if key in manifest:
            contract[key] = manifest[key]
    contract.update(image_shape_chw=[3, *manifest["image_shape_hwc"][:2]], split=split,
                    derived_files_sha256=manifest["files_sha256"],
                    policy_kind=f"frozen_{config.vision_backbone}_small_{args.objective}_head", vision_encoder={
                        "kind": f"{config.vision_backbone}_spatial_patches",
                        "model_id": getattr(config, f"{config.vision_backbone}_model_id"),
                        "revision": getattr(config, f"{config.vision_backbone}_revision"),
                        "trainable_final_blocks": 0})
    seen, update = 0, 0
    best_validation, best_examples = float("inf"), 0
    best_offline_score = float("inf")
    history = []
    effective = args.batch_size * args.grad_accum
    schedule_examples = getattr(args, "schedule_examples", None) or args.examples
    validation_every = getattr(args, "validate_every", 8000)
    if effective <= 0 or args.examples % effective or schedule_examples % effective or validation_every % effective:
        raise ValueError("exposure budgets and validation interval must divide the effective batch exactly")
    schedule_updates = schedule_examples // effective
    warmup = max(1, int(.05 * schedule_updates))
    if schedule_updates < 2 or args.examples > schedule_examples:
        raise ValueError("scheduler horizon must cover at least two updates and the requested exposure")
    identity = {"cache": cache["sha256"], "data": manifest["files_sha256"], "split": split,
                "config": asdict(config), "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                "lr": args.lr, "weight_decay": args.weight_decay, "schedule_examples": schedule_examples,
                "validation_every": validation_every}
    identity["selection_protocol"] = VALIDATION_VERSION
    resume = getattr(args, "resume", None)
    if resume:
        saved = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("flow resume data/config/scheduler identity differs")
        weights = load_file(resume / "model.safetensors")
        policy.load_state_dict({k: v for k, v in weights.items() if not k.startswith("backbone.")}, strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        iterator.load_state_dict(saved["sampler"])
        noise_generator.set_state(saved["noise_generator"])
        restore_rng(saved["rng"])
        seen, update = saved["examples_seen"], saved["update"]
        history = saved["history"]
        best_validation = saved.get("best_validation", float("inf"))
        best_offline_score = saved.get("best_offline_score", float("inf"))
        best_examples = saved.get("best_examples", seen)
        if finalize_only and seen != args.examples:
            raise ValueError("finalize-only requires exactly the requested exposure")
        if not finalize_only and seen >= args.examples:
            raise ValueError("resume checkpoint has already reached the requested exposure")
    while seen < args.examples:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(update + 1, schedule_updates, args.lr,
                                        warmup_steps=warmup, start_factor=.001, min_lr=1e-7)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        target = min(args.batch_size * args.grad_accum, args.examples - seen)
        consumed, loss_sum = 0, 0.
        while consumed < target:
            batch = next(iterator)
            count = min(len(batch["action"]), target - consumed)
            batch = {k: v[:count].to(device) for k, v in batch.items()}
            loss, _ = policy(batch, generator=noise_generator)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite head training loss")
            (loss * count / target).backward()
            loss_sum += loss.item() * count
            consumed += count
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        seen += consumed
        update += 1
        if update == 1 or update % args.log_every == 0 or seen == args.examples:
            entry = {"update": update, "examples": seen, "loss": loss_sum / consumed,
                     "grad_norm": float(norm), "wall_s": time.perf_counter() - started}
            history.append(entry)
            write_json(args.out / "progress.json", entry)
            print(json.dumps(entry), flush=True)
        if seen % validation_every == 0 or seen == args.examples:
            validation, _ = evaluate(policy, val, device, args.batch_size, args.seed, 4)
            score = validation["offline_selection"]["score"]
            best_validation = min(best_validation, validation["strata"]["all"]["model"]["l1"])
            if score < best_offline_score:
                best_offline_score, best_examples = score, seen
            ck = args.out / "checkpoints" / f"{seen:09d}"
            info = {"step": update, "examples_seen": seen, "validation": validation,
                    "best_validation_l1": best_validation, "best_examples": best_examples,
                    "selection_protocol": VALIDATION_VERSION, "best_offline_score": best_offline_score,
                    "objective": args.objective, "identity": identity, "seed": args.seed}
            save_flow_checkpoint(ck, policy, normalizer.stats, contract, info,
                                 backbone_state=load_file(args.cache / "backbone.safetensors"))
            torch.save({"optimizer": optimizer.state_dict(), "sampler": iterator.state_dict(),
                        "rng": rng_state(), "noise_generator": noise_generator.get_state(),
                        "examples_seen": seen, "update": update, "identity": identity, "history": history,
                        "best_validation": best_validation, "best_examples": best_examples,
                        "best_offline_score": best_offline_score},
                       ck / "training_state.pt")
            if getattr(args, "keep_only_best_and_last", False):
                prune_head_checkpoints(args.out / "checkpoints", best_examples, seen)
    synchronize(device)
    training_s = time.perf_counter() - started
    report = {"objective": args.objective, "seed": args.seed, "device": device,
              "examples": seen, "updates": update, "microbatch": args.batch_size,
              "grad_accum": args.grad_accum, "effective_batch": args.batch_size * args.grad_accum,
              "lr": args.lr, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
              "split": split, "trainable_parameters": sum(p.numel() for p in policy.parameters()),
              "feature_cache": str(args.cache.resolve()), "feature_cache_sha256": cache["sha256"],
              "train_wall_s": training_s, "config": asdict(config), "history": history}
    report.update(selection_protocol=VALIDATION_VERSION, best_offline_score=best_offline_score,
                  best_validation_l1=best_validation, best_examples=best_examples,
                  checkpoint_selection_scope="offline shortlist only; bounded development rollouts then reserved evaluation")
    if device.startswith("cuda"):
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    elif device == "mps":
        report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    for sampling_steps in ((4, 8) if args.objective == "flow" else (4,)):
        result, predictions = evaluate(policy, val, device, args.batch_size, args.seed, sampling_steps)
        key = f"inner_val_{sampling_steps}_steps" if args.objective == "flow" else "inner_val"
        report[key] = result
        np.savez_compressed(args.out / f"{key}_predictions.npz", **predictions)
    # Store frozen weights once with the final head; no network access on reload.
    save_flow_checkpoint(args.out / "checkpoint", policy, normalizer.stats, contract, report,
                         backbone_state=load_file(args.cache / "backbone.safetensors"))
    write_json(args.out / "report.json", report)
    return report


def profile_head(args):
    device = resolve_device(args.device)
    policy, normalizer, contract = load_flow_checkpoint(args.checkpoint, device)
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest["files_sha256"] != contract["derived_files_sha256"]:
        raise ValueError("profile data differs from checkpoint data")
    arrays = np.load(args.data / "arrays.npz", allow_pickle=False)
    images = {key: np.load(args.data / manifest["image_files"][key], mmap_mode="r")[0]
              for key in policy.config.camera_keys}
    report = {"device": device, "repeats": args.repeats, "batch_size": 1,
              "latency_includes": "all-camera normalization and input transfer, frozen vision, head/integration, action denormalization and synchronous output transfer",
              "cadence_ms": 1000 * policy.config.n_action_steps / contract["policy_hz"], "checkpoint": str(args.checkpoint)}
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for steps in ((4, 8) if policy.config.objective == "flow" else (4,)):
        timings = []
        for i in range(args.repeats + 3):
            synchronize(device)
            start = time.perf_counter()
            batch = {"observation.state": torch.tensor(normalizer.state(arrays["state"][0]))[None].to(device)}
            for key, img in images.items():
                batch[key] = torch.tensor(normalizer.image(img))[None].to(device)
            prediction = normalizer.unaction(policy.predict_action_chunk(batch, sampling_seed=0, sampling_steps=steps)).cpu()
            synchronize(device)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("nonfinite profiled prediction")
            if i >= 3:
                timings.append((time.perf_counter() - start) * 1000)
        key = f"{steps}_steps" if policy.config.objective == "flow" else "regression"
        report[key] = {"median_ms": float(np.median(timings)), "p95_ms": float(np.quantile(timings, .95)),
                       "max_ms": float(max(timings)), "under_120ms_p95": bool(np.quantile(timings, .95) < 120),
                       "under_200ms_p95": bool(np.quantile(timings, .95) < 200)}
    if device == "mps":
        report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    elif device.startswith("cuda"):
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_json(args.out / "profile.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache")
    cache.add_argument("--vision-config", type=Path, default=Path("configs/act_cartesian_dinov2_pilot.yaml"))
    cache.add_argument("--batch-size", type=int, default=8)
    cache.add_argument("--episodes", type=int, nargs="+")
    cache.add_argument("--chunk-size", type=int, default=20)
    train = commands.add_parser("train")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--objective", choices=("flow", "regression"), required=True)
    train.add_argument("--split-json", type=Path)
    train.add_argument("--examples", type=int, default=32000)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--grad-accum", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=.01)
    train.add_argument("--grad-clip", type=float, default=1.)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--resume", type=Path)
    train.add_argument("--schedule-examples", type=int)
    train.add_argument("--validate-every", type=int, default=8000)
    train.add_argument("--keep-only-best-and-last", action="store_true")
    train.add_argument("--finalize-only", action="store_true",
                       help="restore a full-budget checkpoint and finish evaluation/report without training")
    profile = commands.add_parser("profile")
    profile.add_argument("--checkpoint", type=Path, required=True)
    profile.add_argument("--repeats", type=int, default=20)
    for command in (cache, train, profile):
        command.add_argument("--data", type=Path, default=Path(".cache/derived_cartesian"))
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        command.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    for name in ("threads", "batch_size", "examples", "grad_accum", "log_every", "repeats"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "RUNNING.json", {"pid": os.getpid(), "command": args.command})
    started = time.perf_counter()
    try:
        result = {"cache": build_cache, "train": train_head, "profile": profile_head}[args.command](args)
        write_json(args.out / "COMPLETE.json", {"command": args.command, "wall_s": time.perf_counter() - started,
                   "status": "command_completed", "physical_grasp_success": "not_evaluated",
                   "result_file": "report.json" if args.command == "train" else "manifest.json" if args.command == "cache" else "profile.json"})
        print(json.dumps({"status": "complete", "output": str(args.out), "command": args.command,
                          "wall_s": time.perf_counter() - started, "result_keys": list(result)}), flush=True)
    except BaseException as error:
        write_json(args.out / "FAILED.json", {"command": args.command, "error": repr(error),
                   "traceback": traceback.format_exc(), "wall_s": time.perf_counter() - started})
        raise
    finally:
        (args.out / "RUNNING.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
