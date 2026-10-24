"""Matched auxiliary ball-localization pilot over frozen DINO image patches.

Pseudo labels come from the fixed image detector, never simulator positions. Both
arms predict the features supplied to their action head; pseudo centers are not
fed to the policy at training or inference. Reserved images are never cached.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from real2sim.config import load_yaml
from real2sim.data.raw import sha256_file
from real2sim.data.resample import chunk_indices
from real2sim.policy.dataset import Normalizer
from real2sim.policy.flow_matching import FEATURE_KEY, FlowHeadConfig, build_frozen_backbone
from real2sim.policy.keypoint_bottleneck import (
    KeypointBottleneckConfig,
    KeypointBottleneckPolicy,
    load_keypoint_checkpoint,
    save_keypoint_checkpoint,
)
from real2sim.policy.validation import evaluate_executed_prefix, executed_prefix_anchors
from real2sim.policy.visual_grounding import BallImageFeatures, ball_features


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_source(source: Path):
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest.get("action_representation") != "cartesian"
            or manifest["camera_keys"] != ["observation.images.iphone"]
            or len(manifest["state_names"]) != 13
            or manifest.get("visual_grounding") or manifest.get("observation_variant")):
        raise ValueError("keypoint pilot requires the unchanged canonical13/Cartesian7 iPhone source")
    split = manifest["split"]
    if (set(split["train"]) & set(split["val"])
            or (set(split["train"]) | set(split["val"])) & set(split["reserved"])):
        raise ValueError("source partitions overlap")
    for name, digest in manifest["files_sha256"].items():
        if Path(name).name != name or sha256_file(source / name) != digest:
            raise ValueError(f"source checksum differs: {name}")
    arrays = dict(np.load(source / "arrays.npz", allow_pickle=False))
    if any(np.any(arrays.get(field, False)) for field in ("action_is_invalid", "observation_is_invalid")):
        raise ValueError("this matched pilot uses unchanged targets, without native-feasibility filtering")
    return manifest, arrays, Normalizer.from_json(source / "stats.json")


def select_training_ticks(arrays, episodes, count=1024, seed=20261024):
    """Episode-balanced random observations, fixed before fitting either arm."""
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(np.isin(arrays["episode_index"], episodes)
                                & ~arrays["label_is_extrapolated"])
    if count < len(episodes) or count > len(candidates):
        raise ValueError("cache training count must cover all episodes and fit available observations")
    chosen = []
    per_episode = count // len(episodes)
    for episode in sorted(episodes):
        current = candidates[arrays["episode_index"][candidates] == episode]
        chosen.extend(rng.choice(current, min(per_episode, len(current)), replace=False).tolist())
    remaining = np.setdiff1d(candidates, chosen)
    chosen.extend(rng.choice(remaining, count - len(chosen), replace=False).tolist())
    return np.sort(chosen)


def cache_features(source: Path, out: Path, train_count=1024, batch_size=8, device="cpu"):
    if out.exists():
        raise FileExistsError(out)
    manifest, arrays, norm = check_source(source)
    train_ticks = select_training_ticks(arrays, manifest["split"]["train"], train_count)
    _, val_anchors = executed_prefix_anchors(arrays, manifest["split"]["val"])
    val_ticks = np.unique(val_anchors)
    ticks = np.sort(np.r_[train_ticks, val_ticks])
    model = load_yaml(Path("configs/v2/act_base.yaml"))["model"]
    config = KeypointBottleneckConfig(dinov2_config=model["dinov2_config"])
    backbone = build_frozen_backbone(
        FlowHeadConfig(dinov2_config=config.dinov2_config),
        pretrained_path=model["dinov2_pretrained_path"],
    ).to(device).eval()
    images = np.load(source / manifest["image_files"][config.camera_key], mmap_mode="r")
    spec = BallImageFeatures()
    out.mkdir(parents=True)
    features = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float16,
                                        shape=(len(ticks), config.rows * config.cols, config.feature_dim))
    pseudo = np.empty((len(ticks), 4), np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(ticks), batch_size):
            selected = ticks[start:start + batch_size]
            image = np.stack([norm.image(images[t]) for t in selected])
            values = backbone(torch.from_numpy(image).to(device))["feature_map"].flatten(2).transpose(1, 2)
            values = values.cpu().numpy().astype(np.float16)
            if not np.isfinite(values).all():
                raise FloatingPointError("nonfinite cached DINO patches")
            features[start:start + len(selected)] = values
            pseudo[start:start + len(selected)] = np.stack([ball_features(images[t], spec) for t in selected])
            if start % 128 == 0:
                print(f"cache {start + len(selected)}/{len(ticks)} {time.perf_counter()-started:.1f}s", flush=True)
    features.flush()
    for name, values in (("ticks.npy", ticks), ("train_ticks.npy", train_ticks),
                         ("val_ticks.npy", val_ticks), ("pseudo_ball.npy", pseudo)):
        np.save(out / name, values, allow_pickle=False)
    save_file({k: v.detach().cpu().contiguous() for k, v in backbone.state_dict().items()}, out / "backbone.safetensors")
    result = {
        "kind": "FROZEN_DINO_KEYPOINT_PILOT_CACHE", "source": str(source.resolve()),
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "config": asdict(config), "pseudo_detector": spec.to_dict(),
        "training_tick_selection": "Episode-balanced without replacement; seed20261024; excludes episode-terminal placeholders",
        "train_observations": len(train_ticks), "val_observations": len(val_ticks),
        "train_episodes": manifest["split"]["train"], "val_episodes": manifest["split"]["val"],
        "reserved_episodes_used": [], "feature_precision": "float16 round-trip in cached and raw paths",
        "pretrained_backbone": model["dinov2_pretrained_path"],
        "pretrained_weights_sha256": sha256_file(Path(model["dinov2_pretrained_path"]) / "model.safetensors"),
        "device": device, "wall_s": time.perf_counter() - started,
        "sha256": {name: sha256_file(out / name) for name in (
            "features.npy", "ticks.npy", "train_ticks.npy", "val_ticks.npy", "pseudo_ball.npy", "backbone.safetensors")},
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(out / "manifest.json", result)
    return result


class KeypointDataset(Dataset):
    def __init__(self, source: Path, cache: Path, partition: str):
        self.manifest = json.loads((source / "manifest.json").read_text())
        self.arrays = dict(np.load(source / "arrays.npz", allow_pickle=False))
        self.normalizer = Normalizer.from_json(source / "stats.json")
        cached = json.loads((cache / "manifest.json").read_text())
        if sha256_file(source / "manifest.json") != cached["source_manifest_sha256"]:
            raise ValueError("feature cache source differs")
        if partition not in ("train", "val"):
            raise ValueError("keypoint pilot may only access training/validation observations")
        for name, digest in cached["sha256"].items():
            if sha256_file(cache / name) != digest:
                raise ValueError(f"feature cache payload differs: {name}")
        self.indices = np.load(cache / f"{partition}_ticks.npy")
        if not np.isin(self.arrays["episode_index"][self.indices], self.manifest["split"][partition]).all():
            raise ValueError("cached ticks cross a declared partition")
        self.features = np.load(cache / "features.npy", mmap_mode="r")
        self.pseudo = np.load(cache / "pseudo_ball.npy", mmap_mode="r")
        self.lookup = {int(t): i for i, t in enumerate(np.load(cache / "ticks.npy"))}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        tick = int(self.indices[item])
        indices, pad = chunk_indices(self.arrays["ep_start"], self.arrays["ep_len"], tick, 2)
        pad |= self.arrays["label_is_extrapolated"][indices]
        cached = self.lookup[tick]
        return {
            "observation.state": torch.from_numpy(self.normalizer.state(self.arrays["state"][tick]).astype(np.float32)),
            FEATURE_KEY: torch.from_numpy(np.array(self.features[cached], dtype=np.float32)),
            "pseudo_ball": torch.from_numpy(np.array(self.pseudo[cached], dtype=np.float32)),
            "action": torch.from_numpy(self.normalizer.action(self.arrays["action"][indices]).astype(np.float32)),
            "action_is_pad": torch.from_numpy(pad), "tick": tick,
            "episode_index": int(self.arrays["episode_index"][tick]),
        }


@torch.inference_mode()
def localization_and_dependence(policy, dataset, batch_size=32):
    predicted, target, probabilities = [], [], []
    feature_changes, oracle_changes = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        action, uv, logits, _ = policy.predictions(batch)
        prob = logits.sigmoid()
        predicted.extend(uv.numpy())
        target.extend(batch["pseudo_ball"].numpy())
        probabilities.extend(prob.numpy())
        # Deliberate single-input interventions, kept separate from deployable inference.
        swapped = policy.actions_from_keypoints(batch["observation.state"], uv.roll(1, 0), prob)
        oracle = policy.actions_from_keypoints(batch["observation.state"], batch["pseudo_ball"][:, :2], batch["pseudo_ball"][:, 3])
        for values, changed in ((feature_changes, swapped), (oracle_changes, oracle)):
            delta = dataset.normalizer.unaction(changed) - dataset.normalizer.unaction(action)
            values.extend(torch.linalg.vector_norm(delta[:, :, :3], dim=-1).flatten().numpy() * 1000)
    predicted, target, probabilities = map(np.asarray, (predicted, target, probabilities))
    visible = target[:, 3] > .5
    pixel_error = np.linalg.norm((predicted[visible] - target[visible, :2]) * [256, 192], axis=1)
    return {"pseudo_visible_anchors": int(visible.sum()), "anchors": len(target),
            "center_error_mean_px_against_pseudo": float(pixel_error.mean()),
            "center_error_p95_px_against_pseudo": float(np.quantile(pixel_error, .95)),
            "detector_event_classification_accuracy_at_half": float(((probabilities >= .5) == visible).mean()),
            "predicted_visibility_probability_mean": float(probabilities.mean()),
            "center_swap_translation_change_mean_mm": float(np.mean(feature_changes)),
            "pseudo_keypoint_intervention_translation_change_mean_mm": float(np.mean(oracle_changes)),
            "interpretation": "Pseudo detector agreement and input sensitivity; not annotation accuracy or causal task success. Swapping only predicted centers deliberately breaks image-feature consistency."}


def train_matched(source: Path, cache: Path, out: Path, steps=256, batch_size=16):
    if out.exists():
        raise FileExistsError(out)
    if not 1 <= steps <= 512 or batch_size < 1:
        raise ValueError("bounded positive pilot budget required")
    manifest, arrays, norm = check_source(source)
    cached = json.loads((cache / "manifest.json").read_text())
    config = KeypointBottleneckConfig(**cached["config"])
    train, val = KeypointDataset(source, cache, "train"), KeypointDataset(source, cache, "val")
    out.mkdir(parents=True)
    contract = copy.deepcopy(manifest)
    contract.update(image_shape_chw=[3, 192, 256], derived_files_sha256=manifest["files_sha256"],
                    policy_kind="frozen_dino_ball_keypoint_bottleneck",
                    keypoint_bottleneck={"config": asdict(config), "feature_precision": cached["feature_precision"],
                                         "pseudo_detector": cached["pseudo_detector"],
                                         "action_input": "Only predicted ball uv times predicted detector-event probability; pseudo targets are auxiliary loss only"})
    backbone = load_file(cache / "backbone.safetensors")
    report = {"kind": "MATCHED_AUXILIARY_IMAGE_KEYPOINT_PILOT", "steps": steps,
              "batch_size": batch_size, "examples_per_model": steps * batch_size,
              "training_seed": 0, "training_distinct_observations": len(train),
              "validation_distinct_observations": len(val), "cache": str(cache),
              "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
              "reserved_episodes_used": [], "physical_calibration_accepted": False,
              "checkpoint_promoted": False, "models": {},
              "limitations": "One seed, small new bottleneck architecture and fixed pseudo-label detector. Uses unchanged provisional FK targets and bounded training subset. Independent object localization, geometric metrology and task success are not established."}
    for name, auxiliary_weight in (("action_only", 0.), ("auxiliary_ball", 1.)):
        torch.manual_seed(0)
        policy = KeypointBottleneckPolicy(config)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-4, weight_decay=1e-4)
        rng = np.random.default_rng(0)
        history = []
        started = time.perf_counter()
        for step in range(1, steps + 1):
            selected = rng.integers(len(train), size=batch_size)
            batch = torch.utils.data.default_collate([train[int(i)] for i in selected])
            optimizer.zero_grad(set_to_none=True)
            loss, detail = policy(batch, auxiliary_weight=auxiliary_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite keypoint training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            optimizer.step()
            if step % 64 == 0 or step == steps:
                history.append({"step": step, "loss": float(loss), **detail})
                print(name, history[-1], flush=True)
        policy.eval()
        metrics, _ = evaluate_executed_prefix(policy, val, "cpu", batch_size=32)
        dependence = localization_and_dependence(policy, val)
        path = out / name
        info = {"step": steps, "auxiliary_weight": auxiliary_weight, "seed": 0,
                "trainable_parameters": sum(p.numel() for p in policy.parameters()),
                "history": history, "validation": metrics, "localization": dependence,
                "wall_s": time.perf_counter() - started}
        save_keypoint_checkpoint(path, policy, norm, contract, info, backbone)
        # A portable checkpoint must reproduce the cached policy from raw image input.
        loaded, loaded_norm, loaded_contract = load_keypoint_checkpoint(path)
        images = np.load(source / manifest["image_files"][config.camera_key], mmap_mode="r")
        parity = []
        for index in np.unique(np.linspace(0, len(val) - 1, 6).astype(int)):
            item = val[int(index)]
            batch = torch.utils.data.default_collate([item])
            tick = int(item["tick"])
            raw_batch = {"observation.state": batch["observation.state"],
                         config.camera_key: torch.from_numpy(loaded_norm.image(images[tick]))[None]}
            with torch.inference_mode():
                expected = policy.predict_action_chunk(batch)
                observed = loaded.predict_action_chunk(raw_batch)
            parity.append(float(torch.max(torch.abs(expected - observed))))
        if loaded_contract != contract or max(parity) > 1e-4:
            raise ValueError("portable raw/cached keypoint inference parity failed")
        info.update(checkpoint=str(path), checkpoint_sha256=sha256_file(path / "model.safetensors"),
                    raw_cached_normalized_action_max_abs_difference=max(parity))
        report["models"][name] = info
        write_json(out / "report.json", report)
        del loaded, policy
    if report["models"]["action_only"]["trainable_parameters"] != report["models"]["auxiliary_ball"]["trainable_parameters"]:
        raise ValueError("matched auxiliary pilot has different parameter counts")
    report["source_sha256"] = {str(path): sha256_file(path) for path in (
        Path(__file__), Path("src/real2sim/policy/keypoint_bottleneck.py"))}
    write_json(out / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("cache", "train"))
    p.add_argument("--source", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone"))
    p.add_argument("--cache", type=Path, default=Path(".cache/v2_keypoint_bottleneck_20261024"))
    p.add_argument("--out", type=Path)
    p.add_argument("--train-count", type=int, default=1024)
    p.add_argument("--steps", type=int, default=256)
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.stage == "cache":
        result = cache_features(args.source, args.cache, args.train_count)
        print(json.dumps({key: result[key] for key in ("train_observations", "val_observations", "wall_s")}, indent=2))
    else:
        if args.out is None:
            raise ValueError("training requires a new --out directory")
        result = train_matched(args.source, args.cache, args.out, args.steps)
        print(json.dumps({name: value["validation"]["scores"]["model"]["all"] for name, value in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
