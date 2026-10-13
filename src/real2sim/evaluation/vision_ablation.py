"""Paired clean/blank/cross-episode-image diagnostics of frozen ACT, never training.

Includes clean training episodes in inference mode so the train/validation comparison does
not confuse the training VAE posterior with the inference zero latent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from real2sim.evaluation.run import sha256_file
from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset


def stratified_indices(ds, per_episode: int) -> np.ndarray:
    if per_episode < 1:
        raise ValueError("samples per episode must be positive")
    eps = ds.arrays["episode_index"][ds.indices]
    return np.concatenate([
        (local := np.flatnonzero(eps == ep))[
            np.linspace(0, len(local) - 1, min(per_episode, len(local)), dtype=int)]
        for ep in np.unique(eps)
    ])


def cross_episode_donors(episodes: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic donor mapping, with every donor from a different episode."""
    rng = np.random.default_rng(seed)
    if len(np.unique(episodes)) < 2:
        raise ValueError("cross-episode shuffle requires at least two episodes")
    return np.array([rng.choice(np.flatnonzero(episodes != ep)) for ep in episodes])


class ImageCondition(Dataset):
    def __init__(self, ds, indices, condition, donors=None):
        self.ds, self.indices, self.condition, self.donors = ds, indices, condition, donors

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.ds[int(self.indices[i])]
        if self.condition == "blank":
            for key in self.ds.camera_keys:
                blank = (np.full_like(self.ds.normalizer.img_mean, 114 / 255)
                         - self.ds.normalizer.img_mean) / self.ds.normalizer.img_std
                item[key] = torch.as_tensor(blank).expand_as(item[key]).clone()
        elif self.condition == "shuffled":
            donor = self.ds[int(self.indices[self.donors[i]])]
            for key in self.ds.camera_keys:
                item[key] = donor[key]
        elif self.condition != "clean":
            raise ValueError(f"unknown image condition: {self.condition}")
        return item


@torch.inference_mode()
def score(policy, dataset, device, action_std):
    policy.eval()
    sums, counts, predictions = {}, {}, []
    for batch in DataLoader(dataset, batch_size=16, shuffle=False):
        obs = {k: v.to(device) for k, v in batch.items() if k.startswith("observation.")}
        pred = policy.predict_action_chunk(obs).cpu()
        if not torch.isfinite(pred).all():
            raise FloatingPointError("non-finite ablation prediction")
        valid = ~batch["action_is_pad"]
        errors = ((pred - batch["action"]).abs() * valid[:, :, None]).numpy()
        for i, ep in enumerate(batch["episode_index"].tolist()):
            sums[ep] = sums.get(ep, np.zeros(len(action_std))) + errors[i].sum(0)
            counts[ep] = counts.get(ep, 0) + int(valid[i].sum())
        predictions.append(pred.numpy())
    total = sum(counts.values())
    per_dim = sum(sums.values()) / total
    return {
        "l1_normalized": float(per_dim.mean()),
        "l1_raw_per_dimension": (per_dim * action_std).tolist(),
        "samples": len(dataset), "valid_action_ticks": total,
        "per_episode_l1_normalized": {str(ep): float((sums[ep] / counts[ep]).mean())
                                      for ep in sums if counts[ep]},
    }, np.concatenate(predictions)


def run(checkpoint: Path, data: Path, out: Path, *, device="cpu", per_episode=32, seed=0):
    policy, norm, contract = load_checkpoint(checkpoint, device)
    manifest = json.loads((data / "manifest.json").read_text())
    for key in ("camera_keys", "state_names", "action_names", "label_semantics", "split"):
        if manifest[key] != contract[key]:
            raise ValueError(f"checkpoint/data mismatch: {key}")
    if contract["derived_files_sha256"] != manifest["files_sha256"]:
        raise ValueError("checkpoint was trained on different derived data")
    results, provenance = {}, {}
    for split in ("train", "val"):
        ds = DerivedTickDataset(data, manifest["split"][split], policy.config.chunk_size, norm)
        indices = stratified_indices(ds, per_episode)
        episodes = ds.arrays["episode_index"][ds.indices[indices]]
        donors = cross_episode_donors(episodes, seed)
        provenance[split] = {"ticks": ds.indices[indices].tolist(),
                             "donor_ticks": ds.indices[indices[donors]].tolist()}
        reference = None
        for condition in (("clean",) if split == "train" else ("clean", "blank", "shuffled")):
            metrics, pred = score(policy, ImageCondition(ds, indices, condition, donors),
                                  device, norm.action_std)
            if reference is None:
                reference = pred
            else:
                # Include actual prediction movement; similar loss can conceal different errors.
                metrics["prediction_change_normalized_l1"] = float(np.abs(pred - reference).mean())
                metrics["loss_ratio_to_clean"] = metrics["l1_normalized"] / results["val_clean"]["l1_normalized"]
            results[f"{split}_{condition}"] = metrics
            print(split, condition, metrics["l1_normalized"], flush=True)
    report = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint / "model.safetensors"),
              "dataset_revision": contract["source"]["revision"], "split_sha256": contract["split"]["sha256"],
              "device": device, "seed": seed, "samples_per_episode": per_episode,
              "action_names": contract["action_names"], "results": results,
              "sampling": provenance, "sampling_sha256": hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
              "note": "All conditions use eval mode, zero VAE latent, identical state/labels/masks. Blank is RGB 114. Shuffled donors come from another held-out episode. Inference-mode train error is the fair comparison. Image sensitivity does not prove ball localisation; blank images are out of distribution. Diagnostic, not task success."}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--per-episode", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.checkpoint, args.data, args.out, device=args.device,
        per_episode=args.per_episode, seed=args.seed)


if __name__ == "__main__":
    main()
