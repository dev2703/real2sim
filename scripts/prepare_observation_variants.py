"""Materialize immutable observation-only ablations with one train-only inner split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from real2sim.data.prepare import compute_stats
from real2sim.data.raw import sha256_file
from real2sim.policy.observation_variants import (
    ObservationVariant,
    fit_state_statistics,
    state_names,
    transform_observations,
)


def inner_split(original: dict, seed: int = 20261016) -> dict:
    episodes = np.array(sorted(original["train"]), dtype=np.int64)
    if len(episodes) < 3:
        raise ValueError("need at least three training episodes for an inner split")
    rng = np.random.default_rng(seed)
    val = sorted(int(e) for e in rng.permutation(episodes)[:max(1, int(np.ceil(len(episodes) * .2)))])
    split = {"seed": seed, "train": sorted(set(episodes.tolist()) - set(val)), "val": val,
             "historical_val": list(original["val"]), "excluded": list(original.get("excluded", []))}
    split["sha256"] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    return split


def prepare_variants(source: Path, out: Path, seed: int = 20261016) -> dict:
    if out.exists():
        raise FileExistsError(out)
    source = source.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["mode"] != "cartesian" or len(manifest["state_names"]) != 13 or manifest.get("observation_variant"):
        raise ValueError("source must be the untransformed canonical Cartesian dataset")
    stats = json.loads((source / "stats.json").read_text())
    with np.load(source / "arrays.npz") as loaded:
        arrays = {k: loaded[k] for k in loaded.files}
    # Verify the entire immutable input, including image content shared by symlinks.
    for name, expected in manifest["files_sha256"].items():
        if sha256_file(source / name) != expected:
            raise ValueError(f"source hash mismatch: {name}")
    split = inner_split(manifest["split"], seed)
    action_mask = np.isin(arrays["episode_index"], split["train"]) & ~arrays["label_is_extrapolated"]
    action_stats = compute_stats(arrays["action"][action_mask])
    hz = float(manifest["policy_hz"])
    variants = {
        "canonical": ObservationVariant(policy_hz=hz),
        "no_velocity": ObservationVariant("no_velocity", policy_hz=hz),
        "ema_velocity_05": ObservationVariant("ema_velocity", ema_alpha=.5, policy_hz=hz),
        "ema_velocity_075": ObservationVariant("ema_velocity", ema_alpha=.75, policy_hz=hz),
        "joints": ObservationVariant("joints", policy_hz=hz),
        "canonical_joints": ObservationVariant("canonical_joints", policy_hz=hz),
        "history3": ObservationVariant(history=3, policy_hz=hz),
    }
    out.mkdir(parents=True)
    report = {"source": str(source), "source_manifest_sha256": sha256_file(source / "manifest.json"),
              "split": split, "source_split": manifest["split"], "variants": {},
              "action_targets": "unchanged, including leader-gripper semantics",
              "historical_val": "retained in arrays for comparison; excluded from inner selection and fitting"}
    for name, variant in variants.items():
        target = out / name
        target.mkdir()
        transformed = transform_observations(arrays, variant)
        output_arrays = {**arrays, "canonical_state": arrays["state"], "state": transformed}
        np.savez(target / "arrays.npz", **output_arrays)
        output_stats = copy.deepcopy(stats)
        output_stats.update({"observation.state": fit_state_statistics(transformed, arrays["episode_index"], split["train"], variant),
                             "action": action_stats, "train_episodes": split["train"],
                             "computed_on": "inner training episodes only; terminal extrapolated actions excluded"})
        (target / "stats.json").write_text(json.dumps(output_stats, indent=2) + "\n")
        for image_file in manifest["image_files"].values():
            (target / image_file).symlink_to(source / image_file)
        output_manifest = copy.deepcopy(manifest)
        output_manifest.update({"prepare_version": "observation_variants_v1", "split": split,
                                "source_split": manifest["split"], "observation_variant": variant.to_dict(),
                                "state_names": state_names(variant),
                                "source_derived": {"path": str(source), "manifest_sha256": report["source_manifest_sha256"],
                                                   "files_sha256": manifest["files_sha256"]},
                                "observation_semantics": "causal observation-only transform; measured joint inputs, no applied-command history",
                                "n_ticks_train": int(np.isin(arrays["episode_index"], split["train"]).sum()),
                                "n_ticks_val": int(np.isin(arrays["episode_index"], split["val"]).sum()),
                                "n_ticks_historical_val": int(np.isin(arrays["episode_index"], split["historical_val"]).sum())})
        output_manifest["files_sha256"] = {**manifest["files_sha256"],
                                           "arrays.npz": sha256_file(target / "arrays.npz"),
                                           "stats.json": sha256_file(target / "stats.json")}
        (target / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
        report["variants"][name] = {"path": str(target.resolve()), "state_dim": transformed.shape[1],
                                    "config": variant.to_dict(), "manifest_sha256": sha256_file(target / "manifest.json")}
        print(f"{name}: state_dim={transformed.shape[1]}, train={output_manifest['n_ticks_train']}, val={output_manifest['n_ticks_val']}", flush=True)
    (out / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".cache/derived_cartesian"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    prepare_variants(args.source, args.out, args.seed)
