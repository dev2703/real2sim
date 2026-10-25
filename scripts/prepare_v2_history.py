"""Immutable causal state history with the v2 split, labels and normalization preserved."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from real2sim.data.raw import sha256_file
from real2sim.policy.observation_variants import (
    ObservationVariant,
    state_names,
    transform_observations,
)


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, str(value.dtype))).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def shared_history_statistics(source_stats, features, episodes, training, history):
    """Reuse source sensor statistics exactly; age/valid retain seconds/binary units."""
    current = source_stats["observation.state"]
    if len(current["mean"]) != 13 or len(current["std"]) != 13:
        raise ValueError("source normalization must describe 13 canonical features")
    selected = features[np.isin(episodes, training)]
    if not len(selected):
        raise ValueError("no training observations")
    result = copy.deepcopy(source_stats)
    result["observation.state"] = {
        "mean": np.tile([*current["mean"], 0., 0.], history).tolist(),
        "std": np.tile([*current["std"], 1., 1.], history).tolist(),
        "min": selected.min(0).tolist(), "max": selected.max(0).tolist(),
        "count": current["count"],
    }
    result["state_normalization_lineage"] = "Exact source canonical mean/std repeated per history slot; age/valid have mean0/std1. No new action statistics."
    return result


def prepare(source, out, history=4):
    if out.exists():
        raise FileExistsError(out)
    source = source.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if (len(manifest["state_names"]) != 13 or manifest.get("observation_variant")
            or manifest.get("visual_grounding") or manifest["action_representation"] != "cartesian"):
        raise ValueError("untransformed canonical Cartesian source required")
    variant = ObservationVariant(history=history, policy_hz=float(manifest["policy_hz"]))
    if variant.history < 2:
        raise ValueError("this preparation requires at least two observations")
    for name, digest in manifest["files_sha256"].items():
        if Path(name).name != name or sha256_file(source / name) != digest:
            raise ValueError(f"source payload mismatch: {name}")
    arrays = dict(np.load(source / "arrays.npz", allow_pickle=False))
    if "canonical_state" in arrays or arrays["state"].shape[1] != 13:
        raise ValueError("source has already been transformed")
    stats = json.loads((source / "stats.json").read_text())
    transformed = transform_observations(arrays, variant)
    output_arrays = {**arrays, "canonical_state": arrays["state"], "state": transformed}
    output_stats = shared_history_statistics(stats, transformed, arrays["episode_index"],
                                             manifest["split"]["train"], history)
    out.mkdir(parents=True)
    np.savez(out / "arrays.npz", **output_arrays)
    (out / "stats.json").write_text(json.dumps(output_stats, indent=2, allow_nan=False) + "\n")
    for filename in set(manifest["image_files"].values()):
        (out / filename).symlink_to(source / filename)
    result = copy.deepcopy(manifest)
    result.update(prepare_version="v2_causal_state_history_v1",
                  contract_version=manifest["contract_version"] + f".history{history}_v1",
                  observation_variant=variant.to_dict(), state_names=state_names(variant),
                  source_manifest_sha256=sha256_file(source / "manifest.json"),
                  source_derived={"path": str(source), "files_sha256": manifest["files_sha256"]},
                  observation_semantics="Causal observed states, oldest first, with seconds of age and valid flags. No action or future observation input.")
    result["files_sha256"] = {**manifest["files_sha256"],
                              "arrays.npz": sha256_file(out / "arrays.npz"),
                              "stats.json": sha256_file(out / "stats.json")}
    (out / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = {"schema": "v2_causal_history_preparation_v1", "source": str(source),
              "output": str(out.resolve()), "history": variant.to_dict(),
              "state_dimension": transformed.shape[1], "split_unchanged": result["split"] == manifest["split"],
              "action_statistics_unchanged": output_stats["action"] == stats["action"],
              "unchanged_arrays": {name: array_hash(value) for name, value in arrays.items() if name != "state"},
              "canonical_state_sha256": array_hash(arrays["state"]),
              "reserved_frames_decoded_or_scored": False,
              "scope": "All source rows are preserved/transformed causally; no reserved images decoded or labels scored. Sensor normalization remains the source training-only contract.",
              "source_sha256": {str(p): sha256_file(p) for p in [Path(__file__), source / "manifest.json", source / "stats.json",
                  Path("src/real2sim/policy/observation_variants.py")]}}
    (out / "preparation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out, args.history), indent=2))
