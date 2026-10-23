"""Version a ball-image-feature dataset; fit added statistics on training images only."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from real2sim.data.prepare import compute_stats
from real2sim.data.raw import sha256_file
from real2sim.policy.visual_grounding import (
    BallImageFeatures,
    ball_features,
    grounded_state_names,
)


def prepare_grounding(source: Path, out: Path, spec: BallImageFeatures | None = None) -> dict:
    spec = spec or BallImageFeatures()
    source = source.resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest.get("action_representation") != "cartesian"
            or "observation_variant" in manifest or "visual_grounding" in manifest
            or spec.camera_key not in manifest["camera_keys"]):
        raise ValueError("grounding source must be the original canonical Cartesian dataset")
    split = manifest["split"]
    sets = [set(split[key]) for key in ("train", "val", "reserved")]
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("training, validation and reserved episodes must be disjoint")
    for name, digest in manifest["files_sha256"].items():
        if Path(name).name != name or sha256_file(source / name) != digest:
            raise ValueError(f"source checksum mismatch: {name}")
    with np.load(source / "arrays.npz") as arrays:
        episodes = arrays["episode_index"]
        canonical = arrays["state"]
    if canonical.shape != (len(episodes), 13):
        raise ValueError("source must store canonical proprioception[13]")
    images = np.load(source / manifest["image_files"][spec.camera_key], mmap_mode="r")
    indices = np.flatnonzero(np.isin(episodes, split["train"]))
    features = np.asarray([ball_features(images[i], spec) for i in indices])
    extra = compute_stats(features)
    # Visibility is binary and should retain its meaning, even for always-visible data.
    extra["mean"][-1], extra["std"][-1] = 0.0, 1.0
    stats = json.loads((source / "stats.json").read_text())
    for field in ("mean", "std", "min", "max"):
        stats["observation.state"][field] += extra[field]
    output = copy.deepcopy(manifest)
    output.update(visual_grounding=spec.to_dict(), state_names=grounded_state_names(),
                  prepare_version="ball_visual_grounding_v1",
                  source_manifest_sha256=sha256_file(source / "manifest.json"),
                  observation_semantics="Canonical proprioception plus current-image ball coordinates/radius/visibility; no metric backprojection or privileged fallback",
                  visual_features_storage="Computed on demand by the same detector as inference; arrays.state retains canonical 13 fields",
                  visual_feature_statistics_episodes=split["train"])
    out.mkdir(parents=True)
    for name in ("arrays.npz", *manifest["image_files"].values()):
        (out / name).symlink_to(source / name)
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    output["files_sha256"] = {**manifest["files_sha256"], "stats.json": sha256_file(out / "stats.json")}
    (out / "manifest.json").write_text(json.dumps(output, indent=2) + "\n")
    report = {
        "source": str(source), "out": str(out.resolve()), "detector": spec.to_dict(),
        "source_manifest_sha256": output["source_manifest_sha256"],
        "manifest_sha256": sha256_file(out / "manifest.json"),
        "train_frames": len(indices), "train_visible_fraction": float(features[:, -1].mean()),
        "reserved_episodes_read": [], "action_labels_changed": False,
        "physical_calibration_accepted": False,
        "limitations": "Fixed colour/shape proposal, not supervised object localization. Visibility indicates a matching contour, not calibrated confidence. Occlusion can remove detections. Source FK labels remain provisional.",
    }
    (out / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--disabled-control", action="store_true", help="Matched 17-state control with four constant-zero ball slots")
    args = p.parse_args()
    print(json.dumps(prepare_grounding(args.source, args.out, BallImageFeatures(enabled=not args.disabled_control)), indent=2))
