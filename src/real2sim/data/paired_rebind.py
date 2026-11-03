"""Reuse rendered images across derived action schemas only after exact tick/image checks."""

import copy
import json
from pathlib import Path

import numpy as np

from real2sim.artifacts import sha256_file as file_sha256


def rebind(source_pairs, source_derived, target_derived, out):
    source_pairs, source_derived, target_derived, out = map(
        Path, (source_pairs, source_derived, target_derived, out),
    )
    if out.exists():
        raise FileExistsError(out)
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (source_derived, target_derived)]
    source, target = manifests
    paired = json.loads((source_pairs / "manifest.json").read_text())
    for key in ("source", "split", "files_sha256"):
        if paired["derived_manifest"][key] != source[key]:
            raise ValueError(f"source pairs differ from source derived data: {key}")
    for key in ("source", "split", "camera_keys", "image_shape_hwc", "letterbox", "policy_hz"):
        if source[key] != target[key]:
            raise ValueError(f"cannot rebind pairs across different {key}")
    for root, manifest in zip((source_derived, target_derived), manifests, strict=True):
        for name, checksum in manifest["files_sha256"].items():
            if file_sha256(root / name) != checksum:
                raise ValueError(f"derived checksum mismatch: {root / name}")
    if source["image_files"] != target["image_files"]:
        raise ValueError("camera image files differ")
    for name in source["image_files"].values():
        if source["files_sha256"][name] != target["files_sha256"][name]:
            raise ValueError("real images differ")
    with np.load(source_derived / "arrays.npz") as a, np.load(target_derived / "arrays.npz") as b:
        for key in ("episode_index", "frame_index"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"tick alignment mismatch: {key}")
        ticks = np.load(source_pairs / "ticks.npy", allow_pickle=False)
        if not np.isin(b["episode_index"][ticks], target["split"][paired["split_name"]]).all():
            raise ValueError("paired ticks cross split")
    if set(paired["sha256"]) != {"images.npy", "ticks.npy"}:
        raise ValueError("unexpected paired files")
    for name, checksum in paired["sha256"].items():
        if file_sha256(source_pairs / name) != checksum:
            raise ValueError(f"paired checksum mismatch: {name}")
    result = copy.deepcopy(paired)
    result["derived_manifest"] = target
    result["rebind_provenance"] = {
        "source_pairs": str(source_pairs.resolve()),
        "source_manifest_sha256": file_sha256(source_pairs / "manifest.json"),
        "source_derived_sha256": source["files_sha256"],
        "verified": "identical episode/frame rows, raw images, split, camera and timing; no label edits",
    }
    out.mkdir(parents=True, exist_ok=False)
    for name in paired["sha256"]:
        (out / name).symlink_to((source_pairs / name).resolve())
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
