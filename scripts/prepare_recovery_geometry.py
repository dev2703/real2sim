"""Prepare immutable grasp-relative recovery packs from recorded observations.

This changes only the ball/bowl reference channels. It uses measured TCP pose,
RGB references and the same declared tool offsets as native execution. Phases,
actions, episode roles and all other observations retain their original values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.artifacts import sha256_file
from real2sim.policy.progress_geometry import (
    observation_geometry_names,
    transform_observation_geometry,
)
from real2sim.serialization import write_json

if __package__:
    from scripts.run_recovery_dagger import snapshot_sources
else:
    from run_recovery_dagger import snapshot_sources


def prepare(directory, out, *, geometry="grasp"):
    directory, out = Path(directory), Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "complete" or manifest.get("runtime_source_changes")
            or manifest.get("base_weights_frozen") is not True
            or manifest.get("lighting_changed") is not False):
        raise ValueError("Recovery geometry preparation requires a completed unchanged collection")
    original_names = manifest.get("observation_names")
    names = observation_geometry_names(original_names, geometry=geometry)
    paths = (Path(__file__), Path("src/real2sim/policy/progress_geometry.py"),
             Path("src/real2sim/policy/common_recovery.py"), manifest_path,
             directory / "train.npz", directory / "val.npz")
    before = {str(path): sha256_file(path) for path in paths}
    converted = {}
    for partition in ("train", "val"):
        with np.load(directory / f"{partition}.npz", allow_pickle=False) as payload:
            rows = {key: payload[key] for key in payload.files}
        rows["observations"] = transform_observation_geometry(rows["observations"], original_names, geometry=geometry)
        converted[partition] = rows
    out.mkdir(parents=True)
    ledger = snapshot_sources(out, before)
    for partition, rows in converted.items():
        np.savez_compressed(out / f"{partition}.npz", **rows)
    after = {path: sha256_file(Path(path)) for path in before}
    if before != after:
        raise RuntimeError("Recovery geometry inputs changed during preparation")
    report = {**manifest, "observation_names": names, "observation_geometry": geometry,
        "geometry_preparation": {"schema": "declared_tool_grasp_relative_rgb_v1",
            "source_directory": str(directory), "source_manifest_sha256": before[str(manifest_path)],
            "actions_or_phases_changed": False, "object_truth_used": False},
        "source_sha256_before": before, "source_sha256_after": after,
        "runtime_source_changes": [], "executed_sources_manifest": ledger}
    write_json(out / "manifest.json", report, atomic=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--geometry", choices=("tcp", "grasp"), default="grasp")
    args = parser.parse_args()
    result = prepare(args.data, args.out, geometry=args.geometry)
    print(json.dumps({"status": result["status"], "observation_geometry": result["observation_geometry"],
                      "rows": result.get("supervised_rows")}))
