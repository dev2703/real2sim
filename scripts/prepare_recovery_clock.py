"""Remove the initial-opening episode clock from recorded achieved histories.

After the first explicitly issued close, normal jaw-switch dwell is retained.
The conversion uses only issued commands and recorded timer observations, never
teacher phases or simulator object state. Consecutive recorded ticks are required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.artifacts import sha256_file
from real2sim.policy.progress_geometry import observation_geometry_names
from real2sim.serialization import write_json

if __package__:
    from scripts.run_recovery_dagger import snapshot_sources
else:
    from run_recovery_dagger import snapshot_sources


def task_event_observations(rows):
    observations = np.asarray(rows["observations"])
    keys = [np.asarray(rows[name]) for name in ("robot_id", "episode_index", "tick")]
    n = len(observations)
    if (not n or observations.shape != (n, 99) or not np.isfinite(observations).all()
            or any(column.shape != (n,) or not np.issubdtype(column.dtype, np.integer)
                   for column in keys)):
        raise ValueError("Task-event reconstruction requires finite 99D rows and integer sequence keys")
    lookup = {}
    for index, key in enumerate(zip(*keys, strict=True)):
        if key in lookup or key[-1] < 0:
            raise ValueError("Duplicate or negative recovery control tick")
        lookup[key] = index
    clocks, closed, previous = {}, {}, {}
    for key, index in sorted(lookup.items()):
        group, tick = key[:2], key[-1]
        if tick != previous.get(group, -1) + 1:
            raise ValueError("Task-event clock conversion requires consecutive ticks starting at zero")
        current = observations[index, -33:]
        if not 0 <= current[29] <= 1 or current[32] < 0:
            raise ValueError("Issued jaw history and elapsed time must be valid")
        # The encoder's initial zero previous-action vector is not an issued
        # closure and must not start an event clock at episode initialization.
        closed[group] = closed.get(group, False) or (tick > 0 and current[29] < .5)
        clocks[key] = current[32] if closed[group] else 0.
        previous[group] = tick
    result = observations.copy()
    for key, index in lookup.items():
        for history in range(3):
            source_key = (*key[:2], max(0, key[-1] - 2 + history))
            result[index, history * 33 + 32] = clocks[source_key]
    return result


def prepare(directory, out):
    directory, out = Path(directory), Path(out)
    if out.exists():
        raise FileExistsError(out)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if (manifest.get("status") != "complete" or manifest.get("runtime_source_changes")
            or manifest.get("base_weights_frozen") is not True or manifest.get("lighting_changed") is not False):
        raise ValueError("Task-event preparation requires a completed unchanged collection")
    names = manifest.get("observation_names")
    if not names or not any("previous_achieved_" in name for name in names):
        raise ValueError("Task-event clocks require recorded achieved motion and issued jaw history")
    geometry = manifest.get("observation_geometry", "grasp" if any("_grasp_relative_" in name for name in names) else "tcp")
    observation_geometry_names(names, geometry)  # Validate the known source schema.
    converted_names = [name.replace("seconds_since_issued_gripper_switch", "seconds_since_task_gripper_event")
                       for name in names]
    observation_geometry_names(converted_names, geometry)
    sources = {str(source): sha256_file(source) for source in (
        Path(__file__), Path("src/real2sim/policy/common_recovery.py"),
        Path("src/real2sim/policy/progress_geometry.py"), path,
        directory / "train.npz", directory / "val.npz")}
    converted = {}
    for split in ("train", "val"):
        with np.load(directory / f"{split}.npz", allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["observations"] = task_event_observations(arrays)
        converted[split] = arrays
    out.mkdir(parents=True)
    ledger = snapshot_sources(out, sources)
    for split, arrays in converted.items():
        np.savez_compressed(out / f"{split}.npz", **arrays)
    after = {name: sha256_file(Path(name)) for name in sources}
    if after != sources:
        raise RuntimeError("Recovery clock inputs changed during preparation")
    report = {**manifest, "gripper_clock": "task_events", "observation_names": converted_names,
        "clock_preparation": {"schema": "issued_task_gripper_event_clock_v1",
            "source_directory": str(directory), "source_manifest_sha256": sources[str(path)],
            "teacher_phase_or_object_truth_used": False,
            "initial_opening_clock": "Suppressed until the first explicitly issued close",
            "close_release_retry_dwell": "Preserved after first close"},
        "source_sha256_before": sources, "source_sha256_after": after,
        "runtime_source_changes": [], "executed_sources_manifest": ledger}
    write_json(out / "manifest.json", report, atomic=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.data, args.out)
    print(json.dumps({"status": report["status"], "gripper_clock": report["gripper_clock"],
                      "rows": report.get("supervised_rows")}))
