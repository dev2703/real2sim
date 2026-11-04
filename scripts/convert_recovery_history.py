"""Reconstruct achieved histories from immutable, rejection-free expert packs.

Only pure-expert runs with consecutive recorded measured poses and no rejected
or clipped native commands are eligible. Their selected jaw equals the issued
jaw; measured pose differences replace selected motion. Mixed DAgger packs need
live achieved-history collection because their issued jaw cannot be inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.artifacts import sha256_file
from real2sim.geometry.so3 import delta_action, make_T, matrix_from_rot6d
from real2sim.policy.common_recovery import (
    ACHIEVED_OBSERVATION_NAMES,
    OBSERVATION_NAMES,
    TRACKED_ACHIEVED_OBSERVATION_NAMES,
    TRACKED_OBSERVATION_NAMES,
)
from real2sim.serialization import write_json

if __package__:
    from scripts.run_recovery_dagger import snapshot_sources
else:
    from run_recovery_dagger import snapshot_sources


def validate_manifest(manifest):
    if (manifest.get("status") != "complete" or manifest.get("runtime_source_changes")
            or manifest.get("checkpoint_unchanged") is not True
            or manifest.get("base_weights_frozen") is not True
            or manifest.get("lighting_changed") is not False):
        raise ValueError("Conversion requires a completed unchanged frozen-checkpoint collection")
    if (manifest.get("beta_teacher_execution") != 1.
            or manifest.get("expert_phase_label_source") != "issued_command"):
        raise ValueError("Only pure-expert issued-command collections can reconstruct issued jaw history")
    trials = manifest.get("trials", [])
    if not trials or any(row.get("abort") is not None or row.get("ik_rejections", 1) != 0
                         or row.get("learner_executed_ticks", 1) != 0
                         or row.get("max_action_saturation", 1.) != 0. for row in trials):
        raise ValueError("Rejected, clipped or learner-executed commands require live feedback collection")
    names = manifest.get("observation_names")
    if names == TRACKED_OBSERVATION_NAMES:
        return TRACKED_ACHIEVED_OBSERVATION_NAMES
    if names == OBSERVATION_NAMES:
        return ACHIEVED_OBSERVATION_NAMES
    raise ValueError("Expected the original selected-action recovery observation schema")


def convert_observations(rows):
    observations = np.asarray(rows["observations"])
    ticks, episodes, robots = (np.asarray(rows[key]) for key in ("tick", "episode_index", "robot_id"))
    target = np.asarray(rows["target_actions"])
    n = len(observations)
    if (not n or observations.shape != (n, 99) or target.shape != (n, 7)
            or not np.isfinite(observations).all() or not np.isfinite(target).all()
            or any(column.shape != (n,) or not np.issubdtype(column.dtype, np.integer)
                   for column in (ticks, episodes, robots))):
        raise ValueError("Finite 99-value observations, actions and integer sequence keys required")
    result = observations.copy()
    lookup = {}
    for index, key in enumerate(zip(robots, episodes, ticks, strict=True)):
        if key in lookup or key[-1] < 0:
            raise ValueError("Duplicate or negative recovery sequence tick")
        lookup[key] = index
    feedback = {}
    for key, index in sorted(lookup.items()):
        current = observations[index, -33:]
        if key[-1] == 0:
            action = np.zeros(7)
        else:
            previous_key = (*key[:2], key[-1] - 1)
            if previous_key not in lookup:
                raise ValueError("Cannot reconstruct achieved motion across a missing control tick")
            previous_index = lookup[previous_key]
            previous = observations[previous_index, -33:]
            before = make_T(matrix_from_rot6d(previous[3:9]), previous[:3])
            after = make_T(matrix_from_rot6d(current[3:9]), current[:3])
            jaw = target[previous_index, 6]
            if not 0. <= jaw <= 1. or not np.isclose(current[29], jaw, atol=1e-6, rtol=0.):
                raise ValueError("Recorded selected jaw differs from the accepted expert command")
            action = np.r_[delta_action(before, after), jaw]
        feedback[key] = action
    for key, index in lookup.items():
        for history in range(3):
            frame_tick = max(0, key[-1] - 2 + history)
            frame_key = (*key[:2], frame_tick)
            if frame_key not in lookup:
                raise ValueError("Missing measured pose for a recovery history frame")
            frame = observations[index, history * 33:(history + 1) * 33]
            recorded = observations[lookup[frame_key], -33:]
            if not np.allclose(frame[:9], recorded[:9], atol=1e-6, rtol=0.):
                raise ValueError("History pose differs from its recorded causal control tick")
            result[index, history * 33 + 23:history * 33 + 30] = feedback[frame_key]
    return result


def convert(directory, out):
    directory, out = Path(directory), Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = validate_manifest(manifest)
    sources = {str(path): sha256_file(path) for path in (
        Path(__file__), Path("src/real2sim/policy/common_recovery.py"),
        Path("src/real2sim/geometry/so3.py"), manifest_path,
        directory / "train.npz", directory / "val.npz")}
    converted = {}
    for partition in ("train", "val"):
        with np.load(directory / f"{partition}.npz", allow_pickle=False) as payload:
            rows = {key: payload[key] for key in payload.files}
        rows["observations"] = convert_observations(rows)
        converted[partition] = rows
    out.mkdir(parents=True)
    ledger = snapshot_sources(out, sources)
    for partition, rows in converted.items():
        np.savez_compressed(out / f"{partition}.npz", **rows)
    after = {path: sha256_file(Path(path)) for path in sources}
    if after != sources:
        raise RuntimeError("Recovery conversion inputs changed during reconstruction")
    report = {**manifest, "history_mode": "achieved", "observation_names": names,
        "conversion": {"schema": "pure_expert_measured_motion_history_v1",
            "source_directory": str(directory), "source_manifest_sha256": sources[str(manifest_path)],
            "motion": "Consecutive recorded measured TCP poses, without clipping",
            "jaw": "Accepted pure-expert command; no rejected or clipped commands allowed",
            "gripper_timer": "Original selected and issued jaw switches coincide under these constraints"},
        "source_sha256_before": sources, "source_sha256_after": after,
        "runtime_source_changes": [], "executed_sources_manifest": ledger}
    write_json(out / "manifest.json", report, atomic=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = convert(args.data, args.out)
    print(json.dumps({"status": result["status"], "history_mode": result["history_mode"],
                      "rows": result.get("supervised_rows")}))
