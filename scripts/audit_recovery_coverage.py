"""Audit corrective-state coverage without running policies or changing artifacts.

Compare completed collection packs against a frozen head's training normalizers.
All trial attempts stay in the audit, including aborts and unlabelled terminal
tails. Phase IDs are training labels; only recorded observable inputs enter the
extrapolation calculation. Reserved evaluation artifacts are never opened.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from real2sim.artifacts import sha256_file
from real2sim.policy.phase_recovery_mlp import N_CONTROL_PHASES, PHASE_NAMES
from real2sim.serialization import write_json

ARRAY_NAMES = ("observations", "base_actions", "features", "target_actions", "robot_id",
               "episode_index", "expert_phase_id", "tick")


def distribution(values):
    values = np.asarray(values, float)
    if not values.size:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {"count": int(values.size), "min": float(values.min()),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def normalized_coverage(values, mean, std, names):
    """Report raw pre-clamp normalization tails, never substitute simulator state."""
    values, mean, std = np.asarray(values), np.asarray(mean), np.asarray(std)
    if (values.ndim != 2 or mean.shape != (values.shape[1],) or std.shape != mean.shape
            or len(names) != len(mean) or np.any(std <= 0)
            or not all(np.isfinite(value).all() for value in (values, mean, std))):
        raise ValueError("Coverage requires finite matrices and matching positive normalizers")
    normalized = np.abs((values - mean) / std)
    if not len(values):
        return {"rows": 0, "max_abs_training_z": None,
                "rows_any_abs_z_gt_5": 0, "rows_any_abs_z_gt_10": 0,
                "largest_dimensions": []}
    maximum = normalized.max(axis=0)
    largest = np.argsort(-maximum, kind="stable")[:5]
    return {"rows": len(values), "max_abs_training_z": float(maximum.max()),
        "rows_any_abs_z_gt_5": int(np.any(normalized > 5, axis=1).sum()),
        "rows_any_abs_z_gt_10": int(np.any(normalized > 10, axis=1).sum()),
        "largest_dimensions": [{"name": names[index], "max_abs_training_z": float(maximum[index]),
            "rows_abs_z_gt_10": int(np.sum(normalized[:, index] > 10)),
            "observed_min": float(values[:, index].min()),
            "observed_max": float(values[:, index].max())} for index in largest]}


def validate_partition(arrays, manifest, partition, names, feature_dim):
    if set(ARRAY_NAMES) - arrays.keys():
        raise ValueError("Coverage requires saved observations, action/phase labels and actual ticks")
    count = len(arrays["observations"])
    widths = {"observations": len(names), "features": feature_dim,
              "base_actions": 7, "target_actions": 7}
    for name, width in widths.items():
        if arrays[name].shape != (count, width) or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Invalid finite {name} matrix")
    for name in ("robot_id", "episode_index", "expert_phase_id", "tick"):
        values = arrays[name]
        if (values.shape != (count,) or not np.issubdtype(values.dtype, np.integer)
                or np.any(values < 0)):
            raise ValueError(f"Invalid per-row integer {name}")
    expected_robot = {"so101": 0, "panda": 1}[manifest["robot"]]
    if np.any(arrays["robot_id"] != expected_robot):
        raise ValueError("Robot IDs differ from the collection embodiment")
    if np.any(arrays["expert_phase_id"] >= N_CONTROL_PHASES):
        raise ValueError("Coverage requires nonterminal command-phase labels")
    declared = manifest["split"][partition]
    if len(set(declared)) != len(declared) or not set(arrays["episode_index"]) <= set(declared):
        raise ValueError("Saved rows do not match declared episode partition")
    identities = list(zip(arrays["episode_index"], arrays["tick"], strict=True))
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate episode/tick rows would inflate state coverage")
    trials = [trial for trial in manifest["trials"] if trial["partition"] == partition]
    if len(trials) != len(declared) or {trial["seed"] for trial in trials} != set(declared):
        raise ValueError("Every declared attempt must have exactly one scored trial")
    if count != manifest["supervised_rows"][partition]:
        raise ValueError("Supervised row total differs from the manifest")
    episodes = []
    for trial in trials:
        ticks = np.sort(arrays["tick"][arrays["episode_index"] == trial["seed"]])
        if len(ticks) != trial["supervised_rows"] or (len(ticks) and ticks[-1] >= trial["ticks"]):
            raise ValueError("Trial supervised count/tick coverage differs from saved arrays")
        episodes.append({"seed": trial["seed"], "executed_ticks": trial["ticks"],
            "supervised_rows": len(ticks), "first_label_tick": int(ticks[0]) if len(ticks) else None,
            "last_label_tick": int(ticks[-1]) if len(ticks) else None,
            "missing_ticks_inside_label_span": int(np.maximum(np.diff(ticks) - 1, 0).sum()),
            "unlabelled_final_ticks": int(trial["ticks"] - ticks[-1] - 1) if len(ticks) else trial["ticks"],
            "learner_executed_ticks": trial["learner_executed_ticks"],
            "abort": trial["abort"], "abort_source_projection": trial.get("abort_source_projection")})
    return episodes


def snapshot_integrity(directory, manifest):
    before, after = manifest["source_sha256_before"], manifest["source_sha256_after"]
    if before != after:
        raise ValueError("Collection source hashes changed during execution")
    ledger_path = Path(manifest["executed_sources_manifest"])
    ledger = json.loads(ledger_path.read_text())
    recorded = {entry["source"]: entry["sha256"] for entry in ledger["entries"]}
    if recorded != before:
        raise ValueError("Executed source ledger differs from declared source hashes")
    verified = 0
    for entry in ledger["entries"]:
        if entry["snapshot"] is not None:
            if sha256_file(directory / entry["snapshot"]) != entry["sha256"]:
                raise ValueError("Preserved executable source snapshot checksum mismatch")
            verified += 1
    return {"manifest": str(ledger_path), "manifest_sha256": sha256_file(ledger_path),
            "preserved_text_sources_verified": verified,
            "binary_references": len(ledger["entries"]) - verified}


def audit_pack(directory, metadata, state):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("runtime_source_changes"):
        raise ValueError("Coverage audit requires completed collection with unchanged sources")
    if (manifest.get("checkpoint_unchanged") is not True
            or manifest.get("base_weights_frozen") is not True
            or manifest.get("lighting_changed") is not False
            or manifest.get("base_checkpoint_sha256") != metadata["base_checkpoint_sha256"]):
        raise ValueError("Collection frozen-checkpoint provenance differs")
    names = metadata["observation_names"]
    if (manifest.get("observation_names") != names
            or manifest.get("expert_phase_label_source") != "issued_command"):
        raise ValueError("Coverage comparison requires matching observables and issued-command labels")
    snapshots = snapshot_integrity(directory, manifest)
    arrays_by_split, summaries = {}, {}
    for partition in ("train", "val"):
        path = directory / f"{partition}.npz"
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in ARRAY_NAMES if name in saved}
        episodes = validate_partition(arrays, manifest, partition, names, len(state["feature_mean"]))
        arrays_by_split[partition] = arrays
        summaries[partition] = {"array": str(path), "sha256": sha256_file(path),
                               "rows": len(arrays["observations"]), "episodes": episodes}
    reasons = Counter()
    for trial in manifest["trials"]:
        reasons.update(trial["command_reasons"])
    row = {"directory": str(directory), "manifest_sha256": sha256_file(manifest_path),
        "robot": manifest["robot"], "split": manifest["split"], "snapshots": snapshots,
        "teacher_execution_probability": manifest["beta_teacher_execution"],
        "expert_queries_actual_learner_states": manifest["expert_queries_actual_learner_states"],
        "learner_executed_ticks": sum(t["learner_executed_ticks"] for t in manifest["trials"]),
        "attempts": len(manifest["trials"]),
        "aborted_attempts": sum(t["abort"] is not None for t in manifest["trials"]),
        "command_reasons": dict(reasons),
        "source_mapping_guard_declared": manifest.get("panda_actions_require_source_mapping"),
        "commands_finally_rejected_for_source_mapping": sum(value for key, value in reasons.items()
                                                           if key.startswith("virtual_source_")),
        "source_projection_aborts": [t.get("abort_source_projection") for t in manifest["trials"]
                                     if t.get("abort_source_projection") is not None],
        "partitions": summaries}
    return row, arrays_by_split


def summarize_group(groups, names, state):
    values = {name: np.concatenate([group[name] for group in groups]) for name in ARRAY_NAMES}
    phases = np.bincount(values["expert_phase_id"], minlength=N_CONTROL_PHASES)
    timing = {}
    for suffix in ("ball_observation_age_s", "bowl_observation_age_s",
                   "seconds_since_selected_gripper_switch"):
        name = "history_2_" + suffix
        if name in names:
            timing[name] = distribution(values["observations"][:, names.index(name)])
    return {"rows": len(values["observations"]),
        "episodes": sorted(set(values["episode_index"].tolist())),
        "phase_rows": {PHASE_NAMES[index]: int(count) for index, count in enumerate(phases)},
        "observable_timing": timing,
        "observation_extrapolation": normalized_coverage(values["observations"],
            state["observation_mean"], state["observation_std"], names),
        "act_feature_extrapolation": normalized_coverage(values["features"],
            state["feature_mean"], state["feature_std"],
            [f"decoder_{index}" for index in range(len(state["feature_mean"]))])}


def audit(prior_packs, new_packs, head):
    head = Path(head)
    metadata_path, weights = head / "head.json", head / "head.pt"
    metadata = json.loads(metadata_path.read_text())
    if sha256_file(weights) != metadata["head_sha256"]:
        raise ValueError("Frozen head checksum mismatch")
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    state = {name: saved[name].cpu().numpy() for name in
             ("observation_mean", "observation_std", "feature_mean", "feature_std")}
    seen_directories, all_splits, reserved = set(), {"train": set(), "val": set()}, set()
    cohorts = {}
    for name, directories in (("prior", prior_packs), ("new", new_packs)):
        packs, grouped = [], {}
        for directory in directories:
            resolved = Path(directory).resolve()
            if resolved in seen_directories:
                raise ValueError("A pack cannot be repeated across comparison cohorts")
            seen_directories.add(resolved)
            row, arrays = audit_pack(directory, metadata, state)
            packs.append(row)
            reserved.update(row["split"].get("evaluation", []))
            for partition in ("train", "val"):
                all_splits[partition].update(row["split"][partition])
                grouped.setdefault(row["robot"], {}).setdefault(partition, []).append(arrays[partition])
        if not packs:
            raise ValueError("Both prior and new cohorts require completed packs")
        cohorts[name] = {"packs": packs, "robots": {robot: {
            partition: summarize_group(groups, metadata["observation_names"], state)
            for partition, groups in partitions.items()} for robot, partitions in grouped.items()}}
    if all_splits["train"] & all_splits["val"]:
        raise ValueError("Train/validation episode overlap across compared packs")
    if (all_splits["train"] | all_splits["val"]) & reserved:
        raise ValueError("Reserved evaluation episode entered collection")
    return {"schema": "recovery_corrective_coverage_audit_v1", "status": "complete",
        "head": str(head), "head_sha256": metadata["head_sha256"],
        "head_metadata_sha256": sha256_file(metadata_path),
        "base_checkpoint_sha256": metadata["base_checkpoint_sha256"],
        "audit_source_sha256": sha256_file(Path(__file__)),
        "scope": "Coverage of saved corrective labels and their causal observable inputs. All attempted trials retained; this is not learned-only task success or a workspace-completeness claim.",
        "normalization": "Exact saved old-head training normalizers, before its inference clamp",
        "limitations": ["Unlabelled terminal tails are counted but have no stored model-input rows.",
            "Learner-executed ticks prove learner exposure; row-level teacher/learner selection is not saved.",
            "Manifest command reasons retain final outcomes; backtracked candidate mapping rejections are not counted.",
            "Zero mapping aborts or rejections in these trajectories cannot establish coverage of every Panda pose."],
        "split_integrity": {"train_validation_disjoint": True,
                            "reserved_episode_overlap": [], "reserved_artifacts_read": False},
        "cohorts": cohorts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-packs", type=Path, nargs="+", required=True)
    parser.add_argument("--new-packs", type=Path, nargs="+", required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    write_json(args.out, audit(args.prior_packs, args.new_packs, args.head), atomic=True)
    print(str(args.out))


if __name__ == "__main__":
    main()
