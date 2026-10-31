"""Probe train-fitted phase sequence context and input sensitivity offline.

This reads immutable heads and their exact recorded collection packs. It does
not update a head, run a controller, or establish closed-loop task success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from real2sim.artifacts import sha256_file
from real2sim.policy.phase_recovery_mlp import N_CONTROL_PHASES, PHASE_NAMES, PhaseRecoveryMLPPolicy
from real2sim.policy.phase_sequence_decoder import (
    LearnedPhaseSequenceDecoder,
    fit_phase_sequence_decoder,
)
from real2sim.policy.phase_sequences import sequence_metrics
from real2sim.policy.recovery_mlp import ACTION_SCALES, bound_common_action

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_phase_recovery_mlp import load_phase_collections  # noqa: E402
from scripts.train_recovery_mlp import robot_episode_weights  # noqa: E402


def reference(path):
    return {"path": str(path), "sha256": sha256_file(Path(path))}


def keys_from_packs(directories, partition):
    collections, robots, episodes, ticks = [], [], [], []
    for collection, directory in enumerate(directories):
        with np.load(directory / f"{partition}.npz", allow_pickle=False) as saved:
            ticks.append(saved["tick"])
            robots.append(saved["robot_id"].astype(np.int64))
            episodes.append(saved["episode_index"])
            collections.append(np.full(len(saved["tick"]), collection, np.int64))
    return [np.concatenate(values) for values in (collections, robots, episodes, ticks)]


def filter_rollouts(metadata, keys, emissions, power):
    filtered = np.empty_like(emissions)
    groups = {}
    for row, key in enumerate(zip(*keys[:3], strict=True)):
        groups.setdefault(key, []).append(row)
    for indices in groups.values():
        order = sorted(indices, key=lambda index: keys[3][index])
        ticks = keys[3][order]
        if ticks[0] != 0 or np.any(np.diff(ticks) != 1):
            raise ValueError("Causal validation requires complete tick-zero control sequences")
        decoder = LearnedPhaseSequenceDecoder(metadata, emission_power=power)
        for index in order:
            filtered[index] = decoder.step(emissions[index])
    return filtered


def score(actions, predicted_phases, rows, weights):
    error = np.abs(actions - rows["target_actions"]) / np.asarray(ACTION_SCALES)
    bounded_error = np.abs(bound_common_action(actions) - rows["target_actions"]) / np.asarray(ACTION_SCALES)
    return {"robot_episode_balanced_action_mae_normalized": float(np.sum(weights * error.mean(-1))),
            "robot_episode_balanced_bounded_action_mae_normalized": float(np.sum(weights * bounded_error.mean(-1))),
            "phase_accuracy": float(np.mean(predicted_phases == rows["expert_phase_id"])),
            "robot_episode_balanced_phase_accuracy": float(np.sum(weights * (predicted_phases == rows["expert_phase_id"]))),
            "sequence": sequence_metrics(predicted_phases, rows["expert_phase_id"], rows["consecutive_pairs"])}


@torch.inference_mode()
def predictions(model, observations, base, features):
    action, logits, components = model(torch.from_numpy(observations.astype(np.float32)), base, features)
    probabilities = torch.softmax(logits / model.temperature, -1).cpu().numpy()[:, :N_CONTROL_PHASES]
    return action.cpu().numpy(), probabilities, components.cpu().numpy()[:, :N_CONTROL_PHASES]


def probe_head(directory, powers):
    policy = PhaseRecoveryMLPPolicy(directory)
    training_report = json.loads(Path(policy.metadata["training_report"]).read_text())
    directories = [Path(item["directory"]) for item in training_report["collections"]]
    data, _, sources = load_phase_collections(directories,
        expected_base_sha256=policy.metadata["base_checkpoint_sha256"],
        require_command_labels=True, require_sequences=True)
    train_keys, val_keys = (keys_from_packs(directories, name) for name in ("train", "val"))
    sequence = fit_phase_sequence_decoder(*train_keys, data["train"]["expert_phase_id"],
                                         n_phases=N_CONTROL_PHASES)
    rows = data["val"]
    weights = robot_episode_weights(rows["robot_id"], rows["episode_index"]).numpy()
    base, features = (torch.from_numpy(rows[name].astype(np.float32)) for name in ("base_actions", "features"))
    action, probabilities, components = predictions(policy.model, rows["observations"], base, features)
    raw_phase = probabilities.argmax(-1)
    raw_hard = components[np.arange(len(raw_phase)), raw_phase]
    settings = []
    for power in powers:
        filtered = filter_rollouts(sequence, val_keys, probabilities, power)
        phases = filtered.argmax(-1)
        hard_action = components[np.arange(len(phases)), phases]
        soft_action = np.sum(filtered[..., None] * components, axis=1)
        settings.append({"emission_power": power,
            "hard_routed": score(hard_action, phases, rows, weights),
            "soft_routed": score(soft_action, phases, rows, weights)})
    names = policy.metadata.get("observation_names") or []
    timer = [index for index, name in enumerate(names) if name.endswith((
        "_seconds_since_selected_gripper_switch", "_seconds_since_issued_gripper_switch"))]
    jaw = [index for index, name in enumerate(names) if name.endswith((
        "_previous_gripper_open_fraction", "_previous_issued_gripper_open_fraction"))]
    ball = {axis: [index for index, name in enumerate(names)
                  if name.endswith((f"_initial_rgb_ball_relative_{axis}", f"_fresh_rgb_ball_relative_{axis}"))]
            for axis in "xyz"}
    sensitivity = []
    variants = [("timer_zero", timer, 0., True), ("previous_jaw_zero", jaw, 0., True),
                ("ball_reference_plus10mm_x", ball["x"], .01, False),
                ("ball_reference_plus10mm_y", ball["y"], .01, False)]
    for name, columns, value, replace in variants:
        if not columns:
            continue
        changed = rows["observations"].copy()
        changed[:, columns] = value if replace else changed[:, columns] + value
        changed_action, changed_probabilities, _ = predictions(policy.model, changed, base, features)
        flips = changed_probabilities.argmax(-1) != raw_phase
        per_phase = {PHASE_NAMES[int(phase)]: float(flips[rows["expert_phase_id"] == phase].mean())
                     for phase in np.unique(rows["expert_phase_id"])}
        difference = np.abs(changed_action - action)
        sensitivity.append({"input_change": name, "all_history_frames_changed_together": True,
            "changed_columns": [names[index] for index in columns],
            "phase_prediction_changed_fraction": float(flips.mean()),
            "phase_changed_fraction_by_actual_phase": per_phase,
            "mean_translation_change_mm": float(difference[:, :3].mean() * 1000),
            "mean_gripper_change_fraction": float(difference[:, 6].mean()),
            "not_a_valid_control_rollout_or_task_score": True})
    counts = np.asarray(sequence["transition_counts"])
    learned_edges = [{"from": PHASE_NAMES[left], "to": PHASE_NAMES[right], "count": int(counts[left, right])}
                     for left, right in zip(*np.nonzero(counts), strict=True)]
    result = {"head": reference(directory / "head.json"), "head_sha256": policy.head_sha256,
        "base_checkpoint_sha256": policy.metadata["base_checkpoint_sha256"],
        "use_act": policy.metadata["use_act"], "action_routing": policy.model.action_routing,
        "collections": [str(path) for path in directories],
        "collection_artifact_sha256": sources,
        "training_rows": len(data["train"]["observations"]), "validation_rows": len(rows["observations"]),
        "learned_initial_counts": sequence["initial_counts"], "learned_edges": learned_edges,
        "learned_retry_edges": [edge for edge in learned_edges if PHASE_NAMES.index(edge["to"]) < PHASE_NAMES.index(edge["from"])],
        "raw_model_route": score(action, raw_phase, rows, weights),
        "raw_hard_route": score(raw_hard, raw_phase, rows, weights),
        "causal_sequence_routes": settings, "input_sensitivity": sensitivity}
    if any(sha256_file(Path(path)) != digest for path, digest in sources.items()):
        raise RuntimeError("Collection artifact changed during the offline probe")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", nargs="+", type=Path, required=True)
    parser.add_argument("--emission-powers", nargs="+", type=float, default=[1., .5, .25])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    result = {"schema": "phase_sequence_offline_probe_v1",
        "scope": "Held-out phase/action validation and input sensitivity only; no runtime task-success claim",
        "selection_data": "Training-only transition fit; held-out validation-only emission-power comparison",
        "heads": [probe_head(head, args.emission_powers) for head in args.heads]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for head in result["heads"]:
        print(json.dumps({"head": head["head"]["path"], "raw_hard": head["raw_hard_route"],
            "sequence": [{"power": entry["emission_power"], **entry["hard_routed"]}
                         for entry in head["causal_sequence_routes"]]}, allow_nan=False))


if __name__ == "__main__":
    main()
