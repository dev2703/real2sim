"""Training-only supervision between consecutive, auditable recovery rows.

No state carries across collections, embodiments, episodes, or missing ticks.
Teacher phase changes remain legal: consistency applies only while the teacher
phase is unchanged, and action differences are compared to labelled differences.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def consecutive_pairs(collection_ids, robots, episodes, ticks):
    """Return row-index pairs separated by exactly one tick in the same rollout."""
    columns = [np.asarray(value) for value in (collection_ids, robots, episodes, ticks)]
    if (any(value.ndim != 1 or value.shape != columns[0].shape for value in columns)
            or any(not np.issubdtype(value.dtype, np.integer) for value in columns)
            or any(np.any(value < 0) for value in columns)):
        raise ValueError("Sequence keys must be matching nonnegative integer vectors")
    lookup = {}
    for index, key in enumerate(zip(*columns, strict=True)):
        if key in lookup:
            raise ValueError("Duplicate collection/robot/episode/tick recovery row")
        lookup[key] = index
    pairs = []
    for key, index in lookup.items():
        following = (*key[:3], key[3] + 1)
        if following in lookup:
            pairs.append((index, lookup[following]))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def temporal_phase_loss(model, tensors, pairs, *, balance_transitions=False):
    """Encourage phase stability and supervised changes in common actions.

    This loss never forces a monotonic task sequence: retries, release changes,
    and corrective motions follow the two expert labels. It adds no inference
    state and changes neither the model's observation nor its prediction API.
    """
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
        raise ValueError("Temporal supervision requires nonempty [pairs,2] row indices")
    left, right = pairs[:, 0], pairs[:, 1]
    actions, logits, _ = model(*(tensors[name][pairs.reshape(-1)] for name in
                                 ("observations", "base_actions", "features")))
    actions = actions.reshape(-1, 2, 7)
    probabilities = torch.softmax(logits / model.temperature, dim=-1).reshape(len(pairs), 2, -1)
    same = tensors["expert_phase_id"][left] == tensors["expert_phase_id"][right]
    # Symmetric Jensen-Shannon divergence has finite gradients at small class
    # probabilities. Ground-truth boundaries are excluded, not smoothed away.
    mean = probabilities.mean(dim=1, keepdim=True)
    divergence = (probabilities * (probabilities.clamp_min(1e-8).log()
                                   - mean.clamp_min(1e-8).log())).sum(-1).mean(-1)
    stability = divergence[same].mean() if same.any() else divergence.sum() * 0.
    predicted_change = (actions[:, 1] - actions[:, 0]) / model.action_scales
    labelled_change = (tensors["target_actions"][right] - tensors["target_actions"][left]) / model.action_scales
    pair_error = F.smooth_l1_loss(predicted_change, labelled_change, beta=.1,
                                reduction="none").mean(-1)
    # When enabled, long dwell segments cannot dilute the comparatively rare
    # action changes at a labelled transition. Real retry boundaries remain
    # fully supervised, without a monotonic phase assumption.
    change_loss = (.5 * (pair_error[same].mean() + pair_error[~same].mean())
                   if balance_transitions and same.any() and (~same).any()
                   else pair_error.mean())
    return stability + change_loss, {"phase_stability_loss": stability,
                                     "action_change_loss": change_loss}


def phase_transition_loss(model, tensors, pairs):
    """Supervise both sides of actual command-phase boundaries explicitly.

    Labels enter only this training loss. Evaluation predicts from the same
    observable input API; no phase graph or teacher state controls inference.
    """
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
        raise ValueError("Transition supervision requires nonempty [pairs,2] indices")
    phases = tensors["expert_phase_id"][pairs]
    if torch.any(phases[:, 0] == phases[:, 1]):
        raise ValueError("Transition supervision requires actual phase boundaries")
    _, logits, _ = model(*(tensors[name][pairs.reshape(-1)] for name in
                          ("observations", "base_actions", "features")))
    return F.cross_entropy(logits / model.temperature, phases.reshape(-1))


def sequence_metrics(predicted_phases, labelled_phases, pairs):
    predicted, labelled = np.asarray(predicted_phases), np.asarray(labelled_phases)
    if not len(pairs):
        return {"consecutive_pairs": 0, "same_phase_switch_rate": None,
                "transition_pair_accuracy": None, "unjustified_regression_rate": None}
    left, right = pairs.T
    same = labelled[left] == labelled[right]
    transitions = ~same
    forward = labelled[right] >= labelled[left]
    return {"consecutive_pairs": len(pairs), "same_phase_pairs": int(same.sum()),
        "transition_pairs": int(transitions.sum()),
        "same_phase_switch_rate": (float((predicted[left][same] != predicted[right][same]).mean())
                                   if same.any() else None),
        "transition_pair_accuracy": (float(((predicted[left] == labelled[left])
            & (predicted[right] == labelled[right]))[transitions].mean())
            if transitions.any() else None),
        "unjustified_regression_rate": (float((predicted[right] < predicted[left])[forward].mean())
                                        if forward.any() else None)}
