"""Optional intervention classifier, distinct from the recovery action MLP.

The gate consumes the same observable inputs as the recovery head. Corrective
expert actions create labels during training only. Single-class evidence cannot
identify when ACT is sufficient, so deployment conservatively keeps recovery on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from real2sim.artifacts import sha256_file
from real2sim.policy.recovery_mlp import ACTION_SCALES

LABEL_THRESHOLDS = {"translation_l2_m": .002, "rotation_l2_rad": .02,
                    "gripper_fraction": .05}


def correction_needed(base_actions, expert_actions):
    """Supervision from expert disagreement, never a runtime policy input."""
    base, expert = np.asarray(base_actions, float), np.asarray(expert_actions, float)
    if (base.shape != expert.shape or base.shape[-1:] != (7,)
            or not np.isfinite(base).all() or not np.isfinite(expert).all()):
        raise ValueError("Gate labels require matching finite seven-value actions")
    delta = expert - base
    return ((np.linalg.norm(delta[..., :3], axis=-1) > LABEL_THRESHOLDS["translation_l2_m"])
            | (np.linalg.norm(delta[..., 3:6], axis=-1) > LABEL_THRESHOLDS["rotation_l2_rad"])
            | (np.abs(delta[..., 6]) > LABEL_THRESHOLDS["gripper_fraction"]))


def classification_metrics(probabilities, labels, threshold):
    probabilities, labels = np.asarray(probabilities, float), np.asarray(labels, bool)
    if (probabilities.ndim != 1 or probabilities.shape != labels.shape or not len(labels)
            or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))
            or not 0 <= threshold <= 1):
        raise ValueError("Finite probabilities and matching nonempty binary labels required")
    predicted = probabilities >= threshold
    tp, fp = int(np.sum(predicted & labels)), int(np.sum(predicted & ~labels))
    tn, fn = int(np.sum(~predicted & ~labels)), int(np.sum(~predicted & labels))
    return {"n": len(labels), "positive_labels": int(labels.sum()),
            "negative_labels": int((~labels).sum()),
            "positive_label_rate": float(labels.mean()),
            "intervention_rate": float(predicted.mean()),
            "true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "brier_score": float(np.mean((probabilities - labels.astype(float)) ** 2))}


def calibrate_threshold(probabilities, labels, training_labels, *, minimum_recall=.995, robots=None):
    """Choose a conservative held-out threshold; require both classes to gate off.

    When validation has both classes, choose the largest threshold retaining the
    requested recall in each represented robot's positive validation examples.
    One-class train or validation data cannot establish the negative boundary.
    """
    probabilities, labels = np.asarray(probabilities, float), np.asarray(labels, bool)
    training_labels = np.asarray(training_labels, bool)
    if not 0 < minimum_recall <= 1 or training_labels.ndim != 1 or not len(training_labels):
        raise ValueError("Nonempty training labels and recall target in0..1 required")
    classification_metrics(probabilities, labels, .5)
    single_class = len(np.unique(labels)) < 2 or len(np.unique(training_labels)) < 2
    if single_class:
        return {"threshold": 0., "decision_mode": "always_recover_single_class",
                "minimum_validation_recall_target": minimum_recall,
                "disabling_recovery_supported_by_data": False,
                "limitation": "Training or validation contains only one label class; no reliable ACT-sufficient/recovery-needed boundary is established."}
    groups = np.zeros(len(labels), int) if robots is None else np.asarray(robots)
    if groups.shape != labels.shape:
        raise ValueError("Robot groups must match validation labels")
    thresholds = []
    for robot in np.unique(groups):
        positive_probabilities = np.sort(probabilities[(groups == robot) & labels])
        if len(positive_probabilities):
            allowed_misses = int(np.floor((1 - minimum_recall) * len(positive_probabilities) + 1e-10))
            thresholds.append(float(positive_probabilities[allowed_misses]))
    return {"threshold": min(thresholds), "decision_mode": "learned_threshold",
            "minimum_validation_recall_target": minimum_recall,
            "disabling_recovery_supported_by_data": True,
            "limitation": "Recall calibration uses held-out collected states; it does not establish closed-loop task success or out-of-distribution reliability."}


class InterventionGateMLP(nn.Module):
    def __init__(self, observation_mean, observation_std, feature_mean, feature_std,
                 *, hidden_sizes=(64, 32)):
        super().__init__()
        observations, features = np.asarray(observation_mean), np.asarray(feature_mean)
        obs_std, feature_std = np.asarray(observation_std), np.asarray(feature_std)
        if (observations.ndim != 1 or features.ndim != 1 or not len(observations)
                or observations.shape != obs_std.shape or features.shape != feature_std.shape
                or not np.isfinite(np.r_[observations, features, obs_std, feature_std]).all()
                or np.any(obs_std < 0) or np.any(feature_std < 0)):
            raise ValueError("Gate normalizers must be finite matching vectors")
        self.observation_dim, self.feature_dim = len(observations), len(features)
        self.hidden_sizes = tuple(hidden_sizes)
        for name, value in (("observation_mean", observations),
                            ("observation_std", np.maximum(obs_std, 1e-4)),
                            ("feature_mean", features),
                            ("feature_std", np.maximum(feature_std, 1e-3)),
                            ("action_scales", np.asarray(ACTION_SCALES))):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        sizes = (self.observation_dim + self.feature_dim + 7, *self.hidden_sizes, 1)
        layers = []
        for left, right in zip(sizes[:-2], sizes[1:-1], strict=True):
            layers.extend([nn.Linear(left, right), nn.SiLU()])
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, observations, base_actions, features):
        if (observations.shape[-1] != self.observation_dim or base_actions.shape[-1] != 7
                or features.shape[-1] != self.feature_dim
                or observations.shape[:-1] != base_actions.shape[:-1]
                or observations.shape[:-1] != features.shape[:-1]):
            raise ValueError("Gate input dimensions differ from its saved contract")
        observation = ((observations - self.observation_mean) / self.observation_std).clamp(-10., 10.)
        feature = ((features - self.feature_mean) / self.feature_std).clamp(-10., 10.)
        prior = base_actions / self.action_scales
        return self.network(torch.cat([observation, prior, feature], dim=-1)).squeeze(-1)


class RecoveryInterventionGate:
    """Load an independently stored gate bound to both ACT and recovery weights."""

    def __init__(self, directory, *, expected_base_sha256, expected_recovery_head_sha256,
                 device="cpu"):
        directory = Path(directory)
        self.metadata = json.loads((directory / "gate.json").read_text())
        if self.metadata.get("schema") != "frozen_act_recovery_intervention_gate_v1":
            raise ValueError("Unsupported recovery intervention gate artifact")
        if self.metadata["base_checkpoint_sha256"] != expected_base_sha256:
            raise ValueError("Gate belongs to a different frozen ACT checkpoint")
        if self.metadata["recovery_head_sha256"] != expected_recovery_head_sha256:
            raise ValueError("Gate belongs to a different recovery MLP head")
        checkpoint = directory / "gate.pt"
        if sha256_file(checkpoint) != self.metadata["gate_sha256"]:
            raise ValueError("Intervention gate checksum mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = InterventionGateMLP(state["observation_mean"].numpy(),
            state["observation_std"].numpy(), state["feature_mean"].numpy(),
            state["feature_std"].numpy(), hidden_sizes=self.metadata["hidden_sizes"])
        self.model.load_state_dict(state, strict=True)
        if (self.model.observation_dim != self.metadata["observation_dim"]
                or self.model.feature_dim != self.metadata["act_feature_dim"]):
            raise ValueError("Intervention gate input metadata differs from saved weights")
        calibration = self.metadata["calibration"]
        self.threshold = float(calibration["threshold"])
        self.decision_mode = calibration["decision_mode"]
        if (not 0 <= self.threshold <= 1
                or self.decision_mode not in ("learned_threshold", "always_recover_single_class")
                or (self.decision_mode == "always_recover_single_class" and self.threshold != 0)):
            raise ValueError("Unsupported gate calibration or nonconservative single-class threshold")
        self.device = device
        self.model.to(device).eval().requires_grad_(False)
        self.gate_sha256 = self.metadata["gate_sha256"]

    @torch.inference_mode()
    def probability(self, observations, base_actions, features):
        values = [np.asarray(x, np.float32) for x in (observations, base_actions, features)]
        if not all(np.isfinite(x).all() for x in values):
            raise ValueError("Intervention gate inputs must be finite")
        tensors = [torch.from_numpy(x.copy()).to(self.device) for x in values]
        probabilities = torch.sigmoid(self.model(*tensors)).cpu().numpy()
        return float(probabilities) if probabilities.ndim == 0 else probabilities

    def needs_recovery(self, observations, base_actions, features):
        probability = self.probability(observations, base_actions, features)
        return np.asarray(probability) >= self.threshold
