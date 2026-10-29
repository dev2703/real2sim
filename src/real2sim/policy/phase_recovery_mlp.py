"""Phase-supervised mixture of recovery action heads around frozen ACT.

Expert phase is auxiliary training supervision. Inference predicts its own phase
probabilities from observable history, the frozen ACT proposal and its decoder
features. No teacher phase, waypoint or simulator task flag is an input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from real2sim.artifacts import sha256_file
from real2sim.policy.recovery_mlp import ACTION_NAMES, ACTION_SCALES, bound_common_action

PHASE_NAMES = ["ACQUIRE", "PRE_GRASP", "DESCEND", "CLOSE", "LIFT", "TRANSPORT",
               "LOWER", "RELEASE", "RETREAT", "DONE", "FAILED"]
N_CONTROL_PHASES = 9


class PhaseRecoveryMLP(nn.Module):
    """Predict phase logits and mix seven-value per-phase residual actions."""

    def __init__(self, observation_mean, observation_std, feature_mean, feature_std,
                 *, hidden_sizes=(256, 128), temperature=.5, use_act=True):
        super().__init__()
        observations, features = np.asarray(observation_mean), np.asarray(feature_mean)
        obs_std, feat_std = np.asarray(observation_std), np.asarray(feature_std)
        if (observations.ndim != 1 or features.ndim != 1 or not len(observations)
                or observations.shape != obs_std.shape or features.shape != feat_std.shape
                or not np.isfinite(np.r_[observations, features, obs_std, feat_std]).all()
                or np.any(obs_std < 0) or np.any(feat_std < 0)):
            raise ValueError("Phase-recovery normalizers must be finite matching vectors")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Positive finite phase temperature required")
        self.observation_dim, self.feature_dim = len(observations), len(features)
        self.hidden_sizes = tuple(int(n) for n in hidden_sizes)
        self.temperature = float(temperature)
        self.use_act = bool(use_act)
        for name, value in (("observation_mean", observations),
                            ("observation_std", np.maximum(obs_std, 1e-4)),
                            ("feature_mean", features),
                            ("feature_std", np.maximum(feat_std, 1e-3)),
                            ("action_scales", np.asarray(ACTION_SCALES))):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        dimensions = (self.observation_dim + self.feature_dim + 7, *self.hidden_sizes)
        layers = []
        for left, right in zip(dimensions[:-1], dimensions[1:], strict=True):
            layers.extend([nn.Linear(left, right), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.phase_head = nn.Linear(dimensions[-1], len(PHASE_NAMES))
        self.action_heads = nn.Linear(dimensions[-1], len(PHASE_NAMES) * 7)
        nn.init.zeros_(self.action_heads.weight)
        nn.init.zeros_(self.action_heads.bias)

    def forward(self, observations, base_actions, features):
        if (observations.shape[-1] != self.observation_dim or base_actions.shape[-1] != 7
                or features.shape[-1] != self.feature_dim
                or observations.shape[:-1] != base_actions.shape[:-1]
                or observations.shape[:-1] != features.shape[:-1]):
            raise ValueError("Phase-recovery observable input dimensions differ")
        observation = ((observations - self.observation_mean) / self.observation_std).clamp(-10., 10.)
        if self.use_act:
            feature = ((features - self.feature_mean) / self.feature_std).clamp(-10., 10.)
            prior = base_actions / self.action_scales
        else:
            feature = torch.zeros_like(features)
            prior = torch.zeros_like(base_actions)
        encoded = self.trunk(torch.cat([observation, prior, feature], dim=-1))
        raw_logits = self.phase_head(encoded)
        # Terminal phases supply no valid corrective labels and cannot drive
        # an action. The nonterminal components are inferred, never supplied.
        control_logits = raw_logits[..., :N_CONTROL_PHASES]
        masked_logits = torch.cat([control_logits, torch.full_like(raw_logits[..., N_CONTROL_PHASES:], -1e9)], dim=-1)
        phase_probabilities = torch.softmax(masked_logits / self.temperature, dim=-1)
        residuals = self.action_heads(encoded).reshape(*encoded.shape[:-1], len(PHASE_NAMES), 7)
        phase_actions = (prior.unsqueeze(-2) + residuals) * self.action_scales
        mixed_action = (phase_probabilities.unsqueeze(-1) * phase_actions).sum(dim=-2)
        return mixed_action, masked_logits, phase_actions


def phase_training_loss(model, observations, base_actions, features, target_actions,
                        expert_phase_ids, *, phase_weight=.2, class_weights=None):
    """Only this supervised loss receives expert phase labels."""
    if (expert_phase_ids.shape != observations.shape[:-1]
            or expert_phase_ids.dtype != torch.long
            or torch.any(expert_phase_ids < 0) or torch.any(expert_phase_ids >= N_CONTROL_PHASES)):
        raise ValueError("Expert phase audit labels must be nonterminal integer IDs")
    mixed, logits, per_phase = model(observations, base_actions, features)
    selected = per_phase.gather(-2, expert_phase_ids[..., None, None].expand(*expert_phase_ids.shape, 1, 7)).squeeze(-2)
    scale = model.action_scales
    mixed_loss = nn.functional.smooth_l1_loss(mixed / scale, target_actions / scale, beta=.1)
    selected_loss = nn.functional.smooth_l1_loss(selected / scale, target_actions / scale, beta=.1)
    phase_loss = nn.functional.cross_entropy(logits[..., :N_CONTROL_PHASES], expert_phase_ids,
                                             weight=class_weights)
    total = .5 * mixed_loss + .5 * selected_loss + phase_weight * phase_loss
    return total, {"mixed_action_loss": mixed_loss, "expert_component_action_loss": selected_loss,
                   "phase_classification_loss": phase_loss}


class PhaseRecoveryMLPPolicy:
    """Independent artifact with the same three-input action API as recovery MLP."""

    def __init__(self, directory, *, expected_base_sha256=None, device="cpu"):
        directory = Path(directory)
        self.metadata = json.loads((directory / "head.json").read_text())
        if self.metadata.get("schema") != "frozen_act_phase_supervised_recovery_mlp_v1":
            raise ValueError("Unsupported phase-supervised recovery artifact")
        if (self.metadata.get("action_names") != ACTION_NAMES
                or self.metadata.get("phase_names") != PHASE_NAMES
                or not np.allclose(self.metadata.get("action_scales", []), ACTION_SCALES)
                or not isinstance(self.metadata.get("use_act"), bool)
                or self.metadata.get("expert_phase_is_runtime_input") is not False):
            raise ValueError("Phase-recovery action or observable-input contract differs")
        if expected_base_sha256 is not None and self.metadata["base_checkpoint_sha256"] != expected_base_sha256:
            raise ValueError("Phase-recovery head belongs to a different frozen ACT checkpoint")
        checkpoint = directory / "head.pt"
        if sha256_file(checkpoint) != self.metadata["head_sha256"]:
            raise ValueError("Phase-recovery head checksum mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = PhaseRecoveryMLP(state["observation_mean"].numpy(), state["observation_std"].numpy(),
            state["feature_mean"].numpy(), state["feature_std"].numpy(),
            hidden_sizes=self.metadata["hidden_sizes"], temperature=self.metadata["phase_temperature"],
            use_act=self.metadata["use_act"])
        self.model.load_state_dict(state, strict=True)
        if (self.model.observation_dim != self.metadata["observation_dim"]
                or self.model.feature_dim != self.metadata["act_feature_dim"]):
            raise ValueError("Phase-recovery input metadata differs from saved weights")
        self.model.to(device).eval().requires_grad_(False)
        self.device = device
        self.head_sha256 = self.metadata["head_sha256"]
        self.effective_policy_identifier = self.metadata["effective_policy_identifier"]
        self.last_phase_id = None
        self.last_phase_probabilities = None

    @torch.inference_mode()
    def predict(self, observations, base_actions, features):
        values = [np.asarray(x, np.float32) for x in (observations, base_actions, features)]
        if not all(np.isfinite(x).all() for x in values):
            raise ValueError("Phase-recovery observable inputs must be finite")
        tensors = [torch.from_numpy(x.copy()).to(self.device) for x in values]
        action, logits, _ = self.model(*tensors)
        phase = torch.softmax(logits / self.model.temperature, dim=-1).cpu().numpy()
        self.last_phase_probabilities = phase.tolist()
        self.last_phase_id = np.argmax(phase, axis=-1).tolist()
        return bound_common_action(action.cpu().numpy())
