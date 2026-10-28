"""A separate learned Cartesian recovery head around a frozen ACT policy.

DAgger describes how corrective supervision is collected. ``RecoveryMLP`` is
the model fitted to that supervision. It never instantiates a geometric teacher
or consumes task-success, simulator object-pose or teacher-phase inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from real2sim.artifacts import sha256_file

ACTION_NAMES = ["tcp_dx_m", "tcp_dy_m", "tcp_dz_m", "world_drx_rad", "world_dry_rad",
                "world_drz_rad", "gripper_open_fraction"]
ACTION_SCALES = [.03, .03, .03, .3, .3, .3, 1.]


def bound_common_action(action):
    """Enforce Euclidean translation/rotation caps and semantic jaw endpoints."""
    result = np.asarray(action, np.float64).copy()
    if result.shape[-1:] != (7,) or not np.isfinite(result).all():
        raise ValueError("Common recovery action must contain seven finite values")
    for part, limit in ((slice(0, 3), .03), (slice(3, 6), .3)):
        lengths = np.linalg.norm(result[..., part], axis=-1, keepdims=True)
        result[..., part] *= np.minimum(1., limit / np.maximum(lengths, 1e-12))
    result[..., 6] = np.clip(result[..., 6], 0., 1.)
    return result


class RecoveryMLP(nn.Module):
    """Train a residual using common observations and optional frozen ACT input.

    ``use_act=False`` removes both the base-action prior and decoder features
    before learned computation. That ablation predicts a direct common action
    from exactly the same causal recovery observations.
    """

    def __init__(self, observation_mean, observation_std, feature_mean, feature_std,
                 *, use_act=True, hidden_sizes=(256, 128), action_scales=ACTION_SCALES):
        super().__init__()
        self.use_act = bool(use_act)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        means = (np.asarray(observation_mean), np.asarray(feature_mean))
        stds = (np.asarray(observation_std), np.asarray(feature_std))
        if any(mean.ndim != 1 or mean.shape != std.shape for mean, std in zip(means, stds, strict=True)):
            raise ValueError("Observation and feature normalizers must be matching vectors")
        if not len(means[0]) or not np.isfinite(np.concatenate([*means, *stds])).all():
            raise ValueError("Finite nonempty recovery normalizers required")
        if any(np.any(std < 0) for std in stds):
            raise ValueError("Recovery standard deviations cannot be negative")
        scale = np.asarray(action_scales, np.float32)
        if scale.shape != (7,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Seven positive finite action scales required")
        self.observation_dim = len(means[0])
        self.feature_dim = len(means[1])
        for name, value in (("observation_mean", means[0]), ("feature_mean", means[1]),
                            ("observation_std", np.maximum(stds[0], 1e-4)),
                            ("feature_std", np.maximum(stds[1], 1e-3)),
                            ("action_scales", scale)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        dimensions = (self.observation_dim + self.feature_dim + 7, *self.hidden_sizes, 7)
        layers = []
        for left, right in zip(dimensions[:-2], dimensions[1:-1], strict=True):
            layers.extend([nn.Linear(left, right), nn.SiLU()])
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observations, base_actions, features):
        if (observations.shape[-1] != self.observation_dim or base_actions.shape[-1] != 7
                or features.shape[-1] != self.feature_dim
                or observations.shape[:-1] != base_actions.shape[:-1]
                or observations.shape[:-1] != features.shape[:-1]):
            raise ValueError("Recovery observation, prior and feature dimensions differ")
        normalized_observations = (observations - self.observation_mean) / self.observation_std
        if self.use_act:
            prior = base_actions / self.action_scales
            normalized_features = (features - self.feature_mean) / self.feature_std
        else:
            prior = torch.zeros_like(base_actions)
            normalized_features = torch.zeros_like(features)
        inputs = torch.cat([normalized_observations, prior, normalized_features], dim=-1)
        return (prior + self.network(inputs)) * self.action_scales


class RecoveryMLPPolicy:
    """Load an independently stored recovery head with exact base provenance."""

    def __init__(self, directory, *, expected_base_sha256=None, device="cpu"):
        directory = Path(directory)
        self.metadata = json.loads((directory / "head.json").read_text())
        if self.metadata.get("schema") != "frozen_act_cartesian_recovery_mlp_v1":
            raise ValueError("Unsupported recovery MLP artifact")
        if (self.metadata.get("action_names") != ACTION_NAMES
                or not np.allclose(self.metadata.get("action_scales", []), ACTION_SCALES)):
            raise ValueError("Recovery MLP common action contract differs")
        if expected_base_sha256 is not None and self.metadata["base_checkpoint_sha256"] != expected_base_sha256:
            raise ValueError("Recovery MLP belongs to a different frozen ACT checkpoint")
        checkpoint = directory / "head.pt"
        if sha256_file(checkpoint) != self.metadata["head_sha256"]:
            raise ValueError("Recovery MLP checksum mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = RecoveryMLP(state["observation_mean"].numpy(), state["observation_std"].numpy(),
            state["feature_mean"].numpy(), state["feature_std"].numpy(),
            use_act=self.metadata["use_act"], hidden_sizes=self.metadata["hidden_sizes"],
            action_scales=self.metadata["action_scales"])
        self.model.load_state_dict(state, strict=True)
        if (self.model.observation_dim != self.metadata["observation_dim"]
                or self.model.feature_dim != self.metadata["act_feature_dim"]):
            raise ValueError("Recovery MLP input metadata differs from saved weights")
        self.model.to(device).eval().requires_grad_(False)
        self.device = device
        self.head_sha256 = self.metadata["head_sha256"]
        self.effective_policy_identifier = self.metadata["effective_policy_identifier"]

    @torch.inference_mode()
    def predict(self, observations, base_actions, features):
        values = [np.asarray(x, np.float32) for x in (observations, base_actions, features)]
        if not all(np.isfinite(x).all() for x in values):
            raise ValueError("Recovery inputs must be finite")
        tensors = [torch.from_numpy(x.copy()).to(self.device) for x in values]
        action = self.model(*tensors).cpu().numpy()
        return bound_common_action(action)
