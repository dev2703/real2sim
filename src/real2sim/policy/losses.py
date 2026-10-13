"""Training-only ACT reconstruction losses; inference and saved model schemas stay identical."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch.nn.functional as F
from lerobot.utils.constants import ACTION, OBS_IMAGES


@dataclass(frozen=True)
class ReconstructionLoss:
    kind: str = "l1"
    huber_delta: float = 1.0

    def __post_init__(self):
        if self.kind not in {"l1", "l2", "huber"}:
            raise ValueError(f"unknown reconstruction loss: {self.kind}")
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive (normalised action units)")


def masked_reconstruction(pred, target, action_is_pad, config: ReconstructionLoss):
    valid = (~action_is_pad).unsqueeze(-1)
    denominator = (valid.sum() * pred.shape[-1]).clamp_min(1)
    absolute = (pred - target).abs()
    if config.kind == "l1":
        errors = absolute
    elif config.kind == "l2":
        errors = (pred - target).square()
    else:
        errors = F.huber_loss(pred, target, reduction="none", delta=config.huber_delta)
    return (errors * valid).sum() / denominator, (absolute * valid).sum() / denominator


def act_training_loss(policy, batch: dict, config: ReconstructionLoss):
    # Preserve the exact installed LeRobot path for every existing/default L1 run.
    if config.kind == "l1":
        loss, metrics = policy.forward(batch)
        return loss, {**metrics, "reconstruction_loss": metrics["l1_loss"]}
    model_batch = dict(batch)
    if policy.config.image_features:
        model_batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]
    predicted, (mu, log_variance) = policy.model(model_batch)
    reconstruction, l1 = masked_reconstruction(
        predicted, batch[ACTION], batch["action_is_pad"], config,
    )
    loss = reconstruction
    metrics = {"reconstruction_loss": reconstruction.item(), "l1_loss": l1.item()}
    if policy.config.use_vae and log_variance is not None:
        # Same latent reduction and beta as LeRobot 0.6.1 ACTPolicy.forward.
        kld = (-0.5 * (1 + log_variance - mu.pow(2) - log_variance.exp())).sum(-1).mean()
        loss = loss + policy.config.kl_weight * kld
        metrics["kld_loss"] = kld.item()
    return loss, metrics
