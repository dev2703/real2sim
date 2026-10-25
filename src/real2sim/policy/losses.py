"""Training-only ACT reconstruction losses; inference and saved model schemas stay identical."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from lerobot.utils.constants import ACTION, OBS_IMAGES


@dataclass(frozen=True)
class ReconstructionLoss:
    kind: str = "l1"
    huber_delta: float = 1.0
    trajectory_weight: float = 0.0
    translation_scale_m: float = .024
    rotation_scale_rad: float = .24
    supervised_prefix_steps: int | None = None

    def __post_init__(self):
        if self.kind not in {"l1", "l2", "huber"}:
            raise ValueError(f"unknown reconstruction loss: {self.kind}")
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive (normalised action units)")
        if not math.isfinite(self.trajectory_weight) or self.trajectory_weight < 0:
            raise ValueError("trajectory_weight must be finite and nonnegative")
        if any(not math.isfinite(x) or x <= 0 for x in (self.translation_scale_m, self.rotation_scale_rad)):
            raise ValueError("trajectory scales must be finite and positive")
        if self.supervised_prefix_steps is not None and (
                isinstance(self.supervised_prefix_steps, bool)
                or not isinstance(self.supervised_prefix_steps, int)
                or self.supervised_prefix_steps < 1):
            raise ValueError("supervised_prefix_steps must be a positive integer or None")


def _supervision_slice(pred, target, action_is_pad, config):
    """Keep the executed action offsets, without changing the predicted chunk."""
    steps = config.supervised_prefix_steps
    if steps is None:
        return pred, target, action_is_pad
    if pred.ndim != 3 or pred.shape != target.shape or action_is_pad.shape != pred.shape[:2]:
        raise ValueError("prefix supervision requires aligned batch/action chunks and padding")
    if steps > pred.shape[1]:
        raise ValueError("supervised_prefix_steps exceeds the predicted action chunk")
    return pred[:, :steps], target[:, :steps], action_is_pad[:, :steps]


def masked_reconstruction(pred, target, action_is_pad, config: ReconstructionLoss):
    pred, target, action_is_pad = _supervision_slice(pred, target, action_is_pad, config)
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


def _rotation_from_vector(vector):
    """Rodrigues exponential with finite derivatives at zero, including on MPS."""
    squared_angle = vector.square().sum(-1)
    # Keep the inactive analytic branch finite too: autograd evaluates both branches.
    angle = squared_angle.clamp_min(1e-8).sqrt()
    small = squared_angle < 1e-4
    a = torch.where(
        small, 1 - squared_angle / 6 + squared_angle.square() / 120,
        angle.sin() / angle,
    )
    half_angle = angle / 2
    b = torch.where(
        small, .5 - squared_angle / 24 + squared_angle.square() / 720,
        .5 * (half_angle.sin() / half_angle).square(),
    )
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(
        *vector.shape[:-1], 3, 3,
    )
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return identity + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def integrated_tool_loss(pred, target, pad, *, tool_frame, translation_scale, rotation_scale):
    """Differentiable rigid-motion composition; no simulator ground truth or robot joints.

    Compare composed predicted and demonstrated trajectories from a common identity pose.
    This penalizes accumulated translation/rotation error; it is not LEGATO's DHB loss.
    """
    if pred.shape != target.shape or pred.shape[-1] != 7 or pad.shape != pred.shape[:2]:
        raise ValueError("trajectory consistency requires aligned seven-dimensional chunks")
    if torch.any(pad[:, :-1] & ~pad[:, 1:]):
        raise ValueError("trajectory padding must be terminal")
    valid = ~pad
    # Rotation exponentials and repeated rotation products are unstable in half precision.
    # Disable autocast as well as promoting inputs: autocast would otherwise cast the
    # composition matmuls back down. Casts remain differentiable to the model outputs.
    dtype = torch.promote_types(torch.promote_types(pred.dtype, target.dtype), torch.float32)
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred, target = (x.to(dtype).masked_fill(pad[..., None], 0) for x in (pred, target))

        def compose(actions):
            rotation = torch.eye(3, device=actions.device, dtype=actions.dtype).expand(len(actions), 3, 3)
            position = actions.new_zeros(len(actions), 3)
            positions, rotations = [], []
            for action in actions.unbind(1):
                # linalg_matrix_exp has no MPS kernel. Rodrigues has the same SO(3)
                # semantics and keeps this training loss on the selected device.
                step = _rotation_from_vector(action[:, 3:6])
                dp = action[:, :3]
                if tool_frame:
                    position = position + (rotation @ dp[..., None]).squeeze(-1)
                    rotation = rotation @ step
                else:
                    position = position + dp
                    rotation = step @ rotation
                positions.append(position)
                rotations.append(rotation)
            return torch.stack(positions, 1), torch.stack(rotations, 1)

        p, r = compose(pred)
        tp, tr = compose(target)
        position_error = (p - tp).square().sum(-1) / translation_scale**2
        # Stable chordal SO(3) error: ||R1-R2||_F^2/2, approximately angle^2 near zero.
        rotation_error = (r - tr).square().sum((-1, -2)) / (2 * rotation_scale**2)
        return ((position_error + rotation_error) * valid).sum() / valid.sum().clamp_min(1)


def act_training_loss(policy, batch: dict, config: ReconstructionLoss, *,
                      action_mean=None, action_std=None, action_representation=None):
    # Preserve the exact installed LeRobot path for every existing/default L1 run.
    if config.kind == "l1" and not config.trajectory_weight and config.supervised_prefix_steps is None:
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
    if config.trajectory_weight:
        if action_representation not in ("cartesian", "cartesian_tool") or action_mean is None or action_std is None:
            raise ValueError("trajectory loss requires delta actions and training-only normalization")
        dtype = torch.promote_types(predicted.dtype, torch.float32)
        with torch.autocast(device_type=predicted.device.type, enabled=False):
            mean = torch.as_tensor(action_mean, device=predicted.device, dtype=dtype)
            std = torch.as_tensor(action_std, device=predicted.device, dtype=dtype)
            trajectory_pred, trajectory_target, trajectory_pad = _supervision_slice(
                predicted, batch[ACTION], batch["action_is_pad"], config,
            )
            trajectory = integrated_tool_loss(
                trajectory_pred.to(dtype) * std + mean, trajectory_target.to(dtype) * std + mean, trajectory_pad,
                tool_frame=action_representation == "cartesian_tool",
                translation_scale=config.translation_scale_m, rotation_scale=config.rotation_scale_rad,
            )
        loss = loss + config.trajectory_weight * trajectory
        metrics["trajectory_loss"] = trajectory.item()
    if policy.config.use_vae and log_variance is not None:
        # Same latent reduction and beta as LeRobot 0.6.1 ACTPolicy.forward.
        kld = (-0.5 * (1 + log_variance - mu.pow(2) - log_variance.exp())).sum(-1).mean()
        loss = loss + policy.config.kl_weight * kld
        metrics["kld_loss"] = kld.item()
    return loss, metrics
