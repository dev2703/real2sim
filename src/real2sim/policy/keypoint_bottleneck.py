"""Experimental image-keypoint bottleneck with shared cached/raw inference.

The supervised arm uses visibility-masked image-derived pseudo centers, never
simulator object positions. Action inference receives predicted centers only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from real2sim.policy.dataset import Normalizer
from real2sim.policy.flow_matching import FEATURE_KEY, FlowHeadConfig, build_frozen_backbone


@dataclass
class KeypointBottleneckConfig:
    feature_dim: int = 384
    rows: int = 14
    cols: int = 19
    patch_size: int = 14
    image_height: int = 192
    image_width: int = 256
    state_dim: int = 13
    action_dim: int = 7
    chunk_size: int = 2
    n_action_steps: int = 2
    object_names: list[str] = field(default_factory=lambda: ["ball"])
    camera_key: str = "observation.images.iphone"
    dinov2_config: dict = field(default_factory=dict)
    dinov2_model_id: str = "facebook/dinov2-small"
    dinov2_revision: str = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    type: str = "keypoint_bottleneck_v1"

    def __post_init__(self):
        if (self.state_dim != 13 or self.action_dim != 7 or self.chunk_size != 2
                or self.n_action_steps != 2 or self.object_names != ["ball"]
                or self.type != "keypoint_bottleneck_v1"):
            raise ValueError("this bounded experiment requires canonical13, Cartesian7, two actions and one ball")
        if min(self.rows, self.cols, self.patch_size, self.feature_dim) < 1:
            raise ValueError("positive image feature grid required")


class KeypointBottleneckPolicy(nn.Module):
    def __init__(self, config: KeypointBottleneckConfig, backbone=None):
        super().__init__()
        self.config, self.backbone = config, backbone
        self.localization = nn.Sequential(nn.Linear(config.feature_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.visibility = nn.Sequential(nn.Linear(config.feature_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.actions = nn.Sequential(nn.Linear(config.state_dim + 3, 128), nn.GELU(),
                                     nn.Linear(128, 128), nn.GELU(),
                                     nn.Linear(128, config.chunk_size * config.action_dim))
        yy, xx = torch.meshgrid(torch.arange(config.rows), torch.arange(config.cols), indexing="ij")
        coordinates = torch.stack([((xx + .5) * config.patch_size / config.image_width).clamp(max=1),
                                   ((yy + .5) * config.patch_size / config.image_height).clamp(max=1)], -1)
        self.register_buffer("coordinates", coordinates.reshape(-1, 2).float())
        if backbone is not None:
            backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def features(self, batch):
        if FEATURE_KEY in batch:
            return batch[FEATURE_KEY]
        if self.backbone is None:
            raise ValueError("raw-image inference requires the saved frozen backbone")
        with torch.no_grad():
            return self.backbone(batch[self.config.camera_key])["feature_map"].flatten(2).transpose(1, 2)

    def predictions(self, batch):
        features = self.features(batch)
        c = self.config
        if features.shape[1:] != (c.rows * c.cols, c.feature_dim):
            raise ValueError("keypoint feature grid differs from checkpoint")
        logits = self.localization(features).squeeze(-1)
        uv = logits.softmax(-1) @ self.coordinates
        visibility_logits = self.visibility(features.mean(1)).squeeze(-1)
        probability = visibility_logits.sigmoid()
        # Predicted features are used identically during action training/inference.
        grounded = torch.cat([uv * probability[:, None], probability[:, None]], -1)
        actions = self.actions(torch.cat([batch["observation.state"], grounded], -1))
        return actions.reshape(-1, c.chunk_size, c.action_dim), uv, visibility_logits, logits

    def forward(self, batch, *, auxiliary_weight=0.):
        action, uv, visible_logits, logits = self.predictions(batch)
        valid = ~batch["action_is_pad"]
        action_loss = ((action - batch["action"]).abs() * valid[..., None]).sum() / (valid.sum().clamp(min=1) * self.config.action_dim)
        target = batch["pseudo_ball"]
        visible = target[:, 3] > .5
        # Missing detector rows supply no center supervision. Availability is a
        # pseudo detector event, not measured object visibility or occlusion.
        visibility_loss = F.binary_cross_entropy_with_logits(visible_logits, visible.float())
        if visible.any():
            scale = uv.new_tensor([self.config.image_width / self.config.patch_size,
                                   self.config.image_height / self.config.patch_size])
            distances = ((self.coordinates[None] - target[visible, None, :2]) * scale).square().sum(-1)
            heatmap = (-.5 * distances).softmax(-1)
            localization_loss = -(heatmap * logits[visible].log_softmax(-1)).sum(-1).mean() / torch.log(logits.new_tensor(logits.shape[-1]))
            center_loss = F.smooth_l1_loss(uv[visible], target[visible, :2]) * 10
        else:
            localization_loss = center_loss = logits.sum() * 0
        auxiliary = localization_loss + center_loss + visibility_loss
        return action_loss + auxiliary_weight * auxiliary, {
            "action_l1": float(action_loss.detach()), "auxiliary": float(auxiliary.detach()),
            "visible_pseudo_targets": int(visible.sum()),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return self.predictions(batch)[0]


def save_keypoint_checkpoint(path: Path, policy, normalizer, contract, info, backbone_state):
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=False)
    weights = {k: v.detach().cpu().contiguous() for k, v in policy.state_dict().items()}
    if policy.backbone is None:
        weights.update({f"backbone.{k}": v.detach().cpu().contiguous() for k, v in backbone_state.items()})
    save_file(weights, path / "model.safetensors")
    for name, value in (("config.json", asdict(policy.config)), ("stats.json", normalizer.stats),
                        ("real2sim_contract.json", contract), ("run_info.json", info)):
        (path / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_keypoint_checkpoint(path: Path, device="cpu"):
    from safetensors.torch import load_file

    config = KeypointBottleneckConfig(**json.loads((path / "config.json").read_text()))
    backbone = build_frozen_backbone(FlowHeadConfig(dinov2_config=config.dinov2_config))
    policy = KeypointBottleneckPolicy(config, backbone)
    policy.load_state_dict(load_file(path / "model.safetensors"), strict=True)
    return policy.to(device).eval(), Normalizer.from_json(path / "stats.json"), json.loads((path / "real2sim_contract.json").read_text())
