"""Shared cached/image inference and masked pseudo-label semantics."""

import numpy as np
import pytest
import torch
from torch import nn

from real2sim.policy.flow_matching import FEATURE_KEY
from real2sim.policy.keypoint_bottleneck import (
    KeypointBottleneckConfig,
    KeypointBottleneckPolicy,
    load_keypoint_checkpoint,
    save_keypoint_checkpoint,
)


def configuration():
    return KeypointBottleneckConfig(feature_dim=4, rows=2, cols=3, patch_size=14,
                                    image_height=28, image_width=42)


class FixedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("features", torch.linspace(-1.1, 1.2, 24).reshape(1, 4, 2, 3))

    def forward(self, image):
        return {"feature_map": self.features.expand(len(image), -1, -1, -1)}


def test_cached_and_raw_paths_share_precision_and_predicted_bottleneck():
    torch.manual_seed(1)
    backbone = FixedBackbone()
    policy = KeypointBottleneckPolicy(configuration(), backbone).eval()
    state = torch.randn(2, 13)
    raw = {"observation.state": state, "observation.images.iphone": torch.zeros(2, 3, 28, 42)}
    features = backbone.features.expand(2, -1, -1, -1).flatten(2).transpose(1, 2).half()
    cached = {"observation.state": state, FEATURE_KEY: features}
    torch.testing.assert_close(policy.predict_action_chunk(raw), policy.predict_action_chunk(cached), atol=0, rtol=0)
    prediction, uv, visibility, _ = policy.predictions(cached)
    torch.testing.assert_close(prediction, policy.actions_from_keypoints(state, uv, visibility.sigmoid()))
    policy.reset()


def test_missing_pseudo_center_coordinates_cannot_supervise_localization():
    torch.manual_seed(2)
    policy = KeypointBottleneckPolicy(configuration())
    batch = {"observation.state": torch.zeros(2, 13), FEATURE_KEY: torch.randn(2, 6, 4),
             "action": torch.zeros(2, 2, 7), "action_is_pad": torch.tensor([[False, True], [False, False]]),
             "pseudo_ball": torch.zeros(2, 4)}
    original = policy(batch, auxiliary_weight=1)[0]
    batch["pseudo_ball"][:, :3] = 999
    changed = policy(batch, auxiliary_weight=1)[0]
    torch.testing.assert_close(original, changed, atol=0, rtol=0)
    changed.backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_only_nonpadding_actions_contribute_and_pseudo_centers_not_action_inputs():
    torch.manual_seed(3)
    policy = KeypointBottleneckPolicy(configuration())
    batch = {"observation.state": torch.zeros(1, 13), FEATURE_KEY: torch.randn(1, 6, 4),
             "action": torch.zeros(1, 2, 7), "action_is_pad": torch.tensor([[False, True]]),
             "pseudo_ball": torch.tensor([[.3, .4, .05, 1.]])}
    action = policy.predict_action_chunk(batch)
    original = policy(batch, auxiliary_weight=0)[0]
    batch["action"][:, 1] = 999
    batch["pseudo_ball"][:] = 0
    torch.testing.assert_close(policy(batch, auxiliary_weight=0)[0], original, atol=0, rtol=0)
    torch.testing.assert_close(policy.predict_action_chunk(batch), action, atol=0, rtol=0)


def test_feature_grid_contract_rejects_missing_or_incompatible_inputs():
    policy = KeypointBottleneckPolicy(configuration())
    with pytest.raises(ValueError, match="raw-image"):
        policy.predict_action_chunk({"observation.state": torch.zeros(1, 13),
                                      "observation.images.iphone": torch.zeros(1, 3, 28, 42)})
    with pytest.raises(ValueError, match="feature grid"):
        policy.predict_action_chunk({"observation.state": torch.zeros(1, 13), FEATURE_KEY: torch.zeros(1, 7, 4)})
    with pytest.raises(ValueError, match="feature grid"):
        features = torch.zeros(1, 6, 4)
        features[0, 0, 0] = np.nan
        policy.predict_action_chunk({"observation.state": torch.zeros(1, 13), FEATURE_KEY: features})


def test_portable_checkpoint_reloads_backbone_and_raw_image_contract(tmp_path):
    from transformers import Dinov2Config

    from real2sim.policy.dataset import Normalizer
    from real2sim.policy.flow_matching import FlowHeadConfig, build_frozen_backbone

    torch.manual_seed(4)
    architecture = Dinov2Config(hidden_size=12, num_hidden_layers=1,
                                num_attention_heads=3, image_size=28, patch_size=14, mlp_ratio=2).to_dict()
    cfg = KeypointBottleneckConfig(feature_dim=12, rows=2, cols=3, patch_size=14,
                                   image_height=28, image_width=42, dinov2_config=architecture)
    backbone = build_frozen_backbone(FlowHeadConfig(dinov2_config=architecture))
    policy = KeypointBottleneckPolicy(cfg, backbone).eval()
    norm = Normalizer({"observation.state": {"mean": [0] * 13, "std": [1] * 13},
                       "action": {"mean": [0] * 7, "std": [1] * 7},
                       "image_normalization": {"mean": [0] * 3, "std": [1] * 3}})
    contract = {"keypoint_bottleneck": {"config": "test"}, "camera_keys": [cfg.camera_key]}
    checkpoint = tmp_path / "checkpoint"
    save_keypoint_checkpoint(checkpoint, policy, norm, contract, {"step": 1}, backbone.state_dict())
    loaded, reloaded_norm, reloaded_contract = load_keypoint_checkpoint(checkpoint)
    batch = {"observation.state": torch.randn(1, 13), cfg.camera_key: torch.randn(1, 3, 28, 42)}
    torch.testing.assert_close(loaded.predict_action_chunk(batch), policy.predict_action_chunk(batch), atol=0, rtol=0)
    assert reloaded_contract == contract and reloaded_norm.stats == norm.stats
