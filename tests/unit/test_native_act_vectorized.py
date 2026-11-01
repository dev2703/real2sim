"""Vectorized frozen inference preserves ACT actions and caches only fixed positions."""

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from real2sim.policy.native_lerobot_act import vectorized_act_actions


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("with_environment", [False, True])
def test_frozen_vectorized_forward_preserves_all_actions_and_camera_order(
        batch_size, with_environment):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.iphone": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
    }
    if with_environment:
        features["observation.environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(3,))
    config = ACTConfig(input_features=features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=50, n_action_steps=10, dim_model=32, n_heads=4, dim_feedforward=64,
        n_encoder_layers=1, n_decoder_layers=1, pretrained_backbone_weights=None,
        dropout=0., device="cpu")
    policy = ACTPolicy(config).eval().requires_grad_(False)
    cameras = list(config.image_features)
    cache = {}
    previous = None
    with torch.inference_mode():
        for _ in range(2):
            batch = {"observation.state": torch.randn(batch_size, 6),
                     **{key: torch.randn(batch_size, 3, 32, 32) for key in cameras}}
            if with_environment:
                batch["observation.environment_state"] = torch.randn(batch_size, 3)
            reference = policy.predict_action_chunk(batch)
            actual = vectorized_act_actions(policy.model, batch, cameras, cache)
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            assert actual.shape == (batch_size, 50, 6)
            assert len(cache) == 1  # Shared shape, not cached observations/features.
            if previous is not None:
                assert not torch.equal(actual, previous)
            previous = actual
    policy.train()
    with pytest.raises(ValueError, match="frozen evaluation"):
        vectorized_act_actions(policy.model, batch, cameras, cache)
