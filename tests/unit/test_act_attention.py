"""SDPA must preserve ACT outputs, padded training gradients and checkpoint behavior."""

import copy

import pytest
import torch
from torch import nn

from real2sim.policy.attention import SDPAMultiheadAttention, enable_sdpa


@pytest.mark.parametrize("cross", [False, True])
def test_sdpa_outputs_and_gradients_with_masks(cross):
    torch.manual_seed(42)
    reference = nn.Sequential(nn.MultiheadAttention(24, 3, dropout=0)).double()
    optimized = copy.deepcopy(reference)
    rng = torch.get_rng_state().clone()
    assert enable_sdpa(optimized) == 1
    assert torch.equal(rng, torch.get_rng_state())
    assert isinstance(optimized[0], SDPAMultiheadAttention)
    q = torch.randn(4, 2, 24, dtype=torch.float64)
    k = torch.randn(6 if cross else 4, 2, 24, dtype=torch.float64)
    mask = torch.zeros(2, k.shape[0], dtype=torch.bool)
    mask[0, -2:] = True
    attn = torch.zeros(q.shape[0], k.shape[0], dtype=torch.bool)
    attn[0, 1] = True
    outs, grads = [], []
    for model in (reference[0], optimized[0]):
        query, key = q.clone().requires_grad_(), k.clone().requires_grad_()
        result, _ = model(query, key, key, key_padding_mask=mask, attn_mask=attn)
        result.square().sum().backward()
        outs.append(result)
        grads.append([query.grad, key.grad, *(p.grad for p in model.parameters())])
    torch.testing.assert_close(*outs, rtol=1e-10, atol=1e-10)
    for left, right in zip(*grads, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-10)
    assert enable_sdpa(optimized) == 0


def test_all_act_attention_and_roundtrip(tmp_path):
    from lerobot.configs.types import FeatureType, PolicyFeature

    from real2sim.policy.attention import ExperimentACTConfig, ExperimentACTPolicy

    config = ExperimentACTConfig(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(13,)),
                        "observation.images.iphone": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        pretrained_backbone_weights=None, dim_model=48, n_heads=12, dim_feedforward=64,
        chunk_size=4, n_action_steps=2, dropout=0, freeze_backbone=True, device="cpu",
    )
    policy = ExperimentACTPolicy(config).eval()
    assert sum(isinstance(m, SDPAMultiheadAttention) for m in policy.modules()) == 10
    obs = {"observation.state": torch.randn(2, 13),
           "observation.images.iphone": torch.randn(2, 3, 32, 32)}
    expected = policy.predict_action_chunk(obs)
    policy.save_pretrained(tmp_path)
    loaded = ExperimentACTPolicy.from_pretrained(str(tmp_path), local_files_only=True).eval()
    torch.testing.assert_close(expected, loaded.predict_action_chunk(obs), rtol=0, atol=0)
    loaded.train()
    assert not loaded.model.backbone.training
    assert not any(p.requires_grad for p in loaded.model.backbone.parameters())
