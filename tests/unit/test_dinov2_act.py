"""Offline tiny-model tests for DINO gradients, spatial tokens and portable checkpoints."""

import json

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from real2sim.policy.act import load_checkpoint, save_checkpoint
from real2sim.policy.dinov2_act import DinoACTConfig, DinoACTPolicy, warmstart_nonvisual
from real2sim.policy.schedule import learning_rate


def configuration():
    return DinoACTConfig(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(13,)),
                        "observation.images.iphone": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 42))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        dim_model=32, n_heads=4, dim_feedforward=64, n_encoder_layers=1,
        n_decoder_layers=1, n_vae_encoder_layers=1, chunk_size=4, n_action_steps=2,
        dinov2_config={"hidden_size": 24, "num_hidden_layers": 3, "num_attention_heads": 3,
                       "image_size": 28, "patch_size": 14, "mlp_ratio": 2},
        dinov2_trainable_blocks=2, device="cpu",
    )


def batch():
    return {"observation.state": torch.randn(2, 13),
            "observation.images.iphone": torch.randn(2, 3, 28, 42),
            "action": torch.randn(2, 4, 7), "action_is_pad": torch.zeros(2, 4, dtype=torch.bool)}


def test_spatial_features_and_only_selected_layers_update():
    p = DinoACTPolicy(configuration())
    bb = p.model.backbone
    assert bb(torch.randn(2, 3, 29, 43))["feature_map"].shape == (2, 24, 3, 4)
    before = {n: v.detach().clone() for n, v in bb.named_parameters()}
    optimizer = torch.optim.AdamW(p.get_optim_params(), lr=1e-3)
    loss, _ = p(batch())
    loss.backward()
    assert torch.isfinite(loss)
    for name, value in bb.named_parameters():
        allowed = name.startswith(("dino.encoder.layer.1.", "dino.encoder.layer.2.", "dino.layernorm."))
        assert value.requires_grad == allowed
        assert (value.grad is not None) == allowed
        if allowed:
            assert torch.isfinite(value.grad).all()
    optimizer.step()
    assert torch.equal(before["dino.embeddings.position_embeddings"], bb.dino.embeddings.position_embeddings)
    assert not torch.equal(before["dino.layernorm.weight"], bb.dino.layernorm.weight)
    p.train()
    assert not bb.dino.encoder.layer[0].training and bb.dino.encoder.layer[1].training


def test_warmstart_excludes_vision_and_preserves_every_other_tensor():
    config = configuration()
    kwargs = {k: getattr(config, k) for k in ACTConfig.__dataclass_fields__ if k in vars(config)}
    kwargs.update(vision_backbone="resnet18", pretrained_backbone_weights=None)
    parent = ACTPolicy(ACTConfig(**kwargs))
    policy = DinoACTPolicy(config)
    vision_before = policy.model.encoder_img_feat_input_proj.weight.detach().clone()
    result = warmstart_nonvisual(policy, parent)
    assert result["copied_tensors"] > 0
    for name, value in policy.state_dict().items():
        if not name.startswith(tuple(result["excluded_prefixes"])):
            assert torch.equal(value, parent.state_dict()[name])
    assert torch.equal(vision_before, policy.model.encoder_img_feat_input_proj.weight)


def test_checkpoint_roundtrip_needs_no_hf_download(tmp_path, monkeypatch):
    p = DinoACTPolicy(configuration()).eval()
    x = batch()
    expected = p.predict_action_chunk(x)
    stats = {"observation.state": {"mean": [0] * 13, "std": [1] * 13},
             "action": {"mean": [0] * 7, "std": [1] * 7},
             "image_normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]}}
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats))
    manifest = {"camera_keys": ["observation.images.iphone"], "image_shape_hwc": [28, 42, 3],
                "letterbox": {}, "state_names": [str(i) for i in range(13)],
                "action_names": [str(i) for i in range(7)], "label_semantics": "test",
                "policy_hz": 10, "source": {}, "split": {}, "files_sha256": {}, "prepare_version": "test"}
    save_checkpoint(tmp_path / "checkpoint", p, stats_path, manifest, {"step": 0})
    def forbid(*args, **kwargs):
        raise AssertionError("checkpoint must not fetch pretrained DINO weights")
    monkeypatch.setattr("real2sim.policy.dinov2_act.Dinov2Model.from_pretrained", forbid)
    loaded, _, contract = load_checkpoint(tmp_path / "checkpoint", "cpu")
    assert isinstance(loaded, DinoACTPolicy)
    assert contract["vision_encoder"]["trainable_final_blocks"] == 2
    torch.testing.assert_close(loaded.predict_action_chunk(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("peak", [1e-4, 1e-5])
def test_schedule_boundaries_monotonicity_and_absolute_step_resume(peak):
    rates = [learning_rate(s, 2000, peak, warmup_steps=100, start_factor=.001, min_lr=1e-7)
             for s in range(1, 2001)]
    assert rates[0] == pytest.approx(peak * .001)
    assert rates[99] == pytest.approx(peak)
    assert rates[-1] == pytest.approx(1e-7)
    assert all(a <= b for a, b in zip(rates[:99], rates[1:100], strict=True))
    assert all(a >= b for a, b in zip(rates[99:-1], rates[100:], strict=True))
    assert learning_rate(251, 2000, peak, warmup_steps=100, start_factor=.001, min_lr=1e-7) == rates[250]


def test_invalid_unfreeze_is_rejected():
    cfg = configuration()
    cfg.dinov2_trainable_blocks = 4
    with pytest.raises(ValueError, match="unfrozen"):
        cfg.__post_init__()
