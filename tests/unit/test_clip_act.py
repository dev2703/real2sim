"""Offline CLIP spatial features, fine-tuning boundaries and portable two-view reload."""

import json

import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from real2sim.policy.act import load_checkpoint, save_checkpoint
from real2sim.policy.clip_act import ClipACTConfig, ClipACTPolicy
from real2sim.policy.flow_matching import FlowHeadConfig, FlowHeadPolicy, build_frozen_backbone

VISION = dict(hidden_size=24, num_hidden_layers=3, num_attention_heads=3,
              intermediate_size=48, image_size=32, patch_size=16, hidden_act="quick_gelu")
CAMERAS = ["observation.images.iphone", "observation.images.top"]


def configuration():
    return ClipACTConfig(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(13,)),
                        **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 48)) for k in CAMERAS}},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        dim_model=32, n_heads=4, dim_feedforward=64, n_encoder_layers=1,
        n_decoder_layers=1, n_vae_encoder_layers=1, chunk_size=4, n_action_steps=2,
        clip_config=VISION, clip_trainable_blocks=2, device="cpu",
    )


def batch():
    return {"observation.state": torch.randn(2, 13),
            **{k: torch.randn(2, 3, 32, 48) for k in CAMERAS},
            "action": torch.randn(2, 4, 7), "action_is_pad": torch.zeros(2, 4, dtype=torch.bool)}


def test_clip_full_view_normalization_and_finetune_gradients():
    policy = ClipACTPolicy(configuration())
    bb = policy.model.backbone
    assert bb(torch.randn(2, 3, 33, 49))["feature_map"].shape == (2, 24, 3, 4)
    rgb = torch.rand(2, 3, 32, 48)
    x = (rgb - bb.imagenet_mean) / bb.imagenet_std
    torch.testing.assert_close(bb.preprocess(x), (rgb - bb.clip_mean) / bb.clip_std)
    loss, _ = policy(batch())
    loss.backward()
    for name, value in bb.named_parameters():
        allowed = any(k in name for k in ("encoder.layers.1.", "encoder.layers.2.", "post_layernorm."))
        assert value.requires_grad == allowed
        assert (value.grad is not None) == allowed
        if allowed:
            assert torch.isfinite(value.grad).all()
    policy.train()
    assert not bb.vision.encoder.layers[0].training
    assert bb.vision.encoder.layers[1].training


def test_clip_both_camera_checkpoint_reload_is_offline(tmp_path, monkeypatch):
    policy = ClipACTPolicy(configuration()).eval()
    x = batch()
    expected = policy.predict_action_chunk(x)
    stats = {"observation.state": {"mean": [0] * 13, "std": [1] * 13},
             "action": {"mean": [0] * 7, "std": [1] * 7},
             "image_normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]}}
    sp = tmp_path / "stats.json"
    sp.write_text(json.dumps(stats))
    manifest = dict(camera_keys=CAMERAS, image_shape_hwc=[32, 48, 3], letterbox={},
                    state_names=list(map(str, range(13))), action_names=list(map(str, range(7))),
                    label_semantics="test", policy_hz=12.5, source={}, split={}, files_sha256={},
                    prepare_version="test")
    save_checkpoint(tmp_path / "checkpoint", policy, sp, manifest, {"step": 0})

    def forbidden(*args, **kwargs):
        raise AssertionError("reload must not request external vision weights")

    monkeypatch.setattr("real2sim.policy.clip_act.CLIPVisionModel.from_pretrained", forbidden)
    loaded, _, contract = load_checkpoint(tmp_path / "checkpoint")
    assert contract["camera_keys"] == CAMERAS
    assert contract["vision_encoder"]["trainable_final_blocks"] == 2
    torch.testing.assert_close(loaded.predict_action_chunk(x), expected, rtol=0, atol=0)


def test_clip_flow_and_regression_share_frozen_features_and_parameter_schema():
    c = FlowHeadConfig(vision_backbone="clip", clip_config=VISION, feature_dim=24,
                       feature_tokens=12, camera_keys=CAMERAS, chunk_size=4,
                       dim_model=32, n_heads=4, n_layers=1, dim_feedforward=64)
    p = FlowHeadPolicy(c, backbone=build_frozen_backbone(c))
    x = batch()
    features = p.features(x)
    assert features.shape == (2, 12, 24)
    before = set(p.state_dict())
    for objective in ("flow", "regression"):
        p.config.objective = objective
        loss, _ = p(x)
        loss.backward()
        assert torch.isfinite(loss)
        assert set(p.state_dict()) == before
        assert all(v.grad is None for v in p.backbone.parameters())
        assert p.predict_action_chunk(x).shape == (2, 4, 7)
