"""Numerical checks for the small flow head, masking and deployable checkpoints."""

from dataclasses import replace

import pytest
import torch

from real2sim.policy.flow_matching import (
    FEATURE_KEY,
    ChunkHead,
    FlowHeadConfig,
    FlowHeadPolicy,
    build_frozen_backbone,
    load_flow_checkpoint,
    save_flow_checkpoint,
    seeded_noise,
)


def configuration(**changes):
    return FlowHeadConfig(
        state_dim=3, action_dim=2, chunk_size=4, n_action_steps=2, feature_dim=12,
        feature_tokens=6, dim_model=16, n_heads=4, n_layers=2, dim_feedforward=32,
        dinov2_config={"hidden_size": 12, "num_hidden_layers": 1, "num_attention_heads": 3,
                       "image_size": 28, "patch_size": 14, "mlp_ratio": 2}, **changes,
    )


def batch():
    return {"observation.state": torch.randn(2, 3), FEATURE_KEY: torch.randn(2, 6, 12),
            "action": torch.randn(2, 4, 2),
            "action_is_pad": torch.tensor([[False, False, True, True], [False] * 4])}


@pytest.mark.parametrize("change", [{"dim_model": 18}, {"n_heads": 0}, {"sampling_steps": 3},
                                     {"n_action_steps": 5}, {"objective": "unknown"}])
def test_config_rejects_incompatible_shapes_and_sampling(change):
    with pytest.raises(ValueError):
        replace(configuration(), **change)


def test_flow_and_regression_have_identical_initial_parameters():
    torch.manual_seed(20)
    flow = ChunkHead(configuration())
    torch.manual_seed(20)
    regression = ChunkHead(configuration(objective="regression"))
    assert flow.state_dict().keys() == regression.state_dict().keys()
    for key, value in flow.state_dict().items():
        torch.testing.assert_close(value, regression.state_dict()[key], rtol=0, atol=0)


def test_masked_actions_cannot_leak_into_flow_predictions_or_gradients():
    model = ChunkHead(configuration())
    data = batch()
    noise = torch.randn(2, 4, 2)
    time = torch.tensor([.25, .75])
    loss, info = model.training_loss(data, noise=noise, time=time)
    loss.backward()
    gradients = {k: p.grad.clone() for k, p in model.named_parameters()}
    assert info["valid_components"] == 12
    model.zero_grad(set_to_none=True)
    changed = {**data, "action": data["action"].clone()}
    changed["action"][data["action_is_pad"]] = float("nan")
    noisy = noise.clone()
    noisy[data["action_is_pad"]] = 1e9
    loss_changed, _ = model.training_loss(changed, noise=noisy, time=time)
    loss_changed.backward()
    torch.testing.assert_close(loss, loss_changed, atol=0, rtol=0)
    for key, p in model.named_parameters():
        torch.testing.assert_close(gradients[key], p.grad, atol=0, rtol=0)
    assert model.state_projection.weight.grad.abs().sum() > 0
    assert model.feature_projection.weight.grad.abs().sum() > 0


def test_all_padding_has_finite_differentiable_zero_loss():
    model = ChunkHead(configuration())
    data = batch()
    data["action_is_pad"][:] = True
    data["action"][:] = float("nan")
    loss, info = model.training_loss(data)
    loss.backward()
    assert loss.item() == 0 and info["valid_components"] == 0
    assert all(p.grad is not None and (p.grad == 0).all() for p in model.parameters())


def test_flow_loss_uses_correct_interpolation_and_vector_target(monkeypatch):
    model = ChunkHead(configuration())
    data = batch()
    noise, times = torch.randn(2, 4, 2), torch.tensor([.2, .8])
    pad = data["action_is_pad"][..., None]
    action, clean_noise = data["action"].masked_fill(pad, 0), noise.masked_fill(pad, 0)

    def correct_field(x, t, memory, action_is_pad):
        torch.testing.assert_close(x, clean_noise + times[:, None, None] * (action - clean_noise))
        torch.testing.assert_close(t, times)
        return action - clean_noise

    monkeypatch.setattr(model, "field", correct_field)
    loss, _ = model.training_loss(data, noise=noise, time=times)
    assert loss == 0


@pytest.mark.parametrize("steps", [4, 8])
def test_euler_integrates_in_correct_direction_and_reuses_condition(monkeypatch, steps):
    model = ChunkHead(configuration())
    data = batch()
    memory_ids, call_times = [], []

    def constant_field(x, time, memory, action_is_pad=None):
        memory_ids.append(id(memory))
        call_times.append(time[0].item())
        return torch.ones_like(x) * 2

    monkeypatch.setattr(model, "field", constant_field)
    prediction = model.sample(data["observation.state"], data[FEATURE_KEY], steps=steps, seed=44)
    initial = seeded_noise((2, 4, 2), device="cpu", dtype=torch.float32, seed=44)
    torch.testing.assert_close(prediction, initial + 2)
    assert len(set(memory_ids)) == 1 and call_times == [i / steps for i in range(steps)]


def test_per_anchor_seed_is_independent_of_batch_boundaries_and_reset_repeats():
    policy = FlowHeadPolicy(configuration()).eval()
    data = batch()
    full = policy.predict_action_chunk(data, sampling_seed=[91, 105])
    singleton = {k: v[1:] for k, v in data.items()}
    one = policy.predict_action_chunk(singleton, sampling_seed=[105])
    torch.testing.assert_close(full[1:], one, atol=2e-6, rtol=2e-6)
    policy.reset(27)
    first = policy.select_action(singleton)
    second = policy.select_action(singleton)
    assert policy._calls == 1
    policy.reset(27)
    repeat = policy.predict_action_chunk(singleton)
    torch.testing.assert_close(first, repeat[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(second, repeat[:, 1], rtol=0, atol=0)


def test_regression_is_seed_independent_and_can_fit_a_training_segment():
    torch.manual_seed(2)
    policy = FlowHeadPolicy(configuration(objective="regression"))
    data = batch()
    data["action"].fill_(.3)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.003)
    initial, _ = policy(data)
    for _ in range(25):
        optimizer.zero_grad()
        loss, _ = policy(data)
        loss.backward()
        optimizer.step()
    final, _ = policy(data)
    assert final < initial * .1
    torch.testing.assert_close(policy.predict_action_chunk(data, sampling_seed=1),
                               policy.predict_action_chunk(data, sampling_seed=33), rtol=0, atol=0)


def test_checkpoint_contains_vision_and_reloads_offline(tmp_path, monkeypatch):
    config = configuration()
    policy = FlowHeadPolicy(config, backbone=build_frozen_backbone(config)).eval()
    data = {"observation.state": torch.randn(1, 3),
            config.camera_key: torch.randn(1, 3, 28, 42)}
    expected = policy.predict_action_chunk(data, sampling_seed=55)
    stats = {"observation.state": {"mean": [0] * 3, "std": [1] * 3},
             "action": {"mean": [0] * 2, "std": [1] * 2},
             "image_normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]}}
    save_flow_checkpoint(tmp_path, policy, stats, {"kind": "test"}, {})

    def forbid(*args, **kwargs):
        raise AssertionError("must reload without downloading pretrained vision")

    monkeypatch.setattr("real2sim.policy.dinov2_act.Dinov2Model.from_pretrained", forbid)
    loaded, normalizer, contract = load_flow_checkpoint(tmp_path)
    torch.testing.assert_close(loaded.predict_action_chunk(data, sampling_seed=55), expected, rtol=0, atol=0)
    assert contract == {"kind": "test"} and len(normalizer.action_std) == 2
    loaded.train()
    assert not loaded.backbone.training and not any(p.requires_grad for p in loaded.backbone.parameters())
