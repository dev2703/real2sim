"""Phase is a supervised prediction, never an oracle inference input."""

import inspect
import json

import numpy as np
import pytest
import torch

from real2sim.artifacts import sha256_file
from real2sim.policy.common_recovery import ACTION_NAMES, OBSERVATION_NAMES
from real2sim.policy.phase_recovery_mlp import (
    N_CONTROL_PHASES,
    PhaseRecoveryMLP,
    PhaseRecoveryMLPPolicy,
    phase_training_loss,
)
from scripts.train_phase_recovery_mlp import (
    corrupt_action_history,
    drop_approach_timers,
    load_phase_collections,
    train,
)


def model():
    return PhaseRecoveryMLP(np.zeros(6), np.ones(6), np.zeros(4), np.ones(4), hidden_sizes=(16, 8))


def test_untrained_phase_mixture_preserves_act_prior_and_masks_terminals():
    network = model()
    observations, actions, features = torch.randn(5, 6), torch.randn(5, 7), torch.randn(5, 4)
    prediction, logits, phase_actions = network(observations, actions, features)
    torch.testing.assert_close(prediction, actions)
    assert phase_actions.shape == (5, 11, 7)
    assert logits.argmax(-1).max() < N_CONTROL_PHASES
    assert torch.softmax(logits, dim=-1)[:, N_CONTROL_PHASES:].count_nonzero() == 0
    assert list(inspect.signature(network.forward).parameters) == ["observations", "base_actions", "features"]
    assert list(inspect.signature(PhaseRecoveryMLPPolicy.predict).parameters) == [
        "self", "observations", "base_actions", "features"]
    with pytest.raises(TypeError):
        network(observations, actions, features, expert_phase_ids=torch.zeros(5, dtype=torch.long))


def test_training_shapes_both_phase_prediction_and_specialized_action_components():
    torch.manual_seed(7)
    network = model()
    # Make phase-conditioned action predictions distinct so action loss itself
    # must propagate through predicted phase probabilities, not an oracle route.
    torch.nn.init.normal_(network.action_heads.weight, std=.1)
    observations, actions, features = torch.randn(5, 6), torch.randn(5, 7), torch.randn(5, 4)
    targets = torch.zeros(5, 7)
    phases = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    loss, terms = phase_training_loss(network, observations, actions, features, targets,
                                     phases, phase_weight=0.)
    loss.backward()
    assert network.phase_head.weight.grad.abs().sum() > 0
    assert network.action_heads.weight.grad.abs().sum() > 0
    assert network.trunk[0].weight.grad.abs().sum() > 0
    assert terms["phase_classification_loss"] > 0
    with pytest.raises(ValueError, match="nonterminal"):
        phase_training_loss(network, observations, actions, features, targets,
                            torch.full((5,), 9, dtype=torch.long))


def test_no_act_phase_head_cannot_use_action_or_decoder_inputs():
    network = PhaseRecoveryMLP(np.zeros(6), np.ones(6), np.zeros(4), np.ones(4),
                               hidden_sizes=(16, 8), use_act=False)
    torch.nn.init.normal_(network.action_heads.weight, std=.1)
    observations, actions, features = torch.randn(5, 6), torch.randn(5, 7), torch.randn(5, 4)
    first = network(observations, actions, features)
    second = network(observations, actions * 100 + 7, features * -100 - 3)
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_direct_act_head_keeps_frozen_inputs_without_forcing_their_action_prior():
    torch.manual_seed(72)
    direct = PhaseRecoveryMLP(np.zeros(6), np.ones(6), np.zeros(4), np.ones(4),
        hidden_sizes=(16, 8), use_act=True, residual_prior=False)
    residual = model()
    residual.load_state_dict(direct.state_dict())
    inputs = torch.randn(3, 6), torch.ones(3, 7) * .01, torch.randn(3, 4)
    direct_action = direct(*inputs)[0]
    torch.testing.assert_close(direct_action, torch.zeros_like(direct_action))
    torch.testing.assert_close(residual(*inputs)[0], inputs[1])
    torch.nn.init.normal_(direct.action_heads.weight, std=.1)
    first = direct(*inputs)[0]
    second = direct(inputs[0], inputs[1] * 2, inputs[2] + 1.)[0]
    assert not torch.allclose(first, second)


def test_approach_timer_dropout_keeps_geometry_and_close_release_dwell():
    observations = torch.arange(4 * 99).reshape(4, 99).float()
    original = observations.clone()
    mean = torch.arange(99).float() / 10.
    phases = torch.tensor([1, 2, 3, 7])
    altered = drop_approach_timers(observations, OBSERVATION_NAMES, phases, mean, probability=1.)
    columns = [32, 65, 98]
    torch.testing.assert_close(altered[:2, columns], mean[columns].expand(2, -1))
    torch.testing.assert_close(altered[2:], original[2:])
    remaining = [i for i in range(99) if i not in columns]
    torch.testing.assert_close(altered[:, remaining], original[:, remaining])
    torch.testing.assert_close(observations, original)
    with pytest.raises(ValueError, match="dropout"):
        drop_approach_timers(observations, OBSERVATION_NAMES, phases, mean, probability=1.1)


def test_hard_routing_executes_one_predicted_component_without_losing_phase_gradient():
    soft, hard = model(), model()
    for network in (soft, hard):
        network.use_act = False
        torch.nn.init.zeros_(network.phase_head.weight)
        with torch.no_grad():
            network.phase_head.bias.fill_(-20.)
            network.phase_head.bias[1:3] = torch.tensor([2., 1.9])
            network.action_heads.bias.reshape(11, 7)[2, 6] = 1.
    hard.load_state_dict(soft.state_dict())
    hard.action_routing = "hard"
    inputs = torch.randn(3, 6), torch.zeros(3, 7), torch.randn(3, 4)
    soft_action, _, _ = soft(*inputs)
    hard_action, logits, components = hard(*inputs)
    assert torch.all(soft_action[:, 6] > .4)
    torch.testing.assert_close(hard_action, components[:, 1], atol=1e-7, rtol=0.)
    assert logits.argmax(-1).tolist() == [1, 1, 1]
    targets = torch.zeros(3, 7)
    targets[:, 6] = 1.
    loss, _ = phase_training_loss(hard, *inputs, targets, torch.ones(3, dtype=torch.long),
                                 phase_weight=0.)
    loss.backward()
    assert hard.phase_head.bias.grad.abs().sum() > 0
    assert torch.isfinite(hard.phase_head.bias.grad).all()


@pytest.mark.parametrize("history", ["selected", "achieved"])
def test_history_corruption_preserves_pose_rgb_and_declared_action_bounds(history):
    names = list(OBSERVATION_NAMES)
    if history == "achieved":
        names = [name.replace("_previous_", "_previous_achieved_")
                 if any(name.endswith("_previous_" + action) for action in ACTION_NAMES[:6])
                 else name.replace("_previous_gripper_open_fraction", "_previous_issued_gripper_open_fraction")
                 for name in names]
    observations = torch.zeros(128, len(names))
    original = observations.clone()
    generator = torch.Generator().manual_seed(23)
    corrupted = corrupt_action_history(observations, names, noise=2., dropout=.5,
                                       generator=generator)
    torch.testing.assert_close(observations, original, atol=0, rtol=0)
    history_columns = [index for index, name in enumerate(names) if "_previous_" in name]
    remaining = [index for index in range(len(names)) if index not in history_columns]
    torch.testing.assert_close(corrupted[:, remaining], original[:, remaining], atol=0, rtol=0)
    for start in range(0, 21, 7):
        values = corrupted[:, history_columns[start:start + 7]]
        assert torch.all(values[:, :3].norm(dim=-1) <= .03000001)
        assert torch.all(values[:, 3:6].norm(dim=-1) <= .3000001)
        assert torch.all((values[:, 6] >= 0) & (values[:, 6] <= 1))
        assert 0 < torch.all(values == 0, dim=-1).sum() < 128
    repeated = corrupt_action_history(observations, names, noise=2., dropout=.5,
        generator=torch.Generator().manual_seed(23))
    torch.testing.assert_close(corrupted, repeated, atol=0, rtol=0)


def test_history_corruption_requires_complete_named_frames():
    observations = torch.zeros(4, 6)
    with pytest.raises(ValueError, match="declared observation"):
        corrupt_action_history(observations, None, noise=.1)
    with pytest.raises(ValueError, match="complete"):
        corrupt_action_history(observations, ["history_0_previous_tcp_dx_m"] * 6, noise=.1)
    with pytest.raises(ValueError, match="dropout"):
        corrupt_action_history(observations, None, dropout=1.)


def collection(tmp_path, *, audit=True, terminal=False):
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"immutable actual frozen ACT")
    base_hash = sha256_file(checkpoint / "model.safetensors")
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "manifest.json").write_text(json.dumps({
        "status": "complete", "base_checkpoint_sha256": base_hash,
        "lighting_changed": False, "base_weights_frozen": True, "runtime_source_changes": [],
        "expert_phase_is_audit_label_only": audit, "iteration": 2}))
    rng = np.random.default_rng(6)
    for partition, episode in (("train", 10000), ("val", 11000)):
        observations = rng.normal(size=(12, 6)).astype(np.float32)
        if partition == "val":
            observations += 7
        data = {"observations": observations,
                "base_actions": np.zeros((12, 7), np.float32),
                "target_actions": rng.normal(size=(12, 7)).astype(np.float32) * .001,
                "features": rng.normal(size=(12, 4)).astype(np.float32),
                "robot_id": np.arange(12) % 2, "episode_index": np.full(12, episode)}
        if audit:
            data["expert_phase_id"] = np.full(12, 9) if terminal else np.tile([1, 2, 3], 4)
        np.savez(pack / f"{partition}.npz", **data)
    return checkpoint, base_hash, pack


def test_missing_actual_phase_audits_cannot_be_replaced_with_invented_labels(tmp_path):
    _, base_hash, pack = collection(tmp_path, audit=False)
    with pytest.raises(ValueError, match="actual audit"):
        load_phase_collections([pack], expected_base_sha256=base_hash)


def test_terminal_expert_labels_are_rejected_from_the_action_model(tmp_path):
    _, base_hash, pack = collection(tmp_path, terminal=True)
    with pytest.raises(ValueError, match="nonterminal"):
        load_phase_collections([pack], expected_base_sha256=base_hash)


def test_audit_phase_cannot_be_copied_into_the_observation_contract(tmp_path):
    _, base_hash, pack = collection(tmp_path)
    path = pack / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["observation_names"] = ["expert_phase_id", *[f"observable_{i}" for i in range(5)]]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cannot be a recovery observation"):
        load_phase_collections([pack], expected_base_sha256=base_hash)


@pytest.mark.parametrize("use_act", [True, False])
@pytest.mark.parametrize("action_routing", ["soft", "hard"])
@pytest.mark.parametrize("residual_prior", [True, False])
def test_saved_phase_head_uses_training_only_stats_and_frozen_provenance(tmp_path, use_act, action_routing,
                                                                      residual_prior):
    checkpoint, base_hash, pack = collection(tmp_path)
    before = sha256_file(checkpoint / "model.safetensors")
    report = train([pack], tmp_path / "head", snapshot=checkpoint, steps=2, use_act=use_act,
                   action_routing=action_routing, residual_prior=residual_prior)
    assert report["status"] == "complete" and report["runtime_source_changes"] == []
    assert report["expert_phase_is_runtime_input"] is False
    policy = PhaseRecoveryMLPPolicy(tmp_path / "head", expected_base_sha256=base_hash)
    assert policy.model.use_act is use_act
    assert policy.model.action_routing == action_routing
    assert policy.model.residual_prior is residual_prior
    with np.load(pack / "train.npz") as rows:
        np.testing.assert_allclose(policy.model.observation_mean.numpy(), rows["observations"].mean(0))
        output = policy.predict(rows["observations"][0], rows["base_actions"][0], rows["features"][0])
        assert output.shape == (7,) and np.isfinite(output).all()
        assert 0 <= policy.last_phase_id < N_CONTROL_PHASES
        assert np.sum(policy.last_phase_probabilities[N_CONTROL_PHASES:]) == 0
    assert sha256_file(checkpoint / "model.safetensors") == before
    with pytest.raises(ValueError, match="different frozen ACT"):
        PhaseRecoveryMLPPolicy(tmp_path / "head", expected_base_sha256="new ACT")
    with (tmp_path / "head/head.pt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        PhaseRecoveryMLPPolicy(tmp_path / "head", expected_base_sha256=base_hash)
