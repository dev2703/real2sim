"""Phase is a supervised prediction, never an oracle inference input."""

import inspect
import json

import numpy as np
import pytest
import torch

from real2sim.artifacts import sha256_file
from real2sim.policy.phase_recovery_mlp import (
    N_CONTROL_PHASES,
    PhaseRecoveryMLP,
    PhaseRecoveryMLPPolicy,
    phase_training_loss,
)
from scripts.train_phase_recovery_mlp import load_phase_collections, train


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


def test_saved_phase_head_uses_training_only_stats_and_frozen_provenance(tmp_path):
    checkpoint, base_hash, pack = collection(tmp_path)
    before = sha256_file(checkpoint / "model.safetensors")
    report = train([pack], tmp_path / "head", snapshot=checkpoint, steps=2)
    assert report["status"] == "complete" and report["runtime_source_changes"] == []
    assert report["expert_phase_is_runtime_input"] is False
    policy = PhaseRecoveryMLPPolicy(tmp_path / "head", expected_base_sha256=base_hash)
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
