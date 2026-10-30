"""Temporal supervision cannot cross rollout boundaries or erase real phase changes."""

import json

import numpy as np
import pytest
import torch

from real2sim.policy.phase_sequences import (
    consecutive_pairs,
    phase_transition_loss,
    sequence_metrics,
    temporal_phase_loss,
)
from scripts.train_phase_recovery_mlp import load_phase_collections, train
from tests.unit.test_phase_recovery_mlp import collection


def test_pairs_are_independent_of_storage_order_and_never_cross_rollouts_or_missing_ticks():
    pairs = consecutive_pairs(
        np.array([0, 0, 0, 0, 1, 0, 0]), np.array([0, 0, 0, 0, 0, 1, 0]),
        np.array([10, 10, 10, 10, 10, 10, 11]), np.array([2, 0, 1, 4, 3, 3, 3]))
    assert set(map(tuple, pairs)) == {(1, 2), (2, 0)}


def test_duplicate_row_identity_is_rejected():
    values = np.array([0, 0])
    with pytest.raises(ValueError, match="Duplicate"):
        consecutive_pairs(values, values, values, values)


class FixedPredictions(torch.nn.Module):
    temperature = .5

    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([[8., -8.], [-8., 8.]]))
        self.actions = torch.nn.Parameter(torch.tensor([[0.] * 7, [.03] + [0.] * 6]))
        self.action_scales = torch.tensor([.03] * 3 + [.3] * 3 + [1.])

    def forward(self, observations, base_actions, features):
        index = observations[:, 0].long()
        return self.actions[index], self.logits[index], None


def payload(phases):
    return {"observations": torch.tensor([[0.], [1.]]),
            "base_actions": torch.zeros(2, 7), "features": torch.zeros(2, 1),
            "target_actions": torch.tensor([[0.] * 7, [.03] + [0.] * 6]),
            "expert_phase_id": torch.tensor(phases)}


def test_teacher_phase_boundary_and_correct_action_change_are_not_smoothed():
    model = FixedPredictions()
    loss, terms = temporal_phase_loss(model, payload([5, 1]), torch.tensor([[0, 1]]))
    assert terms["phase_stability_loss"] == 0
    assert terms["action_change_loss"] == 0
    assert loss == 0


def test_false_phase_switch_has_finite_training_gradient():
    model = FixedPredictions()
    loss, terms = temporal_phase_loss(model, payload([1, 1]), torch.tensor([[0, 1]]))
    assert terms["phase_stability_loss"] > .6
    loss.backward()
    assert torch.isfinite(model.logits.grad).all()
    assert model.logits.grad.abs().sum() > 0


def test_boundary_supervision_learns_both_sides_without_a_monotonic_phase_rule():
    network = FixedPredictions()
    pairs = torch.tensor([[0, 1]])
    correct = phase_transition_loss(network, payload([0, 1]), pairs)
    retry = phase_transition_loss(network, payload([1, 0]), pairs)
    assert correct < .001 and retry > 10
    retry.backward()
    assert torch.isfinite(network.logits.grad).all()
    assert network.logits.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="actual phase boundaries"):
        phase_transition_loss(network, payload([1, 1]), pairs)


def test_metrics_distinguish_real_retry_from_unjustified_regression():
    report = sequence_metrics(np.array([4, 3, 1, 2]), np.array([4, 4, 1, 2]),
                              np.array([[0, 1], [1, 2], [2, 3]]))
    assert report["same_phase_switch_rate"] == 1.
    assert report["transition_pair_accuracy"] == .5
    assert report["unjustified_regression_rate"] == .5


def test_temporal_training_requires_issued_labels_and_real_ticks(tmp_path):
    checkpoint, base_hash, pack = collection(tmp_path)
    with pytest.raises(ValueError, match="issued-command"):
        load_phase_collections([pack], expected_base_sha256=base_hash, require_sequences=True)
    path = pack / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["expert_phase_label_source"] = "issued_command"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="saved control ticks"):
        load_phase_collections([pack], expected_base_sha256=base_hash, require_sequences=True)
    for split in ("train", "val"):
        path = pack / f"{split}.npz"
        with np.load(path) as saved:
            arrays = dict(saved)
        arrays["tick"] = np.arange(12) // 2
        np.savez(path, **arrays)
    report = train([pack], tmp_path / "temporal_head", snapshot=checkpoint, steps=2,
                   temporal_weight=.1, require_command_labels=True)
    assert report["train_consecutive_pairs"] == 10
    assert report["require_issued_command_labels"]
    assert report["status"] == "complete"
    assert report["head"]["temporal_loss_weight"] == .1
    assert report["validation_metrics"]["phase_sequence_metrics"]["consecutive_pairs"] == 10
    assert report["runtime_source_changes"] == []
