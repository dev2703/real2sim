"""Independent checks for sequence isolation and episode-balanced selection."""

import json

import numpy as np
import pytest
import torch

from real2sim.policy.phase_sequences import temporal_phase_loss
from scripts.train_phase_recovery_mlp import evaluate, load_phase_collections
from tests.unit.test_phase_recovery_mlp import collection


def test_loader_keeps_same_seed_in_different_collection_rollouts_separate(tmp_path):
    directories = []
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir()
        _, base_hash, pack = collection(root)
        manifest_path = pack / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["expert_phase_label_source"] = "issued_command"
        manifest_path.write_text(json.dumps(manifest))
        for split in ("train", "val"):
            path = pack / f"{split}.npz"
            with np.load(path) as saved:
                arrays = dict(saved)
            arrays["tick"] = np.repeat(np.arange(6), 2)
            np.savez(path, **arrays)
        directories.append(pack)
    data, _, _ = load_phase_collections(directories, expected_base_sha256=base_hash,
                                       require_sequences=True)
    for split in data.values():
        pairs = split["consecutive_pairs"]
        assert len(pairs) == 20
        np.testing.assert_array_equal(pairs[:, 0] // 12, pairs[:, 1] // 12)
        np.testing.assert_array_equal(split["robot_id"][pairs[:, 0]],
                                      split["robot_id"][pairs[:, 1]])


class LookupModel(torch.nn.Module):
    temperature = .5

    def __init__(self, actions, logits):
        super().__init__()
        self.actions = torch.nn.Parameter(actions.clone())
        self.logits = torch.nn.Parameter(logits.clone())
        self.action_scales = torch.tensor([.03, .03, .03, .3, .3, .3, 1.])

    def forward(self, observations, base_actions, features):
        index = observations[:, 0].long()
        actions, logits = self.actions[index], self.logits[index]
        components = actions[:, None, :].expand(-1, logits.shape[-1], -1)
        return actions, logits, components


def test_temporal_loss_keeps_order_for_permuted_noncontiguous_pairs():
    torch.manual_seed(31)
    actions, logits = torch.randn(6, 7), torch.randn(6, 9)
    network = LookupModel(actions, logits)
    payload = {"observations": torch.arange(6).float()[:, None],
               "base_actions": torch.zeros(6, 7), "features": torch.zeros(6, 1),
               "target_actions": actions.clone(), "expert_phase_id": torch.arange(6)}
    loss, terms = temporal_phase_loss(network, payload, torch.tensor([[4, 0], [1, 5], [3, 2]]))
    assert terms["phase_stability_loss"] == 0
    assert terms["action_change_loss"] == 0
    assert loss == 0


def test_episode_balanced_selection_does_not_let_long_rollout_dominate():
    normalized_errors = torch.tensor([0.] * 9 + [1., .5, .5])
    scales = torch.tensor([.03, .03, .03, .3, .3, .3, 1.])
    network = LookupModel(normalized_errors[:, None] * scales, torch.zeros(12, 11))
    payload = {"observations": torch.arange(12).float()[:, None],
               "base_actions": torch.zeros(12, 7), "features": torch.zeros(12, 1),
               "target_actions": torch.zeros(12, 7), "expert_phase_id": torch.zeros(12).long()}
    report = evaluate(network, payload, np.array([0] * 10 + [1, 1]),
                      episodes=np.array([10] * 9 + [11, 20, 20]))
    assert report["robot_balanced_mae_normalized"] == pytest.approx(.3)
    assert report["robot_episode_balanced_mae_normalized"] == pytest.approx(.5)
