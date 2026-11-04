"""A local residual must preserve and identify its exact frozen HF base."""

import json

import pytest
import torch
from torch import nn

from real2sim.artifacts import sha256_file
from real2sim.policy.local_hf_head import LocalHeadACT, ResidualActionHead
from real2sim.policy.native_lerobot_act import NativeLeRobotACT


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_zero_residual_is_bit_exact_original_output(kind):
    torch.manual_seed(7)
    original = nn.Linear(256, 6)
    features = torch.randn(2, 50, 256)
    head = ResidualActionHead(original, torch.randn(256), torch.rand(256), kind)
    assert torch.equal(head(features), original(features))
    assert all(not parameter.requires_grad for parameter in head.original.parameters())
    assert head.original.weight.data_ptr() != original.weight.data_ptr()


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_training_residual_leaves_original_weights_and_gradients_unchanged(kind):
    torch.manual_seed(8)
    original = nn.Linear(256, 6)
    features = torch.randn(32, 256)
    head = ResidualActionHead(original, torch.zeros(256), torch.ones(256), kind)
    before = {key: tensor.clone() for key, tensor in head.original.state_dict().items()}
    residual_before = {key: tensor.clone() for key, tensor in head.residual.state_dict().items()}
    optimizer = torch.optim.SGD(head.residual.parameters(), lr=.05)
    target = original(features).detach() + .25
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(head(features), target)
        loss.backward()
        optimizer.step()
    for key, tensor in head.original.state_dict().items():
        assert torch.equal(tensor, before[key])
        assert torch.equal(tensor, original.state_dict()[key])
    assert all(parameter.grad is None for parameter in head.original.parameters())
    assert any(not torch.equal(tensor, residual_before[key])
               for key, tensor in head.residual.state_dict().items())


def fake_native_init(self, path, *, device="cpu"):
    """Avoid loading a full visual model for metadata and checksum contracts."""
    self.weights_sha256 = "expected-base-checkpoint"
    self.policy = nn.Module()
    self.policy.model = nn.Module()
    self.policy.model.action_head = nn.Linear(256, 6)


def write_head(directory, *, base="expected-base-checkpoint"):
    head = ResidualActionHead(nn.Linear(256, 6), torch.zeros(256), torch.ones(256), "linear")
    checkpoint = directory / "head.pt"
    torch.save(head.state_dict(), checkpoint)
    metadata = {"kind": "linear", "base_checkpoint_sha256": base,
                "head_sha256": sha256_file(checkpoint)}
    (directory / "head.json").write_text(json.dumps(metadata))
    return metadata


def test_local_head_rejects_different_base_checkpoint_before_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(NativeLeRobotACT, "__init__", fake_native_init)
    write_head(tmp_path, base="different-base-checkpoint")
    with pytest.raises(ValueError, match="different HF checkpoint"):
        LocalHeadACT("unused", tmp_path)


def test_local_head_rejects_changed_checkpoint_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(NativeLeRobotACT, "__init__", fake_native_init)
    write_head(tmp_path)
    with (tmp_path / "head.pt").open("ab") as handle:
        handle.write(b"changed-after-metadata")
    with pytest.raises(ValueError, match="Local head checksum mismatch"):
        LocalHeadACT("unused", tmp_path)


def test_checked_local_head_loads_frozen_and_records_exact_checksum(monkeypatch, tmp_path):
    monkeypatch.setattr(NativeLeRobotACT, "__init__", fake_native_init)
    metadata = write_head(tmp_path)
    loaded = LocalHeadACT("unused", tmp_path)
    assert loaded.local_head_sha256 == metadata["head_sha256"]
    assert isinstance(loaded.policy.model.action_head, ResidualActionHead)
    assert not loaded.policy.training
    assert all(not parameter.requires_grad for parameter in loaded.policy.parameters())
