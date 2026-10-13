"""Loss formulas/gradients, padded labels, common metrics and safe checkpoint retention."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.policies.act.modeling_act import ACTPolicy

from real2sim.policy.losses import ReconstructionLoss, act_training_loss, masked_reconstruction
from real2sim.policy.train import evaluate, prune_checkpoints, update_checkpoint_link


@pytest.mark.parametrize("kind,expected,gradient", [
    ("l1", 1.25, [0.5, -0.5]),
    ("l2", 2.125, [0.5, -2.0]),
    ("huber", 0.8125, [0.25, -0.5]),
])
def test_masked_loss_values_and_gradients(kind, expected, gradient):
    pred = torch.tensor([[[0.5, -2.0], [100.0, -100.0]]], requires_grad=True)
    loss, l1 = masked_reconstruction(pred, torch.zeros_like(pred), torch.tensor([[False, True]]), ReconstructionLoss(kind))
    assert loss.item() == pytest.approx(expected)
    assert l1.item() == pytest.approx(1.25)
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[gradient, [0.0, 0.0]]]))


def test_huber_delta_is_in_normalised_units_and_empty_mask_is_zero():
    pred = torch.tensor([[[2.0]]], requires_grad=True)
    loss, _ = masked_reconstruction(pred, pred * 0, torch.tensor([[False]]), ReconstructionLoss("huber", 0.5))
    assert loss.item() == pytest.approx(0.875)
    masked, _ = masked_reconstruction(pred, pred * 0, torch.tensor([[True]]), ReconstructionLoss("l2"))
    masked.backward()
    assert masked.item() == 0 and pred.grad.item() == 0


@pytest.mark.parametrize("cfg", [{"kind": "mystery"}, {"huber_delta": 0}, {"huber_delta": float("nan")}])
def test_invalid_loss_configuration(cfg):
    with pytest.raises(ValueError):
        ReconstructionLoss(**cfg)


class TinyACT(torch.nn.Module):
    """Use the installed upstream loss with a deterministic tiny model to test the adapter."""

    forward = ACTPolicy.forward

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(image_features={"observation.images.iphone": None}, use_vae=True, kl_weight=10.0)
        self.pred = torch.nn.Parameter(torch.tensor([[[0.5, -2.0], [3.0, 4.0]]]))
        self.mu = torch.nn.Parameter(torch.tensor([[0.5, 1.0]]))

    def model(self, batch):
        assert len(batch["observation.images"]) == 1
        return self.pred, (self.mu, torch.zeros_like(self.mu))


@pytest.mark.parametrize("kind", ["l1", "l2", "huber"])
def test_adapter_preserves_kl_and_does_not_mutate_batch(kind):
    policy = TinyACT()
    batch = {"action": torch.zeros_like(policy.pred), "action_is_pad": torch.tensor([[False, True]]),
             "observation.images.iphone": torch.zeros(1, 3, 8, 8)}
    upstream, upstream_metrics = policy.forward(batch)
    loss, metrics = act_training_loss(policy, batch, ReconstructionLoss(kind))
    assert "observation.images" not in batch
    assert metrics["kld_loss"] == upstream_metrics["kld_loss"] == 0.625
    assert metrics["l1_loss"] == upstream_metrics["l1_loss"]
    assert loss.item() == pytest.approx(metrics["reconstruction_loss"] + 10 * metrics["kld_loss"])
    if kind == "l1":
        torch.testing.assert_close(loss, upstream, rtol=0, atol=0)
        actual = torch.autograd.grad(loss, tuple(policy.parameters()), retain_graph=True)
        expected = torch.autograd.grad(upstream, tuple(policy.parameters()))
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
    else:
        loss.backward()
        torch.testing.assert_close(policy.mu.grad, 10 * policy.mu.detach())
        assert torch.isfinite(policy.pred.grad).all()


def test_metrics_ignore_padding_and_report_per_dimension_units():
    class FixedPolicy(torch.nn.Module):
        def predict_action_chunk(self, batch):
            return batch["prediction"]

    loader = [{"prediction": torch.tensor([[[1.0, 2.0], [3.0, 4.0], [1e6, 1e6]]]),
               "action": torch.zeros(1, 3, 2), "action_is_pad": torch.tensor([[False, False, True]])}]
    result = evaluate(FixedPolicy(), loader, "cpu", np.array([10., 100.]), None, detailed=True)
    assert result["l1"] == 2.5
    assert result["rmse"] == pytest.approx(np.sqrt(7.5))
    assert result["p95_abs"] == pytest.approx(3.85)
    assert result["l1_raw_per_dim"] == [20., 300.]
    assert result["rmse_raw_per_dim"] == pytest.approx([np.sqrt(5) * 10, np.sqrt(10) * 100])
    assert result["p95_abs_raw_per_dim"] == pytest.approx([29., 390.])
    assert result["valid_action_components"] == 4


def test_retention_only_removes_this_runs_obsolete_checkpoints(tmp_path):
    for name in ("000001", "000002", "000003", "000999"):
        (tmp_path / name).mkdir()
    update_checkpoint_link(tmp_path, "000001", "best")
    update_checkpoint_link(tmp_path, "000003", "last")
    created = {"000001", "000002", "000003"}
    prune_checkpoints(tmp_path, created)
    assert created == {"000001", "000003"}
    assert not (tmp_path / "000002").exists()
    assert (tmp_path / "000999").is_dir()  # never touch artifacts outside this train call
    assert (tmp_path / "best").is_dir() and (tmp_path / "last").is_dir()
