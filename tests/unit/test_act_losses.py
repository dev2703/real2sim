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


@pytest.mark.parametrize("cfg", [{"kind": "mystery"}, {"huber_delta": 0}, {"huber_delta": float("nan")},
                                 {"supervised_prefix_steps": 0}, {"supervised_prefix_steps": -1},
                                 {"supervised_prefix_steps": 1.5}, {"supervised_prefix_steps": True}])
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


@pytest.mark.parametrize("kind", ["l1", "l2", "huber"])
def test_prefix_reconstruction_uses_actual_valid_prefix_components_and_zero_tail_gradient(kind):
    # One terminally padded prefix action, with much larger unexecuted tail values.
    pred = torch.tensor([[[1., -3.], [2., 4.], [100., -100.]],
                         [[3., 5.], [900., 900.], [800., 800.]]], requires_grad=True)
    target = torch.zeros_like(pred)
    pad = torch.tensor([[False, False, False], [False, True, True]])
    config = ReconstructionLoss(kind=kind, supervised_prefix_steps=2)
    loss, l1 = masked_reconstruction(pred, target, pad, config)
    reference, reference_l1 = masked_reconstruction(pred[:, :2], target[:, :2], pad[:, :2],
                                                    ReconstructionLoss(kind=kind))
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(l1, reference_l1)
    assert l1.item() == pytest.approx(3.)  # six real prefix components
    loss.backward()
    assert torch.count_nonzero(pred.grad[:, 2:]) == 0
    assert torch.count_nonzero(pred.grad[1, 1:]) == 0
    assert torch.count_nonzero(pred.grad[0, :2]) == 4


def test_prefix_rejects_overlong_or_misaligned_chunks_and_all_pad_is_finite():
    pred = torch.ones((2, 3, 2), requires_grad=True)
    target = torch.zeros_like(pred)
    config = ReconstructionLoss(supervised_prefix_steps=2)
    loss, l1 = masked_reconstruction(pred, target, torch.ones((2, 3), dtype=torch.bool), config)
    loss.backward()
    assert loss.item() == l1.item() == 0.
    assert torch.isfinite(pred.grad).all() and torch.count_nonzero(pred.grad) == 0
    with pytest.raises(ValueError, match="exceeds"):
        masked_reconstruction(pred, target, torch.zeros((2, 3), dtype=torch.bool),
                              ReconstructionLoss(supervised_prefix_steps=4))
    with pytest.raises(ValueError, match="aligned"):
        masked_reconstruction(pred, target, torch.zeros((2, 2), dtype=torch.bool), config)


def test_prefix_adapter_keeps_kl_and_full_prediction_shape():
    policy = TinyACT()
    batch = {"action": torch.zeros_like(policy.pred), "action_is_pad": torch.zeros((1, 2), dtype=torch.bool),
             "observation.images.iphone": torch.zeros(1, 3, 8, 8)}
    loss, metrics = act_training_loss(policy, batch, ReconstructionLoss(supervised_prefix_steps=1))
    assert policy.pred.shape == (1, 2, 2)
    assert metrics["reconstruction_loss"] == metrics["l1_loss"] == 1.25
    assert metrics["kld_loss"] == .625
    assert loss.item() == pytest.approx(1.25 + 10 * .625)
    loss.backward()
    torch.testing.assert_close(policy.mu.grad, 10 * policy.mu.detach())
    assert torch.count_nonzero(policy.pred.grad[:, 1:]) == 0
    assert "observation.images" not in batch


def test_prefix_trajectory_loss_also_ignores_unexecuted_tail():
    class TrajectoryACT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(image_features={}, use_vae=False)
            self.pred = torch.nn.Parameter(torch.zeros((1, 4, 7)))
            with torch.no_grad():
                self.pred[0, 0, 0] = .01
                self.pred[0, 2:, :6] = 100.

        def model(self, batch):
            return self.pred, (None, None)

    policy = TrajectoryACT()
    batch = {"action": torch.zeros_like(policy.pred), "action_is_pad": torch.zeros((1, 4), dtype=torch.bool)}
    config = ReconstructionLoss(trajectory_weight=.5, supervised_prefix_steps=2)
    loss, metrics = act_training_loss(policy, batch, config, action_mean=torch.zeros(7),
                                     action_std=torch.ones(7), action_representation="cartesian")
    assert metrics["trajectory_loss"] == pytest.approx((.01 / config.translation_scale_m)**2)
    loss.backward()
    assert torch.isfinite(policy.pred.grad).all()
    assert torch.count_nonzero(policy.pred.grad[:, 2:]) == 0
    assert policy.pred.grad[0, 0, 0] > 0


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
