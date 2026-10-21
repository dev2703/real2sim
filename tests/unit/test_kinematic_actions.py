from types import SimpleNamespace

import numpy as np
import pytest
import torch

from real2sim.data.labels import episode_labels, tool_frame_actions
from real2sim.geometry.so3 import apply_delta, delta_action, exp_so3
from real2sim.policy.adapters import ApplyResult
from real2sim.policy.kinematic_actions import KinematicDeltaAdapter, bounded_joint_step
from real2sim.policy.losses import ReconstructionLoss, act_training_loss, integrated_tool_loss


def test_tool_labels_and_execution_roundtrip_under_rotated_base():
    start = np.eye(4)
    start[:3, :3] = exp_so3(np.array([.3, -.7, 1.2]))
    end = apply_delta(start, np.array([.01, -.02, .005, .03, .01, -.02]))
    lab = episode_labels(np.stack([start, end]), [50, 50], [60, 60], 12.5, "leader_action")
    tool = tool_frame_actions(lab.poses, lab.action)

    class Adapter:
        def __init__(self):
            self.env = self

        def ee_pose(self):
            return start

        def apply(self, a):
            self.received = a
            return ApplyResult({}, {}, False, 0)

    wrapped = Adapter()
    KinematicDeltaAdapter(wrapped, tool_frame=True).apply(tool[0])
    np.testing.assert_allclose(wrapped.received, lab.action[0], atol=1e-12)
    np.testing.assert_allclose(apply_delta(start, wrapped.received[:6]), end, atol=1e-12)
    # Left-multiplying both poses by a rotated/translated task frame preserves tool labels.
    base = np.eye(4)
    base[:3, :3], base[:3, 3] = exp_so3(np.array([-.5, .2, -.9])), [.1, .2, .3]
    pose2 = np.stack([base @ start, base @ end])
    action2 = np.array([np.r_[delta_action(*pose2), .6], np.r_[np.zeros(6), .6]])
    np.testing.assert_allclose(tool_frame_actions(pose2, action2), tool, atol=1e-12)


def test_bounded_projection_respects_joint_margins_and_missing_dof():
    j = np.eye(6)[:, :5]
    q = np.array([.99, 0, 0, 0, 0])
    bounds = np.tile([-1., 1.], (5, 1))
    d = np.array([.1, .1, -.1, 0, 0, .3])
    dq = bounded_joint_step(j, d, q, bounds, np.full(5, .02))
    assert np.all(q + dq >= bounds[:, 0]) and np.all(q + dq <= bounds[:, 1])
    assert np.max(np.abs(dq)) <= .02
    assert dq[0] <= .01
    assert (j @ dq)[5] == 0  # impossible sixth direction is observable as residual
    zero = bounded_joint_step(j, np.zeros(6), np.zeros(5), bounds, np.full(5, .02))
    np.testing.assert_allclose(zero, 0, atol=1e-12)
    with pytest.raises(ValueError, match="never silently clip"):
        bounded_joint_step(j, d, np.full(5, 1.01), bounds, np.full(5, .02))


def test_trajectory_loss_accumulation_mask_and_rotation_gradients():
    target = torch.zeros(1, 4, 7)
    pred = target.clone()
    pred[:, :, 0] = .001
    pred[:, :, 5] = .01
    pred.requires_grad_()
    pad = torch.tensor([[False, False, False, True]])
    kwargs = dict(tool_frame=True, translation_scale=.024, rotation_scale=.24)
    loss = integrated_tool_loss(pred, target, pad, **kwargs)
    assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0, :3, 0].abs().sum() > 0
    assert pred.grad[0, :3, 5].abs().sum() > 0
    assert not pred.grad[0, 3].any()
    modified = pred.detach().clone()
    modified[:, 3] = 1e8
    torch.testing.assert_close(integrated_tool_loss(modified, target, pad, **kwargs), loss.detach())
    zero = target.clone().requires_grad_()
    exact = integrated_tool_loss(zero, target, pad, **kwargs)
    exact.backward()
    assert exact.item() == 0 and torch.isfinite(zero.grad).all()
    assert integrated_tool_loss(pred, target, torch.ones_like(pad), **kwargs).item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tool_frame", [False, True])
def test_half_precision_trajectory_matches_float32_with_finite_gradients(dtype, tool_frame):
    pred = torch.zeros(1, 4, 7, dtype=dtype)
    pred[:, :, 0], pred[:, :, 5] = .001, .01
    pred.requires_grad_()
    target = torch.zeros_like(pred)
    pad = torch.tensor([[False, False, False, True]])
    kwargs = dict(tool_frame=tool_frame, translation_scale=.024, rotation_scale=.24)
    expected = integrated_tool_loss(pred.detach().float(), target.float(), pad, **kwargs)
    actual = integrated_tool_loss(pred, target, pad, **kwargs)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0, :3, 0].abs().sum() > 0
    assert pred.grad[0, :3, 5].abs().sum() > 0
    assert not pred.grad[0, 3].any()


@pytest.mark.parametrize("representation", ["cartesian", "cartesian_tool"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_act_trajectory_uses_full_precision_composition_under_autocast(representation, device):
    class TinyPolicy(torch.nn.Module):
        config = SimpleNamespace(image_features={}, use_vae=False)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(
                [[.05], [0], [0], [0], [0], [.04], [0]], device=device))

        def model(self, batch):
            return torch.nn.functional.linear(batch["probe"], self.weight), (None, None)

    policy = TinyPolicy()
    batch = {"action": torch.zeros(1, 4, 7, device=device),
             "action_is_pad": torch.tensor([[False, False, False, True]], device=device),
             "probe": torch.ones(1, 4, 1, device=device)}
    mean = torch.tensor([.0003, -.0002, 0, .0005, 0, -.0003, .6], device=device)
    std = torch.tensor([.02, .02, .02, .24, .24, .24, .2], device=device)
    config = ReconstructionLoss(trajectory_weight=.1)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        predicted, _ = policy.model(batch)
        assert predicted.dtype == torch.bfloat16
        loss, metrics = act_training_loss(policy, batch, config, action_mean=mean,
                                         action_std=std, action_representation=representation)
    expected = integrated_tool_loss(
        predicted.float() * std + mean, batch["action"] * std + mean, batch["action_is_pad"],
        tool_frame=representation == "cartesian_tool", translation_scale=.024, rotation_scale=.24,
    )
    assert metrics["trajectory_loss"] == pytest.approx(expected.item(), rel=1e-6)
    assert loss.item() == pytest.approx(metrics["reconstruction_loss"] + .1 * expected.item())
    loss.backward()
    assert torch.isfinite(policy.weight.grad).all()
    assert policy.weight.grad[0].abs().item() > 0
    assert policy.weight.grad[5].abs().item() > 0


def test_new_matrix_controls_are_distinct_and_exposure_matched():
    from real2sim.config import load_yaml
    from real2sim.experiments.v2 import experiment_matrix

    matrix = experiment_matrix(load_yaml("configs/v2/act_base.yaml"),
                               clip_model=load_yaml("configs/v2/clip_backbone.yaml")["model"])
    arms = {a["id"]: a for a in matrix}
    assert "reuse" not in arms["K3_tool_trajectory"]
    for name in ("K1_tool_delta", "K3_tool_trajectory", "L7_clip_frozen", "L8_clip_flow",
                 "L8_clip_regression", "L7_clip_last2_lr3e-06", "L7_clip_last2_lr1e-05"):
        arm = arms[name]
        assert arm["examples"] == 64000
        cfg = arm["config"]
        assert cfg["train"]["schedule_steps"] * arm["effective_batch"] == 320000
    assert arms["K1_tool_delta"]["action_representation"] == "cartesian_tool"
    # A later measured CLIP winner can use the same batch/LR screen without losing its 1e-4 control.
    adaptive = experiment_matrix(load_yaml("configs/v2/act_base.yaml"), backbone="clip",
                                 effective_batch=64, head_lr=3e-5,
                                 clip_model=load_yaml("configs/v2/clip_backbone.yaml")["model"])
    control = next(a for a in adaptive if a["id"] == "L3_clip_b16_lr0.0001")
    assert control["config"]["model"]["vision_backbone"] == "clip"
    assert control["effective_batch"] == 16
