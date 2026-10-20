import numpy as np
import pytest

from real2sim.data.labels import episode_labels, tool_frame_actions
from real2sim.geometry.so3 import apply_delta, delta_action, exp_so3
from real2sim.policy.adapters import ApplyResult
from real2sim.policy.kinematic_actions import KinematicDeltaAdapter, bounded_joint_step


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
