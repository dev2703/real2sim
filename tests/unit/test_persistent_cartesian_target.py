"""Persistent Cartesian execution must arrest bias without winding up rejected motion."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.geometry.so3 import apply_delta, delta_action, exp_so3, log_so3
from real2sim.policy.adapters import CartesianActionAdapter
from real2sim.sim.kinematics import IKResult


class BiasedServo:
    """Deterministic toy servo: every commanded position has a fixed measured bias."""

    def __init__(self, bias=0.001, frozen=False):
        self.mapping = SimpleNamespace(gripper_real_to_sim_rad=lambda percent: percent / 100)
        self.limits = np.tile([-2., 2.], (6, 1))
        self.pose = np.eye(4)
        self.q = np.zeros(6)
        self.commands = []
        self.bias = bias
        self.frozen = frozen
        self.fail_step = False

    def ee_pose(self):
        return self.pose.copy()

    def sim_state(self):
        return self.q.copy()

    def step(self, q):
        if self.fail_step:
            raise RuntimeError("failed step")
        self.commands.append(q.copy())
        if not self.frozen:
            self.q = q.copy()
            self.q[0] -= self.bias
            self.pose[:3, 3] = self.q[:3]
            self.pose[:3, :3] = exp_so3(np.r_[self.q[3:5], 0.])
        return {}, 0., False, False, {"control": {"applied_radians": q.tolist()}}


class ToyIK:
    def __init__(self):
        self.accept = True
        self.targets = []
        self.seeds = []

    def solve(self, target, seed, *, gripper):
        self.targets.append(target.copy())
        self.seeds.append(seed.copy())
        q = np.r_[target[:3, 3], log_so3(target[:3, :3])[:2], gripper]
        if not self.accept:
            q[:] = 100  # A rejected solver result must never reach native control.
        return IKResult(q, self.accept, 0., 0., 1,
                        "accepted" if self.accept else "residual_rejected")


def controller(*, mode="accepted_target", **env_kwargs):
    env, ik = BiasedServo(**env_kwargs), ToyIK()
    return env, ik, CartesianActionAdapter(env, ik, target_reference=mode)


def test_persistent_zero_action_arrests_servo_bias_while_default_reproduces_drift():
    current, old_ik = BiasedServo(), ToyIK()
    default = CartesianActionAdapter(current, old_ik)
    held, _, persistent = controller()
    for _ in range(200):
        default.apply(np.zeros(7))
        persistent.apply(np.zeros(7))
    assert default.target_reference == "current_pose"
    assert default.accepted_target is None
    assert current.pose[0, 3] == pytest.approx(-.2)
    assert held.pose[0, 3] == pytest.approx(-.001)
    np.testing.assert_allclose(persistent.accepted_target, np.eye(4), atol=1e-15)


def test_increments_compose_on_accepted_target_in_task_frame():
    env, ik, adapter = controller()
    first = np.array([.005, .002, 0., .02, 0., 0., .4])
    second = np.array([.002, .003, 0., 0., .03, 0., .5])
    adapter.apply(first)
    expected = apply_delta(apply_delta(np.eye(4), first[:6]), second[:6])
    result = adapter.apply(second)
    np.testing.assert_allclose(ik.targets[-1], expected, atol=1e-14)
    np.testing.assert_allclose(adapter.accepted_target, expected, atol=1e-14)
    assert result.detail["target_advanced"]
    assert env.commands[-1][5] == .5


def test_native_gripper_clipping_after_calibration_is_reported():
    env, _, adapter = controller()
    env.mapping = adapter.mapping = SimpleNamespace(
        gripper_real_to_sim_rad=lambda percent: .03 * percent)
    result = adapter.apply(np.r_[np.zeros(6), 1.])
    assert result.detail["requested_gripper_rad"] == 3.
    assert result.detail["bounded_gripper_rad"] == 2.
    assert result.detail["native_gripper_clipped"]
    assert result.detail["clamped"][6]
    assert not result.detail["action_clamped"][6]
    assert result.saturation == pytest.approx(1 / 7)
    assert env.commands[-1][5] == 2.


def test_feedback_antiwindup_discards_excess_under_a_stalled_servo():
    env, ik, adapter = controller(frozen=True)
    action = np.array([.02, 0., 0., .2, 0., 0., .4])
    for _ in range(100):
        result = adapter.apply(action)
        feedback = delta_action(env.pose, ik.targets[-1])
        assert np.linalg.norm(feedback[:3]) <= adapter.max_step_m + 1e-14
        assert np.linalg.norm(feedback[3:]) <= adapter.max_step_rad + 1e-14
    np.testing.assert_allclose(adapter.accepted_target[:3, 3], [.03, 0., 0.])
    assert np.linalg.norm(log_so3(adapter.accepted_target[:3, :3])) == pytest.approx(.3)
    assert result.detail["feedback_clamped"] == [True] * 6
    assert result.saturation == pytest.approx(6 / 7)
    adapter.apply(np.zeros(7))
    np.testing.assert_allclose(adapter.accepted_target[:3, 3], [.03, 0., 0.])
    assert np.linalg.norm(log_so3(adapter.accepted_target[:3, :3])) == pytest.approx(.3)


def test_action_and_gripper_caps_remain_in_force():
    env, ik, adapter = controller()
    result = adapter.apply(np.array([3., 4., 0., 0., 0., 5., 2.]))
    feedback = delta_action(np.eye(4), ik.targets[-1])
    assert np.linalg.norm(feedback[:3]) == pytest.approx(.03)
    assert np.linalg.norm(feedback[3:]) == pytest.approx(.3)
    assert result.detail["action_clamped"] == [True] * 7
    assert result.saturation == 1
    assert env.commands[-1][5] == 1


def test_rejection_keeps_reference_and_does_not_replay_rejected_increment():
    env, ik, adapter = controller()
    adapter.apply(np.array([.005, 0., 0., 0., 0., 0., .2]))
    previous = adapter.accepted_target
    seed = env.sim_state()
    ik.accept = False
    rejected = adapter.apply(np.array([.02, 0., 0., 0., 0., 0., .8]))
    assert rejected.ik_rejected and not rejected.detail["target_advanced"]
    assert adapter.ik_rejections == 1
    np.testing.assert_array_equal(adapter.accepted_target, previous)
    np.testing.assert_array_equal(env.commands[-1][:5], seed[:5])
    assert env.commands[-1][5] == .8
    ik.accept = True
    adapter.apply(np.zeros(7))
    np.testing.assert_allclose(ik.targets[-1], previous, atol=1e-14)


def test_failed_native_step_does_not_advance_reference():
    env, _, adapter = controller()
    adapter.apply(np.zeros(7))
    previous = adapter.accepted_target
    env.fail_step = True
    with pytest.raises(RuntimeError, match="failed step"):
        adapter.apply(np.array([.02, 0., 0., 0., 0., 0., 0.]))
    np.testing.assert_array_equal(adapter.accepted_target, previous)


def test_reset_and_explicit_rebase_discard_old_target_without_reusing_old_pose():
    env, ik, adapter = controller()
    adapter.apply(np.zeros(7))
    ik.accept = False
    adapter.apply(np.zeros(7))
    adapter.reset()
    assert adapter.accepted_target is None
    assert adapter.ik_rejections == 0 and adapter.residuals == []
    env.pose[0, 3] = .1  # Environment reset may happen after adapter.reset().
    ik.accept = True
    adapter.apply(np.zeros(7))
    assert adapter.accepted_target[0, 3] == pytest.approx(.1)
    env.pose[0, 3] = .2
    stats = list(adapter.residuals)
    returned = adapter.rebase_target()
    returned[0, 3] = 20  # Callers cannot mutate the internal target.
    public = adapter.accepted_target
    public[0, 3] = 30
    assert adapter.accepted_target[0, 3] == pytest.approx(.2)
    assert adapter.residuals == stats
    adapter.apply(np.zeros(7))
    assert ik.targets[-1][0, 3] == pytest.approx(.2)


def test_default_mode_rebase_is_refused():
    _, _, adapter = controller(mode="current_pose")
    with pytest.raises(ValueError, match="rebasing requires"):
        adapter.rebase_target()


@pytest.mark.parametrize("kwargs", [{"max_step_m": 0}, {"max_step_rad": -1},
                                    {"max_step_m": np.nan}, {"max_step_rad": np.inf},
                                    {"target_reference": "absolute_future_reference"}])
def test_invalid_controller_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        CartesianActionAdapter(BiasedServo(), ToyIK(), **kwargs)


@pytest.mark.parametrize("action", [np.zeros(6), np.zeros(8), np.full(7, np.nan)])
def test_malformed_action_cannot_mutate_reference(action):
    env, ik, adapter = controller()
    with pytest.raises(ValueError, match="seven finite"):
        adapter.apply(action)
    assert adapter.accepted_target is None
    assert not env.commands and not ik.targets


@pytest.mark.parametrize("invalid", ["translation_nan", "rotation_invalid", "homogeneous_invalid"])
def test_invalid_observed_pose_cannot_anchor_controller(invalid):
    env, _, adapter = controller()
    if invalid == "translation_nan":
        env.pose[0, 3] = np.nan
    elif invalid == "rotation_invalid":
        env.pose[0, 0] = 2
    else:
        env.pose[3, 3] = 0
    with pytest.raises(RuntimeError, match="finite proper"):
        adapter.apply(np.zeros(7))
    assert adapter.accepted_target is None


def test_soft_joint_overshoot_remains_clipped_only_for_ik_seed():
    env, ik, adapter = controller()
    env.q[0] = 2.001
    ik.accept = False
    adapter.apply(np.zeros(7))
    assert ik.seeds[-1][0] == 2
    assert env.commands[-1][0] == 2
    assert adapter.ik_rejections == 1
