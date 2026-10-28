"""A learned step can shrink for feasibility without inventing an object goal."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.policy.common_recovery import CommonCartesianExecutor, soft_stop_seed


def executor(accept_distance):
    commands, targets = [], []

    def step(command):
        commands.append(command.copy())
        return {}, 0., False, False, {"control": {"limit_clipped": [False] * 8}}

    def solve(target, seed, **kwargs):
        targets.append(target.copy())
        accepted = np.linalg.norm(target[:3, 3]) <= accept_distance
        return SimpleNamespace(accepted=accepted, reason="accepted" if accepted else "collision",
                               qpos=seed.copy())

    result = object.__new__(CommonCartesianExecutor)
    result.robot = "panda"
    result.env = SimpleNamespace(ee_pose=lambda: np.eye(4), arm_qpos=lambda: np.zeros(7),
        max_opening=.08, gripper_opening=lambda: .08, step=step,
        limits=np.tile([-1., 1.], (7, 1)))
    result.guard = SimpleNamespace(solve=solve)
    return result, commands, targets


def test_backtracking_rechecks_each_smaller_step_and_keeps_its_direction():
    controller, commands, targets = executor(.005)
    result = controller.step(np.array([.03, 0., 0., 0., 0., 0., 1.]))
    assert [target[0, 3] for target in targets] == [.03, .015, .0075, .00375]
    assert result.detail["feasibility_fraction"] == .125
    assert not result.ik_rejected
    assert result.detail["common_action"][0] == .00375
    assert result.detail["requested_common_action"][0] == .03
    assert commands[0][-1] == .08


def test_failed_candidates_hold_the_arm_and_remain_rejections():
    controller, commands, targets = executor(0.)
    result = controller.step(np.array([.03, 0., 0., 0., 0., 0., 0.]))
    assert len(targets) == 5 and result.ik_rejected
    assert result.detail["feasibility_fraction"] == 0.
    np.testing.assert_array_equal(commands[0], np.zeros(8))


def test_soft_stop_projection_only_accepts_small_measured_seed_error():
    measured = np.array([1.0001, -.2])
    limits = np.array([[-1., 1.], [-1., 1.]])
    np.testing.assert_array_equal(soft_stop_seed(measured, limits), [1., -.2])
    assert measured[0] == 1.0001
    with pytest.raises(ValueError, match="soft-stop"):
        soft_stop_seed(np.array([1.002, 0.]), limits)
