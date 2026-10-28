"""DAgger's expert clock follows executed jaw intent, not hypothetical labels."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_hf_recovery import StaticBowlRecovery
from scripts.run_recovery_dagger import CartesianConvergenceTeacher, teacher_common_action


@pytest.mark.parametrize("phase,selected,paused", [
    ("CLOSE", 1., True), ("CLOSE", 0., False),
    ("RELEASE", 0., True), ("RELEASE", 1., False),
])
def test_expert_dwell_does_not_count_opposing_executed_command(monkeypatch, phase, selected, paused):
    teacher = object.__new__(CartesianConvergenceTeacher)
    teacher.phase, teacher.phase_tick = phase, 12
    teacher.target_q = None
    teacher.executed_semantic_gripper = selected

    def step(self, obs, env):
        self.phase_tick += 1
        return self.phase_tick

    monkeypatch.setattr(StaticBowlRecovery, "step", step)
    assert teacher.step({}, object()) == (12 if paused else 13)


def test_panda_equivalent_tcp_allows_phase_progress_without_original_joint_branch(monkeypatch):
    teacher = object.__new__(CartesianConvergenceTeacher)
    teacher.phase, teacher.phase_tick = "PRE_GRASP", 12
    teacher.target_q = np.arange(7, dtype=float)
    teacher.n = 7
    measured = np.zeros(7)
    teacher._native = lambda: measured
    teacher._grip = lambda opened: .08 if opened else 0.
    teacher._forward = lambda *_: np.eye(4)
    monkeypatch.setattr(StaticBowlRecovery, "step", lambda *_: None)
    teacher.step({}, SimpleNamespace(ee_pose=lambda: np.eye(4)))
    np.testing.assert_array_equal(teacher.target_q, measured)


def test_panda_expert_cartesian_endpoint_is_not_native_nullspace_motion():
    target = np.eye(4)
    target[:3, 3] = [.1, .2, .3]
    teacher = SimpleNamespace(robot="panda", phase="PRE_GRASP", target_q=np.zeros(7),
        _grip=lambda _: .08, _forward=lambda *_: target)
    executor = SimpleNamespace(env=SimpleNamespace(ee_pose=lambda: np.eye(4)),
        native_to_common=lambda _: np.r_[np.zeros(6), 1.])
    action = teacher_common_action(teacher, SimpleNamespace(detail={"requested_native": np.zeros(8)}), executor)
    assert np.linalg.norm(action[:3]) == pytest.approx(.03)
    np.testing.assert_allclose(action[:3] / np.linalg.norm(action[:3]),
                               target[:3, 3] / np.linalg.norm(target[:3, 3]))
    assert action[6] == 1.
