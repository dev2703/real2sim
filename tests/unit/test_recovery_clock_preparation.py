"""A task-event clock removes only the initial-opening time shortcut."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.geometry.so3 import delta_action, make_T, matrix_from_rot6d
from real2sim.policy.common_recovery import RecoveryObservation
from scripts.convert_recovery_history import convert_observations
from scripts.prepare_recovery_clock import task_event_observations
from tests.unit.test_recovery_history_conversion import sequence


def test_recorded_task_clock_matches_live_encoder_without_teacher_phase_input():
    rows, _ = sequence()
    rows["observations"] = convert_observations(rows)
    original = rows["observations"].copy()
    actual = task_event_observations(rows)
    encoder = RecoveryObservation("panda", 25., action_history="achieved", gripper_clock="task_events")
    pose = np.eye(4)
    env = SimpleNamespace(ee_pose=lambda: pose.copy(), gripper_semantic=lambda: .7)
    expected = []
    for index in range(len(actual)):
        current = original[index, -33:]
        pose[:] = make_T(matrix_from_rot6d(current[3:9]), current[:3])
        expected.append(encoder.encode(env, [.1, .2, .03], [.3, .1, .03], index))
        if index + 1 < len(actual):
            following = original[index + 1, -33:]
            after = make_T(matrix_from_rot6d(following[3:9]), following[:3])
            encoder.executed_action(np.r_[delta_action(pose, after), rows["target_actions"][index, 6]], index)
    np.testing.assert_allclose(actual, np.stack(expected), atol=1e-6)
    np.testing.assert_array_equal(rows["observations"], original)
    assert original[1, -1] == pytest.approx(.04)
    assert actual[1, -1] == 0.
    assert actual[2, -1] == pytest.approx(.04)
    assert actual[3, -1] == pytest.approx(.08)


def test_release_clock_still_runs_after_an_actual_close_event():
    observations = np.zeros((6, 99), np.float32)
    jaws = [0., 1., 0., 0., 1., 1.]
    ages = [0., .04, .04, .08, .04, .08]
    for row in range(6):
        for history in range(3):
            at = max(0, row - 2 + history)
            observations[row, history * 33 + 29] = jaws[at]
            observations[row, history * 33 + 32] = ages[at]
    rows = {"observations": observations, "robot_id": np.ones(6, np.int64),
        "episode_index": np.ones(6, np.int64), "tick": np.arange(6)}
    converted = task_event_observations(rows)
    np.testing.assert_allclose(converted[:, -1], [0., 0., .04, .08, .04, .08])


@pytest.mark.parametrize("ticks", [[0, 2, 3, 4], [0, 0, 1, 2], [1, 2, 3, 4]])
def test_missing_or_duplicate_command_history_cannot_invent_a_first_close(ticks):
    rows, _ = sequence()
    rows["tick"] = np.array(ticks)
    with pytest.raises(ValueError):
        task_event_observations(rows)
