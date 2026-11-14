"""A five-joint source command moves the Panda grasp point; it is not a seven-joint copy."""

import numpy as np
import pytest

from real2sim.geometry.so3 import exp_so3
from real2sim.panda.model import POINTING_DOWN
from scripts.diag_v2_learned_panda_retarget_20261027 import (
    panda_target_from_source_command,
    target_from_source,
)


class Source:
    """Wrist flex swings the tool point; one arm joint and the jaw do not."""

    def __init__(self):
        self.panda_offset = np.array([-.008, 0., 0.])

    def ball_point(self, q):
        q = np.asarray(q, float)
        if q.shape != (6,):
            raise AssertionError(f"grasp point expected the source command, got {q.shape}")
        return np.array([.30 + q[0], -.05 + .03 * q[3], .12 + q[1]])


def grasp(pose, offset):
    return pose[:3, 3] + pose[:3, :3] @ offset


def measured_pose():
    pose = np.eye(4)
    pose[:3, :3] = exp_so3(np.array([.2, -.1, .05])) @ POINTING_DOWN
    pose[:3, 3] = np.array([.26, .02, .18])
    return pose


def test_held_source_command_does_not_execute_the_projection_residual():
    source = Source()
    measured = measured_pose()
    reference = np.array([.1, -.2, .3, .4, -.5, .6])
    mapped = panda_target_from_source_command(source, reference, measured, reference)
    np.testing.assert_allclose(grasp(mapped, source.panda_offset), grasp(measured, source.panda_offset))
    np.testing.assert_allclose(mapped[:3, :3], POINTING_DOWN)
    absolute = target_from_source(source, reference)
    residual = grasp(absolute, source.panda_offset) - grasp(measured, source.panda_offset)
    assert np.linalg.norm(residual) > .01


def test_source_wrist_motion_moves_the_measured_grasp_point_in_task_frame():
    source = Source()
    measured = measured_pose()
    reference = np.zeros(6)
    commanded = np.array([.04, -.02, .7, .5, -.3, .2])
    mapped = panda_target_from_source_command(source, commanded, measured, reference)
    displacement = source.ball_point(commanded) - source.ball_point(reference)
    np.testing.assert_allclose(grasp(mapped, source.panda_offset),
                               grasp(measured, source.panda_offset) + displacement)
    np.testing.assert_allclose(mapped[:3, :3], POINTING_DOWN)
    # The wrist term is not a Cartesian translation, and it is not dropped.
    assert abs(displacement[1]) > .01
    assert not np.allclose(displacement, commanded[:3])


def test_null_arm_joint_and_gripper_do_not_move_the_panda_target():
    source = Source()
    measured = measured_pose()
    reference = np.array([.1, -.2, .3, .4, -.5, 10.])
    commanded = reference.copy()
    commanded[2] += .8
    commanded[5] = 0.
    held = panda_target_from_source_command(source, reference, measured, reference)
    mapped = panda_target_from_source_command(source, commanded, measured, reference)
    np.testing.assert_allclose(mapped, held)


@pytest.mark.parametrize("command", [np.zeros(5), np.zeros(7), np.zeros(8), np.zeros((6, 1))])
def test_panda_joint_vector_is_not_read_as_a_source_command(command):
    source = Source()
    with pytest.raises(ValueError, match="five arm joints"):
        panda_target_from_source_command(source, command, measured_pose(), np.zeros(6))


def test_nonfinite_command_and_improper_pose_are_rejected():
    source = Source()
    reference = np.zeros(6)
    bad = reference.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        panda_target_from_source_command(source, bad, measured_pose(), reference)
    pose = measured_pose()
    pose[0, 0] = 2.
    with pytest.raises(ValueError, match="proper transform"):
        panda_target_from_source_command(source, reference, pose, reference)
