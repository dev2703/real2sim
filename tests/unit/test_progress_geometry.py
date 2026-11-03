"""Grasp progress is explicit, frame-correct and separate from saved contracts."""

import numpy as np
import pytest

from real2sim.geometry.so3 import exp_so3, rot6d_from_matrix
from real2sim.policy.common_recovery import (
    ACHIEVED_OBSERVATION_NAMES,
    OBSERVATION_NAMES,
    TASK_EVENT_OBSERVATION_NAMES,
    TRACKED_ACHIEVED_OBSERVATION_NAMES,
    TRACKED_OBSERVATION_NAMES,
    TRACKED_TASK_EVENT_OBSERVATION_NAMES,
)
from real2sim.policy.progress_geometry import (
    PANDA_GRASP_OFFSET_M,
    SOURCE_GRASP_OFFSET_M,
    observation_geometry_names,
    transform_observation_geometry,
)


def observation(robot, rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    frames = np.zeros((3, 33), dtype=np.float64)
    frames[:, :3] = [.1, .2, .3]
    frames[:, 3:9] = rot6d_from_matrix(rotation)
    frames[:, 13:16] = [.04, -.02, -.1]
    frames[:, 16:19] = [.2, .1, -.1]
    frames[:, 30 + robot] = 1.
    frames[:, 21:23] = [.4, .8]
    frames[:, 23:30] = [.001, .002, .003, .01, .02, .03, .8]
    frames[:, 32] = .16
    return frames.reshape(99)


@pytest.mark.parametrize("robot,offset", [(0, SOURCE_GRASP_OFFSET_M), (1, PANDA_GRASP_OFFSET_M)])
def test_reference_error_uses_rotated_grasp_point_and_preserves_other_channels(robot, offset):
    rotation = exp_so3(np.array([.2, -.4, .7]))
    values = observation(robot, rotation)
    original = values.copy()
    result = transform_observation_geometry(values, TRACKED_ACHIEVED_OBSERVATION_NAMES)
    frame = result.reshape(3, 33)
    old = values.reshape(3, 33)
    np.testing.assert_allclose(frame[:, 13:16], old[:, 13:16] - rotation @ offset)
    np.testing.assert_allclose(frame[:, 16:19], old[:, 16:19] - rotation @ offset)
    unchanged = np.r_[np.arange(13), np.arange(19, 33)]
    np.testing.assert_array_equal(frame[:, unchanged], old[:, unchanged])
    np.testing.assert_array_equal(values, original)
    assert not np.shares_memory(result, values)


def test_source_schema_and_native_panda_agree_for_the_same_grasp_point():
    source_rotation = exp_so3(np.array([0., .6, 0.]))
    panda_rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    point = np.array([.05, .33, .02])
    ball = np.array([.052, .329, .021])
    source, panda = observation(0, source_rotation), observation(1, panda_rotation)
    for values, rotation, offset in ((source, source_rotation, SOURCE_GRASP_OFFSET_M),
                                     (panda, panda_rotation, PANDA_GRASP_OFFSET_M)):
        frames = values.reshape(3, 33)
        tcp = point - rotation @ offset
        frames[:, :3] = tcp
        frames[:, 13:16] = ball - tcp
    converted = transform_observation_geometry(np.stack([source, panda]), OBSERVATION_NAMES)
    np.testing.assert_allclose(converted[:, -33 + 13:-33 + 16],
                               np.broadcast_to(ball - point, (2, 3)), atol=1e-14)


@pytest.mark.parametrize("names", [OBSERVATION_NAMES, TRACKED_OBSERVATION_NAMES,
                                    ACHIEVED_OBSERVATION_NAMES, TRACKED_ACHIEVED_OBSERVATION_NAMES,
                                    TASK_EVENT_OBSERVATION_NAMES, TRACKED_TASK_EVENT_OBSERVATION_NAMES])
def test_geometry_contract_is_explicit_and_conversion_is_reversible_and_idempotent(names):
    values = np.stack([observation(0), observation(1)]).astype(np.float32)
    grasp_names = observation_geometry_names(names, "grasp")
    assert "_rgb_ball_grasp_relative_" in grasp_names[13]
    assert "_rgb_bowl_grasp_relative_" in grasp_names[16]
    assert len(grasp_names) == 99 and grasp_names != list(names)
    assert observation_geometry_names(grasp_names, "tcp") == list(names)
    result = transform_observation_geometry(values, names)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(transform_observation_geometry(result, grasp_names), result)
    np.testing.assert_allclose(transform_observation_geometry(result, grasp_names, "tcp"),
                               values, atol=1e-8)


@pytest.mark.parametrize("bits", [[0., 0.], [1., 1.], [.5, .5], [0., 2.]])
def test_unknown_or_ambiguous_encoded_embodiment_fails_before_geometry(bits):
    values = observation(0)
    values.reshape(3, 33)[1, 30:32] = bits
    with pytest.raises(ValueError, match="one-hot"):
        transform_observation_geometry(values, OBSERVATION_NAMES)


def test_invalid_contract_shape_and_rotation_are_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="99D"):
        transform_observation_geometry(observation(0), ["unknown"] * 99)
    with pytest.raises(ValueError, match="finite"):
        transform_observation_geometry(np.zeros(98), OBSERVATION_NAMES)
    with pytest.raises(ValueError, match="tcp or grasp"):
        observation_geometry_names(OBSERVATION_NAMES, "world")
    values = observation(0)
    values[3:9] = 0.
    with pytest.raises(ValueError, match="rotation6d"):
        transform_observation_geometry(values, OBSERVATION_NAMES)


def test_empty_dataset_batch_keeps_shape_without_inventing_robot_rows():
    result = transform_observation_geometry(np.empty((0, 99), np.float32), OBSERVATION_NAMES)
    assert result.shape == (0, 99) and result.dtype == np.float32
