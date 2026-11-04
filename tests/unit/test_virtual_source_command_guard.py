"""Invalid source actions remain inspectable without corrupting causal memory."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.geometry.so101_fk import ARM_JOINTS, JointMapping
from real2sim.sim.env import real_from_sim
from scripts.diag_v2_learned_panda_retarget_20261027 import (
    VirtualSource,
    VirtualSourceCommandError,
)


@pytest.fixture
def source():
    value = VirtualSource.__new__(VirtualSource)
    value.mapping = JointMapping()
    value.limits = np.array([[-1., 1.]] * 5 + [[-.1745, 1.7454]])
    value.q = np.array([0., -.1, .2, -.3, .4, .5])
    value.last_target = value.q.copy()
    value.source = SimpleNamespace(config={"joint_names": [*ARM_JOINTS, "gripper"]})
    return value


@pytest.mark.parametrize("joint", range(5))
@pytest.mark.parametrize("direction", [-1, 1])
def test_rejected_arm_target_exposes_unclipped_command_and_preserves_memory(
        source, joint, direction):
    measured, previous, limits = source.q.copy(), source.last_target.copy(), source.limits.copy()
    requested_native = measured.copy()
    requested_native[joint] = direction * 1.2
    action = real_from_sim(source.mapping, requested_native)
    with pytest.raises(VirtualSourceCommandError, match="unchanged SO-101 limits") as caught:
        source.source_action(action)
    error = caught.value
    assert isinstance(error, RuntimeError)  # Existing strict callers still abort.
    details = error.diagnostics()
    assert details["reason"] == "source_target_native_limits"
    assert details["violating_arm_joint_indices"] == [joint]
    assert details["violating_arm_joint_names"] == [ARM_JOINTS[joint]]
    np.testing.assert_array_equal(details["raw_hf_action_degree_percent"], action)
    np.testing.assert_allclose(details["requested_source_native_rad"], requested_native)
    np.testing.assert_allclose(details["gripper_bounded_source_native_rad"], requested_native)
    np.testing.assert_array_equal(source.q, measured)
    np.testing.assert_array_equal(source.last_target, previous)
    np.testing.assert_array_equal(source.limits, limits)
    action[:] = 0.
    source.limits[:] = 0.
    assert error.action[joint] != 0.  # Error evidence owns copies of its inputs.
    np.testing.assert_array_equal(error.limits, limits)


def test_valid_arm_target_and_saturated_jaw_preserve_existing_action_semantics(source):
    before = source.q.copy()
    action = np.array([10., -20., 30., -40., 50., 120.])
    requested, accepted = source.source_action(action)
    np.testing.assert_allclose(requested[:5], np.deg2rad(action[:5]))
    np.testing.assert_array_equal(accepted[:5], requested[:5])
    assert requested[5] > source.limits[5, 1]
    assert accepted[5] == source.limits[5, 1]
    np.testing.assert_array_equal(source.last_target, accepted)
    np.testing.assert_array_equal(source.q, before)


def test_nonfinite_source_action_remains_a_fatal_input_error(source):
    previous = source.last_target.copy()
    with pytest.raises(ValueError, match="finite"):
        source.source_action(np.full(6, np.nan))
    np.testing.assert_array_equal(source.last_target, previous)
