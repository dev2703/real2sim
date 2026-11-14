"""Out-of-limit ACT commands hold measured joints instead of aborting the episode."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.geometry.so101_fk import ARM_JOINTS, JointMapping
from real2sim.sim.env import real_from_sim, sim_from_real
from scripts.diag_v2_learned_panda_retarget_20261027 import (
    VirtualSource,
    VirtualSourceProjectionError,
)
from scripts.evaluate_hf_act_panda_20261028 import source_action_or_hold


@pytest.fixture
def source():
    value = VirtualSource.__new__(VirtualSource)
    value.mapping = JointMapping()
    value.limits = np.array([[-1., 1.]] * 5 + [[-.1745, 1.7454]])
    value.q = np.array([0., -.1, .2, -.3, .4, .5])
    value.last_target = value.q.copy()
    value.source = SimpleNamespace(config={"joint_names": [*ARM_JOINTS, "gripper"]})
    return value


def test_out_of_limit_command_holds_measured_joints_and_keeps_source_memory(source):
    measured, previous = source.q.copy(), source.last_target.copy()
    invalid = measured.copy()
    invalid[2] = 1.2
    action = real_from_sim(source.mapping, invalid)
    action[5] = 0.
    hold = np.array([.1, -.2, .3, -2., .1, 1.5, .5, .031])

    decision = source_action_or_hold(source, action, hold)

    assert not decision.accepted
    assert decision.reason == "source_target_native_limits"
    np.testing.assert_array_equal(decision.command, hold)
    assert decision.command is not hold
    assert decision.rejection["violating_arm_joint_indices"] == [2]
    assert decision.rejection["violating_arm_joint_names"] == ["elbow_flex"]
    np.testing.assert_array_equal(decision.rejection["raw_hf_action_degree_percent"], action)
    np.testing.assert_allclose(decision.requested, sim_from_real(source.mapping, action))
    np.testing.assert_array_equal(source.q, measured)
    np.testing.assert_array_equal(source.last_target, previous)


def test_in_limit_command_is_applied_and_returns_no_hold(source):
    action = np.array([10., -20., 30., -40., 50., 60.])
    decision = source_action_or_hold(source, action, np.zeros(8))
    assert decision.accepted and decision.reason is None and decision.rejection is None
    assert decision.command is None
    np.testing.assert_array_equal(source.last_target, decision.bounded)


def test_unrelated_runtime_error_propagates(source):
    previous = source.last_target.copy()

    def broken(_action):
        raise RuntimeError("unexpected source failure")

    source.source_action = broken
    with pytest.raises(RuntimeError, match="unexpected source failure"):
        source_action_or_hold(source, np.zeros(6), np.zeros(8))
    np.testing.assert_array_equal(source.last_target, previous)


def test_projection_error_is_not_converted_to_a_hold(source):
    previous = source.last_target.copy()
    failure = VirtualSourceProjectionError(SimpleNamespace(error_m=.01))

    def broken(_action):
        raise failure

    source.source_action = broken
    with pytest.raises(VirtualSourceProjectionError) as caught:
        source_action_or_hold(source, np.zeros(6), np.zeros(8))
    assert caught.value is failure
    np.testing.assert_array_equal(source.last_target, previous)


def test_nonfinite_action_propagates(source):
    previous = source.last_target.copy()
    with pytest.raises(ValueError, match="finite"):
        source_action_or_hold(source, np.full(6, np.nan), np.zeros(8))
    np.testing.assert_array_equal(source.last_target, previous)
