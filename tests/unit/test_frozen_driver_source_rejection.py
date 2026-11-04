"""Panda rejects one unsafe source command without losing the remaining chunk."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.geometry.so101_fk import ARM_JOINTS, JointMapping
from real2sim.sim.env import real_from_sim, sim_from_real
from scripts import benchmark_frozen_act as benchmark
from scripts.diag_v2_learned_panda_retarget_20261027 import (
    VirtualSource,
    VirtualSourceProjection,
    VirtualSourceProjectionError,
)


@pytest.fixture
def setup(monkeypatch):
    source = VirtualSource.__new__(VirtualSource)
    source.mapping = JointMapping()
    source.limits = np.array([[-1., 1.]] * 5 + [[-.1745, 1.7454]])
    source.q = np.array([0., -.1, .2, -.3, .4, .5])
    source.last_target = source.q.copy()
    source.source = SimpleNamespace(config={"joint_names": [*ARM_JOINTS, "gripper"]})

    class Environment:
        max_opening = .08

        def __init__(self):
            self.arm = np.array([.1, -.2, .3, -2., .1, 1.5, .5])
            self.opening = .031
            self.commands = []

        def arm_qpos(self):
            return self.arm.copy()

        def gripper_opening(self):
            return self.opening

        def ee_pose(self):
            return np.eye(4)

        def step(self, command):
            self.commands.append(command.copy())
            return {}, 0., False, False, {"control": {"limit_clipped": np.zeros(8)}}

    class Guard:
        def __init__(self):
            self.calls = []

        def solve(self, desired, measured, *, finger_opening):
            self.calls.append((desired.copy(), measured.copy(), finger_opening))
            return SimpleNamespace(accepted=True, reason="accepted", qpos=measured + .001)

    env, guard = Environment(), Guard()
    policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50, n_action_steps=10),
                             reset=lambda: None)
    driver = benchmark.FrozenDriver(SimpleNamespace(env=env, source=source, robot="panda",
                                                    guard=guard), policy, execute_steps=50)
    monkeypatch.setattr(benchmark.transfer.retarget, "target_from_source", lambda *_: np.eye(4))
    return driver, env, source, guard


def test_limit_rejection_holds_measured_arm_and_jaw_then_executes_next_valid_action(setup):
    driver, env, source, guard = setup
    measured_source, prior = source.q.copy(), source.last_target.copy()
    invalid_native = source.q.copy()
    invalid_native[2] = 1.2
    invalid_action = real_from_sim(source.mapping, invalid_native)
    invalid_action[5] = 0.  # A rejected arm command must not close the Panda jaw.
    valid_action = np.array([5., -10., 15., -20., 25., 60.])
    driver.queue = [invalid_action.copy(), valid_action.copy()]

    held = driver.step({}, env)
    expected_hold = np.r_[env.arm, env.opening]
    np.testing.assert_array_equal(env.commands[0], expected_hold)
    np.testing.assert_array_equal(held.detail["requested_native"], expected_hold)
    np.testing.assert_array_equal(held.detail["raw_hf_action"], invalid_action)
    assert held.ik_rejected and held.detail["guard_reason"] == "source_target_native_limits"
    assert held.detail["uses_act"] and not held.detail["intervention"]
    details = held.detail["source_command_rejection"]
    assert details["violating_arm_joint_indices"] == [2]
    assert details["violating_arm_joint_names"] == ["elbow_flex"]
    np.testing.assert_array_equal(details["raw_hf_action_degree_percent"], invalid_action)
    np.testing.assert_allclose(details["requested_source_native_rad"],
                               sim_from_real(source.mapping, invalid_action))
    assert guard.calls == []
    np.testing.assert_array_equal(source.q, measured_source)
    np.testing.assert_array_equal(source.last_target, prior)
    assert len(driver.queue) == 1  # Reject a single command, retain the action chunk.

    accepted = driver.step({}, env)
    assert not accepted.ik_rejected
    assert accepted.detail["source_command_rejection"] is None
    assert len(guard.calls) == 1 and not driver.queue
    expected_width = valid_action[5] / 90. * env.max_opening
    np.testing.assert_array_equal(env.commands[1][:7], env.arm + .001)
    assert env.commands[1][7] == pytest.approx(expected_width)
    np.testing.assert_allclose(source.last_target, sim_from_real(source.mapping, valid_action))


def test_nonfinite_action_still_propagates_without_step_or_memory_change(setup):
    driver, env, source, guard = setup
    before = source.last_target.copy()
    driver.queue = [np.full(6, np.nan)]
    with pytest.raises(ValueError, match="finite"):
        driver.step({}, env)
    assert not env.commands and not guard.calls
    np.testing.assert_array_equal(source.last_target, before)


def test_source_projection_error_remains_fatal_before_inference(setup):
    driver, env, source, guard = setup
    before = source.last_target.copy()
    failure = VirtualSourceProjectionError(VirtualSourceProjection(source.q[:5], .01, False))

    def invalid_observation(_):
        raise failure

    source.source_state = invalid_observation
    with pytest.raises(VirtualSourceProjectionError) as caught:
        driver.step({}, env)
    assert caught.value is failure
    assert not env.commands and not guard.calls
    np.testing.assert_array_equal(source.last_target, before)


def test_unrelated_runtime_error_is_not_converted_to_a_hold(setup):
    driver, env, source, guard = setup
    driver.queue = [np.zeros(6)]

    def broken_source_action(_):
        raise RuntimeError("unexpected source failure")

    source.source_action = broken_source_action
    with pytest.raises(RuntimeError, match="unexpected source failure"):
        driver.step({}, env)
    assert not env.commands and not guard.calls
