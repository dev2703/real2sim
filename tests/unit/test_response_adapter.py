import numpy as np
import pytest

from real2sim.evaluation.response_adapter import ApproachResponseAdapter
from real2sim.sim.follower_response import FollowerResponse


class JointOnlyEnv:
    """No object poses, contacts or phase information exposed to adapter."""
    control_hz = 10.

    def __init__(self):
        self.state = np.zeros(6)
        self.sent = []

    def real_state(self):
        return self.state.copy()

    def step_real_joint_targets(self, target):
        self.sent.append(target.copy())
        self.state = target.copy()
        return {}, 0., False, False, {"control": {"limit_clipped": [False] * 6,
                                                   "rate_limited": [False] * 6}}


def setup(active_ticks=2, delay=0):
    env = JointOnlyEnv()
    response = FollowerResponse([.2] * 6, [0.] * 6, [delay] * 6, [-100.] * 6, [100.] * 6)
    adapter = ApproachResponseAdapter(env, response, active_ticks=active_ticks)
    adapter.reset()
    return env, adapter


def test_three_raw_ticks_match_offline_response_without_privileged_inputs():
    env, adapter = setup()
    action = np.ones(6) * 10
    expected = adapter.model.rollout(env.state, np.repeat(action[None], 3, axis=0))[-1]
    adapter.apply(action)
    np.testing.assert_allclose(env.sent[0], expected)


def test_horizon_falls_back_exactly_and_reset_clears_history():
    env, adapter = setup(active_ticks=1)
    adapter.apply(np.ones(6) * 10)
    adapter.apply(np.ones(6) * -20)
    np.testing.assert_array_equal(env.sent[-1], np.ones(6) * -20)
    assert [r["response_active"] for r in adapter.trace] == [True, False]
    env.state[:] = 0
    adapter.reset()
    adapter.apply(np.ones(6) * 10)
    np.testing.assert_array_equal(env.sent[0], env.sent[-1])


def test_measured_state_feedback_overrides_predicted_path():
    env, adapter = setup()
    adapter.apply(np.ones(6) * 10)
    env.state[:] = -30  # external discrepancy/contact, not hidden reference trajectory
    adapter.apply(np.ones(6) * 10)
    assert np.all(env.sent[-1] < -25)
    np.testing.assert_array_equal(adapter.trace[-1]["observed_state"], np.ones(6) * -30)


def test_delay_uses_only_past_executed_commands():
    env, adapter = setup(active_ticks=10, delay=4)
    adapter.apply(np.zeros(6))
    adapter.apply(np.ones(6) * 10)
    np.testing.assert_array_equal(env.sent[-1], np.zeros(6))
    adapter.apply(np.ones(6) * 10)
    assert np.all(env.sent[-1] > 0)


def test_invalid_input_cannot_advance_environment():
    env, adapter = setup()
    with pytest.raises(ValueError):
        adapter.apply(np.full(6, np.nan))
    assert len(env.sent) == 0 and adapter.tick == 0


def test_nonintegral_clock_rejected():
    env, adapter = setup()
    env.control_hz = 11.
    with pytest.raises(ValueError):
        ApproachResponseAdapter(env, adapter.model, active_ticks=10)
