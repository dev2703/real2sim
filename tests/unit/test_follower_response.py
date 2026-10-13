import numpy as np
import pytest

from real2sim.sim.follower_response import FollowerResponse, fit_joint


def model(**kwargs):
    return FollowerResponse(**{**dict(gain=[.2], inertia=[.4], delay=[0],
                                      lower=[-100.], upper=[100.]), **kwargs})


def test_rollout_is_causal_and_resets_memory():
    m = model(delay=[2])
    a = np.zeros((20, 1))
    b = a.copy()
    b[8:] = 20
    before = m.rollout([0], a)
    changed = m.rollout([0], b)
    np.testing.assert_array_equal(before[:11], changed[:11])
    assert changed[11, 0] > 0
    np.testing.assert_array_equal(m.rollout([0], a), before)


def test_position_and_velocity_are_bounded():
    m = model(gain=[1.], inertia=[.98], lower=[-8.], upper=[8.])
    a = np.repeat([[1e6], [-1e6]], 30, axis=0)
    trajectory = m.rollout([0], a)
    assert np.max(abs(trajectory)) <= 8
    assert np.max(abs(np.diff(trajectory, axis=0))) <= 5


def test_fit_recovers_known_dynamics_without_crossing_episode_boundaries():
    rng = np.random.default_rng(3)
    truth = model(gain=[.16], inertia=[.55])
    episodes = []
    for initial in (-40., 40.):
        actions = np.repeat(rng.uniform(-50, 50, size=(80, 1)), 5, axis=0)
        state = truth.rollout([initial], actions[:-1])
        episodes.append((state, actions))
    gain, inertia = fit_joint(episodes, 0, 0, momentum=True)
    assert gain == pytest.approx(.16, abs=1e-8)
    assert inertia == pytest.approx(.55, abs=1e-8)


@pytest.mark.parametrize("kwargs", [dict(delay=[-1]), dict(delay=[.5]), dict(gain=[1.1]),
                                   dict(inertia=[1.]), dict(lower=[101.]), dict(hz=0)])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        model(**kwargs)


def test_nonfinite_or_outside_inputs_rejected():
    m = model()
    with pytest.raises(ValueError):
        m.rollout([0], [[float("nan")]])
    with pytest.raises(ValueError):
        m.rollout([101], [[0]])
    with pytest.raises(ValueError):
        m.advance([0], [float("inf")], [1])


def test_serialization_preserves_predictions():
    m = model(delay=[3])
    restored = FollowerResponse(**m.to_dict())
    a = np.arange(15.).reshape(-1, 1)
    np.testing.assert_array_equal(m.rollout([0], a), restored.rollout([0], a))
