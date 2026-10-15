import numpy as np
import pytest

from real2sim.policy.observation_variants import (
    KINDS,
    ObservationVariant,
    StreamingObservationTransform,
    fit_state_statistics,
    state_names,
    transform_observations,
)


def observations():
    rng = np.random.default_rng(94)
    states = rng.normal(size=(9, 13)).astype(np.float32)
    states[[0, 5], 9:12] = 0
    return {"state": states, "joint_state": rng.normal(size=(9, 6)).astype(np.float32),
            "episode_index": np.array([0] * 5 + [1] * 4),
            "timestamp": np.r_[np.arange(5), np.arange(4)] / 10,
            "action": rng.normal(size=(9, 7)).astype(np.float32)}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("history", [1, 3])
def test_offline_online_equivalence_and_future_independence(kind, history):
    arrays = observations()
    cfg = ObservationVariant(kind, history=history)
    expected = transform_observations(arrays, cfg)
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["state"][3:] += 100
    changed["joint_state"][3:] -= 20
    changed["action"][:] += 800
    np.testing.assert_array_equal(expected[:3], transform_observations(changed, cfg)[:3])
    online = StreamingObservationTransform(cfg.to_dict())
    actual = []
    for i in range(len(arrays["state"])):
        if i in (0, 5):
            online.reset()
        actual.append(online.update(arrays["state"][i], arrays["joint_state"][i]))
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert expected.shape[1] == len(state_names(cfg))


def test_exact_channel_semantics_and_inputs_untouched():
    arrays = observations()
    original = {k: v.copy() for k, v in arrays.items()}
    np.testing.assert_array_equal(transform_observations(arrays, ObservationVariant()), arrays["state"])
    no_vel = transform_observations(arrays, ObservationVariant("no_velocity"))
    np.testing.assert_array_equal(no_vel, arrays["state"][:, [*range(9), 12]])
    joints = transform_observations(arrays, ObservationVariant("joints"))
    np.testing.assert_array_equal(joints[:, :5], arrays["joint_state"][:, :5])
    np.testing.assert_allclose(joints[:, 5], arrays["joint_state"][:, 5] / 100)
    both = transform_observations(arrays, ObservationVariant("canonical_joints"))
    np.testing.assert_array_equal(both[:, :13], arrays["state"])
    np.testing.assert_array_equal(both[:, 13:], arrays["joint_state"][:, :5])
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], original[key])


def test_ema_resets_and_preserves_gripper_events():
    arrays = observations()
    arrays["state"][:, 9:12] = 4
    arrays["state"][:, 12] = [1, 1, .3, .3, 0, 1, .4, .4, 0]
    result = transform_observations(arrays, ObservationVariant("ema_velocity", ema_alpha=.5))
    np.testing.assert_allclose(result[:, 9], [0, 2, 3, 3.5, 3.75, 0, 2, 3, 3.5])
    np.testing.assert_array_equal(result[:, :9], arrays["state"][:, :9])
    np.testing.assert_array_equal(result[:, 12], arrays["state"][:, 12])


def test_history_masks_ages_and_boundary_values():
    arrays = observations()
    result = transform_observations(arrays, ObservationVariant(history=3)).reshape(9, 3, 15)
    np.testing.assert_array_equal(result[0, :, -1], [0, 0, 1])
    np.testing.assert_array_equal(result[1, :, -1], [0, 1, 1])
    np.testing.assert_array_equal(result[5, :, -1], [0, 0, 1])
    np.testing.assert_allclose(result[2, :, -2], [.2, .1, 0])
    np.testing.assert_array_equal(result[2, :, :13], arrays["state"][:3])
    np.testing.assert_array_equal(result[5, 0, :13], arrays["state"][5])
    np.testing.assert_array_equal(result[5, 1, :13], arrays["state"][5])


@pytest.mark.parametrize("history", [1, 3])
def test_normalization_is_train_only_and_history_metadata_remains_unscaled(history):
    arrays = observations()
    cfg = ObservationVariant(history=history)
    transformed = transform_observations(arrays, cfg)
    stats = fit_state_statistics(transformed, arrays["episode_index"], [0], cfg)
    transformed[5:] += 1000
    assert stats == fit_state_statistics(transformed, arrays["episode_index"], [0], cfg)
    expected_mean = arrays["state"][:5].astype(np.float64).mean(axis=0)
    if history == 1:
        np.testing.assert_allclose(stats["mean"], expected_mean)
    else:
        np.testing.assert_allclose(np.array(stats["mean"]).reshape(3, 15)[:, :13], np.tile(expected_mean, (3, 1)))
        np.testing.assert_array_equal(np.array(stats["mean"]).reshape(3, 15)[:, -2:], 0)
        np.testing.assert_array_equal(np.array(stats["std"]).reshape(3, 15)[:, -2:], 1)
    assert stats["count"] == 5


def test_invalid_configs_and_streams_fail_explicitly():
    for kw in ({"kind": "bad"}, {"history": 0}, {"history": 1.2}, {"ema_alpha": 0}, {"policy_hz": 0}):
        with pytest.raises(ValueError):
            ObservationVariant(**kw)
    stream = StreamingObservationTransform()
    stream.update(np.zeros(13), timestamp=3)
    with pytest.raises(ValueError, match="timestamps"):
        stream.update(np.zeros(13), timestamp=2)
    with pytest.raises(ValueError, match="joint_state"):
        StreamingObservationTransform({"kind": "joints"}).update(np.zeros(13))
    arrays = observations()
    arrays["episode_index"][-1] = 0
    with pytest.raises(ValueError, match="contiguous"):
        transform_observations(arrays, ObservationVariant())
