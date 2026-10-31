import numpy as np
import pytest

from real2sim.policy.temporal_ensemble import BoundedTemporalEnsembler


def test_overlapping_predictions_use_same_absolute_action_tick():
    ensemble = BoundedTemporalEnsembler(4, coefficient=0)
    ensemble.add_chunk(np.array([[0.], [10.], [20.], [30.]]), start_tick=0)
    np.testing.assert_array_equal(ensemble.action_at(tick=0), [0.])
    # A delayed second inference still starts at observation tick 1, not execution tick 2.
    ensemble.add_chunk(np.array([[100.], [200.], [300.], [400.]]), start_tick=1)
    np.testing.assert_allclose(ensemble.action_at(tick=2), [110.])
    assert ensemble.last_selection["offsets"] == [2, 1]
    assert ensemble.last_selection["start_ticks"] == [0, 1]


@pytest.mark.parametrize("coefficient", [0., .01, 1., -1., 1e308, -1e308])
def test_oldest_weighting_and_numerical_stability(coefficient):
    ensemble = BoundedTemporalEnsembler(3, coefficient)
    for tick, value in enumerate([10., 20., 30.]):
        ensemble.add_chunk(np.full((3, 1), value), start_tick=tick)
    action = ensemble.action_at(tick=2)[0]
    if coefficient == 0:
        assert action == pytest.approx(20.)
    elif coefficient > 0:
        assert 10. <= action < 20.
    else:
        assert 20. < action <= 30.
    assert np.isfinite(action)
    assert sum(ensemble.last_selection["weights"]) == pytest.approx(1.)


def test_supervised_prefix_blocks_tail_and_expired_actions():
    ensemble = BoundedTemporalEnsembler(4, 0., supervised_prefix_steps=2)
    chunk = np.array([[1.], [2.], [1e30], [-1e30]])
    ensemble.add_chunk(chunk, start_tick=0)
    np.testing.assert_array_equal(ensemble.action_at(tick=0), [1.])
    ensemble.add_chunk(np.array([[3.], [4.], [-1e30], [1e30]]), start_tick=1)
    np.testing.assert_allclose(ensemble.action_at(tick=1), [2.5])
    np.testing.assert_array_equal(ensemble.action_at(tick=2), [4.])
    with pytest.raises(ValueError, match="no supervised prediction"):
        ensemble.action_at(tick=3)
    assert ensemble.pending_chunks == 0


def test_reset_drops_episode_history_and_dimension():
    ensemble = BoundedTemporalEnsembler(2, 0.)
    ensemble.add_chunk(np.ones((2, 1)), start_tick=8)
    ensemble.action_at(tick=8)
    ensemble.reset()
    assert ensemble.pending_chunks == 0 and ensemble.last_selection is None
    with pytest.raises(ValueError, match="no supervised prediction"):
        ensemble.action_at(tick=0)
    ensemble.add_chunk(np.full((2, 3), 5.), start_tick=0)
    np.testing.assert_array_equal(ensemble.action_at(tick=0), [5., 5., 5.])


def test_missing_or_repeated_ticks_never_reuse_stale_action():
    ensemble = BoundedTemporalEnsembler(2, 0.)
    ensemble.add_chunk(np.ones((2, 1)), start_tick=4)
    with pytest.raises(ValueError, match="precedes"):
        ensemble.action_at(tick=3)
    ensemble.action_at(tick=4)
    with pytest.raises(ValueError, match="increase strictly"):
        ensemble.action_at(tick=4)
    with pytest.raises(ValueError, match="no supervised prediction"):
        ensemble.action_at(tick=6)
    # Failure did not consume tick 6; fresh inference can cover it.
    ensemble.add_chunk(np.full((2, 1), 7.), start_tick=6)
    np.testing.assert_array_equal(ensemble.action_at(tick=6), [7.])


def test_rejected_chunks_leave_valid_history_and_clock_unchanged():
    ensemble = BoundedTemporalEnsembler(2, 0.)
    ensemble.add_chunk(np.ones((2, 1)), start_tick=0)
    for bad in [np.full((2, 1), np.nan), np.full((2, 1), np.inf), np.ones((2, 2)),
                np.ones((3, 1)), np.ones((1, 2, 1))]:
        with pytest.raises(ValueError):
            ensemble.add_chunk(bad, start_tick=1)
    assert ensemble.pending_chunks == 1
    ensemble.add_chunk(np.full((2, 1), 3.), start_tick=1)
    np.testing.assert_array_equal(ensemble.action_at(tick=1), [2.])


def test_duplicate_and_out_of_order_chunks_are_rejected():
    ensemble = BoundedTemporalEnsembler(5, 0., supervised_prefix_steps=2)
    ensemble.add_chunk(np.zeros((5, 1)), start_tick=0)
    ensemble.add_chunk(np.zeros((5, 1)), start_tick=3)
    ensemble.action_at(tick=4)
    for tick in [0, 3]:
        with pytest.raises(ValueError, match="increase strictly"):
            ensemble.add_chunk(np.ones((5, 1)), start_tick=tick)
    # A later valid observation can continue from its correctly aligned prefix.
    ensemble.add_chunk(np.zeros((5, 1)), start_tick=4)
    ensemble.action_at(tick=5)


def test_memory_is_bounded_and_input_is_copied():
    ensemble = BoundedTemporalEnsembler(25, 0., supervised_prefix_steps=2)
    chunk = np.ones((25, 1))
    for tick in range(100):
        ensemble.add_chunk(chunk, start_tick=tick)
        assert ensemble.pending_chunks <= 2
    chunk[:] = 123.
    np.testing.assert_array_equal(ensemble.action_at(tick=99), [1.])


def test_extreme_finite_values_stay_finite():
    ensemble = BoundedTemporalEnsembler(2, 0.)
    values = np.array([[1e308, -1e308], [1e308, -1e308]])
    ensemble.add_chunk(values, start_tick=0)
    ensemble.add_chunk(values, start_tick=1)
    np.testing.assert_array_equal(ensemble.action_at(tick=1), [1e308, -1e308])


@pytest.mark.parametrize("kwargs", [
    {"chunk_size": True, "coefficient": 0.},
    {"chunk_size": 0, "coefficient": 0.},
    {"chunk_size": 2, "coefficient": np.nan},
    {"chunk_size": 2, "coefficient": np.inf},
    {"chunk_size": 2, "coefficient": True},
    {"chunk_size": 2, "coefficient": "0.01"},
    {"chunk_size": 2, "coefficient": 0., "supervised_prefix_steps": 3},
    {"chunk_size": 2, "coefficient": 0., "supervised_prefix_steps": 0},
    {"chunk_size": 2, "coefficient": 0., "supervised_prefix_steps": 1.5},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BoundedTemporalEnsembler(**kwargs)


@pytest.mark.parametrize("tick", [-1, True, 1.5])
def test_nonintegral_or_negative_clock_is_rejected(tick):
    ensemble = BoundedTemporalEnsembler(2, 0.)
    with pytest.raises(ValueError):
        ensemble.add_chunk(np.zeros((2, 1)), start_tick=tick)
    with pytest.raises(ValueError):
        ensemble.action_at(tick=tick)
