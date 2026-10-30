"""Learned sequence context constrains emissions without an oracle phase graph."""

import inspect
import json

import numpy as np
import pytest

from real2sim.policy.phase_sequence_decoder import (
    LearnedPhaseSequenceDecoder,
    fit_phase_sequence_decoder,
)


def fit(labels, **kwargs):
    n = len(labels)
    return fit_phase_sequence_decoder(np.zeros(n, np.int64), np.zeros(n, np.int64),
        np.zeros(n, np.int64), np.arange(n), np.asarray(labels), n_phases=5, **kwargs)


def test_learned_support_blocks_an_unobserved_skip_and_allows_seen_progression():
    metadata = fit([1, 1, 2, 2, 3, 3, 4])
    decoder = LearnedPhaseSequenceDecoder(metadata)
    assert list(inspect.signature(decoder.step).parameters) == ["emissions"]
    decoder.step([0., 1., 0., 0., 0.])
    posterior = decoder.step([0., 0., 0., 0., 1.])
    assert posterior[4] == 0.  # No direct training edge from PRE_GRASP to RETREAT.
    assert decoder.last_phase_id == 1
    decoder.step([0., 0., 1., 0., 0.])
    assert decoder.last_phase_id == 2
    assert decoder.last_diagnostics["teacher_phase_is_runtime_input"] is False


def test_a_recorded_retry_backedge_is_not_replaced_by_a_monotonic_rule():
    decoder = LearnedPhaseSequenceDecoder(fit([1, 2, 3, 1, 1, 2]))
    for phase in [1, 2, 3, 1]:
        emissions = np.zeros(5)
        emissions[phase] = 1.
        decoder.step(emissions)
        assert decoder.last_phase_id == phase
    assert decoder.transition[3, 1] == 1.


def test_initial_prior_uses_actual_tick_zero_and_edges_never_cross_rollouts():
    metadata = fit_phase_sequence_decoder(
        np.array([0, 0, 0, 1, 0, 0]), np.array([0, 0, 0, 0, 1, 0]),
        np.array([10, 10, 11, 10, 10, 10]), np.array([0, 1, 0, 1, 2, 3]),
        np.array([1, 2, 3, 4, 0, 4]), n_phases=5)
    assert metadata["consecutive_pairs"] == 1
    counts = np.asarray(metadata["transition_counts"])
    assert counts.sum() == 1 and counts[1, 2] == 1
    np.testing.assert_array_equal(metadata["initial_counts"], [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(metadata["initial_prior"], [0., .5, 0., .5, 0.])
    assert metadata["tick_zero_rows"] == 2


def test_unseen_rows_are_absorbing_and_pseudocounts_do_not_invent_edges():
    metadata = fit([1, 1, 2, 2], transition_pseudocount=.5, initial_pseudocount=.5)
    transition = np.asarray(metadata["transition_matrix"])
    np.testing.assert_array_equal(transition[0], [1., 0., 0., 0., 0.])
    assert transition[1, 3] == 0. and transition[1, 4] == 0.
    np.testing.assert_allclose(transition.sum(-1), 1.)
    assert metadata["initial_prior"] == [0., 1., 0., 0., 0.]
    restored = LearnedPhaseSequenceDecoder(json.loads(json.dumps(metadata)))
    assert restored.n_phases == 5


def test_episode_reset_clears_causal_context_and_callers_cannot_rewrite_it():
    decoder = LearnedPhaseSequenceDecoder(fit([1, 2, 3]))
    decoder.step([0., 1., 0., 0., 0.])
    decoder.step([0., 0., 1., 0., 0.])
    assert decoder.last_phase_id == 2
    decoder.posterior[:] = 99.
    assert decoder.posterior.sum() == pytest.approx(1.)
    decoder.reset()
    assert decoder.last_phase_id is None and decoder.steps == 0
    decoder.step([0., 0., 0., 1., 0.])
    assert decoder.last_phase_id == 1


def test_zero_emissions_on_reachable_support_remain_finite_and_do_not_create_edges():
    decoder = LearnedPhaseSequenceDecoder(fit([1, 1, 2, 2]), emission_power=100.)
    for _ in range(50):
        posterior = decoder.step([0., 0., 0., 0., 1.])
        assert np.isfinite(posterior).all() and posterior.sum() == pytest.approx(1.)
        assert posterior[4] == 0.
    sharp = LearnedPhaseSequenceDecoder(fit([1, 1, 2, 2]), emission_power=1e308)
    for _ in range(4):
        posterior = sharp.step([1e308] * 5)
        assert np.isfinite(posterior).all() and posterior.sum() == pytest.approx(1.)


def test_missing_initial_rows_and_duplicate_row_keys_are_rejected():
    values = np.zeros(2, np.int64)
    with pytest.raises(ValueError, match="tick-zero"):
        fit_phase_sequence_decoder(values, values, values, np.array([1, 2]),
                                   np.array([1, 2]), n_phases=5)
    with pytest.raises(ValueError, match="Duplicate"):
        fit_phase_sequence_decoder(values, values, values, values, np.array([1, 2]), n_phases=5)


def test_invalid_saved_probabilities_and_runtime_emissions_fail_closed():
    metadata = fit([1, 2, 2])
    metadata["transition_matrix"][1][2] = .7
    with pytest.raises(ValueError, match="stochastic"):
        LearnedPhaseSequenceDecoder(metadata)
    decoder = LearnedPhaseSequenceDecoder(fit([1, 2, 2]))
    for emissions in ([0.] * 5, [np.nan] * 5, [-1.] * 5, [1.] * 4):
        with pytest.raises(ValueError, match="neural phase distribution"):
            decoder.step(emissions)
