"""Causal history and descriptive future-label residual partitions stay separate."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit():
    path = Path(__file__).resolve().parents[2] / "scripts/audit_v2_history_context.py"
    spec = importlib.util.spec_from_file_location("history_context_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arrays(length=6, episodes=1):
    n = length * episodes
    return {"state": np.zeros((n, 13), np.float32),
            "joint_state": np.zeros((n, 6), np.float32),
            "action": np.zeros((n, 7), np.float32),
            "episode_index": np.repeat(np.arange(episodes), length),
            "ep_start": np.repeat(np.arange(episodes) * length, length),
            "ep_len": np.full(n, length), "frame_index": np.tile(np.arange(length), episodes),
            "label_is_extrapolated": np.zeros(n, bool)}


def test_history_is_oldest_first_and_future_observations_cannot_change_it(audit):
    state = np.zeros((5, 13), np.float32)
    state[:, 0] = np.arange(5)
    history = audit.causal_state_history(state, 4).reshape(5, 4, 13)
    np.testing.assert_array_equal(history[0, :, 0], [0, 0, 0, 0])
    np.testing.assert_array_equal(history[2, :, 0], [0, 0, 1, 2])
    changed = state.copy()
    changed[3:] = 1000
    np.testing.assert_array_equal(audit.causal_state_history(changed, 4)[:3], history.reshape(5, -1)[:3])
    # A new episode is supplied independently; no previous episode can fill its slots.
    np.testing.assert_array_equal(audit.causal_state_history(state[4:], 4)[0, ::13], [4, 4, 4, 4])


def test_past_observations_resolve_a_current_state_collision_without_reading_labels(audit):
    source = arrays(4)
    source["state"][:, 0] = [0, 1, 0, 2]
    source["action"][2, 0] = .001
    result = audit.history_conflicts(source, 0, histories=(1, 2))
    assert [r["label_conflict_groups"] for r in result["histories"]] == [1, 0]
    features = audit.causal_state_history(source["state"], 2)
    source["action"][:] = 100
    np.testing.assert_array_equal(features, audit.causal_state_history(source["state"], 2))


def test_terminal_labels_cannot_create_a_history_conflict(audit):
    source = arrays(4)
    source["state"][:, 0] = [0, 1, 0, 2]
    source["action"][2, 0] = .01
    source["label_is_extrapolated"][2] = True
    result = audit.history_conflicts(source, 0, histories=(1,))
    assert result["histories"][0]["label_conflict_groups"] == 0
    assert result["histories"][0]["eligible_target_ticks"] == 3


def test_future_labels_change_only_descriptive_strata_and_never_cross_episodes(audit):
    source = arrays(6, 2)
    ticks = np.array([3, 4, 5])
    before = audit.stationary_strata(source, ticks)
    source["action"][5, 0] = .001
    after = audit.stationary_strata(source, ticks)
    np.testing.assert_array_equal(before["stationary_causal_low_motion_history"][:2],
                                  after["stationary_causal_low_motion_history"][:2])
    assert not before["stationary_near_future_label_motion"][0]
    assert after["stationary_near_future_label_motion"][0]
    source["action"][:] = 0
    source["action"][6, 0] = .01
    boundaries = audit.stationary_strata(source, ticks)
    assert not boundaries["stationary_near_future_label_motion"].any()


def test_residual_statistics_use_physical_units_and_missing_strata_are_explicit(audit):
    predicted, labels = np.zeros((2, 7)), np.zeros((2, 7))
    predicted[0, 0] = .001
    result = audit.residual_statistics(predicted, labels, [True, False])
    assert result["predicted_step_mean_mm"] == result["target_error_mean_mm"] == pytest.approx(1.)
    assert result["error_vector_sum_xyz_mm"] == [1., 0., 0.]
    empty = audit.residual_statistics(predicted, labels, [False, False])
    assert empty["n"] == 0 and empty["predicted_step_mean_mm"] is None
    json.dumps(empty, allow_nan=False)
