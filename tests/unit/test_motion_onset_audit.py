"""Causal onset descriptions and future-tail comparisons keep real tick offsets."""

import importlib.util
from pathlib import Path

import numpy as np


def diagnostic():
    path = Path(__file__).resolve().parents[2] / "scripts/audit_v2_motion_onsets.py"
    spec = importlib.util.spec_from_file_location("onset_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example():
    n = 12
    arrays = {"episode_index": np.array([11] * 8 + [99] * 4),
              "frame_index": np.arange(n) * 2, "state": np.zeros((n, 13), np.float32),
              "joint_state": np.zeros((n, 6), np.float32),
              "action": np.zeros((n, 7), np.float32), "label_is_extrapolated": np.zeros(n, bool)}
    arrays["label_is_extrapolated"][[7, 11]] = True
    arrays["action"][4, 0] = .004
    arrays["action"][8:, 0] = np.nan  # unaudited episode cannot enter statistics
    images = np.zeros((n, 4, 5, 3), np.uint8)
    images[4] = 1
    return arrays, images


def test_next_motion_is_outcome_after_causal_hold_and_future_tail_is_separate():
    module = diagnostic()
    arrays, images = example()
    result = module.audit_episode(arrays, images, 11)
    assert result["valid_ticks"] == 7
    assert result["causal_low_motion_history_followed_by_moving_label"] == 1
    assert result["onsets"][0]["tick"] == 4
    assert result["onsets"][0]["causal_history_ticks"] == [1, 2, 3, 4]
    assert result["stationary_two_step_prefixes_with_motion_in_chunk_tail"] == 3
    assert result["exact_state_conflict_pairs"][0]["ticks"] == [4, 6]
    assert result["exact_state_conflict_pairs"][0]["current_image_mae_uint8"] == 1


def test_causal_history_rejects_prior_motion_without_reading_future_as_input():
    module = diagnostic()
    arrays, images = example()
    arrays["state"][2, 9] = .01
    result = module.audit_episode(arrays, images, 11)
    assert result["causal_low_motion_history_followed_by_moving_label"] == 0
    # A later achieved motion is irrelevant to tick4's known history.
    arrays["state"][2, 9] = 0
    arrays["state"][5, 9] = .1
    assert module.audit_episode(arrays, images, 11)["onsets"][0]["tick"] == 4


def test_pair_comparison_keeps_offsets_and_never_crosses_episode_or_invalid_labels():
    module = diagnostic()
    arrays, images = example()
    arrays["action_is_invalid"] = np.zeros(12, bool)
    arrays["action_is_invalid"][5] = True
    valid = module.valid_mask(arrays)
    pair = module.pair_metrics(arrays, images, 4, 6, valid, module.RULES)
    assert pair["prefix_valid_offsets"] == 1
    assert pair["chunk_valid_offsets"] == 1
    assert pair["first_action_translation_difference_mm"] == 4


def test_near_state_differences_are_explicit_and_image_similarity_is_descriptive():
    module = diagnostic()
    arrays, images = example()
    arrays["joint_state"][4, 0] = .1
    result = module.audit_episode(arrays, images, 11)
    near = result["adjacent_near_state_conflict_pairs"]
    assert len(near) == 2
    assert all(p["similar_current_image"] for p in near)
    arrays["joint_state"][4, 0] = 2
    assert not module.audit_episode(arrays, images, 11)["adjacent_near_state_conflict_pairs"]
