"""Matched probes must isolate inputs without leaking reserved or future labels."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("matched_actions", Path("scripts/diag_v2_matched_object_actions.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def synthetic():
    episodes = [7, 8, 9, 11, 13, 22, 23, 40]
    size = 12
    arrays = {
        "episode_index": np.repeat(episodes, size),
        "joint_state": np.zeros((len(episodes) * size, 6)),
        "state": np.zeros((len(episodes) * size, 13)),
        "label_is_extrapolated": np.tile([False] * (size - 1) + [True], len(episodes)),
    }
    split = {"train": [7, 8, 11, 23], "val": [9, 13, 22], "reserved": [40],
             "conditions": {str(e): "ball_shift" if e in (8, 9) else "bowl_shift" if e in (22, 23) else "baseline" for e in episodes}}
    return arrays, split


def test_selection_never_uses_targets_or_reserved_and_keeps_partitions_separate():
    arrays, split = synthetic()
    pairs, support = module.select_pairs(arrays, split)
    assert len(pairs) == 6
    for pair in pairs:
        assert pair["anchor_episode"] in split[pair["partition"]]
        assert pair["donor_episode"] in split[pair["partition"]]
        assert pair["context"] == "stationary_start"
    assert sum(row["selected_pairs"] for row in support) == len(pairs)
    arrays["action"] = np.full((len(arrays["state"]), 7), np.nan)
    assert module.select_pairs(arrays, split)[0] == pairs
    split["val"].append(40)
    with pytest.raises(ValueError, match="overlap"):
        module.select_pairs(arrays, split)


def test_stationarity_requires_causal_history_and_real_action_successors():
    arrays, _ = synthetic()
    selected = module.candidate_ticks(arrays, 7, "stationary_start")
    assert selected.tolist() == [4, 6, 8]
    arrays["joint_state"][3, 0] = 1.
    assert module.candidate_ticks(arrays, 7, "stationary_start").tolist() == [8]
    arrays["state"][8, 9] = .1
    assert not len(module.candidate_ticks(arrays, 7, "stationary_start"))


def test_joint_and_velocity_mismatch_cannot_be_hidden_by_small_mean_difference():
    arrays, split = synthetic()
    arrays["joint_state"][arrays["episode_index"] == 8, 4] = 6.
    pairs, _ = module.select_pairs(arrays, split)
    assert not any(p["donor_episode"] == 8 for p in pairs)
    arrays["state"][:, 9] = .05
    arrays["state"][arrays["episode_index"] == 23, 9] = .1
    pairs, _ = module.select_pairs(arrays, split)
    assert not any(p["donor_episode"] == 23 for p in pairs)


def test_object_swap_changes_union_region_only_and_does_not_mutate_sources():
    a = np.zeros((12, 12, 3), np.uint8)
    b = np.ones_like(a) * 255
    ma, mb = np.zeros((12, 12), np.uint8), np.zeros((12, 12), np.uint8)
    ma[2, 2], mb[7, 7] = 1, 1
    output, mask = module.object_composite(a, b, ma, mb, 1)
    assert mask.sum() == 18
    assert np.all(output[mask] == 255) and np.all(output[~mask] == 0)
    assert not np.any(a)


def test_direction_and_amplitude_require_informative_target_contrast():
    a, b = np.zeros((2, 7)), np.zeros((2, 7))
    b[:, 0] = .002
    result = module.contrast_metrics(a, b, a, b * .1, b * .2)
    assert result["informative_target_contrast"]
    assert result["fixed_state_swap_effect"]["target_contrast_amplitude_ratio"] == pytest.approx(.2)
    assert result["fixed_state_swap_effect"]["target_contrast_cosine"] == pytest.approx(1.)
    opposite = module.contrast_metrics(a, b, a, b, -b)
    assert opposite["fixed_state_swap_effect"]["target_contrast_cosine"] == pytest.approx(-1.)
    zero = module.contrast_metrics(a, a, a, b, b)
    assert zero["fixed_state_swap_effect"]["target_contrast_amplitude_ratio"] is None
    assert zero["fixed_state_swap_effect"]["target_contrast_cosine"] is None


def test_missing_object_and_small_displacement_report_unsupported():
    pair = {"object": "bowl", "anchor_tick": 1, "donor_tick": 2}
    frames = {1: {"objects": {}}, 2: {"objects": {}}}
    assert module.object_support(pair, frames)["reason"] == "object_missing"
    for tick in frames:
        frames[tick]["objects"]["bowl"] = {"center_xy": [tick * 2., 1.], "mask_area_px": 100}
    assert not module.object_support(pair, frames)["supported"]
    frames[2]["objects"]["bowl"]["center_xy"] = [20., 1.]
    assert module.object_support(pair, frames)["supported"]
