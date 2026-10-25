"""A paired object comparison must retain the exact support and target context."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


def load_summary():
    path = Path(__file__).resolve().parents[2] / "scripts/summarize_v2_prefix_object_actions.py"
    spec = importlib.util.spec_from_file_location("prefix_object_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example():
    return {"schema": "canonical_matched_object_probe_v1",
            "plan": {"rules": {"prefix": 2}, "pairs": [{"anchor_tick": 4, "donor_tick": 8}],
                     "actual_training_episodes": [8, 11, 23], "data": "same data"},
            "selected_pairs": 1, "supported_pairs": 1,
            "groups": [{"partition": "val", "object": "ball", "context": "active", "mode": "paired_object_region",
                        "pairs": 1, "reciprocal_directions": 2, "informative_pairs": 1,
                        "mean_target_contrast_prefix_norm_mm": 4., "mean_original_target_step_mm": 3.}]}


def test_equal_predictions_are_not_required_but_identical_pair_selection_is():
    summary = load_summary()
    baseline = example()
    prefix = deepcopy(baseline)
    prefix["groups"][0]["mean_swap_effect_prefix_norm_mm"] = .5
    summary.verify_paired_probe_reports(baseline, prefix)
    prefix["plan"]["pairs"][0]["anchor_tick"] = 5
    with pytest.raises(ValueError, match="inputs differ: pairs"):
        summary.verify_paired_probe_reports(baseline, prefix)


@pytest.mark.parametrize("field,value", [("informative_pairs", 0),
                                         ("mean_target_contrast_prefix_norm_mm", 5.)])
def test_changed_object_support_or_target_cannot_be_hidden_in_group_average(field, value):
    baseline, prefix = example(), example()
    prefix["groups"][0][field] = value
    with pytest.raises(ValueError, match="targets/support differ"):
        load_summary().verify_paired_probe_reports(baseline, prefix)
