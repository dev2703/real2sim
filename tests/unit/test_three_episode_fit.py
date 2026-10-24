"""Capacity-gate checks must require success on all three training conditions."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("tiny_fit_runner", Path("scripts/run_v2_three_episode_fit.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report(translation, gripper):
    episodes = {str(e): {"translation_mean_mm": x, "gripper_mae": g,
                         "movement": {"n": 20, "vector_error_mean": .5},
                         "stationary_drift": {"n": 10, "summed_error_per_run_mean": .02,
                                              "summed_error_per_run_max": .04}}
                for e, x, g in zip((11, 8, 23), translation, gripper, strict=True)}
    result = {"scores": {"model": {"per_episode": episodes}}}
    for name in ("hold", "persistence"):
        baseline = deepcopy(episodes)
        for row in baseline.values():
            row["movement"]["vector_error_mean"] = 2.
            row["stationary_drift"].update(summed_error_per_run_mean=.1, summed_error_per_run_max=.2)
        result["scores"][name] = {"per_episode": baseline}
    return result


def test_all_conditions_must_meet_strict_physical_thresholds():
    assert module.gate_from_validation(report([.2, .9, .8], [.01, .02, .029]))["pass"]
    assert not module.gate_from_validation(report([.2, 1., .8], [.01, .02, .02]))["pass"]
    assert not module.gate_from_validation(report([.2, .9, .8], [.01, .03, .02]))["pass"]


def test_two_conditions_cannot_pass_three_condition_gate():
    value = report([.2, .9, .8], [.01, .02, .02])
    del value["scores"]["model"]["per_episode"]["23"]
    assert not module.gate_from_validation(value)["pass"]


@pytest.mark.parametrize("baseline", ["hold", "persistence"])
@pytest.mark.parametrize("group,metric", [
    ("movement", "vector_error_mean"),
    ("stationary_drift", "summed_error_per_run_mean"),
    ("stationary_drift", "summed_error_per_run_max"),
])
def test_good_aggregate_fit_cannot_hide_a_condition_worse_than_either_baseline(baseline, group, metric):
    value = report([.2, .3, .4], [.01] * 3)
    value["scores"][baseline]["per_episode"]["23"][group][metric] = 0.
    assert not module.gate_from_validation(value)["pass"]


@pytest.mark.parametrize("group", ["movement", "stationary_drift"])
def test_missing_or_mismatched_motion_strata_fail(group):
    value = report([.2] * 3, [.01] * 3)
    value["scores"]["model"]["per_episode"]["11"][group]["n"] = 0
    assert not module.gate_from_validation(value)["pass"]
    value["scores"]["model"]["per_episode"]["11"][group]["n"] = 999
    assert not module.gate_from_validation(value)["pass"]


def test_zero_stationary_error_floor_can_tie_but_nonzero_ties_fail():
    value = report([.2] * 3, [.01] * 3)
    for score in value["scores"].values():
        for row in score["per_episode"].values():
            row["stationary_drift"].update(summed_error_per_run_mean=0., summed_error_per_run_max=0.)
    assert module.gate_from_validation(value)["pass"]
    for score in value["scores"].values():
        score["per_episode"]["11"]["stationary_drift"]["summed_error_per_run_mean"] = .1
    assert not module.gate_from_validation(value)["pass"]


def test_nonfinite_error_and_wrong_episode_set_fail():
    value = report([.2] * 3, [.01] * 3)
    value["scores"]["model"]["per_episode"]["11"]["movement"]["vector_error_mean"] = float("nan")
    assert not module.gate_from_validation(value)["pass"]
    value = report([.2] * 3, [.01] * 3)
    for score in value["scores"].values():
        score["per_episode"]["12"] = score["per_episode"].pop("11")
    assert not module.gate_from_validation(value)["pass"]
