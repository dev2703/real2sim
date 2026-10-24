"""Capacity-gate checks must require success on all three training conditions."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("tiny_fit_runner", Path("scripts/run_v2_three_episode_fit.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report(translation, gripper):
    return {"scores": {"model": {"per_episode": {
        str(e): {"translation_mean_mm": x, "gripper_mae": g}
        for e, x, g in zip((11, 8, 23), translation, gripper, strict=True)}}}}


def test_all_conditions_must_meet_strict_physical_thresholds():
    assert module.gate_from_validation(report([.2, .9, .8], [.01, .02, .029]))["pass"]
    assert not module.gate_from_validation(report([.2, 1., .8], [.01, .02, .02]))["pass"]
    assert not module.gate_from_validation(report([.2, .9, .8], [.01, .03, .02]))["pass"]


def test_two_conditions_cannot_pass_three_condition_gate():
    value = report([.2, .9, .8], [.01, .02, .02])
    del value["scores"]["model"]["per_episode"]["23"]
    assert not module.gate_from_validation(value)["pass"]
