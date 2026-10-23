"""An apparently low average error must not disguise missed or reversed movement."""

import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "policy_audit", Path(__file__).resolve().parents[2] / "scripts/audit_v2_policy.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stationary_ticks_cannot_hide_missed_motion_or_direction():
    label = np.zeros((100, 7))
    label[-1, 1] = .01
    stopped = MODULE.metrics(np.zeros_like(label), label, action_kind="cartesian")
    assert stopped["translation_mean_mm"] == .1
    assert stopped["translation_motion"]["n"] == 1
    assert stopped["translation_motion"]["translation_mean_mm"] == 10.
    assert stopped["translation_motion"]["median_predicted_to_label_step_ratio"] == 0.
    reversed_motion = MODULE.metrics(-label, label, action_kind="cartesian")
    assert reversed_motion["translation_motion"]["median_predicted_to_label_step_ratio"] == 1.
    assert reversed_motion["translation_motion"]["positive_direction_fraction"] == 0.
    accurate = MODULE.metrics(label, label, action_kind="cartesian")
    assert accurate["translation_skill_vs_zero"] == 1.
    assert accurate["translation_motion"]["positive_direction_fraction"] == 1.


def test_delta_baseline_is_not_applied_to_absolute_poses():
    poses = np.zeros((2, 10))
    poses[:, 3:9] = [1, 0, 0, 0, 1, 0]
    report = MODULE.metrics(poses, poses, action_kind="cartesian_absolute")
    assert "translation_motion" not in report
    assert "zero_delta_translation_mean_mm" not in report
    stationary = MODULE.metrics(np.zeros((2, 7)), np.zeros((2, 7)), action_kind="cartesian")
    assert stationary["translation_skill_vs_zero"] is None
    assert stationary["translation_motion"]["n"] == 0
