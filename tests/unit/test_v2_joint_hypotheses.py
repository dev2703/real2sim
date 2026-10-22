"""Keep diagnostic alternatives from double-applying saved homing offsets."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "joint_hypotheses", Path(__file__).parents[2] / "scripts/diag_v2_joint_hypotheses.py")
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


def test_reported_encoder_homing_offsets_are_not_applied_twice():
    calibration = {name: {"range_min": 1000, "range_max": 3000, "homing_offset": 2000}
                   for name in DIAG.ARM}
    limits = np.tile([-90., 90.], (5, 1))
    train = np.tile([-80., 80.], (5, 1)).T
    before = DIAG.hypotheses(calibration, limits, train)
    for motor in calibration.values():
        motor["homing_offset"] = -1900
    after = DIAG.hypotheses(calibration, limits, train)
    for name in before:
        np.testing.assert_array_equal(before[name]["offset_deg"], after[name]["offset_deg"])
    np.testing.assert_allclose(before["homing_pose_as_zero"]["offset_deg"], -47 * 360 / 4095)


def test_oversized_recorded_span_is_reported_instead_of_silently_scaled():
    calibration = {name: {"range_min": 1000, "range_max": 3000} for name in DIAG.ARM}
    limits = np.tile([-90., 90.], (5, 1))
    train = np.tile([-91., 91.], (5, 1)).T
    with pytest.raises(ValueError, match="cannot fit native limits"):
        DIAG.hypotheses(calibration, limits, train)
