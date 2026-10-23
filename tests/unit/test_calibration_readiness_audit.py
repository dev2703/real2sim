"""Numerical unit evidence must use each signal's own calibration and float64 inversion."""

import copy

import numpy as np

from real2sim.geometry.so101_fk import ARM_JOINTS
from scripts.audit_v2_calibration_readiness import encoder_lattice


def test_lattice_uses_calibration_midpoints_without_fresh_float32_roundoff():
    calibration = {name: {"range_min": 100 + 7 * j, "range_max": 4000 - 9 * j}
                   for j, name in enumerate(ARM_JOINTS)}
    ticks = np.arange(1000, 1101)[:, None]
    midpoint = np.array([(row["range_min"] + row["range_max"]) / 2 for row in calibration.values()])
    real = ((ticks - midpoint) * 360 / 4095).astype(np.float32)
    result = encoder_lattice(real, calibration)
    assert max(row["degree_inverse_tick_residual_max"] for row in result.values()) < 5e-5
    wrong_device = copy.deepcopy(calibration)
    for row in wrong_device.values():
        row["range_max"] += 1
    assert min(row["degree_inverse_tick_residual_p95"] for row in encoder_lattice(real, wrong_device).values()) > .4999
