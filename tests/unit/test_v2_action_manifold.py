"""Physical-model checks for the diagnostic's arm-only spatial Jacobian."""

import importlib.util
from pathlib import Path

import numpy as np

from real2sim.geometry.so3 import delta_action
from real2sim.geometry.so101_fk import So101FK

SPEC = importlib.util.spec_from_file_location(
    "action_manifold", Path(__file__).resolve().parents[2] / "scripts/diag_v2_action_manifold.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_arm_jacobian_matches_finite_fk_motion():
    fk = So101FK()
    q = np.array([.2, -.5, .6, .4, -.3])
    pose, jac = MODULE.task_jacobian(fk, q)
    assert jac.shape == (6, 5)
    for column in range(5):
        step = np.eye(5)[column] * 1e-7
        observed = delta_action(pose, fk.fk_sim(q + step)) / 1e-7
        np.testing.assert_allclose(jac[:, column], observed, atol=1e-7)


def test_local_metric_distinguishes_reachable_from_missing_arm_direction():
    fk = So101FK()
    _, jac = MODULE.task_jacobian(fk, np.array([.2, -.5, .6, .4, -.3]))
    weights = np.r_[np.ones(3), np.full(3, .025)]
    u, singular, _ = np.linalg.svd(weights[:, None] * jac, full_matrices=True)
    assert singular[-1] > 1e-4
    reachable = jac @ np.array([.001, -.002, .001, .003, -.001])
    assert MODULE.local_residual(jac, reachable, .025) < 1e-10
    off_manifold = reachable + .001 * u[:, -1] / weights
    np.testing.assert_allclose(MODULE.local_residual(jac, off_manifold, .025), 1., atol=1e-10)
