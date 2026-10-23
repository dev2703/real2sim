"""Diagnostic comparisons must not conceal infeasible commands with clipping."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from real2sim.geometry.so3 import exp_so3

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/diag_v2_control_contract_comparison.py"
SPEC = importlib.util.spec_from_file_location("control_contract_comparison", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class FakeIK:
    def __init__(self, collision=False, weight=.025):
        self.limits = np.tile([-1., 1.], (6, 1))
        self.config = {"workspace_min": [-.1, -.1, -.1], "workspace_max": [.1, .1, .1],
                       "position_tolerance_m": .003, "orientation_tolerance_rad": .35,
                       "orientation_weight_m_per_rad": weight}
        self.collision = collision
        self.forward_calls = 0

    def forward(self, q):
        self.forward_calls += 1
        pose = np.eye(4)
        pose[0, 3] = q[0]
        pose[:3, :3] = exp_so3(np.array([0., 0., q[1]]))
        return pose

    def _collision(self):
        return self.collision


def test_native_limit_target_is_rejected_before_fk_without_clipping():
    ik = FakeIK()
    target = np.array([1.01, 0, 0, 0, 0, 0])
    row = diagnostic.candidate_gate(ik, target)
    assert row["reason"] == "native_joint_limit" and not row["accepted"]
    assert ik.forward_calls == 0
    assert target[0] == 1.01


@pytest.mark.parametrize("weight", [.025, .0015625])
def test_soft_orientation_cost_retains_original_hard_pose_and_collision_gates(weight):
    ik = FakeIK(weight=weight)
    q = np.array([0., .351, 0., 0., 0., 0.])
    assert diagnostic.candidate_gate(ik, q, np.eye(4))["reason"] == "residual_rejected"
    q[1] = .349
    assert diagnostic.candidate_gate(ik, q, np.eye(4))["accepted"]
    ik.collision = True
    assert diagnostic.candidate_gate(ik, q, np.eye(4))["reason"] == "collision"
    ik.collision = False
    q[0] = .00301
    assert diagnostic.candidate_gate(ik, q, np.eye(4))["reason"] == "residual_rejected"


def test_requested_workspace_gate_survives_near_boundary_feasible_candidate():
    ik = FakeIK()
    q = np.array([.099, 0, 0, 0, 0, 0])
    requested = np.eye(4)
    requested[0, 3] = .1001  # close enough in position, but request is outside workspace
    assert diagnostic.candidate_gate(ik, q, requested)["reason"] == "outside_workspace"


def test_common_progress_preserves_checked_segment_and_all_joint_rate_caps():
    seed = np.array([.2, -.1, .4])
    target = np.array([.5, .1, .45])
    caps = np.array([.08, .02, .1])
    result, fraction = diagnostic.bounded_segment(seed, target, caps)
    np.testing.assert_allclose(result, seed + .1 * (target - seed))
    assert fraction == pytest.approx(.1)
    assert np.all(np.abs(result - seed) <= caps + 1e-12)
    result, fraction = diagnostic.bounded_segment(seed, seed, caps)
    np.testing.assert_array_equal(result, seed)
    assert fraction == 1.
    with pytest.raises(ValueError, match="positive caps"):
        diagnostic.bounded_segment(seed, target, np.zeros(3))
