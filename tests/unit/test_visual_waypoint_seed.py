"""A measured joint slightly past its stop is an IK seed, not an abort."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.policy import visual_waypoint


def test_waypoint_seed_slightly_past_limit_is_clipped(monkeypatch):
    lo, hi = -np.ones(5), np.ones(5)
    measured = np.zeros(5)
    measured[3] = 1.00023
    seen = []

    class IK:
        config = {"position_tolerance_m": 0.01}

        def solve(self, target, seed, gripper=None, finger_opening=None):
            seed = np.asarray(seed, float)
            if np.any(seed[:5] < lo) or np.any(seed[:5] > hi):
                raise ValueError("IK seed exceeds native joint limits")
            seen.append(seed[:5].copy())
            return SimpleNamespace(accepted=False, reason="residual_rejected", qpos=seed)

        def forward(self, q, finger_opening=None):
            return np.eye(4)

        def _collision(self):
            return False

    env = SimpleNamespace(
        limits=np.vstack([np.column_stack([lo, hi]), [-1., 1.]]),
        sim_state=lambda: np.r_[measured, 0.],
        mapping=SimpleNamespace(gripper_real_to_sim_rad=lambda percent: 0.),
        max_velocity=np.ones(6), control_hz=25., task={"ball": {"radius": 0.02}})
    ctrl = object.__new__(visual_waypoint.VisualWaypointController)
    ctrl.env, ctrl.robot, ctrl.n, ctrl.ik = env, "so101", 5, IK()
    ctrl.ball = np.array([.3, 0., .02])
    ctrl.ball_in_tool = np.array([-.01, 0., .03])
    ctrl.grasp_rotation = np.eye(3)
    ctrl.phase = "LIFT"
    ctrl.plan_log, ctrl.ik_calls, ctrl.safety_checks = [], 0, 0
    ctrl.workspace = (np.full(3, -1.), np.full(3, 1.))

    ctrl._plan_grasp()
    assert seen and seen[0][3] == pytest.approx(1.)
    assert np.all(seen[0] <= hi) and np.all(seen[0] >= lo)

    real_least_squares = visual_waypoint.least_squares

    def spy(fun, x0, **kwargs):
        x0 = np.asarray(x0, float)
        if np.any(x0 < lo) or np.any(x0 > hi):
            raise ValueError("Initial guess is outside of provided bounds")
        assert x0[3] == pytest.approx(1.)
        return real_least_squares(fun, x0, **kwargs)

    monkeypatch.setattr(visual_waypoint, "least_squares", spy)
    assert ctrl._plan_carry() in (True, False)
