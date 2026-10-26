"""Carry failure must retain accepted closure without privileged held/contact flags."""

import numpy as np
import pytest

from real2sim.policy.waypoint_failure_hold import FailureHoldWaypointController
from real2sim.sim.env import So101PickPlaceEnv
from tests.integration.test_visual_waypoint import ObservableEnv

CONFIG = "configs/v2/positive_setup_20261026/sim_so101.yaml"
pytestmark = pytest.mark.integration


def test_carry_failure_retains_accepted_closed_command_without_privileged_inputs(monkeypatch):
    with So101PickPlaceEnv(CONFIG, render_images=False) as underlying:
        underlying.reset(seed=0)
        env = ObservableEnv(underlying)
        ctrl = FailureHoldWaypointController(env)
        closed = np.r_[underlying.sim_state()[:5], ctrl._grip(False)]
        # An actually accepted command is the only history used for recovery.
        _, _, _, _, sent = env.step(closed)
        accepted = np.asarray(sent["control"]["applied_radians"])
        ctrl._last_safe_command = accepted.copy()
        ctrl.phase = "TRANSPORT"
        monkeypatch.setattr(ctrl, "_plan_carry", lambda: False)
        ctrl.step({}, env)
        assert ctrl.phase == "FAILED" and ctrl.failure_hold_source_phase == "TRANSPORT"
        for _ in range(3):
            result = ctrl.step({}, env)
            np.testing.assert_array_equal(result.info["control"]["requested_radians"], accepted)
            assert result.detail["failure_hold_armed"] and not result.ik_rejected
        ctrl.reset()
        assert ctrl.failure_hold_command is None and ctrl._last_safe_command is None


def test_failure_before_closure_does_not_arm_loaded_hold():
    with So101PickPlaceEnv(CONFIG, render_images=False) as env:
        env.reset(seed=0)
        ctrl = FailureHoldWaypointController(env)
        ctrl._last_safe_command = np.r_[env.sim_state()[:5], ctrl._grip(True)]
        ctrl.phase = "PRE_GRASP"
        ctrl._transition("FAILED")
        assert ctrl.failure_hold_command is None
