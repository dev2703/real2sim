"""The executable RGB baseline cannot read privileged task state or hide IK use."""

import numpy as np
import pytest

from real2sim.policy import visual_waypoint
from real2sim.policy.visual_targets import VisualTarget
from real2sim.sim.env import So101PickPlaceEnv

CONFIG = "configs/v2/sim_so101.waypoint_20261026.yaml"
pytestmark = pytest.mark.integration


class UnreadableMetrics(dict):
    def __getitem__(self, key):
        raise AssertionError(f"controller read privileged metric {key}")

    def get(self, key, default=None):
        raise AssertionError(f"controller read privileged metric {key}")


class ObservableEnv:
    """Expose actual physics but forbid controller-side diagnostic inspection."""

    def __init__(self, env):
        self.wrapped = env

    def __getattr__(self, name):
        if name == "info":
            raise AssertionError("controller called env.info()")
        return getattr(self.wrapped, name)

    def render(self, camera):
        return np.zeros((360, 640, 3), np.uint8)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.wrapped.step(action)
        info["privileged_metrics"] = UnreadableMetrics()
        return obs, reward, terminated, truncated, info


def test_sparse_ik_execution_uses_no_privileged_object_or_contact_state(monkeypatch):
    # Deterministic frontend fixture, independent of env.info()/body positions.
    ball = VisualTarget(np.array([.3528, .2313, .02034]), np.array([410., 103.]), 2200, .0005, .8)
    bowl = VisualTarget(np.array([.3156, -.0402, .018]), np.array([418., 241.]), 5700, .003, .5)
    monkeypatch.setattr(visual_waypoint, "locate_objects", lambda *a, **k: {"ball": ball, "bowl": bowl})
    with So101PickPlaceEnv(CONFIG, render_images=False) as underlying:
        underlying.reset(seed=0)
        env = ObservableEnv(underlying)
        ctrl = visual_waypoint.VisualWaypointController(env)
        result = ctrl.step({}, env)
        assert ctrl.phase == "PRE_GRASP"
        planned_calls = ctrl.ik_calls
        assert planned_calls > 0
        for _ in range(8):
            result = ctrl.step(result.obs, env)
            assert ctrl.phase == "PRE_GRASP"
        assert ctrl.ik_calls == planned_calls, "joint execution must not rerun IK every tick"
        assert ctrl.safety_checks > 8
        assert ctrl.plan_log[0]["uses_object_ground_truth"] is False


def test_missing_rgb_targets_hold_and_fail_without_nominal_object_center_fallback(monkeypatch):
    monkeypatch.setattr(visual_waypoint, "locate_objects", lambda *a, **k: {"ball": None, "bowl": None})
    with So101PickPlaceEnv(CONFIG, render_images=False) as underlying:
        underlying.reset(seed=0)
        env = ObservableEnv(underlying)
        ctrl = visual_waypoint.VisualWaypointController(env)
        for _ in range(30):
            ctrl.step({}, env)
        assert ctrl.phase == "FAILED"
        assert ctrl.failure == "RGB ball/bowl detection unavailable"
        assert ctrl.ball is None and ctrl.bowl is None
        assert ctrl.plan_log == [] and ctrl.ik_calls == 0
