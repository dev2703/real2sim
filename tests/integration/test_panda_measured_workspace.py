"""Measured task placement remains inside Panda's configured planning box."""

import copy

import numpy as np
import pytest

from real2sim.config import load_yaml
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.panda.workspace import register_measured_task_workspace
from real2sim.policy import visual_waypoint
from real2sim.policy.visual_targets import VisualTarget
from real2sim.policy.visual_waypoint import VisualWaypointController

pytestmark = pytest.mark.integration
SCENE = "configs/v2/measured_registration_20261028/sim_panda_positive_all_scalars.yaml"


def test_measured_envelope_covers_entire_ball_disc_and_keeps_native_gates():
    config = load_yaml(SCENE)
    task = load_yaml(config["task_config"])
    original = copy.deepcopy(config)
    result = register_measured_task_workspace(config, task)
    center = np.asarray(task["ball"]["start_region"]["center_task_xy"])
    radius = task["ball"]["start_region"]["radius"] + max(task["ball"]["radius_interval_m"])
    assert np.all(np.asarray(result["ik"]["workspace_min"][:2]) <= center - radius)
    assert np.all(np.asarray(result["ik"]["workspace_max"][:2]) >= center + radius)
    for name in ("position_tolerance_m", "orientation_tolerance_rad", "penetration_tolerance_m"):
        assert result["ik"][name] == original["ik"][name]
    assert result["ik"]["workspace_min"][2] == original["ik"]["workspace_min"][2]
    assert config == original
    assert register_measured_task_workspace(result, task) == result


@pytest.mark.parametrize("seed", [300, 301, 302, 303, 304])
def test_native_panda_grasp_branches_cover_registered_measured_layouts(seed):
    with PandaPickPlaceEnv(SCENE, render_images=False) as env:
        env.reset(seed=seed)
        controller = VisualWaypointController(env, robot="panda")
        # Feasibility-only targets supplied once, explicitly separate from RGB
        # perception and never represented as an ACT or closed-loop task test.
        metrics = env.info()["privileged_metrics"]
        controller.ball = np.asarray(metrics["ball_xyz_task"])
        assert controller._plan_grasp(), controller.failure
        plan = controller.plan_log[-1]
        assert plan["seed_count"] > 1
        assert plan["candidate_count"] > 0
        assert controller._path_safe(env.arm_qpos(), controller.pre_q, env.max_opening)
        assert controller._path_safe(controller.pre_q, controller.grasp_q, env.max_opening)


def test_panda_multistart_controller_never_requests_privileged_object_metrics(monkeypatch):
    ball = VisualTarget(np.array([.0508, .3302, .02134]), np.zeros(2), 1000, .0001, .9)
    bowl = VisualTarget(np.array([.25631, .10510, .018]), np.zeros(2), 2000, .0001, .9)
    monkeypatch.setattr(visual_waypoint, "locate_objects",
                        lambda *args, **kwargs: {"ball": ball, "bowl": bowl})

    class ObservationOnly:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            if name == "info":
                raise AssertionError("Panda controller requested privileged metrics")
            return getattr(self.wrapped, name)

        def render(self, camera):
            return np.zeros((360, 640, 3), np.uint8)

    with PandaPickPlaceEnv(SCENE, render_images=False) as native:
        native.reset(seed=300)
        env = ObservationOnly(native)
        controller = VisualWaypointController(env, robot="panda")
        for _ in range(3):
            controller.step({}, env)
        assert controller.phase == "PRE_GRASP"
        assert controller.plan_log[-1]["uses_object_ground_truth"] is False
