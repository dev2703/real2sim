"""The diagnostic observer must not change dynamics, including a successful grasp."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from real2sim.evaluation.oracle import OracleGeometryController, settle_start_pose
from real2sim.sim.env import So101PickPlaceEnv

SPEC = importlib.util.spec_from_file_location(
    "geometry_contact_diagnostic", Path("scripts/diag_geometry_contact_sweep.py"))
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
pytestmark = [pytest.mark.integration]


def test_gripper_comparison_preserves_follower_arm_and_raw_data():
    state = np.array([10, -20, 30, 40, 50, 35.0])
    original = state.copy()
    leader = diagnostic.joint_target(state, .03, "leader")
    follower = diagnostic.joint_target(state, .03, "follower")
    np.testing.assert_array_equal(leader[:5], state[:5])
    np.testing.assert_array_equal(follower, state)
    np.testing.assert_array_equal(state, original)
    assert leader[5] == 3.0


def test_substep_force_observer_preserves_successful_oracle_dynamics():
    with So101PickPlaceEnv(Path("configs/sim_so101.yaml"), render_images=False) as env:
        q0, _ = settle_start_pose(env, 0)

        def run():
            controller = OracleGeometryController(env)
            obs, _ = env.reset(seed=0, scenario={"joint_positions": q0})
            controller.reset()
            states = []
            for _ in range(120):
                result = controller.step(obs, env)
                obs = result.obs
                states.append(env.data.qpos.copy())
            return np.array(states)

        baseline = run()
        with diagnostic.contact_monitor(env) as contacts:
            monitored = run()
        np.testing.assert_array_equal(monitored, baseline)
        assert contacts["bilateral_substeps"] > 0
        assert contacts["bilateral_lift_substeps"] > 0
        assert contacts["max_min_jaw_normal_force_n"] > 0
