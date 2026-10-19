"""Headless native physics regression for zero-action hold and the unchanged IK limits."""

import copy

import numpy as np
import pytest

from real2sim.geometry.so3 import delta_action
from real2sim.policy.adapters import CartesianActionAdapter
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("polished", [False, True])
def test_persistent_hold_arrests_drift_with_both_ik_convergence_settings(polished):
    drifts = {}
    for mode in ("current_pose", "accepted_target"):
        with So101PickPlaceEnv(render_images=False) as env:
            state = np.array([0., -45., 45., 45., 0., 40.])
            env.reset(seed=0, scenario={"joint_positions": sim_from_real(env.mapping, state),
                                        "ball_xy": [.5, .3]})
            initial_pose = env.ee_pose()
            initial_config = copy.deepcopy(env.config)
            ik_config = copy.deepcopy(env.config)
            if polished:
                ik_config["ik"].update(position_convergence_m=1e-5,
                                       orientation_convergence_rad=1e-4)
            adapter = CartesianActionAdapter(env, NativeKinematics(env.model, ik_config),
                                              target_reference=mode)
            errors = []
            for _ in range(200):
                result = adapter.apply(np.r_[np.zeros(6), .4])
                errors.append(np.linalg.norm(delta_action(initial_pose, env.ee_pose())[:3]))
                assert not result.ik_rejected
                assert not any(result.info["control"]["limit_clipped"])
                assert result.info["privileged_metrics"]["robot_table_penetration_m"] == 0
                assert not (env._ball_contacts() - {env.table_geom})
            assert env.config == initial_config
            assert adapter.ik.config["position_tolerance_m"] == .003
            assert adapter.ik.config["orientation_tolerance_rad"] == .35
            if mode == "accepted_target":
                np.testing.assert_allclose(adapter.accepted_target, initial_pose, atol=1e-12)
            drifts[mode] = errors
    assert drifts["current_pose"][-1] > .020
    # Legacy convergence deliberately allows a 3 mm deadband. Preserve it; polished
    # convergence must keep hold error below 0.5 mm without relaxing hard acceptance.
    assert max(drifts["accepted_target"]) < (.0005 if polished else .0035)
    assert drifts["accepted_target"][-1] < drifts["current_pose"][-1] / 5
