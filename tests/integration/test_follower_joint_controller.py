"""The explicit joint-target adapter holds live hardware-model bounds unchanged."""

import numpy as np
import pytest

from real2sim.data.follower_joint_targets import VERSION
from real2sim.policy.follower_joint import GuardedJointActionAdapter
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics


@pytest.mark.integration
def test_guarded_joint_hold_and_native_limit_rejection_preserve_model():
    config = "tests/fixtures/geometry_ablation/sim.yaml"
    contract = {"joint_target_semantics": {
        "version": VERSION, "variant": "follower_next_arm_leader_gripper",
        "arm_target": "next_observed_follower", "arm_units": "degrees",
        "gripper_target": "current_recorded_leader_command_percent", "lead_policy_ticks": 1,
        "inference": "guarded_absolute_native_joint_target_v1", "collision_path_samples": 17}}
    with So101PickPlaceEnv(config, render_images=False) as env:
        env.reset(seed=0)
        limits, geoms = env.model.jnt_range.copy(), env.model.geom_pos.copy()
        controller = GuardedJointActionAdapter(env, NativeKinematics(env.model, env.config), contract)
        result = controller.apply(env.real_state())
        assert not result.ik_rejected
        assert result.detail["ik_reason"] == "accepted"
        assert result.detail["collision_path_samples"] == 17
        before = env.real_state()
        requested = before.copy()
        requested[0] = 10_000
        result = controller.apply(requested)
        assert result.ik_rejected
        assert result.detail["ik_reason"] == "target_native_limits"
        assert controller.ik_rejections == 1
        assert result.saturation == 0
        np.testing.assert_array_equal(env.model.jnt_range, limits)
        np.testing.assert_array_equal(env.model.geom_pos, geoms)
