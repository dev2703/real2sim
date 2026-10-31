"""A small gripper endpoint error must not freeze valid learned arm commands."""

import numpy as np
import pytest

from real2sim.policy.bounded_joint import BoundedGripperJointActionAdapter, bound_gripper_command
from real2sim.sim.env import So101PickPlaceEnv

pytestmark = pytest.mark.integration
CONFIG = "configs/v2/sim_so101.waypoint_20261026.yaml"


def test_overclosed_gripper_is_logged_and_valid_arm_command_executes():
    with So101PickPlaceEnv(CONFIG, render_images=False) as env:
        q = np.array([0., 0., 0., 0., 0., .5])
        env.reset(seed=0, scenario={"joint_positions": q})
        limits = env.limits.copy()
        action = env.real_state()
        action[0] += 1.
        action[5] = -.7
        result = BoundedGripperJointActionAdapter(env).apply(action)
        detail = result.detail
        assert not result.ik_rejected
        assert detail["gripper_saturated"] and result.saturation == pytest.approx(1 / 6)
        assert detail["requested_native_rad"][5] < limits[5, 0]
        assert detail["bounded_native_target_rad"][5] >= limits[5, 0]
        assert env.data.ctrl[env.actuator_ids[0]] > 0
        np.testing.assert_array_equal(env.limits, limits)
        assert detail["corrective_ik_calls"] == detail["task_corrections"] == 0


def test_gripper_saturation_does_not_clip_or_accept_impossible_arm_target():
    with So101PickPlaceEnv(CONFIG, render_images=False) as env:
        env.reset(seed=0, scenario={"joint_positions": [0., 0., 0., 0., 0., .5]})
        action = env.real_state()
        action[0], action[5] = 10000., -1.
        raw, bounded = bound_gripper_command(env.mapping, env.limits, action)
        assert raw[0] == bounded[0] and bounded[0] > env.limits[0, 1]
        result = BoundedGripperJointActionAdapter(env).apply(action)
        assert result.ik_rejected and result.detail["guard_reason"] == "target_native_limits"
        assert not result.detail["arm_target_clipping"]


@pytest.mark.parametrize("action", [np.zeros(5), [0., 0., 0., 0., 0., np.nan]])
def test_malformed_joint_predictions_are_refused(action):
    with So101PickPlaceEnv(CONFIG, render_images=False) as env:
        with pytest.raises(ValueError, match="six finite"):
            bound_gripper_command(env.mapping, env.limits, action)
