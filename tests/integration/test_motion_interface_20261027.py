"""The reconstruction probe preserves frames, timing semantics and explicit jaw commands."""

from contextlib import closing

import numpy as np
import pytest

from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics
from scripts.diag_v2_motion_interface_20261027 import (
    SCENE,
    ContinuousPointAdapter,
    command_reference,
    semantic,
)

pytestmark = pytest.mark.integration
OBJECTS = np.array([.34, .2, .315, -.04])


def test_reference_uses_grasp_point_offset_and_commanded_not_readback_gripper():
    with closing(So101PickPlaceEnv(SCENE, render_images=False)) as env:
        env.reset(seed=100)
        fk = NativeKinematics(env.model, env.config)
        native = env.initial_qpos.copy()
        raw = native.copy()
        raw[5] = env.mapping.gripper_real_to_sim_rad(90.)
        reference = command_reference(env, fk, native, raw)
        pose = fk.forward(native)
        np.testing.assert_allclose(reference["point"], pose[:3, 3] + pose[:3, :3] @ semantic.SO_OFFSET)
        np.testing.assert_allclose(reference["rotation"], pose[:3, :3])
        assert reference["gripper_native"] == native[5]
        assert reference["opened"]


def test_continuous_variant_matches_original_adapter_at_binary_endpoint():
    with (closing(So101PickPlaceEnv(SCENE, render_images=False)) as first,
          closing(So101PickPlaceEnv(SCENE, render_images=False)) as second):
        first.reset(seed=100)
        second.reset(seed=100)
        target = semantic.point(first.ee_pose(), semantic.SO_OFFSET)
        original = semantic.PointAdapter(first, "so101", OBJECTS, joint_margin=.01)
        retained = ContinuousPointAdapter(second, "so101", OBJECTS, joint_margin=.01)
        grip = float(np.clip(first.mapping.gripper_real_to_sim_rad(90.), *first.limits[5]))
        one = original.apply(target, True)
        two = retained.apply_continuous(target, grip)
        assert one[3:5] == two[3:5]
        np.testing.assert_allclose(one[2]["control"]["requested_radians"],
                                   two[2]["control"]["requested_radians"], atol=1e-12)
        np.testing.assert_allclose(first.sim_state(), second.sim_state(), atol=1e-12)


def test_intermediate_gripper_is_not_binarized_and_invalid_request_is_not_clipped():
    with closing(So101PickPlaceEnv(SCENE, render_images=False)) as env:
        env.reset(seed=100)
        target = semantic.point(env.ee_pose(), semantic.SO_OFFSET)
        retained = ContinuousPointAdapter(env, "so101", OBJECTS, joint_margin=.01,
                                          orientation_reference=env.ee_pose()[:3, :3])
        grip = float(env.initial_qpos[5])
        result = retained.apply_continuous(target, grip)
        assert result[3], result[4]
        assert result[2]["control"]["requested_radians"][5] == grip
        with pytest.raises(ValueError, match="existing limits"):
            retained.apply_continuous(target, env.limits[5, 1] + .1)
