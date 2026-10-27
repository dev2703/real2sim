"""The native-HF Panda camera follows the hand and preserves the source interface."""

import numpy as np
import pytest

from real2sim.panda.env import PandaPickPlaceEnv
from scripts import evaluate_hf_act_panda_20261028 as transfer

pytestmark = pytest.mark.integration


def test_external_wrist_mount_is_rigid_and_different_from_global_view(tmp_path):
    source_scene, task = transfer.prepare_source(tmp_path, (0.,) * 5)
    source = transfer.retarget.VirtualSource(source_scene)
    try:
        scene, record = transfer.prepare_panda(tmp_path, source, task, -.15)
        with PandaPickPlaceEnv(scene, render_images=False) as env:
            start, width = transfer.panda_start(env, source, -.15)
            env.reset(seed=100, scenario={"joint_positions": start, "gripper_opening": width})
            wrist = env.model.camera(env.camera_bindings["observation.images.top"]).id
            global_camera = env.model.camera("iphone").id
            hand = env.model.body("hand").id
            np.testing.assert_allclose(record["panda_hand_relative_position_m"], [.08, 0., 0.])
            assert env.model.cam_bodyid[wrist] == hand
            assert env.model.cam_bodyid[global_camera] == 0
            before = env.data.cam_xpos[wrist].copy()
            fixed = env.data.cam_xpos[global_camera].copy()
            moved = start.copy()
            moved[0] += .02
            env.reset(seed=100, scenario={"joint_positions": moved, "gripper_opening": width})
            assert np.linalg.norm(env.data.cam_xpos[wrist] - before) > 1e-4
            np.testing.assert_array_equal(env.data.cam_xpos[global_camera], fixed)
            rotation = record["panda_hand_relative_rotation"]
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            assert np.linalg.det(rotation) > .999999
    finally:
        source.close()


def test_joint_registration_changes_raw_state_without_moving_native_start(tmp_path):
    offsets = np.array([-3.5, 7.1, -10., -11.8, -1.4])
    scene, _ = transfer.prepare_source(tmp_path, offsets)
    source = transfer.retarget.VirtualSource(scene)
    try:
        raw = source.source.real_state()
        source.source.reset(seed=0)
        raw = source.source.real_state()
        expected = np.rad2deg(source.source.initial_qpos[:5]) - offsets
        np.testing.assert_allclose(raw[:5], expected, atol=1e-5)
        np.testing.assert_array_equal(source.source.sim_state(), source.source.initial_qpos)
    finally:
        source.close()
