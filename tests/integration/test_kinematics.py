"""Native FK/IK checks; real calibration and independent URDF validation stay unverified."""

import copy

import mujoco
import numpy as np
import pytest

from real2sim.config import load_yaml
from real2sim.geometry.so101_fk import So101FK
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics, xml_forward_kinematics
from real2sim.sim.model import DEFAULT_CONFIG, ROOT, scene_xml, task_frame_offset

pytestmark = pytest.mark.integration


@pytest.fixture
def env():
    with So101PickPlaceEnv(render_images=False) as value:
        value.reset(seed=0)
        yield value


def test_xml_fk_matches_mujoco_at_30_native_poses(env):
    rng = np.random.default_rng(11)
    for q in rng.uniform(env.limits[:, 0], env.limits[:, 1], size=(30, 6)):
        env.data.qpos[env.qadr] = q
        mujoco.mj_forward(env.model, env.data)
        # The XML chain is base-relative; the scene places the base on the table top.
        transform = xml_forward_kinematics(ROOT / env.config["model"],
                                          dict(zip(env.config["joint_names"], q, strict=True)))
        np.testing.assert_allclose(transform, env.ee_pose(), atol=1e-10)
        np.testing.assert_allclose(transform[:3, 3] + env.offset,
                                   env.data.site("gripperframe").xpos, atol=1e-10)


def test_ee_pose_matches_dataset_side_fk_at_10_random_poses(env):
    """Ties the simulator tool frame to real2sim.geometry.so101_fk (dataset-side FK)."""
    fk = So101FK(mapping=env.mapping)
    rng = np.random.default_rng(7)
    for q in rng.uniform(env.limits[:, 0], env.limits[:, 1], size=(10, 6)):
        env.data.qpos[env.qadr] = q
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(env.ee_pose(), fk.fk_real(env.real_state()), atol=1e-6)
    _, info = env.reset(seed=0)
    np.testing.assert_allclose(info["privileged_metrics"]["tool_xyz_task"],
                               fk.fk_real(env.real_state())[:3, 3], atol=1e-6)


def test_ik_reachable_targets_and_no_live_state_mutation(env):
    kin = NativeKinematics(env.model, env.config)
    rng = np.random.default_rng(2)
    # The folded rest pose puts the tool below the table (z < workspace_min); seed mid-workspace.
    seed = np.array([0.0, -0.8, 0.8, 0.8, 0.0, 0.3])
    assert np.all(kin.forward(seed)[:3, 3] > env.config["ik"]["workspace_min"])
    before = env.data.qpos.copy()
    for _ in range(12):
        target_q = seed + rng.uniform(-0.15, 0.15, size=6)
        result = kin.solve(kin.forward(target_q), seed, gripper=target_q[5])
        assert result.accepted, result
        assert result.position_error_m <= 0.003
        assert result.orientation_error_rad <= 0.35
        assert np.all(result.qpos >= kin.limits[:, 0])
        assert np.all(result.qpos <= kin.limits[:, 1])
    np.testing.assert_array_equal(env.data.qpos, before)


def test_unreachable_rejected_without_partial_target(env):
    kin = NativeKinematics(env.model, env.config)
    seed = env.initial_qpos.copy()
    target = kin.forward(seed)
    target[0, 3] = 5
    result = kin.solve(target, seed)
    assert not result.accepted and result.reason == "outside_workspace"
    np.testing.assert_array_equal(result.qpos, seed)
    # Inside the workspace bounding box but beyond the arm's reach.
    target[:3, 3] = [0.5, 0.4, 0.55]
    result = kin.solve(target, seed)
    assert not result.accepted
    np.testing.assert_array_equal(result.qpos, seed)


def test_invalid_ik_input(env):
    kin = NativeKinematics(env.model, env.config)
    seed = env.initial_qpos.copy()
    target = kin.forward(seed)
    target[0, 0] = 9
    with pytest.raises(ValueError, match="rotation"):
        kin.solve(target, seed)
    with pytest.raises(ValueError, match="gripper"):
        kin.solve(np.eye(4), seed, gripper=10)


def test_base_relative_fk_is_invariant_to_base_z_offset(env):
    """With task.yaml robot_so101.base_z_offset set (D-014 option A), base-relative poses from
    NativeKinematics and the dataset-side FK are unchanged; only the world placement moves."""
    config = load_yaml(DEFAULT_CONFIG)
    task = copy.deepcopy(load_yaml(ROOT / config["task_config"]))
    task["robot_so101"]["base_z_offset"] = 0.012
    model = mujoco.MjModel.from_xml_string(scene_xml(config, task))
    lifted = NativeKinematics(model, config)
    reference = NativeKinematics(env.model, config)
    np.testing.assert_allclose(lifted.base_pos - reference.base_pos, [0, 0, 0.012])
    np.testing.assert_allclose(lifted.base_pos, task_frame_offset(task))
    rng = np.random.default_rng(5)
    for q in rng.uniform(env.limits[:, 0], env.limits[:, 1], size=(5, 6)):
        np.testing.assert_allclose(lifted.forward(q), reference.forward(q), atol=1e-12)
        np.testing.assert_allclose(lifted.data.site_xpos[lifted.site] - reference.data.site_xpos[reference.site],
                                   [0, 0, 0.012], atol=1e-12)
