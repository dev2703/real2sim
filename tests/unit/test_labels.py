import numpy as np
import pytest

from real2sim.data.labels import ACTION_NAMES, PROPRIO_NAMES, episode_labels
from real2sim.data.resample import chunk_indices
from real2sim.geometry.so3 import apply_delta, exp_so3, make_T, matrix_from_rot6d


def _trajectory(n=12, seed=0):
    rng = np.random.default_rng(seed)
    T = [make_T(exp_so3(np.array([0.1, -0.2, 0.3])), np.array([0.3, 0.0, 0.1]))]
    for _ in range(n - 1):
        d = np.r_[rng.normal(0, 0.005, 3), rng.normal(0, 0.03, 3)]
        T.append(apply_delta(T[-1], d))
    return np.stack(T)


def test_shapes_names_and_contract_sizes():
    poses = _trajectory()
    lab = episode_labels(poses, np.full(12, 40.0), np.full(12, 2.0), hz=10, gripper_source="leader_action")
    assert lab.proprio.shape == (12, 13) and len(PROPRIO_NAMES) == 13
    assert lab.action.shape == (12, 7) and len(ACTION_NAMES) == 7


def test_actions_reconstruct_achieved_poses():
    poses = _trajectory()
    lab = episode_labels(poses, np.zeros(12), np.zeros(12), hz=10, gripper_source="leader_action")
    for k in range(11):
        np.testing.assert_allclose(apply_delta(poses[k], lab.action[k, :6]), poses[k + 1], atol=1e-9)
    assert lab.label_is_extrapolated.tolist() == [False] * 11 + [True]
    np.testing.assert_allclose(lab.action[-1, :6], 0)


def test_proprio_contents():
    poses = _trajectory()
    lab = episode_labels(poses, np.linspace(0, 100, 12), np.zeros(12), hz=10, gripper_source="leader_action")
    np.testing.assert_allclose(lab.proprio[:, :3], poses[:, :3, 3])
    for k in (0, 5):
        np.testing.assert_allclose(matrix_from_rot6d(lab.proprio[k, 3:9]), poses[k, :3, :3], atol=1e-9)
    np.testing.assert_allclose(lab.proprio[0, 9:12], 0)  # causal: no velocity at the first tick
    np.testing.assert_allclose(lab.proprio[3, 9:12], (poses[3, :3, 3] - poses[2, :3, 3]) * 10)
    np.testing.assert_allclose(lab.proprio[:, 12], np.linspace(0, 1, 12))


def test_gripper_source_semantics():
    poses = _trajectory(4)
    obs = np.array([80.0, 60.0, 35.0, 35.0])  # follower blocked open by the ball
    cmd = np.array([80.0, 0.0, 0.0, 0.0])  # leader commands fully closed
    leader = episode_labels(poses, obs, cmd, hz=10, gripper_source="leader_action")
    follower = episode_labels(poses, obs, cmd, hz=10, gripper_source="follower_state")
    np.testing.assert_allclose(leader.action[:, 6], [0.8, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(follower.action[:, 6], [0.6, 0.35, 0.35, 0.35])
    with pytest.raises(ValueError, match="D-013"):
        episode_labels(poses, obs, cmd, hz=10, gripper_source="unknown")


def test_jump_flag_and_leader_error():
    poses = _trajectory(5)
    poses[3, :3, 3] += 0.2  # 20 cm one-tick teleport
    lead = poses.copy()
    lead[:, 0, 3] += 0.01
    lab = episode_labels(poses, np.zeros(5), np.zeros(5), hz=10, gripper_source="leader_action", leader_poses=lead)
    assert lab.jump_flags[2] and lab.jump_flags[3]
    np.testing.assert_allclose(lab.leader_follower_pos_err, 0.01, atol=1e-12)


def test_no_cross_episode_chunks_for_labels():
    ep_start = np.r_[np.zeros(5, int), np.full(4, 5)]
    ep_len = np.r_[np.full(5, 5), np.full(4, 4)]
    for t in range(9):
        idx, pad = chunk_indices(ep_start, ep_len, t, 20)
        assert set(idx) <= set(range(ep_start[t], ep_start[t] + ep_len[t]))
        assert pad.sum() == 20 - (ep_start[t] + ep_len[t] - t)
