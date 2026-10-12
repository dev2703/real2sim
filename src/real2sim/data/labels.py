"""Canonical 7-D task-space labels and 13-float proprioception (plan §3.3, D-003 C, D-008 A).

Per policy tick ``k`` inside one episode (10 Hz):

* pose ``T_k = FK(q_obs[k])`` (achieved motion, D-003 option C for the Cartesian policy)
* action ``a_k = [delta_action(T_k, T_{k+lead}), gripper_target_k]`` with ``lead`` = 1 tick
* proprio ``s_k = [p_k(3), rot6d(R_k)(6), (p_k - p_{k-1}) * hz (3), gripper_obs_k / 100 (1)]``
  with zero velocity at the first tick (causal, reproducible identically in simulation)

The last ``lead`` ticks of an episode have no successor; their action is a zero motion with the
current gripper target and is flagged ``label_is_extrapolated`` so chunk padding can mask it.

Gripper target semantics are decision D-013 (``gripper_source``):

* ``leader_action``  — recorded leader gripper command at the tick / 100 (operator intent)
* ``follower_state`` — achieved follower gripper at tick ``k+lead`` / 100 (achieved motion)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from real2sim.geometry.so3 import delta_action, rot6d_from_matrix

GRIPPER_SOURCES = ("leader_action", "follower_state")
PROPRIO_NAMES = (
    ["ee_x", "ee_y", "ee_z"]
    + [f"ee_rot6d_{i}" for i in range(6)]
    + ["ee_vx", "ee_vy", "ee_vz", "gripper_obs"]
)
ACTION_NAMES = ["dx", "dy", "dz", "d_rx", "d_ry", "d_rz", "gripper_target"]


@dataclass
class EpisodeLabels:
    poses: np.ndarray  # [n, 4, 4]
    proprio: np.ndarray  # [n, 13]
    action: np.ndarray  # [n, 7]
    label_is_extrapolated: np.ndarray  # [n] bool
    leader_follower_pos_err: np.ndarray  # [n] metres, |FK(action) - FK(state)| (clipping/lag diagnostic)
    jump_flags: np.ndarray  # [n] bool, implausible one-tick motion


def episode_labels(
    poses: np.ndarray,
    gripper_obs_pct: np.ndarray,
    gripper_cmd_pct: np.ndarray,
    hz: float,
    gripper_source: str,
    lead: int = 1,
    leader_poses: np.ndarray | None = None,
    max_step_m: float = 0.05,
    max_step_rad: float = 0.5,
) -> EpisodeLabels:
    """Labels for one episode already sampled at the policy clock.

    ``poses`` are achieved EE poses ``FK(q_obs)``; ``leader_poses`` (optional) ``FK(q_action)``
    feed only the diagnostic error, never the labels.
    """
    if gripper_source not in GRIPPER_SOURCES:
        raise ValueError(f"gripper_source must be one of {GRIPPER_SOURCES} (decision D-013)")
    n = len(poses)
    if not (len(gripper_obs_pct) == len(gripper_cmd_pct) == n):
        raise ValueError("length mismatch")
    p = poses[:, :3, 3]
    vel = np.zeros_like(p)
    vel[1:] = (p[1:] - p[:-1]) * hz
    rot6 = np.stack([rot6d_from_matrix(T[:3, :3]) for T in poses])
    proprio = np.concatenate([p, rot6, vel, (np.asarray(gripper_obs_pct)[:, None] / 100.0)], axis=1)

    action = np.zeros((n, 7))
    extrap = np.zeros(n, dtype=bool)
    jumps = np.zeros(n, dtype=bool)
    g_obs = np.asarray(gripper_obs_pct, np.float64) / 100.0
    g_cmd = np.asarray(gripper_cmd_pct, np.float64) / 100.0
    for k in range(n):
        j = k + lead
        if j < n:
            d = delta_action(poses[k], poses[j])
            g = g_cmd[k] if gripper_source == "leader_action" else g_obs[j]
        else:
            d = np.zeros(6)
            g = g_cmd[k] if gripper_source == "leader_action" else g_obs[k]
            extrap[k] = True
        action[k, :6] = d
        action[k, 6] = np.clip(g, 0.0, 1.0)
        jumps[k] = np.linalg.norm(d[:3]) > max_step_m * lead or np.linalg.norm(d[3:]) > max_step_rad * lead
    err = (
        np.linalg.norm(leader_poses[:, :3, 3] - p, axis=1)
        if leader_poses is not None
        else np.full(n, np.nan)
    )
    return EpisodeLabels(poses, proprio.astype(np.float64), action, extrap, err, jumps)
