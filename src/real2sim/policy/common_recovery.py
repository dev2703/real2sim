"""Observable common Cartesian interface for a separately learned recovery MLP.

The policy sees RGB reference targets, measured proprioception and causal action
history. Native FK/IK and a supplied static collision map execute its seven-value
actions. Teacher phases and simulator dynamic-object state are never inputs.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy

import numpy as np

from real2sim.geometry.so3 import apply_delta, delta_action, rot6d_from_matrix
from real2sim.policy.adapters import ApplyResult
from real2sim.policy.follower_joint import check_joint_request

ACTION_NAMES = ["tcp_dx_m", "tcp_dy_m", "tcp_dz_m", "world_drx_rad", "world_dry_rad",
                "world_drz_rad", "gripper_open_fraction"]
ACTION_SCALE = np.array([.03, .03, .03, .3, .3, .3, 1.], dtype=np.float64)
FRAME_NAMES = ([f"tcp_{a}_m" for a in "xyz"] + [f"rotation6d_{i}" for i in range(6)]
    + [f"velocity_{a}_mps" for a in "xyz"] + ["gripper_measured"]
    + [f"initial_rgb_ball_relative_{a}" for a in "xyz"]
    + [f"initial_rgb_bowl_relative_{a}" for a in "xyz"]
    + ["ball_observed", "bowl_observed", "ball_observation_age_s", "bowl_observation_age_s"]
    + [f"previous_{name}" for name in ACTION_NAMES] + ["robot_so101", "robot_panda"]
    + ["seconds_since_selected_gripper_switch"])
OBSERVATION_NAMES = [f"history_{j}_{name}" for j in range(3) for name in FRAME_NAMES]


def clip_common(action):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Common recovery action must contain seven finite values")
    result = action.copy()
    for part, maximum in ((slice(0, 3), .03), (slice(3, 6), .3)):
        length = np.linalg.norm(result[part])
        if length > maximum:
            result[part] *= maximum / length
    result[6] = np.clip(result[6], 0., 1.)
    return result


def semantic_gripper(env, robot):
    if robot == "panda":
        return float(env.gripper_semantic())
    return float(np.clip(env.mapping.gripper_sim_to_real_percent(env.sim_state()[5]) / 90., 0., 1.))


class RecoveryObservation:
    def __init__(self, robot, hz):
        self.robot, self.hz = robot, float(hz)
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=3)
        self.previous_position = None
        self.previous_action = np.zeros(7)
        self.previous_open = None
        self.gripper_switch_tick = 0

    def selected_action(self, action, tick):
        opened = bool(action[6] >= .5)
        if self.previous_open is None or opened != self.previous_open:
            self.gripper_switch_tick = tick
        self.previous_open = opened
        self.previous_action = np.asarray(action, float).copy()

    def encode(self, env, ball, bowl, tick):
        pose = env.ee_pose()
        position = pose[:3, 3]
        velocity = (np.zeros(3) if self.previous_position is None
                    else (position - self.previous_position) * self.hz)
        self.previous_position = position.copy()
        frame = np.r_[position, rot6d_from_matrix(pose[:3, :3]), velocity,
            semantic_gripper(env, self.robot), np.asarray(ball) - position,
            np.asarray(bowl) - position, 1., 1., tick / self.hz, tick / self.hz,
            self.previous_action, float(self.robot == "so101"), float(self.robot == "panda"),
            (tick - self.gripper_switch_tick) / self.hz]
        if frame.shape != (33,) or not np.isfinite(frame).all():
            raise ValueError("Recovery observation must contain 33 finite observable values")
        if not self.frames:
            self.frames.extend([frame.copy()] * 2)
        self.frames.append(frame)
        return np.concatenate(self.frames).astype(np.float32)


class CommonCartesianExecutor:
    """Execute a learned common action under each robot's existing native gates."""

    def __init__(self, episode):
        self.env, self.robot, self.source = episode.env, episode.robot, episode.source
        config = deepcopy(self.env.config)
        # Numerical polish is necessary for small per-tick targets; retain the
        # same hard acceptance tolerances, native limits and collision gates.
        config["ik"].update(position_convergence_m=1e-5, orientation_convergence_rad=1e-4)
        if self.robot == "so101":
            from real2sim.sim.kinematics import NativeKinematics

            self.guard = NativeKinematics(self.env.model, config)
        else:
            from real2sim.panda.adapter import PandaIK

            self.guard = PandaIK(self.env.model, config, task_origin=self.env.offset)

    def native_to_common(self, native):
        native = np.asarray(native, float)
        if self.robot == "so101":
            target = self.guard.forward(native)
            grip = float(np.clip(self.env.mapping.gripper_sim_to_real_percent(native[5]) / 90., 0., 1.))
        else:
            target = self.guard.forward(native[:7], finger_opening=float(native[7]))
            grip = float(np.clip(native[7] / self.env.max_opening, 0., 1.))
        return clip_common(np.r_[delta_action(self.env.ee_pose(), target), grip])

    def hf_to_common(self, raw):
        _, bounded = self.source.source_action(raw)
        if self.robot == "so101":
            target = self.guard.forward(bounded)
        else:
            from real2sim.panda.model import POINTING_DOWN

            target = np.eye(4)
            target[:3, :3] = POINTING_DOWN
            target[:3, 3] = self.source.ball_point(bounded) - POINTING_DOWN @ self.source.panda_offset
        return clip_common(np.r_[delta_action(self.env.ee_pose(), target), raw[5] / 90.])

    def step(self, action):
        env = self.env
        action = clip_common(action)
        target = apply_delta(env.ee_pose(), action[:6])
        grip = (float(np.clip(env.mapping.gripper_real_to_sim_rad(action[6] * 90.), *env.limits[5]))
                if self.robot == "so101" else action[6] * env.max_opening)
        if self.robot == "so101":
            current = env.sim_state()
            self.guard.data.qpos[:] = env.data.qpos
            # MuJoCo soft joint stops can leave measured readbacks a fraction
            # of a milliradian past a stop. Project only that numerical seed;
            # requested commands still pass the unchanged hard-limit guard.
            seed = np.clip(current, *self.guard.limits.T)
            if np.max(np.abs(seed[:5] - current[:5])) > .001:
                raise ValueError("SO101 measured arm exceeds soft-stop seed tolerance")
            seed[5] = grip
            solved = self.guard.solve(target, seed, gripper=grip)
            accepted, reason = solved.accepted, solved.reason
            command = np.r_[solved.qpos[:5], grip] if accepted else current.copy()
            if accepted:
                accepted, reason, _ = check_joint_request(self.guard, current,
                    env.data.ctrl[env.actuator_ids], command, env.max_velocity / env.control_hz,
                    samples=17)
                if not accepted:
                    command = current.copy()
        else:
            solved = self.guard.solve(target, env.arm_qpos(),
                finger_opening=max(grip, env.gripper_opening()))
            accepted, reason = solved.accepted, solved.reason
            command = np.r_[solved.qpos if accepted else env.arm_qpos(), grip]
        obs, _, _, truncated, info = env.step(command)
        return ApplyResult(obs, info, truncated,
            float(np.mean(info["control"]["limit_clipped"])), ik_rejected=not accepted,
            detail={"common_action": action.tolist(), "requested_native": command.tolist(),
                    "ik_reason": reason, "uses_act": True, "intervention": True,
                    "controller_source": "LEARNED_RECOVERY_MLP"})


class FrozenACTFeatures:
    """Keep actual frozen decoder features aligned with each executed ACT query."""

    def __init__(self, episode, policy):
        self.episode, self.policy = episode, policy
        self.pending = []
        self.calls = 0
        self._captured = None
        self.handle = policy.policy.model.action_head.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        self._captured = inputs[0].detach().cpu().numpy()[0].copy()

    def reset(self):
        self.pending.clear()
        self.calls = 0
        self.policy.reset()

    def next(self):
        env, source = self.episode.env, self.episode.source
        if not self.pending:
            if self.episode.robot == "so101":
                state = env.real_state()
            else:
                state, _ = source.source_state(env)
                state[5] = float(np.clip(env.gripper_opening() / env.max_opening, 0., 1.) * 90.)
            images = {key: env.render(env.camera_bindings[key]) for key in self.policy.camera_keys}
            chunks = self.policy.predict_action_chunk(state, images)
            n = self.policy.config.n_action_steps
            if self._captured is None or self._captured.shape != (50, 256):
                raise ValueError("Frozen ACT decoder feature shape differs from the pinned checkpoint")
            self.pending = list(zip(chunks[:n], self._captured[:n], strict=True))
            self.calls += 1
        raw, features = self.pending.pop(0)
        return raw.copy(), features.astype(np.float32)

    def close(self):
        self.handle.remove()
