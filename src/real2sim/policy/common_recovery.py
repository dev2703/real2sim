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
TRACKED_OBSERVATION_NAMES = [name.replace("initial_rgb_", "fresh_rgb_") for name in OBSERVATION_NAMES]


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


def soft_stop_seed(measured, limits, tolerance=.001):
    """Project tiny soft-stop readback errors, never requested command limits."""
    measured, limits = np.asarray(measured, float), np.asarray(limits, float)
    bounded = np.clip(measured, *limits.T)
    if not np.isfinite(measured).all() or np.max(np.abs(bounded - measured)) > tolerance:
        raise ValueError("Measured joint exceeds soft-stop seed tolerance")
    return bounded


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

    def encode(self, env, ball, bowl, tick, *, visibility=(True, True), last_seen_ticks=None):
        pose = env.ee_pose()
        position = pose[:3, 3]
        velocity = (np.zeros(3) if self.previous_position is None
                    else (position - self.previous_position) * self.hz)
        self.previous_position = position.copy()
        last_seen = (0, 0) if last_seen_ticks is None else last_seen_ticks
        frame = np.r_[position, rot6d_from_matrix(pose[:3, :3]), velocity,
            semantic_gripper(env, self.robot), np.asarray(ball) - position,
            np.asarray(bowl) - position, float(visibility[0]), float(visibility[1]),
            (tick - last_seen[0]) / self.hz, (tick - last_seen[1]) / self.hz,
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
        current_pose = env.ee_pose()
        grip = (float(np.clip(env.mapping.gripper_real_to_sim_rad(action[6] * 90.), *env.limits[5]))
                if self.robot == "so101" else action[6] * env.max_opening)
        accepted, reason, fraction = False, "unattempted", 0.
        current = env.sim_state() if self.robot == "so101" else env.arm_qpos()
        if self.robot == "so101":
            self.guard.data.qpos[:] = env.data.qpos
            # MuJoCo soft joint stops can leave measured readbacks a fraction
            # of a milliradian past a stop. Project only that numerical seed;
            # requested commands still pass the unchanged hard-limit guard.
            arm_seed = soft_stop_seed(current[:5], self.guard.limits[:5])
            # The compliant jaw can cross its soft stop when empty. Its hinge
            # does not move the fixed-jaw TCP; seed it with the bounded command
            # while checking contacts at its actual measured pose below.
            guard_current = np.clip(current, *self.guard.limits.T)
            seed = np.r_[arm_seed, grip]
        # A bounded prediction may still be unreachable or intersect the static
        # map. Backtrack only its magnitude; every candidate retains all native
        # pose, workspace, collision and sampled-path acceptance thresholds.
        # No object goal or geometric corrective direction enters this step.
        attempts = []
        for candidate_fraction in (1., .5, .25, .125, .0625):
            target = apply_delta(current_pose, action[:6] * candidate_fraction)
            if self.robot == "so101":
                solved = self.guard.solve(target, seed, gripper=grip)
            else:
                from real2sim.panda.model import POINTING_DOWN

                # All Panda expert grasps use this task orientation, as does
                # the existing frozen-ACT retarget adapter. Anchor that known
                # action manifold instead of integrating tiny rotation errors.
                target[:3, :3] = POINTING_DOWN
                solved = self.guard.solve(target, current,
                    finger_opening=max(grip, env.gripper_opening()))
            accepted, reason = solved.accepted, solved.reason
            n_arm = 5 if self.robot == "so101" else 7
            if accepted:
                arm = solved.qpos[:n_arm]
                lower, upper = env.limits[:n_arm].T
                if np.any(arm < lower + .003) or np.any(arm > upper - .003):
                    accepted, reason = False, "command_joint_stop_margin"
            if accepted and self.robot == "so101":
                command = np.r_[solved.qpos[:5], grip]
                # Preserve checks on the actual pose as well as the projected
                # numerical path seed, so soft-stop handling cannot hide contact.
                actual = self.guard.forward(current)
                cfg = self.guard.config
                if (np.any(actual[:3, 3] < cfg["workspace_min"])
                        or np.any(actual[:3, 3] > cfg["workspace_max"])
                        or self.guard._collision()):
                    accepted, reason = False, "actual_pose_workspace_or_collision"
                else:
                    accepted, reason, _ = check_joint_request(self.guard, guard_current,
                        env.data.ctrl[env.actuator_ids], command, env.max_velocity / env.control_hz,
                        samples=17)
            attempts.append({"fraction": candidate_fraction, "accepted": accepted, "reason": reason})
            if accepted:
                fraction = candidate_fraction
                break
        if self.robot == "so101":
            command = np.r_[solved.qpos[:5], grip] if accepted else np.clip(current, *env.limits.T)
        else:
            command = np.r_[solved.qpos if accepted else current, grip]
        obs, _, _, truncated, info = env.step(command)
        executed = action.copy()
        executed[:6] = delta_action(current_pose, target) if accepted else 0.
        return ApplyResult(obs, info, truncated,
            float(np.mean(info["control"]["limit_clipped"])), ik_rejected=not accepted,
            detail={"common_action": executed.tolist(), "requested_common_action": action.tolist(),
                    "feasibility_fraction": fraction, "feasibility_attempts": attempts,
                    "arm_joint_stop_command_margin_rad": .003,
                    "panda_orientation_projection": "fixed_top_down_expert_manifold" if self.robot == "panda" else None,
                    "requested_native": command.tolist(),
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
