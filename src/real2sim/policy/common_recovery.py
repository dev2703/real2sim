"""Observable common Cartesian interface for a separately learned recovery MLP.

The policy sees RGB reference targets, measured proprioception and causal action
history. Native FK/IK and a supplied static collision map execute its seven-value
actions. Teacher phases and simulator dynamic-object state are never inputs.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from time import perf_counter

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


def _achieved_history_name(name):
    if "previous_gripper_open_fraction" in name:
        return name.replace("previous_gripper_open_fraction", "previous_issued_gripper_open_fraction")
    if "previous_" in name:
        return name.replace("previous_", "previous_achieved_")
    return name.replace("seconds_since_selected_gripper_switch", "seconds_since_issued_gripper_switch")


ACHIEVED_OBSERVATION_NAMES = [_achieved_history_name(name) for name in OBSERVATION_NAMES]
TRACKED_ACHIEVED_OBSERVATION_NAMES = [_achieved_history_name(name) for name in TRACKED_OBSERVATION_NAMES]


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
    def __init__(self, robot, hz, *, action_history="selected"):
        if action_history not in ("selected", "achieved"):
            raise ValueError("Recovery action history must be selected or achieved")
        self.robot, self.hz = robot, float(hz)
        self.action_history = action_history
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=3)
        self.previous_position = None
        self.previous_action = np.zeros(7)
        self.previous_open = None
        self.gripper_switch_tick = 0

    def selected_action(self, action, tick):
        if self.action_history != "selected":
            raise ValueError("Achieved-history observations require executed_action feedback")
        self._remember_action(action, tick)

    def executed_action(self, action, tick):
        """Remember measured TCP motion and issued jaw intent, without clipping.

        A contact impulse or tracking response can exceed the requested action
        cap. That remains observable feedback and must not be replaced by the
        rejected/ignored proposal or silently clipped to its intended motion.
        """
        if self.action_history != "achieved":
            raise ValueError("Selected-history observations require selected_action feedback")
        self._remember_action(action, tick)

    def _remember_action(self, action, tick):
        action = np.asarray(action, float)
        if (action.shape != (7,) or not np.isfinite(action).all()
                or not 0. <= action[6] <= 1.):
            raise ValueError("Recovery history requires six finite motions and a jaw fraction in 0..1")
        opened = bool(action[6] >= .5)
        if self.previous_open is None or opened != self.previous_open:
            self.gripper_switch_tick = tick
        self.previous_open = opened
        self.previous_action = action.copy()

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

    def __init__(self, episode, *, require_source_mapping=False):
        self.env, self.robot, self.source = episode.env, episode.robot, episode.source
        # Native Panda feasibility does not imply the six-value source ACT can
        # observe that pose. Apply this extra representation constraint only to
        # pipelines which actually require source ACT inputs.
        self.require_source_mapping = bool(require_source_mapping and self.robot == "panda")
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

    def _source_mapping_path(self, current, command, grip):
        """Check bounded native motion without mutating the virtual source.

        A 1 mm command margin sits inside the unchanged 3 mm measured-pose
        acceptance threshold. Samples cover both the measured-to-target joint
        path and the servo's rate-limited command path. This is a kinematic
        guard, not a prediction of contact dynamics or a substitute for checking
        measured feedback at the next ACT query.
        """
        maximum_error, checked = 0., 0
        current_projection = self.source.project_pose(self.env.ee_pose())
        maximum_error = current_projection.error_m
        if not current_projection.accepted:
            return False, {**current_projection.diagnostics(), "path_samples": 1,
                           "command_margin_m": .001, "failed_at": "measured_pose",
                           "reason": "virtual_source_unrepresentable"}
        paths = [(np.asarray(current), np.asarray(command))]
        # env.step limits from the last actuator target, not the measured arm.
        # Check that separately when an actual actuator interface is available.
        if hasattr(self.env, "actuator_ids"):
            previous = self.env.data.ctrl[self.env.actuator_ids].copy()
            applied = previous + np.clip(command - previous,
                -self.env.max_velocity / self.env.control_hz,
                self.env.max_velocity / self.env.control_hz)
            paths.append((previous, applied))
        for start, end in paths:
            for fraction in np.linspace(0., 1., 5):
                pose = self.guard.forward(start + fraction * (end - start), finger_opening=grip)
                projection = self.source.project_pose(pose)
                maximum_error = max(maximum_error, projection.error_m)
                checked += 1
                # Intermediate samples retain the measured bound, allowing a
                # pose near the boundary to move inward. Native target endpoints
                # have a tighter bound to leave tracking headroom.
                tolerance = projection.tolerance_m - (.001 if fraction == 1. else 0.)
                if not projection.accepted or projection.error_m > tolerance:
                    return False, {**projection.diagnostics(), "accepted": False,
                        "within_measured_tolerance": projection.accepted,
                        "command_tolerance_m": tolerance, "path_samples": checked,
                        "reason": ("virtual_source_command_margin" if projection.accepted
                                   else "virtual_source_unrepresentable"),
                        "command_margin_m": .001, "maximum_position_error_m": maximum_error,
                        "failed_at": "sampled_native_path"}
        return True, {"accepted": True, "maximum_position_error_m": maximum_error,
                      "position_tolerance_m": current_projection.tolerance_m,
                      "command_margin_m": .001, "path_samples": checked}

    def native_to_common(self, native):
        return clip_common(self.native_feedback_action(native))

    def native_feedback_action(self, native, *, before_pose=None):
        """FK of an issued/applied native command, separate from actual motion.

        Unlike a learned action request, diagnostic command feedback is not
        capped: clipping would hide a solver residual, branch change or servo
        rate limit. Native acceptance checks remain unchanged upstream.
        """
        native = np.asarray(native, float)
        if self.robot == "so101":
            target = self.guard.forward(native)
            grip = float(np.clip(self.env.mapping.gripper_sim_to_real_percent(native[5]) / 90., 0., 1.))
        else:
            target = self.guard.forward(native[:7], finger_opening=float(native[7]))
            grip = float(np.clip(native[7] / self.env.max_opening, 0., 1.))
        pose = self.env.ee_pose() if before_pose is None else before_pose
        return np.r_[delta_action(pose, target), grip]

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
            projection = None
            if accepted and getattr(self, "require_source_mapping", False):
                accepted, projection = self._source_mapping_path(current, solved.qpos, grip)
                if not accepted:
                    reason = projection["reason"]
            attempts.append({"fraction": candidate_fraction, "accepted": accepted, "reason": reason,
                             **({"source_mapping": projection} if projection is not None else {})})
            if accepted:
                fraction = candidate_fraction
                break
        if self.robot == "so101":
            command = np.r_[solved.qpos[:5], grip] if accepted else np.clip(current, *env.limits.T)
        else:
            command = np.r_[solved.qpos if accepted else current, grip]
        issued_action = self.native_feedback_action(command, before_pose=current_pose)
        obs, _, _, truncated, info = env.step(command)
        applied = info["control"].get("applied_radians" if self.robot == "so101" else "applied", command)
        applied_action = self.native_feedback_action(applied, before_pose=current_pose)
        achieved_action = np.r_[delta_action(current_pose, env.ee_pose()), issued_action[6]]
        executed = action.copy()
        executed[:6] = delta_action(current_pose, target) if accepted else 0.
        return ApplyResult(obs, info, truncated,
            float(np.mean(info["control"]["limit_clipped"])), ik_rejected=not accepted,
            detail={"common_action": executed.tolist(), "requested_common_action": action.tolist(),
                    "issued_common_action": issued_action.tolist(),
                    "applied_common_action": applied_action.tolist(),
                    "achieved_common_action": achieved_action.tolist(),
                    "achieved_action_semantics": "Measured pre/post TCP delta and issued native jaw intent; not clipped",
                    "feasibility_fraction": fraction, "feasibility_attempts": attempts,
                    "arm_joint_stop_command_margin_rad": .003,
                    "requires_virtual_source_mapping": getattr(self, "require_source_mapping", False),
                    "panda_orientation_projection": "fixed_top_down_expert_manifold" if self.robot == "panda" else None,
                    "requested_native": command.tolist(),
                    "ik_reason": reason, "uses_act": True, "intervention": True,
                    "controller_source": "LEARNED_RECOVERY_MLP"})


class FrozenACTFeatures:
    """Keep actual frozen decoder features aligned with each executed ACT query."""

    def __init__(self, episode, policy):
        self.episode, self.policy = episode, policy
        self.pending = deque()
        self.calls = 0
        self._captured = None
        self.last_source_projection = None
        self.last_inference_ms = 0.
        self.invalidations = 0
        self.last_invalidation_reason = None
        self.handle = policy.policy.model.action_head.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        features = inputs[0]
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError("Frozen ACT requires a single batch of decoder features")
        self._captured = features.detach().cpu().numpy()[0].copy()

    def invalidate(self, reason=None):
        """Requery after takeover or a recovery state transition.

        Pending actions and decoder rows describe the same old observation;
        discard them together while retaining causal processor history.
        """
        self.pending.clear()
        self._captured = None
        self.invalidations += 1
        self.last_invalidation_reason = reason

    def reset(self):
        self.pending.clear()
        self.calls = 0
        self.last_source_projection = None
        self.last_inference_ms = 0.
        self.invalidations = 0
        self.last_invalidation_reason = None
        self._captured = None
        self.policy.reset()

    def next(self):
        env, source = self.episode.env, self.episode.source
        self.last_inference_ms = 0.
        if not self.pending:
            started = perf_counter()
            if self.episode.robot == "so101":
                state = env.real_state()
            else:
                state, _ = source.source_state(env)
                self.last_source_projection = source.last_projection.copy()
                state[5] = float(np.clip(env.gripper_opening() / env.max_opening, 0., 1.) * 90.)
            images = {key: env.render(env.camera_bindings[key]) for key in self.policy.camera_keys}
            self._captured = None
            chunks = self.policy.predict_action_chunk(state, images)
            chunks = np.asarray(chunks)
            config = self.policy.config
            n = config.n_action_steps
            if chunks.ndim != 2:
                raise ValueError("Frozen ACT returned an invalid checkpoint action chunk")
            chunk_size = getattr(config, "chunk_size", chunks.shape[0])
            head = self.policy.policy.model.action_head
            feature_dim = getattr(config, "dim_model", getattr(head, "in_features", None))
            if (chunks.shape[0] != chunk_size or not 0 < n <= chunk_size
                    or not np.isfinite(chunks).all()):
                raise ValueError("Frozen ACT returned an invalid checkpoint action chunk")
            if (self._captured is None or self._captured.ndim != 2
                    or self._captured.shape[0] != chunk_size
                    or (feature_dim is not None and self._captured.shape[1] != feature_dim)
                    or not np.isfinite(self._captured).all()):
                raise ValueError("Frozen ACT decoder feature shape differs from the checkpoint")
            self.pending.extend(zip(chunks[:n], self._captured[:n], strict=True))
            self.calls += 1
            self.last_inference_ms = (perf_counter() - started) * 1000.
        raw, features = self.pending.popleft()
        return raw.copy(), features.astype(np.float32)

    def close(self):
        self.handle.remove()
