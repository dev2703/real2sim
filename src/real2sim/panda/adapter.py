"""Panda ``CartesianActionAdapter`` (docs/architecture.md section 4.4, plan M5 step 2).

The 7-D canonical action ``[dx, dy, dz, d_rx, d_ry, d_rz, gripper_target]`` is interpreted
exactly as for the SO-101: translation in task-frame metres and a rotation vector
(``log(R_next R_cur^T)``) in radians, both relative to the **observed** TCP pose of the tick,
followed by a semantic gripper target in [0, 1]. Pipeline per tick:

``clamp (‖dp‖ ≤ 0.03 m, ‖dr‖ ≤ 0.3 rad) → target = apply_delta(T_obs, delta) → workspace box →
full-pose damped-least-squares IK over the 7 arm joints (3 mm / 0.05 rad hard thresholds,
joint limits, penetration check) → gripper width = g * 0.08 m → env.step``.

A rejected target never moves the arm: the tick re-commands the arm joints observed at the start
of the tick (a hold, as the SO-101 adapter does; the gripper command is still applied) and the
rejection is counted. The IK penetration check is run with the fingers at the wider of the
observed and commanded openings, so open fingers meeting the bowl wall are rejected before the
physics sees them.

The IK here is deliberately separate from ``real2sim.sim.kinematics.NativeKinematics``, whose
code is hard-wired to the SO-101's five arm joints plus the gripper hinge at index 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from real2sim.data.labels import ACTION_NAMES
from real2sim.geometry.so3 import apply_delta, delta_action, is_rotation, log_so3, make_T, matrix_from_rot6d
from real2sim.panda.env import N_ARM, PandaPickPlaceEnv
from real2sim.panda.model import BASE_BODY, FINGER_BODIES


@dataclass(frozen=True)
class IKResult:
    qpos: np.ndarray  # 7 arm joints; the seed when rejected
    accepted: bool
    position_error_m: float
    orientation_error_rad: float
    iterations: int
    reason: str  # accepted | outside_workspace | joint_limit | residual_rejected | collision


SEED_LIMIT_TOLERANCE_RAD = 1e-3  # live qpos may sit a few ULP (or soft-limit overshoot) past a limit


class PandaIK:
    """Full-pose damped-least-squares IK over the seven arm joints on an independent MjData.

    ``config`` needs ``joint_names`` (7), ``tool_site`` and an ``ik`` block with
    ``position_tolerance_m, orientation_tolerance_rad, damping, max_iterations,
    max_joint_step_rad, workspace_min, workspace_max, penetration_tolerance_m``. Poses are
    ``T_base_tcp`` with the base body an unrotated translation of the world (= task frame).
    Both residuals are hard acceptance criteria (no soft orientation).
    """

    def __init__(self, model: mujoco.MjModel, config: dict, *, task_origin=None):
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config["ik"]
        self.position_convergence = float(self.config.get(
            "position_convergence_m", 0.1 * self.config["position_tolerance_m"]))
        self.orientation_convergence = float(self.config.get(
            "orientation_convergence_rad", 0.1 * self.config["orientation_tolerance_rad"]))
        for value, limit in ((self.position_convergence, self.config["position_tolerance_m"]),
                             (self.orientation_convergence, self.config["orientation_tolerance_rad"])):
            if not np.isfinite(value) or not 0 < value <= limit:
                raise ValueError("IK convergence must be positive and no looser than acceptance")
        if len(config["joint_names"]) != N_ARM:
            raise ValueError("PandaIK expects seven arm joints")
        ids = np.array([model.joint(name).id for name in config["joint_names"]])
        self.qadr = model.jnt_qposadr[ids]
        self.vadr = model.jnt_dofadr[ids]
        self.limits = model.jnt_range[ids].copy()
        self.site = model.site(config["tool_site"]).id
        base = model.body(BASE_BODY).id
        if not np.allclose(model.body_quat[base], [1, 0, 0, 0]):
            raise ValueError("base-frame kinematics assume an unrotated robot base")
        # Poses are expressed in the shared task frame: world minus the task origin (which is NOT
        # the Panda base position; see real2sim.panda.model.panda_base_world).
        if task_origin is None:
            raise ValueError("PandaIK needs the task-frame origin (env.offset)")
        self.task_origin = np.asarray(task_origin, float).reshape(3)
        self.ball_geoms = {g for g in range(model.ngeom)
                          if model.body(model.geom_bodyid[g]).name == "ball"}
        self.finger_geoms = {g for g in range(model.ngeom)
                             if model.body(model.geom_bodyid[g]).name in FINGER_BODIES}
        self.workspace = (np.asarray(self.config["workspace_min"], float),
                          np.asarray(self.config["workspace_max"], float))

    def forward(self, qpos, *, finger_opening: float | None = None) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (N_ARM,) or not np.all(np.isfinite(qpos)):
            raise ValueError("FK expects seven finite native joint radians")
        self.data.qpos[self.qadr] = qpos
        if finger_opening is not None:
            for name in ("finger_joint1", "finger_joint2"):
                self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = finger_opening / 2
        mujoco.mj_forward(self.model, self.data)
        return make_T(self.data.site_xmat[self.site].reshape(3, 3),
                      self.data.site_xpos[self.site] - self.task_origin)

    def set_finger_opening(self, finger_opening: float) -> None:
        """Pose the IK data's fingers (total opening, m) so penetration checks see them."""
        opening = float(finger_opening)
        if not np.isfinite(opening) or opening < 0:
            raise ValueError("finger_opening must be a finite nonnegative total opening in metres")
        for name in ("finger_joint1", "finger_joint2"):
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = opening / 2

    def _penetrates(self) -> bool:
        """Robot penetrating the table/bowl or itself beyond tolerance (ball contact tolerated)."""
        tol = self.config["penetration_tolerance_m"]
        for contact in self.data.contact[:self.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self.ball_geoms or pair <= self.finger_geoms:
                continue
            if contact.dist < -tol:
                return True
        return False

    def solve(self, target: np.ndarray, seed: np.ndarray, *,
              finger_opening: float | None = None) -> IKResult:
        """Solve ``target`` (T_base_tcp) from ``seed``; ``finger_opening`` (total, m) poses the
        fingers for the penetration check (``None`` keeps the data's current finger pose).

        A seed within ``SEED_LIMIT_TOLERANCE_RAD`` outside a joint limit is clipped onto the
        limit (the live arm routinely sits a few ULP past a limit after a clipped solution);
        anything further out is a caller error. On rejection ``qpos`` is that clipped seed.
        """
        target, seed = np.asarray(target, dtype=float), np.asarray(seed, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)) or \
                not np.allclose(target[3], [0, 0, 0, 1]) or not is_rotation(target[:3, :3]):
            raise ValueError("IK target must be a finite proper 4x4 transform")
        if seed.shape != (N_ARM,) or not np.all(np.isfinite(seed)) or \
                np.any(seed < self.limits[:, 0] - SEED_LIMIT_TOLERANCE_RAD) or \
                np.any(seed > self.limits[:, 1] + SEED_LIMIT_TOLERANCE_RAD):
            raise ValueError("IK seed must be seven finite radians inside the joint limits")
        seed = np.clip(seed, self.limits[:, 0], self.limits[:, 1])
        if finger_opening is not None:
            self.set_finger_opening(finger_opening)
        cfg = self.config
        p_t, R_t = target[:3, 3], target[:3, :3]
        if np.any(p_t < self.workspace[0]) or np.any(p_t > self.workspace[1]):
            T = self.forward(seed)
            return IKResult(seed.copy(), False, float(np.linalg.norm(p_t - T[:3, 3])),
                            float(np.linalg.norm(log_so3(R_t @ T[:3, :3].T))), 0,
                            "outside_workspace")

        def residual(T):
            return np.concatenate([p_t - T[:3, 3], log_so3(R_t @ T[:3, :3].T)])

        pos_tol, rot_tol = cfg["position_tolerance_m"], cfg["orientation_tolerance_rad"]
        q = seed.copy()
        r = residual(self.forward(q))
        iteration = 0
        while iteration < cfg["max_iterations"]:
            pe, re = np.linalg.norm(r[:3]), np.linalg.norm(r[3:])
            if pe <= self.position_convergence and re <= self.orientation_convergence:
                break  # polished well inside the acceptance thresholds
            iteration += 1
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros_like(jp)
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
            jac = np.vstack([jp[:, self.vadr], jr[:, self.vadr]])
            delta = np.linalg.solve(jac.T @ jac + cfg["damping"] ** 2 * np.eye(N_ARM), jac.T @ r)
            delta *= min(1.0, cfg["max_joint_step_rad"] / max(np.max(np.abs(delta)), 1e-12))
            improved = False
            for scale in (1.0, 0.5, 0.25, 0.1):
                candidate = np.clip(q + scale * delta, self.limits[:, 0], self.limits[:, 1])
                rc = residual(self.forward(candidate))
                if np.linalg.norm(rc) < np.linalg.norm(r):
                    q, r, improved = candidate, rc, True
                    break
            if not improved:
                break
        r = residual(self.forward(q))
        pe, re = float(np.linalg.norm(r[:3])), float(np.linalg.norm(r[3:]))
        if pe <= pos_tol and re <= rot_tol:
            if self._penetrates():
                return IKResult(seed.copy(), False, pe, re, iteration, "collision")
            return IKResult(q.copy(), True, pe, re, iteration, "accepted")
        at_limit = np.any(np.abs(q - self.limits[:, 0]) < 1e-6) or np.any(
            np.abs(q - self.limits[:, 1]) < 1e-6)
        return IKResult(seed.copy(), False, pe, re, iteration,
                        "joint_limit" if at_limit else "residual_rejected")


@dataclass
class ControlResult:
    joint_cmd: np.ndarray  # 8 native values sent to env.step (7 rad + opening m)
    target_pose: np.ndarray  # 4x4 T_task_tcp after clamping
    residual_position_m: float
    residual_orientation_rad: float
    rejected: bool
    reason: str
    translation_clamped: bool
    rotation_clamped: bool
    ik_iterations: int
    observation: dict = field(repr=False)
    info: dict = field(repr=False)
    truncated: bool = False


class PandaCartesianActionAdapter:
    """7-D contract -> Panda native joints; see module docstring. Keeps per-episode tallies."""

    def __init__(self, env: PandaPickPlaceEnv, *, target_reference: str = "current_pose"):
        if target_reference not in ("current_pose", "accepted_target"):
            raise ValueError("unknown Cartesian target reference")
        self.target_reference = target_reference
        self.env = env
        self.ik = PandaIK(env.model, env.config, task_origin=env.offset)
        limits = env.config["action_limits"]
        self.max_translation = float(limits["max_translation_m"])
        self.max_rotation = float(limits["max_rotation_rad"])
        self.state_key = env.config["observation"]["state_key"]
        self.reset_stats()

    def reset_stats(self) -> None:
        self._accepted_target = None
        self.stats = {"ticks": 0, "rejected": 0, "reasons": {}, "translation_clamped": 0,
                      "rotation_clamped": 0, "residual_position_m": [],
                      "residual_orientation_rad": []}

    def observe(self) -> dict:
        return self.env.observation()

    @staticmethod
    def pose_from_state(state13: np.ndarray) -> np.ndarray:
        """Observed TCP pose from the canonical 13-float proprioception (no simulator peeking)."""
        s = np.asarray(state13, dtype=float)
        if s.shape != (13,):
            raise ValueError("observation.state must have 13 floats")
        return make_T(matrix_from_rot6d(s[3:9]), s[:3])

    @staticmethod
    def clamp(delta6: np.ndarray, max_translation: float, max_rotation: float):
        d = np.array(delta6, dtype=float)
        tn, rn = np.linalg.norm(d[:3]), np.linalg.norm(d[3:])
        t_clamped, r_clamped = tn > max_translation, rn > max_rotation
        if t_clamped:
            d[:3] *= max_translation / tn
        if r_clamped:
            d[3:] *= max_rotation / rn
        return d, bool(t_clamped), bool(r_clamped)

    def gripper_opening_m(self, g: float) -> float:
        """Semantic target in [0,1] -> total finger opening in metres (0 closed, 1 fully open)."""
        return float(np.clip(g, 0.0, 1.0)) * self.env.max_opening

    def apply(self, action7, obs: dict) -> ControlResult:
        action = np.asarray(action7, dtype=float)
        if action.shape != (len(ACTION_NAMES),) or not np.all(np.isfinite(action)):
            raise ValueError("action must be seven finite floats")
        T_obs = self.pose_from_state(obs[self.state_key])
        delta, t_clamped, r_clamped = self.clamp(action[:6], self.max_translation,
                                                 self.max_rotation)
        reference = T_obs if self._accepted_target is None or self.target_reference == "current_pose" else self._accepted_target
        target = apply_delta(reference, delta)
        if self.target_reference == "accepted_target":
            feedback, feedback_t, feedback_r = self.clamp(delta_action(T_obs, target), self.max_translation, self.max_rotation)
            target = apply_delta(T_obs, feedback)
            t_clamped |= feedback_t
            r_clamped |= feedback_r
        seed = self.env.arm_qpos()
        opening = self.gripper_opening_m(action[6])
        # Penetration check with the fingers at the wider of observed and commanded openings
        # (the rate-limited fingers are somewhere in between during the tick).
        check_opening = max(opening, self.env.gripper_opening())
        result = self.ik.solve(target, seed, finger_opening=check_opening)
        # Rejected: hold the arm joints observed at the start of the tick (result.qpos is the
        # limit-clipped seed), never the previous tick's possibly still-converging ctrl target.
        joint_cmd = np.append(result.qpos, opening)
        observation, _, _, truncated, info = self.env.step(joint_cmd)
        if self.target_reference == "accepted_target" and result.accepted:
            self._accepted_target = target.copy()
        info["adapter"] = {"rejected": not result.accepted, "reason": result.reason,
                           "residual_position_m": result.position_error_m,
                           "residual_orientation_rad": result.orientation_error_rad,
                           "translation_clamped": t_clamped, "rotation_clamped": r_clamped}
        s = self.stats
        s["ticks"] += 1
        s["rejected"] += int(not result.accepted)
        s["reasons"][result.reason] = s["reasons"].get(result.reason, 0) + 1
        s["translation_clamped"] += int(t_clamped)
        s["rotation_clamped"] += int(r_clamped)
        s["residual_position_m"].append(result.position_error_m)
        s["residual_orientation_rad"].append(result.orientation_error_rad)
        return ControlResult(joint_cmd, target, result.position_error_m,
                             result.orientation_error_rad, not result.accepted, result.reason,
                             t_clamped, r_clamped, result.iterations, observation, info, truncated)
