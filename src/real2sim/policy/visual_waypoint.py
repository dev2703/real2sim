"""Image-guided pick/place with sparse IK and measured-joint feedback.

This controller supplies a working geometric baseline, not an ACT policy. Object
positions come exclusively from RGB plus explicitly supplied camera/object geometry.
No measured object poses, held flags or success metrics drive its state machine.
As with the existing IK controllers, collision checks use the supplied scene map,
including its exact static bowl/table geometry. This is not map-free control.
IK selects grasp branches and phase endpoints; execution follows bounded joint
segments, checked with FK and the original workspace/penetration gates.
"""

from __future__ import annotations

import copy
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from real2sim.evaluation.oracle import grasp_orientation
from real2sim.policy.adapters import ApplyResult
from real2sim.policy.visual_targets import camera_pose_task, locate_objects
from real2sim.sim.cameras import camera_intrinsics_px


class VisualWaypointController:
    """RGB targets -> feasible grasp -> bounded native-joint trajectory.

    The SO-101's fixed-jaw-tip site and the Panda's between-pad TCP are different
    physical points. Their ball offsets are explicit model assumptions. They are
    never presented as physical calibration or frozen-policy transfer.
    """

    controller_source = "RGB_WAYPOINT_CONTROLLER"
    phases = ("ACQUIRE", "PRE_GRASP", "DESCEND", "CLOSE", "LIFT", "TRANSPORT",
              "LOWER", "RELEASE", "RETREAT", "DONE")

    def __init__(self, env, *, robot="so101"):
        if robot not in ("so101", "panda"):
            raise ValueError("robot must be so101 or panda")
        self.env, self.robot = env, robot
        cfg = copy.deepcopy(env.config)
        # Polish numerical solves, keeping all hard native acceptance thresholds.
        cfg["ik"].update(position_convergence_m=1e-5, orientation_convergence_rad=1e-4)
        if robot == "so101":
            from real2sim.sim.kinematics import NativeKinematics

            self.ik = NativeKinematics(env.model, cfg)
            self.n = 5
            self.ball_in_tool = np.array([-.010, 0., .030])
        else:
            from real2sim.panda.adapter import PandaIK

            self.ik = PandaIK(env.model, cfg, task_origin=env.offset)
            self.n = 7
            # Seat a spherical object above the short pad centre rather than at
            # its lower edge after gravity loads the grasp. This is an explicit
            # geometric grasp parameter, not a change to contact/actuator force.
            self.ball_in_tool = np.array([-.008, 0., 0.])
        self.max_step_m = .024 if env.control_hz == 12.5 else .03
        self.workspace = (np.asarray(cfg["ik"]["workspace_min"], float),
                          np.asarray(cfg["ik"]["workspace_max"], float))
        self.reset()

    def reset(self):
        self.phase, self.phase_tick, self.tick = "ACQUIRE", 0, 0
        self.failure = None
        self.phase_log, self.plan_log, self.perception_log = [], [], []
        self.ik_calls = self.safety_checks = 0
        self.ball = self.bowl = self.target_q = self.pre_q = self.grasp_q = None
        self.grasp_rotation = None
        self._stable = 0

    def _native(self):
        return (self.env.sim_state() if self.robot == "so101"
                else self.env.arm_qpos())

    def _grip(self, opened):
        if self.robot == "panda":
            return self.env.max_opening if opened else 0.
        requested = self.env.mapping.gripper_real_to_sim_rad(90. if opened else 0.)
        return float(np.clip(requested, *self.env.limits[5]))

    def _forward(self, q, grip):
        if self.robot == "panda":
            return self.ik.forward(q[:7], finger_opening=grip)
        native = np.r_[q[:5], grip]
        return self.ik.forward(native)

    def _safe(self, q, grip):
        """Validate FK endpoints using the unchanged native workspace/contact rules."""
        pose = self._forward(q, grip)
        self.safety_checks += 1
        low, high = self.workspace
        if np.any(pose[:3, 3] < low) or np.any(pose[:3, 3] > high):
            return False
        return not (self.ik._collision() if self.robot == "so101" else self.ik._penetrates())

    def _path_safe(self, start, end, grip):
        cap = self.env.max_velocity[:self.n] / self.env.control_hz
        count = max(1, math.ceil(float(np.max(abs(end[:self.n] - start[:self.n]) / cap))))
        return all(self._safe(start + a * (end - start), grip)
                   for a in np.linspace(0., 1., count + 1))

    def _solve(self, target, seed, grip):
        self.ik_calls += 1
        if self.robot == "panda":
            return self.ik.solve(target, seed[:7], finger_opening=grip)
        return self.ik.solve(target, np.r_[seed[:5], grip], gripper=grip)

    def _acquire(self):
        # Native RGB and camera registration only. Do not inspect object model
        # body positions, depth/segmentation, env.info(), or privileged metrics.
        env = self.env
        image = env.render("iphone")
        camera = env.model.camera("iphone").id
        K = camera_intrinsics_px(env.model, camera, image.shape[1], image.shape[0])
        T = camera_pose_task(env.data.cam_xpos[camera], env.data.cam_xmat[camera], env.offset)
        table_z = env.table_top - env.offset[2]
        support_height = float(env.task["ball"].get("support", {}).get("height", 0))
        found = locate_objects(image, K, T, ball_radius_m=float(env.task["ball"]["radius"]),
            ball_center_height_m=table_z + support_height + env.task["ball"]["radius"],
            bowl_rim_height_m=table_z + env.bowl["height"],
            bowl_outer_radius_m=env.bowl["rim_radius"] + env.bowl["wall"],
            workspace_xy=[self.workspace[0][:2], self.workspace[1][:2]])
        self.perception_log.append({"tick": self.tick,
            "targets": {name: target.to_dict() if target else None for name, target in found.items()}})
        if any(target is None for target in found.values()):
            return False
        self.ball, self.bowl = (found[name].center_task.copy() for name in ("ball", "bowl"))
        return True

    def _plan_grasp(self):
        env, grip = self.env, self._grip(True)
        lo, hi = env.limits[:self.n].T
        current = self._native()[:self.n]
        candidates = []
        if self.robot == "so101":
            pan = float(np.arctan2(self.ball[1], self.ball[0]))
            seeds = [current] + [np.clip([pan, a, b, c, 0.], lo, hi)
                for a, b, c in ((-1.3, 1.3, 1.), (-.8, 1.3, .6), (-.3, .7, .6),
                                (.5, -1.3, 1.), (-.5, .6, -1.))]
            rotations = [grasp_orientation(self.ball[:2], tilt) for tilt in (10., 15., 20., 25.)]
        else:
            from real2sim.panda.model import POINTING_DOWN

            # A seven-joint Panda has several branches for one six-dimensional
            # tool pose. Its folded configured start alone can converge against
            # a wrist stop even when another native branch reaches the same
            # grasp. Fixed robot-joint seeds explore those branches; all results
            # still pass the original pose, native-limit and full-path gates.
            home = np.asarray(env.config["initial_qpos"], float)
            seeds = [current, home]
            for shoulder in (-.8, 0., .8):
                for elbow in (-.9, .9):
                    seed = home.copy()
                    seed[[0, 2, 4]] += [shoulder, elbow, -.5 * elbow]
                    seeds.append(np.clip(seed, lo, hi))
            seeds.append(np.clip([0., -.5, 0., -2.2, 0., 2., -.785398], lo, hi))
            rotations = [POINTING_DOWN]
        attempts = {"pre_rejected": 0, "grasp_rejected": 0, "path_rejected": 0}
        for rotation in rotations:
            target = np.eye(4)
            target[:3, :3] = rotation
            target[:3, 3] = self.ball - rotation @ self.ball_in_tool
            pre = target.copy()
            pre[2, 3] += .08
            for seed in seeds:
                first = self._solve(pre, seed, grip)
                if not first.accepted:
                    attempts["pre_rejected"] += 1
                    continue
                second = self._solve(target, first.qpos, grip)
                if not second.accepted:
                    attempts["grasp_rejected"] += 1
                    continue
                pre_q = np.clip(first.qpos[:self.n], lo, hi)
                grasp_q = np.clip(second.qpos[:self.n], lo, hi)
                if not self._path_safe(current, pre_q, grip) or not self._path_safe(pre_q, grasp_q, grip):
                    attempts["path_rejected"] += 1
                    continue
                score = float(np.linalg.norm(pre_q - current) + np.linalg.norm(grasp_q - pre_q))
                candidates.append((score, pre_q, grasp_q, rotation.copy()))
        if not candidates:
            self.failure = "No collision-free grasp branch satisfies native IK gates"
            return False
        _, self.pre_q, self.grasp_q, self.grasp_rotation = min(candidates, key=lambda c: c[0])
        self.plan_log.append({"phase": "GRASP", "candidate_count": len(candidates),
            "seed_count": len(seeds), "branch_rejections": attempts,
            "pre_qpos": self.pre_q.tolist(), "grasp_qpos": self.grasp_q.tolist(),
            "ball_in_tool_assumption_m": self.ball_in_tool.tolist(),
            "uses_object_ground_truth": False})
        return True

    def _plan_carry(self):
        env, current, grip = self.env, self._native()[:self.n], self._grip(False)
        lo, hi = env.limits[:self.n].T
        # Preserve the RGB-derived ball/tool geometry. Jaw readback, object
        # contacts, and a hidden held-ball transform never enter this estimate.
        pose = self._forward(current, grip)
        ball_now = pose[:3, 3] + pose[:3, :3] @ self.ball_in_tool
        if self.phase == "LIFT":
            goal = self.ball + [0, 0, .08]
        else:
            goal = self.bowl.copy()
            goal[2] += env.task["ball"]["radius"] + .03
            if self.phase == "LOWER":
                goal[2] -= .015
            elif self.phase == "RETREAT":
                goal[2] += .05
        opened = self.phase == "RETREAT"
        grip = self._grip(opened)
        rotation = self.grasp_rotation

        def residual(q):
            T = self._forward(q, grip)
            predicted = T[:3, 3] + T[:3, :3] @ self.ball_in_tool
            dr = Rotation.from_matrix(rotation @ T[:3, :3].T).as_rotvec()
            # SO-101 has five arm DOFs: carry the ball position while allowing
            # rotation to change. Panda retains the exact top-down hand pose.
            weight = .002 if self.robot == "so101" else .1
            return np.r_[predicted - goal, weight * dr, .0001 * (q - current)]

        fitted = least_squares(residual, current, bounds=(lo, hi), max_nfev=100,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        target = fitted.x
        T = self._forward(target, grip)
        position_error = float(np.linalg.norm(T[:3, 3] + T[:3, :3] @ self.ball_in_tool - goal))
        # Exact FK target is checked with the native solver once per phase.
        checked = self._solve(T, target, grip)
        if (not fitted.success or position_error > self.ik.config["position_tolerance_m"] or not checked.accepted
                or not self._path_safe(current, target, grip)):
            self.failure = f"{self.phase} has no feasible bounded path (error {position_error:.4f}m)"
            return False
        self.target_q = target.copy()
        self.plan_log.append({"phase": self.phase, "goal_ball_xyz": goal.tolist(),
            "estimated_start_ball_xyz": ball_now.tolist(), "target_qpos": target.tolist(),
            "planned_position_error_m": position_error, "ik_reason": checked.reason})
        return True

    def _transition(self, phase):
        self.phase_log.append({"phase": self.phase, "end_tick": self.tick,
                               "ticks": self.phase_tick, "next": phase})
        self.phase, self.phase_tick, self._stable = phase, 0, 0
        self.target_q = None

    def step(self, obs, env):
        if env is not self.env:
            raise ValueError("controller is bound to its construction environment")
        del obs  # Acquire uses the explicit native RGB camera; no info reads.
        current = self._native()[:self.n]
        rejected, reason = False, "native_segment"
        if self.phase == "ACQUIRE":
            if self._acquire() and self._plan_grasp():
                self._transition("PRE_GRASP")
            elif self.phase_tick >= round(2 * env.control_hz) or self.failure:
                self.failure = self.failure or "RGB ball/bowl detection unavailable"
                self._transition("FAILED")
        opened = self.phase in ("ACQUIRE", "PRE_GRASP", "DESCEND", "RELEASE", "RETREAT", "DONE", "FAILED")
        grip = self._grip(opened)
        if self.target_q is None:
            if self.phase == "PRE_GRASP":
                self.target_q = self.pre_q.copy()
            elif self.phase == "DESCEND":
                self.target_q = self.grasp_q.copy()
            elif self.phase in ("LIFT", "TRANSPORT", "LOWER", "RETREAT"):
                if not self._plan_carry():
                    self._transition("FAILED")
            else:
                self.target_q = current.copy()  # stable target during close/release/done
        goal = current.copy() if self.target_q is None else self.target_q.copy()
        delta = goal - current
        caps = env.max_velocity[:self.n] / env.control_hz
        delta /= max(1., float(np.max(abs(delta) / caps)))
        T0 = self._forward(current, grip)
        for _ in range(8):
            native = current + delta
            distance = np.linalg.norm(self._forward(native, grip)[:3, 3] - T0[:3, 3])
            if distance <= self.max_step_m:
                break
            delta *= self.max_step_m / distance * .999
        if self.phase not in ("ACQUIRE", "DONE", "FAILED") and not self._safe(native, grip):
            rejected, reason = True, "FK_workspace_or_collision"
            native = current.copy()
            grip = (float(env.sim_state()[5]) if self.robot == "so101"
                    else env.gripper_opening())
        action = np.r_[native, grip]
        observed, _, _, truncated, info = env.step(action)
        self.tick += 1
        self.phase_tick += 1
        before = self.phase
        error = float(np.max(abs(self._native()[:self.n] - goal)))
        reached = error < .012
        self._stable = self._stable + 1 if reached else 0
        if self.phase in ("CLOSE", "RELEASE"):
            if self.phase_tick >= round((.7 if self.phase == "CLOSE" else .8) * env.control_hz):
                self._transition("LIFT" if self.phase == "CLOSE" else "RETREAT")
        elif self.phase in ("PRE_GRASP", "DESCEND", "LIFT", "TRANSPORT", "LOWER", "RETREAT"):
            if self._stable >= 2:
                self._transition(self.phases[self.phases.index(self.phase) + 1])
            elif self.phase_tick >= round(8 * env.control_hz):
                self.failure = f"{self.phase} failed measured-joint convergence"
                self._transition("FAILED")
        return ApplyResult(observed, info, truncated,
            float(np.mean(info["control"]["limit_clipped"])), ik_rejected=rejected,
            detail={"phase": before, "ik_reason": reason, "joint_error_rad": error,
                    "requested_native": action.tolist(), "failure": self.failure,
                    "object_source": "RGB_and_declared_camera_geometry"})
