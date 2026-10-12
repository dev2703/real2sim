"""Seeded MuJoCo environment for the SO-101 pick-and-place task.

Native actions are six absolute hinge positions in radians (including the gripper). Dataset-unit
adapters (LeRobot degrees, gripper 0-100 %) use the provisional mapping in ``configs/task.yaml``
(``robot_so101``) through :class:`real2sim.geometry.so101_fk.JointMapping`; the mapping is
PROVISIONAL until M2 verification and every observation carries ``calibrated: False``.

Observations never contain simulator object poses. Ground truth lives only in
``info()['privileged_metrics']`` for scoring and labelled diagnostics (plan section 7).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from real2sim.config import load_yaml, validate_camera_keys
from real2sim.data.transforms import letterbox_rgb, to_chw_float
from real2sim.geometry.so3 import rot6d_from_matrix
from real2sim.geometry.so101_fk import JointMapping
from real2sim.sim.model import (
    DEFAULT_CONFIG,
    bowl_dimensions,
    load_scene,
    project_path,
    task_frame_offset,
)

GRIPPER_BODIES = ("gripper", "moving_jaw_so101_v1")  # fixed jaw body, moving jaw body


def real_from_sim(mapping: JointMapping, q_sim6: np.ndarray) -> np.ndarray:
    """Inverse of ``JointMapping``: native radians -> LeRobot degrees and gripper percent."""
    q = np.asarray(q_sim6, np.float64)
    arm = (np.rad2deg(q[:5]) - np.asarray(mapping.offset_deg)) / np.asarray(mapping.sign)
    lo, hi = mapping.gripper_sim_range_rad
    return np.concatenate([arm, [(q[5] - lo) / (hi - lo) * 100.0]])


def sim_from_real(mapping: JointMapping, q_real6: np.ndarray) -> np.ndarray:
    q = np.asarray(q_real6, np.float64)
    return np.concatenate([mapping.arm_real_to_sim_rad(q), [mapping.gripper_real_to_sim_rad(q[5])]])


class So101PickPlaceEnv:
    """Gym-style reset/step tuples; reward is zero (no benchmark definition at M2)."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG, *, render_images: bool = True,
                 bowl_inner_depth: float | None = None, contacts: dict | None = None):
        self.model, self.config, self.task = load_scene(
            config_path, bowl_inner_depth=bowl_inner_depth, contacts=contacts)
        self.data = mujoco.MjData(self.model)
        self.render_images = render_images
        self.renderer = None
        self.data_config = load_yaml(project_path(self.config["data_config"]))
        self.camera_keys = validate_camera_keys(self.data_config, [self.task["camera_iphone"]["key"]])
        self.mapping = JointMapping.from_task_config(self.task)
        self.offset = task_frame_offset(self.task)  # p_world = p_task + offset
        self.bowl = bowl_dimensions(self.task, bowl_inner_depth)
        self.joint_ids = np.array([self.model.joint(name).id for name in self.config["joint_names"]])
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([
            self.model.actuator(name).id for name in self.config["joint_names"]
        ])
        self.limits = np.column_stack([
            np.maximum(self.model.jnt_range[self.joint_ids, 0],
                       self.model.actuator_ctrlrange[self.actuator_ids, 0]),
            np.minimum(self.model.jnt_range[self.joint_ids, 1],
                       self.model.actuator_ctrlrange[self.actuator_ids, 1]),
        ])
        ratio = self.config["physics_hz"] / self.config["control_hz"]
        if not ratio.is_integer() or ratio < 1:
            raise ValueError("physics_hz must be an integer multiple of control_hz")
        if self.config["control_hz"] != self.data_config["policy_hz"]:
            raise ValueError("control frequency differs from approved data contract")
        if self.config["control"]["interface"] != "native_mujoco_absolute_joint_positions":
            raise ValueError("only native MuJoCo absolute joint position actions are supported")
        self.substeps = int(ratio)
        self.control_hz = float(self.config["control_hz"])
        self.max_episode_steps = int(round(self.task["episode"]["timeout_s"] * self.control_hz))
        self.max_velocity = self._vector(self.config["control"]["max_velocity"], "max_velocity")
        if np.any(self.max_velocity <= 0):
            raise ValueError("max_velocity must be positive")
        self.initial_qpos = self._vector(self.config["initial_qpos"], "initial_qpos")
        expected = np.clip(sim_from_real(self.mapping, self.config["initial_qpos_real_deg"]),
                           self.limits[:, 0], self.limits[:, 1])
        if not np.allclose(self.initial_qpos, expected, atol=1e-6):
            raise ValueError("initial_qpos is inconsistent with initial_qpos_real_deg under the "
                             f"task.yaml mapping (expected {expected.tolist()})")
        self.ball_geom = self.model.geom("ball_geom").id
        self.ball_body = self.model.body("ball").id
        self.table_geom = self.model.geom("table").id
        self.base_body = self.model.body("base").id
        self.site_id = self.model.site(self.config["tool_site"]).id
        self.gripper_geoms = {
            name: {g for g in range(self.model.ngeom)
                   if self.model.body(self.model.geom_bodyid[g]).name == name
                   and self.model.geom_contype[g] | self.model.geom_conaffinity[g]}
            for name in GRIPPER_BODIES
        }
        ball_joint = self.model.joint("ball_free").id
        self._ball_q = self.model.jnt_qposadr[ball_joint]
        self._ball_v = self.model.jnt_dofadr[ball_joint]
        self.steps = 0
        self._ready = False
        self._bowl_xy_default = np.asarray(self.bowl["center_xy"], float)
        self.start_pose = {}

    @staticmethod
    def _vector(value, name: str, size: int = 6) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite vector of length {size}")
        return result

    # ------------------------------------------------------------------ episode control
    def reset(self, *, seed: int | None = None, scenario: dict | None = None) -> tuple[dict, dict]:
        """Deterministic reset. ``scenario`` keys are task-frame metres (``ball_xy``, ``bowl_xy``)."""
        scenario = dict(scenario or {})
        allowed = {"ball_xy", "bowl_xy", "ball_drop_height", "ball_velocity", "joint_positions"}
        if unknown := scenario.keys() - allowed:
            raise ValueError(f"unknown scenario fields: {sorted(unknown)}")
        rng = np.random.default_rng(seed)
        region = self.task["ball"]["start_region"]
        # Uniform over the start disc: radius ~ sqrt(u) keeps the density uniform in area.
        r = float(region["radius"]) * np.sqrt(rng.uniform())
        theta = rng.uniform(0, 2 * np.pi)
        default_ball = np.asarray(region["center_task_xy"], float) + r * np.array([np.cos(theta),
                                                                                   np.sin(theta)])
        ball_xy = self._vector(scenario.get("ball_xy", default_ball), "ball_xy", 2)
        bowl_xy = self._vector(scenario.get("bowl_xy", self._bowl_xy_default), "bowl_xy", 2)
        q = self._vector(scenario.get("joint_positions", self.initial_qpos), "joint_positions")
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise ValueError("initial joint positions exceed upstream model limits")
        drop = float(scenario.get("ball_drop_height", 0.001))
        if not np.isfinite(drop) or drop < 0:
            raise ValueError("ball_drop_height must be finite and nonnegative")
        velocity = self._vector(scenario.get("ball_velocity", [0, 0, 0]), "ball_velocity", 3)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[self.model.body("bowl").id, :2] = bowl_xy
        self.data.qpos[self.qadr] = q
        self.data.ctrl[self.actuator_ids] = q
        bq, bv = self._ball_q, self._ball_v
        self.data.qpos[bq:bq + 3] = [*(ball_xy + self.offset[:2]),
                                    self.table_top + self.task["ball"]["radius"] + drop]
        self.data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
        self.data.qvel[bv:bv + 3] = velocity
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self._ready = True
        # Known limitation (docs/decisions.md D-014, pending): with the base on the table top the
        # clamped dataset rest pose puts gripper geoms into the table, so the first control tick
        # is a contact-driven jump. Recorded here so every trial carries the confound explicitly.
        self.start_pose = self._robot_table_penetration()
        return self.observation(), self.info()

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        if not self._ready:
            raise RuntimeError("reset is required before step or after truncation")
        requested = self._vector(action, "native joint action (radians)")
        bounded = np.clip(requested, self.limits[:, 0], self.limits[:, 1])
        start = self.data.ctrl[self.actuator_ids].copy()
        max_delta = self.max_velocity / self.control_hz
        target = start + np.clip(bounded - start, -max_delta, max_delta)
        for i in range(self.substeps):
            self.data.ctrl[self.actuator_ids] = start + (i + 1) / self.substeps * (target - start)
            mujoco.mj_step(self.model, self.data)
            if any(w.number for w in self.data.warning):
                self._ready = False
                raise RuntimeError("MuJoCo issued a numerical/capacity warning; episode aborted")
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            self._ready = False
            raise RuntimeError("non-finite physics state")
        # mj_step's derived poses can lag qpos by one integration step.
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        self._ready = not truncated
        info = self.info()
        info["control"] = {"requested_radians": requested.tolist(),
                           "applied_radians": target.tolist(),
                           "limit_clipped": (requested != bounded).tolist(),
                           "rate_limited": (np.abs(bounded - target) > 1e-12).tolist()}
        return self.observation(), 0.0, False, truncated, info

    def step_real_joint_targets(self, target_deg6) -> tuple[dict, float, bool, bool, dict]:
        """Dataset-unit targets (5 joints in degrees, gripper in percent) -> native step."""
        target = self._vector(target_deg6, "real joint target (deg, %)")
        return self.step(sim_from_real(self.mapping, target))

    # ------------------------------------------------------------------ state adapters
    @property
    def table_top(self) -> float:
        return float(self.task["table"]["height"])

    def sim_state(self) -> np.ndarray:
        return self.data.qpos[self.qadr].astype(np.float64).copy()

    def real_state(self) -> np.ndarray:
        """Six joints in dataset units (degrees, gripper percent) via the inverse mapping."""
        return real_from_sim(self.mapping, self.sim_state()).astype(np.float32)

    def ee_pose(self) -> np.ndarray:
        """``T_base_ee`` (4x4) of the ``gripperframe`` site from the current MjData."""
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = self.data.xmat[self.base_body].reshape(3, 3)
        T_world_base[:3, 3] = self.data.xpos[self.base_body]
        T_world_ee = np.eye(4)
        T_world_ee[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        T_world_ee[:3, 3] = self.data.site_xpos[self.site_id]
        R = T_world_base[:3, :3]
        T_base_world = np.eye(4)
        T_base_world[:3, :3] = R.T
        T_base_world[:3, 3] = -R.T @ T_world_base[:3, 3]
        return T_base_world @ T_world_ee

    def canonical_proprio(self, prev_ee_pos: np.ndarray | None = None) -> np.ndarray:
        """13 floats as in ``real2sim.data.labels``: ee_xyz, rot6d, finite-difference velocity
        at control_hz (zeros at the first tick), gripper_obs = percent / 100."""
        T = self.ee_pose()
        p = T[:3, 3]
        vel = np.zeros(3) if prev_ee_pos is None else (p - np.asarray(prev_ee_pos, float)) * self.control_hz
        gripper = real_from_sim(self.mapping, self.sim_state())[5] / 100.0
        return np.concatenate([p, rot6d_from_matrix(T[:3, :3]), vel, [gripper]]).astype(np.float32)

    # ------------------------------------------------------------------ observation / info
    def render(self) -> np.ndarray:
        if self.renderer is None:
            render = self.config["render"]
            self.renderer = mujoco.Renderer(self.model, height=render["height"], width=render["width"])
        self.renderer.update_scene(self.data, camera="iphone")
        return self.renderer.render().copy()

    def observation(self) -> dict:
        result = {
            # Native radians under a distinct key so they can never masquerade as dataset state.
            self.config["observation"]["joint_key"]: self.sim_state().astype(np.float32),
            self.config["observation"]["state_key"]: self.real_state(),
        }
        if self.render_images:
            cfg = self.data_config["image"]
            rgb, _ = letterbox_rgb(self.render(), cfg["height"], cfg["width"], cfg["pad_value"])
            result[self.camera_keys[0]] = to_chw_float(rgb)
        return result

    def _contact_pairs(self) -> list[dict]:
        pairs = []
        for contact in self.data.contact[:self.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            pairs.append({"geom1": self.model.geom(a).name or f"geom_{a}",
                          "geom2": self.model.geom(b).name or f"geom_{b}",
                          "distance_m": float(contact.dist)})
        return pairs

    def _robot_table_penetration(self) -> dict:
        """Robot (non-ball) geoms against the table: count and deepest penetration in metres."""
        count, deepest = 0, 0.0
        for contact in self.data.contact[:self.data.ncon]:
            geoms = (int(contact.geom1), int(contact.geom2))
            if self.table_geom in geoms and self.ball_geom not in geoms:
                count += 1
                deepest = max(deepest, -float(contact.dist))
        return {"robot_table_contacts": count, "robot_table_penetration_m": deepest,
                "tool_height_above_table_m": float(self.data.site_xpos[self.site_id][2] - self.table_top)}

    def _ball_contacts(self) -> set[int]:
        geoms = set()
        for contact in self.data.contact[:self.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if a == self.ball_geom:
                geoms.add(b)
            elif b == self.ball_geom:
                geoms.add(a)
        return geoms

    def info(self) -> dict:
        ball = self.data.xpos[self.ball_body].copy()
        ball_v = self.data.qvel[self._ball_v:self._ball_v + 3].copy()
        bowl = self.data.xpos[self.model.body("bowl").id].copy()
        tool = self.data.site_xpos[self.site_id].copy()
        radius = float(self.task["ball"]["radius"])
        rim_z = bowl[2] + self.bowl["height"]
        floor_z = bowl[2] + self.bowl["floor"]
        touching = self._ball_contacts()
        fixed, moving = (touching & self.gripper_geoms[n] for n in GRIPPER_BODIES)
        above = ball[2] > self.table_top + radius + 0.01
        ball_held = bool((fixed and moving) or ((fixed or moving) and above))
        metrics = {
            "frame": "world (z up; table top at task.table.height); *_task = world - offset",
            "qpos": self.data.qpos.tolist(), "qvel": self.data.qvel.tolist(),
            "ctrl": self.data.ctrl.tolist(),
            "ball_xyz": ball.tolist(), "bowl_xyz": bowl.tolist(), "tool_xyz": tool.tolist(),
            "ball_xyz_task": (ball - self.offset).tolist(),
            "bowl_xyz_task": (bowl - self.offset).tolist(),
            "tool_xyz_task": (tool - self.offset).tolist(),
            "ball_speed_mps": float(np.linalg.norm(ball_v)),
            "ball_in_bowl": bool(np.linalg.norm(ball[:2] - bowl[:2]) < self.bowl["rim_radius"]
                                 and floor_z <= ball[2] <= rim_z + radius),
            "ball_resting": bool(np.linalg.norm(ball_v) < 0.01),
            "ball_held": ball_held,
            "ball_touching_table": self.table_geom in touching,
            "ee_ball_distance_m": float(np.linalg.norm(tool - ball)),
            "ee_bowl_distance_m": float(np.linalg.norm(tool - bowl)),
            "lifted": bool(ball[2] > self.table_top + radius + 0.03),
            "contacts": self._contact_pairs(),
            # Start-pose validity (D-014): penetration at reset and now; see reset().
            "start_pose": dict(self.start_pose),
            **self._robot_table_penetration(),
        }
        return {"controller_source": "NATIVE_SIM_DIAGNOSTIC", "calibrated": False,
                "time_s": float(self.data.time), "step": self.steps,
                "privileged_metrics": metrics}

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self._ready = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
