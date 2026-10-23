"""Seeded MuJoCo environment for the Panda pick-and-place task (plan M5).

Native actions are eight floats: seven absolute arm hinge targets in radians followed by the
commanded total finger opening in metres (0 = closed, 0.08 = fully open: two 0.04 m slide
joints). The upstream tendon actuator ``actuator8`` (ctrlrange 0..255, gain 0.04/255 per
finger) is driven unchanged: ``ctrl = opening / 0.08 * 255``.

Observations never contain simulator object poses. Ground truth lives only in
``info()['privileged_metrics']`` for scoring and labelled diagnostics (plan section 7).
``observation.state`` is the 13-float canonical proprioception of ``real2sim.data.labels``
(D-008): TCP xyz (task frame), rot6d of the TCP rotation, finite-difference TCP velocity at the
control rate (zeros at the first tick after reset) and the semantic gripper opening in [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from real2sim.config import load_yaml, validate_camera_keys
from real2sim.data.labels import PROPRIO_NAMES
from real2sim.data.transforms import letterbox_rgb, to_chw_float
from real2sim.geometry.so3 import log_so3, rot6d_from_matrix
from real2sim.panda.model import (
    BASE_BODY,
    DEFAULT_CONFIG,
    FINGER_BODIES,
    POINTING_DOWN,
    bowl_dimensions,
    load_scene,
    project_path,
    task_frame_offset,
)

N_ARM = 7
N_ACTION = N_ARM + 1


class PandaPickPlaceEnv:
    """Gym-style reset/step tuples; reward is zero (no benchmark definition at M5)."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG, *, render_images: bool = True,
                 bowl_inner_depth: float | None = None, contacts: dict | None = None):
        self.model, self.config, self.task = load_scene(
            config_path, bowl_inner_depth=bowl_inner_depth, contacts=contacts)
        self.data = mujoco.MjData(self.model)
        self.render_images = render_images
        self.renderer = None
        self.data_config = load_yaml(project_path(self.config["data_config"]))
        from real2sim.sim.cameras import camera_bindings

        self.camera_bindings = camera_bindings(self.task, "panda")
        self.camera_keys = validate_camera_keys(self.data_config, list(self.camera_bindings))
        self.offset = task_frame_offset(self.task)  # p_world = p_task + offset
        self.bowl = bowl_dimensions(self.task, bowl_inner_depth)
        model = self.model
        if len(self.config["joint_names"]) != N_ARM:
            raise ValueError("the Panda adapter expects seven arm joints")
        self.joint_ids = np.array([model.joint(n).id for n in self.config["joint_names"]])
        self.qadr = model.jnt_qposadr[self.joint_ids]
        self.vadr = model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([
            model.actuator(f"actuator{i + 1}").id for i in range(N_ARM)])
        for act, joint in zip(self.actuator_ids, self.joint_ids, strict=True):
            if model.actuator_trntype[act] != mujoco.mjtTrn.mjTRN_JOINT or \
                    model.actuator_trnid[act, 0] != joint:
                raise ValueError("upstream arm actuators must drive the configured joints in order")
        self.gripper_actuator = model.actuator(self.config["gripper_actuator"]).id
        if model.actuator_trntype[self.gripper_actuator] != mujoco.mjtTrn.mjTRN_TENDON:
            raise ValueError("expected the upstream tendon gripper actuator")
        self.finger_ids = np.array([model.joint(n).id for n in self.config["finger_joint_names"]])
        self.finger_qadr = model.jnt_qposadr[self.finger_ids]
        self.max_opening = float(self.config["max_finger_opening_m"])
        per_finger = model.jnt_range[self.finger_ids]
        if not np.allclose(per_finger[:, 0], 0) or not np.isclose(per_finger[:, 1].sum(),
                                                                  self.max_opening):
            raise ValueError("max_finger_opening_m must equal the summed finger slide ranges")
        self.gripper_ctrl_range = model.actuator_ctrlrange[self.gripper_actuator].copy()
        # Upstream: force = gain*ctrl + bias1*length + bias2*velocity, so the tendon (= per-finger)
        # position target is gain*ctrl/(-bias1): 0.04/255*255 = 0.04 m at ctrl 255.
        gain = model.actuator_gainprm[self.gripper_actuator, 0]
        kp = -model.actuator_biasprm[self.gripper_actuator, 1]
        if kp <= 0 or not np.isclose(gain * self.gripper_ctrl_range[1] / kp, per_finger[0, 1],
                                     atol=1e-6):
            raise ValueError("gripper actuator gain does not map ctrlrange onto the finger range")
        self.limits = np.column_stack([
            np.maximum(model.jnt_range[self.joint_ids, 0],
                       model.actuator_ctrlrange[self.actuator_ids, 0]),
            np.minimum(model.jnt_range[self.joint_ids, 1],
                       model.actuator_ctrlrange[self.actuator_ids, 1]),
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
        self.max_velocity = self._vector(self.config["control"]["max_velocity"], "max_velocity",
                                         N_ARM)
        self.max_gripper_speed = float(self.config["control"]["max_gripper_speed_mps"])
        self.gravity_compensation = bool(self.config["control"].get("gravity_compensation", False))
        if np.any(self.max_velocity <= 0) or self.max_gripper_speed <= 0:
            raise ValueError("rate limits must be positive")
        self.initial_qpos = self._vector(self.config["initial_qpos"], "initial_qpos", N_ARM)
        if np.any(self.initial_qpos < self.limits[:, 0]) or np.any(
                self.initial_qpos > self.limits[:, 1]):
            raise ValueError("initial_qpos exceeds upstream joint limits")
        self.initial_opening = float(self.config["initial_gripper_opening_m"])
        if not 0 <= self.initial_opening <= self.max_opening:
            raise ValueError("initial_gripper_opening_m outside the finger range")
        self.ball_geom = model.geom("ball_geom").id
        self.ball_body = model.body("ball").id
        self.bowl_body = model.body("bowl").id
        self.table_geom = model.geom("table").id
        self.base_body = model.body(BASE_BODY).id
        self.site_id = model.site(self.config["tool_site"]).id
        self.finger_geoms = {
            name: {g for g in range(model.ngeom)
                   if model.body(model.geom_bodyid[g]).name == name
                   and model.geom_contype[g] | model.geom_conaffinity[g]}
            for name in FINGER_BODIES
        }
        ball_joint = model.joint("ball_free").id
        self._ball_q = model.jnt_qposadr[ball_joint]
        self._ball_v = model.jnt_dofadr[ball_joint]
        self.steps = 0
        self._ready = False
        self._ee_velocity = np.zeros(3)
        self._bowl_xy_default = np.asarray(self.bowl["center_xy"], float)
        self._check_initial_pose()

    @staticmethod
    def _vector(value, name: str, size: int = N_ACTION) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite vector of length {size}")
        return result

    def _check_initial_pose(self) -> None:
        """The configured start pose must put the TCP where the config says, pointing down."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.initial_qpos
        mujoco.mj_forward(self.model, self.data)
        T = self.ee_pose()
        expected = self._vector(self.config["initial_tcp_task_xyz"], "initial_tcp_task_xyz", 3)
        ik = self.config["ik"]
        pos_err = float(np.linalg.norm(T[:3, 3] - expected))
        rot_err = float(np.linalg.norm(log_so3(POINTING_DOWN @ T[:3, :3].T)))
        if pos_err > ik["position_tolerance_m"] or rot_err > ik["orientation_tolerance_rad"]:
            raise ValueError(
                f"initial_qpos puts the TCP at {T[:3, 3].round(4).tolist()} (error {pos_err:.4f} m, "
                f"{rot_err:.3f} rad) instead of {expected.tolist()} pointing down")

    # ------------------------------------------------------------------ episode control
    def reset(self, *, seed: int | None = None, scenario: dict | None = None) -> tuple[dict, dict]:
        """Deterministic reset. ``scenario`` keys are task-frame metres (``ball_xy``, ``bowl_xy``)."""
        scenario = dict(scenario or {})
        allowed = {"ball_xy", "bowl_xy", "ball_drop_height", "ball_velocity", "joint_positions",
                   "gripper_opening", "ball_support_xy"}
        if unknown := scenario.keys() - allowed:
            raise ValueError(f"unknown scenario fields: {sorted(unknown)}")
        rng = np.random.default_rng(seed)
        region = self.task["ball"]["start_region"]
        r = float(region["radius"]) * np.sqrt(rng.uniform())  # uniform in area
        theta = rng.uniform(0, 2 * np.pi)
        default_ball = np.asarray(region["center_task_xy"], float) + r * np.array(
            [np.cos(theta), np.sin(theta)])
        ball_xy = self._vector(scenario.get("ball_xy", default_ball), "ball_xy", 2)
        bowl_xy = self._vector(scenario.get("bowl_xy", self._bowl_xy_default), "bowl_xy", 2)
        q = self._vector(scenario.get("joint_positions", self.initial_qpos), "joint_positions",
                         N_ARM)
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise ValueError("initial joint positions exceed upstream model limits")
        opening = float(scenario.get("gripper_opening", self.initial_opening))
        if not np.isfinite(opening) or not 0 <= opening <= self.max_opening:
            raise ValueError("gripper_opening outside the finger range")
        drop = float(scenario.get("ball_drop_height", 0.001))
        if not np.isfinite(drop) or drop < 0:
            raise ValueError("ball_drop_height must be finite and nonnegative")
        velocity = self._vector(scenario.get("ball_velocity", [0, 0, 0]), "ball_velocity", 3)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[self.bowl_body, :2] = bowl_xy
        self.data.qpos[self.qadr] = q
        self.data.qpos[self.finger_qadr] = opening / 2
        self.data.ctrl[self.actuator_ids] = q
        self.data.ctrl[self.gripper_actuator] = self._gripper_ctrl(opening)
        from real2sim.sim.ball_support import reset_ball_support

        spawn_height = reset_ball_support(self, ball_xy, scenario.get("ball_support_xy"))
        bq, bv = self._ball_q, self._ball_v
        self.data.qpos[bq:bq + 3] = [*(ball_xy + self.offset[:2]),
                                    self.table_top + spawn_height + drop]
        self.data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
        self.data.qvel[bv:bv + 3] = velocity
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self._ready = True
        self._ee_velocity = np.zeros(3)  # first tick: zero velocity, as in the labels
        if self.config.get("strict_initial_state"):
            for contact in self.data.contact[:self.data.ncon]:
                pair = {int(contact.geom1), int(contact.geom2)}
                if self.table_geom in pair and self.ball_geom not in pair and contact.dist < -1e-6:
                    self._ready = False
                    raise ValueError("v2 initial pose penetrates the table")
        return self.observation(), self.info()

    def _gripper_ctrl(self, opening_m: float) -> float:
        lo, hi = self.gripper_ctrl_range
        return float(lo + (hi - lo) * np.clip(opening_m / self.max_opening, 0.0, 1.0))

    def _gripper_opening_from_ctrl(self, ctrl: float) -> float:
        lo, hi = self.gripper_ctrl_range
        return float((ctrl - lo) / (hi - lo) * self.max_opening)

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """``action`` = 7 arm joint targets (rad) + total finger opening target (m)."""
        if not self._ready:
            raise RuntimeError("reset is required before step or after truncation")
        requested = self._vector(action, "native action (7 rad + opening m)")
        lo = np.append(self.limits[:, 0], 0.0)
        hi = np.append(self.limits[:, 1], self.max_opening)
        bounded = np.clip(requested, lo, hi)
        start = np.append(self.data.ctrl[self.actuator_ids].copy(),
                          self._gripper_opening_from_ctrl(self.data.ctrl[self.gripper_actuator]))
        max_delta = np.append(self.max_velocity, self.max_gripper_speed) / self.control_hz
        target = start + np.clip(bounded - start, -max_delta, max_delta)
        before = self.ee_pose()[:3, 3]
        for i in range(self.substeps):
            cmd = start + (i + 1) / self.substeps * (target - start)
            self.data.ctrl[self.actuator_ids] = cmd[:N_ARM]
            self.data.ctrl[self.gripper_actuator] = self._gripper_ctrl(cmd[N_ARM])
            if self.gravity_compensation:
                # Emulate the Franka controller's built-in gravity compensation: feed the arm
                # dofs' bias force (gravity + velocity terms from the last forward pass) back as
                # an applied generalised force. Upstream actuator gains are not touched; without
                # this the position servos sag ~2 mm per tick and a relative-to-observed-pose
                # contract integrates that sag into a drift.
                self.data.qfrc_applied[self.vadr] = self.data.qfrc_bias[self.vadr]
            mujoco.mj_step(self.model, self.data)
            if any(w.number for w in self.data.warning):
                self._ready = False
                raise RuntimeError("MuJoCo issued a numerical/capacity warning; episode aborted")
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            self._ready = False
            raise RuntimeError("non-finite physics state")
        mujoco.mj_forward(self.model, self.data)  # derived poses lag qpos by one step after mj_step
        self._ee_velocity = (self.ee_pose()[:3, 3] - before) * self.control_hz
        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        self._ready = not truncated
        info = self.info()
        info["control"] = {"requested": requested.tolist(), "applied": target.tolist(),
                           "limit_clipped": (requested != bounded).tolist(),
                           "rate_limited": (np.abs(bounded - target) > 1e-12).tolist()}
        return self.observation(), 0.0, False, truncated, info

    # ------------------------------------------------------------------ state adapters
    @property
    def table_top(self) -> float:
        return float(self.task["table"]["height"])

    def sim_state(self) -> np.ndarray:
        """Nine native values: 7 arm radians + 2 finger slide positions (m)."""
        return np.concatenate([self.data.qpos[self.qadr], self.data.qpos[self.finger_qadr]]).astype(
            np.float64)

    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.qadr].astype(np.float64).copy()

    def gripper_opening(self) -> float:
        """Total finger opening in metres (sum of both slide joints)."""
        return float(np.sum(self.data.qpos[self.finger_qadr]))

    def gripper_semantic(self) -> float:
        return float(np.clip(self.gripper_opening() / self.max_opening, 0.0, 1.0))

    def ee_pose(self) -> np.ndarray:
        """``T_task_tcp`` (4x4) of the ``panda_tcp`` site from the current MjData."""
        # The task frame is the shared SO-101 task frame (world translated by ``self.offset``, no
        # rotation), NOT the Panda base frame: the Panda base sits base_z_offset below the origin.
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.site_id] - self.offset
        return T

    def canonical_proprio(self) -> np.ndarray:
        """13 floats ordered as ``real2sim.data.labels.PROPRIO_NAMES``."""
        T = self.ee_pose()
        vec = np.concatenate([T[:3, 3], rot6d_from_matrix(T[:3, :3]), self._ee_velocity,
                              [self.gripper_semantic()]]).astype(np.float32)
        assert vec.shape == (len(PROPRIO_NAMES),)
        return vec

    # ------------------------------------------------------------------ observation / info
    def render(self, camera: str = "iphone") -> np.ndarray:
        if self.renderer is None:
            render = self.config["render"]
            self.renderer = mujoco.Renderer(self.model, height=render["height"], width=render["width"])
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def observation(self) -> dict:
        result = {
            self.config["observation"]["joint_key"]: self.sim_state().astype(np.float32),
            self.config["observation"]["state_key"]: self.canonical_proprio(),
        }
        if self.render_images:
            cfg = self.data_config["image"]
            for key in self.camera_keys:
                rgb, _ = letterbox_rgb(self.render(self.camera_bindings[key]), cfg["height"], cfg["width"], cfg["pad_value"])
                result[key] = to_chw_float(rgb)
        return result

    def _contact_pairs(self) -> list[dict]:
        pairs = []
        for contact in self.data.contact[:self.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            pairs.append({"geom1": self.model.geom(a).name or f"geom_{a}",
                          "geom2": self.model.geom(b).name or f"geom_{b}",
                          "distance_m": float(contact.dist)})
        return pairs

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
        bowl = self.data.xpos[self.bowl_body].copy()
        tool = self.data.site_xpos[self.site_id].copy()
        radius = float(self.task["ball"]["radius"])
        rim_z = bowl[2] + self.bowl["height"]
        floor_z = bowl[2] + self.bowl["floor"]
        touching = self._ball_contacts()
        from real2sim.sim.ball_support import support_contact_metrics

        left, right = (touching & self.finger_geoms[n] for n in FINGER_BODIES)
        above = ball[2] > self.table_top + radius + 0.01
        ball_held = bool((left and right) or ((left or right) and above))
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
            "ball_over_bowl": bool(np.linalg.norm(ball[:2] - bowl[:2]) < self.bowl["rim_radius"] and ball[2] >= floor_z),
            "ball_resting": bool(np.linalg.norm(ball_v) < 0.01),
            "ball_held": ball_held,
            "ball_touching_table": self.table_geom in touching,
            "ee_ball_distance_m": float(np.linalg.norm(tool - ball)),
            "ee_bowl_distance_m": float(np.linalg.norm(tool - bowl)),
            "lifted": bool(ball[2] > self.table_top + radius + 0.03),
            "gripper_opening_m": self.gripper_opening(),
            "contacts": self._contact_pairs(),
            **support_contact_metrics(self.model, self.data),
        }
        return {"controller_source": "NATIVE_SIM_DIAGNOSTIC", "calibrated": False,
                "robot": "panda", "time_s": float(self.data.time), "step": self.steps,
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
