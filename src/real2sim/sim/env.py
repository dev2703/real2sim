"""Seeded MuJoCo diagnostic environment with explicit simulator-native joint controls.

Actions are six absolute hinge positions in radians, including the gripper. They are NOT
dataset degree/percent actions. Real policy adapters remain blocked on calibration.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from real2sim.config import load_yaml, validate_camera_keys
from real2sim.data.transforms import letterbox_rgb, to_chw_float
from real2sim.sim.model import DEFAULT_CONFIG, load_scene, project_path


class So101PickPlaceEnv:
    """Gym-style reset/step tuples; reward is zero (no benchmark definition at M2)."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG, *, render_images: bool = True):
        self.model, self.config, self.task = load_scene(config_path)
        self.data = mujoco.MjData(self.model)
        self.render_images = render_images
        self.renderer = None
        self.data_config = load_yaml(project_path(self.config["data_config"]))
        self.camera_keys = validate_camera_keys(
            self.data_config, [self.task["camera"]["feature_key"]]
        )
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
            raise ValueError("M2 supports only native MuJoCo diagnostic actions")
        self.substeps = int(ratio)
        self.max_velocity = self._vector(self.config["control"]["max_velocity"], "max_velocity")
        if np.any(self.max_velocity <= 0):
            raise ValueError("max_velocity must be positive")
        self.steps = 0
        self._ready = False
        self._bowl_pos = self.model.body("bowl").pos.copy()

    @staticmethod
    def _vector(value, name: str, size: int = 6) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite vector of length {size}")
        return result

    def reset(self, *, seed: int | None = None, scenario: dict | None = None) -> tuple[dict, dict]:
        scenario = dict(scenario or {})
        allowed = {"ball_xy", "bowl_xy", "ball_drop_height", "ball_velocity", "joint_positions"}
        if unknown := scenario.keys() - allowed:
            raise ValueError(f"unknown diagnostic scenario fields: {sorted(unknown)}")
        rng = np.random.default_rng(seed)
        ball_xy = self._vector(scenario.get("ball_xy", np.asarray(self.task["ball"]["start_xy"])
            + rng.uniform(-1, 1, 2) * self.task["ball"]["start_jitter_xy"]), "ball_xy", 2)
        bowl_xy = self._vector(scenario.get("bowl_xy", self._bowl_pos[:2]), "bowl_xy", 2)
        q = self._vector(scenario.get("joint_positions", self.config["initial_qpos"]), "joint_positions")
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise ValueError("initial joint positions exceed upstream model limits")
        drop = float(scenario.get("ball_drop_height", 0.001))
        if not np.isfinite(drop) or drop < 0:
            raise ValueError("ball_drop_height must be finite and nonnegative")
        velocity = self._vector(scenario.get("ball_velocity", [0, 0, 0]), "ball_velocity", 3)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body("bowl").pos[:2] = bowl_xy
        self.data.qpos[self.qadr] = q
        self.data.ctrl[self.actuator_ids] = q
        ball_joint = self.model.joint("ball_free").id
        bq, bv = self.model.jnt_qposadr[ball_joint], self.model.jnt_dofadr[ball_joint]
        self.data.qpos[bq:bq + 3] = [*ball_xy, self.task["table"]["top_z"]
                                    + self.task["ball"]["radius"] + drop]
        self.data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
        self.data.qvel[bv:bv + 3] = velocity
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self._ready = True
        return self.observation(), self.info()

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        if not self._ready:
            raise RuntimeError("reset is required before step or after truncation")
        requested = self._vector(action, "native joint action (radians)")
        bounded = np.clip(requested, self.limits[:, 0], self.limits[:, 1])
        start = self.data.ctrl[self.actuator_ids].copy()
        max_delta = self.max_velocity / self.config["control_hz"]
        target = start + np.clip(bounded - start, -max_delta, max_delta)
        for i in range(self.substeps):
            self.data.ctrl[self.actuator_ids] = start + (i + 1) / self.substeps * (target - start)
            mujoco.mj_step(self.model, self.data)
            if any(w.number for w in self.data.warning):
                self._ready = False
                raise RuntimeError("MuJoCo issued a numerical/capacity warning; diagnostic aborted")
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            self._ready = False
            raise RuntimeError("non-finite physics state")
        # mj_step's derived poses can lag qpos by one integration step.
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        truncated = self.steps >= self.config["max_episode_steps"]
        self._ready = not truncated
        info = self.info()
        info["control"] = {"requested_radians": requested.tolist(),
                           "applied_radians": target.tolist(),
                           "limit_clipped": (requested != bounded).tolist(),
                           "rate_limited": (np.abs(bounded - target) > 1e-12).tolist()}
        return self.observation(), 0.0, False, truncated, info

    def render(self) -> np.ndarray:
        if self.renderer is None:
            camera = self.task["camera"]
            self.renderer = mujoco.Renderer(self.model, height=camera["height"], width=camera["width"])
        self.renderer.update_scene(self.data, camera=self.task["camera"]["name"])
        return self.renderer.render().copy()

    def observation(self) -> dict:
        # A distinct key prevents an uncalibrated radians vector masquerading as dataset state.
        result = {"sim.joint_position": self.data.qpos[self.qadr].astype(np.float32).copy()}
        if self.render_images:
            cfg = self.data_config["image"]
            rgb, _ = letterbox_rgb(self.render(), cfg["height"], cfg["width"], cfg["pad_value"])
            result[self.camera_keys[0]] = to_chw_float(rgb)
        return result

    def info(self) -> dict:
        pairs = []
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            pairs.append({"geom1": self.model.geom(a).name or f"geom_{a}",
                          "geom2": self.model.geom(b).name or f"geom_{b}",
                          "distance_m": float(contact.dist)})
        return {"controller_source": "NATIVE_SIM_DIAGNOSTIC", "calibrated": False,
                "time_s": float(self.data.time), "step": self.steps,
                "privileged_metrics": {"qpos": self.data.qpos.tolist(),
                    "qvel": self.data.qvel.tolist(), "ctrl": self.data.ctrl.tolist(),
                    "ball_xyz": self.data.body("ball").xpos.tolist(),
                    "bowl_xyz": self.data.body("bowl").xpos.tolist(),
                    "tool_xyz": self.data.site(self.config["tool_site"]).xpos.tolist(),
                    "contacts": pairs}}

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self._ready = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
