"""Shared rendering, contacts and task scoring for native MuJoCo robot environments.

Each embodiment owns its action contract, reset/step logic and proprioception. The shared
base operates on the model, data, task configuration and object IDs initialized by it.
"""

from __future__ import annotations

import mujoco
import numpy as np

from real2sim.data.transforms import letterbox_rgb, to_chw_float
from real2sim.sim.ball_support import support_contact_metrics


class PickPlaceEnv:
    """Common camera observations, task metrics and renderer lifetime for both robots."""

    def _observation(self, state: np.ndarray) -> dict:
        result = {
            self.config["observation"]["joint_key"]: self.sim_state().astype(np.float32),
            self.config["observation"]["state_key"]: state,
        }
        if self.render_images:
            cfg = self.data_config["image"]
            for key in self.camera_keys:
                rgb, _ = letterbox_rgb(self.render(self.camera_bindings[key]), cfg["height"],
                                       cfg["width"], cfg["pad_value"])
                result[key] = to_chw_float(rgb)
        return result

    def _task_metrics(self, grasp_geoms: tuple[set[int], set[int]]) -> dict:
        """Privileged scoring state; callers expose it only through info()."""
        ball = self.data.xpos[self.ball_body].copy()
        ball_v = self.data.qvel[self._ball_v:self._ball_v + 3].copy()
        bowl = self.data.xpos[self.model.body("bowl").id].copy()
        tool = self.data.site_xpos[self.site_id].copy()
        radius = float(self.task["ball"]["radius"])
        rest_z = getattr(self, "_ball_rest_center_z", self.table_top + radius)
        rim_z = bowl[2] + self.bowl["height"]
        floor_z = bowl[2] + self.bowl["floor"]
        touching = self._ball_contacts()
        fixed, moving = (touching & geoms for geoms in grasp_geoms)
        above = ball[2] > rest_z + 0.01
        ball_held = bool((fixed and moving) or ((fixed or moving) and above))
        return {
            "frame": "world (z up; table top at task.table.height); *_task = world - offset",
            "qpos": self.data.qpos.tolist(), "qvel": self.data.qvel.tolist(),
            "ctrl": self.data.ctrl.tolist(),
            "ball_xyz": ball.tolist(), "bowl_xyz": bowl.tolist(), "tool_xyz": tool.tolist(),
            "ball_xyz_task": (ball - self.offset).tolist(),
            "bowl_xyz_task": (bowl - self.offset).tolist(),
            "tool_xyz_task": (tool - self.offset).tolist(),
            "ball_speed_mps": float(np.linalg.norm(ball_v)),
            "ball_over_bowl": bool(np.linalg.norm(ball[:2] - bowl[:2]) < self.bowl["rim_radius"] and ball[2] >= floor_z),
            "ball_in_bowl": bool(np.linalg.norm(ball[:2] - bowl[:2]) < self.bowl["rim_radius"]
                                 and floor_z <= ball[2] <= rim_z + radius),
            "ball_resting": bool(np.linalg.norm(ball_v) < 0.01),
            "ball_held": ball_held,
            "ball_touching_table": self.table_geom in touching,
            "ee_ball_distance_m": float(np.linalg.norm(tool - ball)),
            "ee_bowl_distance_m": float(np.linalg.norm(tool - bowl)),
            "lifted": bool(ball[2] > rest_z + 0.03),
            "contacts": self._contact_pairs(),
            **support_contact_metrics(self.model, self.data, self.base_body),
        }

    @property
    def table_top(self) -> float:
        return float(self.task["table"]["height"])

    def render(self, camera: str = "iphone") -> np.ndarray:
        if self.renderer is None:
            render = self.config["render"]
            self.renderer = mujoco.Renderer(self.model, height=render["height"], width=render["width"])
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

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

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self._ready = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

