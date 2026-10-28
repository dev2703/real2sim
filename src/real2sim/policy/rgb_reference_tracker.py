"""Causal RGB ball/bowl references, without simulator dynamic object state."""

from __future__ import annotations

import numpy as np

from real2sim.policy.visual_targets import camera_pose_task, locate_objects
from real2sim.sim.cameras import camera_intrinsics_px


class RGBReferenceTracker:
    """Refresh confident on-table sphere detections and remember occluded goals.

    The frontend fits the observed sphere silhouette against the declared table
    support plane and measured radius. Carried/occluded objects need not yield a
    valid detection; their last RGB reference is retained with an explicit age.
    This is perception memory, not a held-state or task-phase estimator.
    """

    def __init__(self, env, *, refresh_ticks=5):
        self.env = env
        self.refresh_ticks = int(refresh_ticks)
        if self.refresh_ticks < 1:
            raise ValueError("Positive RGB refresh interval required")
        self.ball = self.bowl = None
        self.last_seen = {"ball": None, "bowl": None}
        self.visible = {"ball": False, "bowl": False}
        self.log = []

    def update(self, tick, *, force=False):
        if not force and tick % self.refresh_ticks:
            return None
        env = self.env
        rgb = env.render("iphone")
        camera = env.model.camera("iphone").id
        K = camera_intrinsics_px(env.model, camera, rgb.shape[1], rgb.shape[0])
        pose = camera_pose_task(env.data.cam_xpos[camera], env.data.cam_xmat[camera], env.offset)
        table_z = env.table_top - env.offset[2]
        support = float(env.task["ball"].get("support", {}).get("height", 0.))
        found = locate_objects(rgb, K, pose,
            ball_radius_m=float(env.task["ball"]["radius"]),
            ball_center_height_m=table_z + support + env.task["ball"]["radius"],
            bowl_rim_height_m=table_z + env.bowl["height"],
            bowl_outer_radius_m=env.bowl["rim_radius"] + env.bowl["wall"],
            workspace_xy=[env.config["ik"]["workspace_min"][:2], env.config["ik"]["workspace_max"][:2]])
        previous = None if self.ball is None else self.ball.copy()
        for name, target in found.items():
            self.visible[name] = target is not None
            if target is not None:
                setattr(self, name, target.center_task.copy())
                self.last_seen[name] = tick
        record = {"tick": tick, "targets": {
            name: target.to_dict() if target is not None else None for name, target in found.items()}}
        record["ball_reference_shift_m"] = (
            float(np.linalg.norm(self.ball - previous)) if previous is not None and found["ball"] is not None else 0.)
        self.log.append(record)
        return record

    def observation_metadata(self, tick):
        if self.ball is None or self.bowl is None:
            raise RuntimeError("RGB ball and bowl must each have an observed reference")
        return {"visibility": (self.visible["ball"], self.visible["bowl"]),
                "last_seen_ticks": (self.last_seen["ball"], self.last_seen["bowl"])}

    def expert_needs_replan(self, teacher, update):
        """Recognize a displaced free ball using RGB and measured TCP only."""
        if update is None or update["ball_reference_shift_m"] <= .006 or not self.visible["ball"]:
            return False
        env = self.env
        # A ball observed within the observed bowl footprint should be left
        # there, rather than acquired again by the demonstration expert.
        if np.linalg.norm(self.ball[:2] - self.bowl[:2]) <= env.bowl["rim_radius"]:
            return False
        if teacher.phase in ("ACQUIRE", "PRE_GRASP", "DESCEND"):
            return True
        grasp = env.ee_pose()
        predicted_ball = grasp[:3, 3] + grasp[:3, :3] @ teacher.ball_in_tool
        return bool(np.linalg.norm(self.ball - predicted_ball) > .06)
