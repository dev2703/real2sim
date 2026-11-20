"""Causal RGB ball/bowl references, without simulator dynamic object state."""

from __future__ import annotations

import numpy as np

from real2sim.policy.visual_targets import camera_pose_task, locate_ball_3d, locate_objects
from real2sim.sim.cameras import camera_intrinsics_px


class RGBReferenceTracker:
    """Refresh confident on-table sphere detections and remember occluded goals.

    The frontend fits the observed sphere silhouette against the declared table
    support plane and measured radius. Carried/occluded objects need not yield a
    valid detection; their last RGB reference is retained with an explicit age.
    This is perception memory, not a held-state or task-phase estimator.
    """

    def __init__(self, env, *, refresh_ticks=5, track_ball_3d=False, ball_3d_fit="complete"):
        self.env = env
        if ball_3d_fit not in ("complete", "arcs"):
            raise ValueError("Sphere fit mode must be complete or arcs")
        self.ball_3d_fit = ball_3d_fit
        self.track_ball_3d = bool(track_ball_3d)
        self.observed_ball = None
        self.observed_ball_tick = None
        self.require_supported_ball = False
        self.refresh_ticks = int(refresh_ticks)
        if self.refresh_ticks < 1:
            raise ValueError("Positive RGB refresh interval required")
        self.ball = self.bowl = None
        self.last_seen = {"ball": None, "bowl": None}
        self.visible = {"ball": False, "bowl": False}
        self.log = []

    def update(self, tick, *, force=False):
        refresh_table = force or tick % self.refresh_ticks == 0
        if not refresh_table and not self.track_ball_3d:
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
            workspace_xy=[env.config["ik"]["workspace_min"][:2], env.config["ik"]["workspace_max"][:2]]) if refresh_table else {}
        sphere = None
        if self.track_ball_3d:
            sphere = locate_ball_3d(rgb, K, pose, ball_radius_m=float(env.task["ball"]["radius"]),
                workspace_xy=[env.config["ik"]["workspace_min"][:2], env.config["ik"]["workspace_max"][:2]],
                fit_mode=self.ball_3d_fit)
            self.observed_ball = None if sphere is None else sphere.center_task.copy()
            self.observed_ball_tick = None if sphere is None else tick
            # Keep the trained table-reference contract. An elevated sphere
            # supplies carry feedback, never a fictitious supported pickup goal.
            supported_z = table_z + support + env.task["ball"]["radius"]
            if (refresh_table and self.require_supported_ball
                    and (sphere is None or abs(sphere.center_task[2] - supported_z) > .015)):
                found["ball"] = None
            elif refresh_table and self.require_supported_ball and found.get("ball") is None:
                # Only the guard's post-loss path may seed a new pickup goal
                # from a 3D sphere corroborated at the declared support plane.
                found["ball"] = sphere
            if found.get("ball") is not None:
                self.require_supported_ball = False
        previous = None if self.ball is None else self.ball.copy()
        for name, target in found.items():
            self.visible[name] = target is not None
            if target is not None:
                setattr(self, name, target.center_task.copy())
                self.last_seen[name] = tick
        record = {"tick": tick, "targets": {
            name: target.to_dict() if target is not None else None for name, target in found.items()}}
        if self.track_ball_3d:
            record["ball_3d"] = None if sphere is None else sphere.to_dict()
            record["ball_3d_fit"] = self.ball_3d_fit
        record["ball_reference_shift_m"] = (
            float(np.linalg.norm(self.ball - previous)) if previous is not None and found.get("ball") is not None else 0.)
        self.log.append(record)
        return record

    def invalidate_ball(self):
        """Discard a failed pickup goal; only a new detection can replace it."""
        self.ball = None
        self.last_seen["ball"] = None
        self.visible["ball"] = False
        self.observed_ball = self.observed_ball_tick = None
        self.require_supported_ball = True

    def observation_metadata(self, tick):
        if self.ball is None or self.bowl is None:
            raise RuntimeError("RGB ball and bowl must each have an observed reference")
        return {"visibility": (self.visible["ball"], self.visible["bowl"]),
                "last_seen_ticks": (self.last_seen["ball"], self.last_seen["bowl"])}

    def expert_needs_replan(self, teacher, update):
        """Recognize a displaced free ball using RGB and measured TCP only."""
        if update is None or not self.visible["ball"]:
            return False
        phase = teacher.phase
        # Acquisition already refreshes its own targets. Terminal phases retain
        # their outcome, and intentional release/retreat separation must not
        # silently start another pickup attempt.
        if phase not in ("PRE_GRASP", "DESCEND", "CLOSE", "LIFT", "TRANSPORT", "LOWER"):
            return False
        env = self.env
        # A ball observed within the observed bowl footprint should be left
        # there, rather than acquired again by the demonstration expert.
        if np.linalg.norm(self.ball[:2] - self.bowl[:2]) <= env.bowl["rim_radius"]:
            return False
        if phase in ("PRE_GRASP", "DESCEND"):
            # Compare with the goal used to construct the current grasp plan.
            # Small per-refresh movements may accumulate into a large drift.
            planned_ball = getattr(teacher, "ball", None)
            displacement = (float(np.linalg.norm(self.ball - planned_ball))
                            if planned_ball is not None else update["ball_reference_shift_m"])
            return displacement > .006
        # A missed pickup can leave the ball completely unmoved at the planned
        # table goal. Fresh RGB evidence far from the estimated carried pose is
        # sufficient; requiring goal displacement would hide this failure.
        grasp = env.ee_pose()
        predicted_ball = grasp[:3, 3] + grasp[:3, :3] @ teacher.ball_in_tool
        return bool(np.linalg.norm(self.ball - predicted_ball) > .06)
