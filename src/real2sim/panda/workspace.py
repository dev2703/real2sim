"""Register a Panda task box to the declared physical object layout."""

from __future__ import annotations

import copy

import numpy as np


def register_measured_task_workspace(config, task):
    """Include measured task footprints while retaining native kinematic gates.

    The task box is an operator-configured Cartesian planning region, independent
    of Panda's native joint limits. Its historical x=50 mm floor excludes nominal
    ball draws after the measured ball centre moves to x=50.8 mm. Compute the
    required footprint from declared geometry, never a rollout or success flag.
    """
    result = copy.deepcopy(config)
    if not task.get("operator_measurements"):
        return result
    ball = task["ball"]
    center = np.asarray(ball["start_region"]["center_task_xy"], float)
    jitter = float(ball["start_region"]["radius"])
    radius = max([float(ball["radius"]), *map(float, ball.get("radius_interval_m", []))])
    bowl = task["bowl"]
    bowl_center = np.asarray(bowl["center_task_xy"], float)
    bowl_radius = float(bowl["inner_rim_radius"]) + float(bowl["wall_thickness"])
    if (center.shape != (2,) or bowl_center.shape != (2,)
            or not np.isfinite(np.r_[center, bowl_center, jitter, radius, bowl_radius]).all()
            or min(jitter, radius, bowl_radius) < 0):
        raise ValueError("Finite nonnegative measured task footprint required")
    lower = np.minimum(center - jitter - radius, bowl_center - bowl_radius)
    upper = np.maximum(center + jitter + radius, bowl_center + bowl_radius)
    ik = result["ik"]
    original = ik.get("workspace_registration", {}).get("prior_workspace") or {
        name: list(ik[name]) for name in ("workspace_min", "workspace_max")}
    ik["workspace_min"][:2] = np.minimum(ik["workspace_min"][:2], lower).tolist()
    ik["workspace_max"][:2] = np.maximum(ik["workspace_max"][:2], upper).tolist()
    ik["workspace_registration"] = {
        "source": task["operator_measurements"]["source"],
        "rule": "Union of configured box with measured ball start disc plus upper ball radius and bowl outer-rim footprint; z bounds retained",
        "task_xy_min_m": lower.tolist(), "task_xy_max_m": upper.tolist(),
        "prior_workspace": original,
        "native_joint_collision_and_pose_tolerances_changed": False,
    }
    return result
