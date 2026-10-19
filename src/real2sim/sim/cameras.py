"""Explicit extra-camera mounts shared by both robot scenes.

Local camera axes follow MuJoCo (x right, y up, camera looks along -z).
No placeholder mount is supplied: a wrist view needs a measured body transform.
"""

import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np


def shared_task_hash(task: dict) -> str:
    """Freeze shared task/optics and SO-101 calibration, permitting a later Panda mount.

    Only the added Panda camera-to-link mounts are excluded. Geometry, coordinates,
    iPhone calibration, camera roles/order, intrinsics and SO-101 mounts remain pinned.
    """
    common = copy.deepcopy(task)
    for camera in common.get("additional_cameras", []):
        camera.get("mounts", {}).pop("panda", None)
    return hashlib.sha256(json.dumps(common, sort_keys=True, allow_nan=False).encode()).hexdigest()


def camera_bindings(task: dict, robot: str) -> dict[str, str]:
    result = {task["camera_iphone"]["key"]: "iphone"}
    names = {"iphone"}
    for camera in task.get("additional_cameras", []):
        key, name = camera["key"], camera["name"]
        if not key.startswith("observation.images.") or key in result or name in names:
            raise ValueError("camera keys and names must be unique declared image inputs")
        if robot not in camera.get("mounts", {}):
            raise ValueError(f"missing {robot} mount for camera {key}")
        result[key] = name
        names.add(name)
    return result


def add_link_cameras(root: ET.Element, task: dict, robot: str) -> None:
    camera_bindings(task, robot)
    for camera in task.get("additional_cameras", []):
        mount = camera["mounts"][robot]
        body = root.find(f".//body[@name='{mount['body']}']")
        if body is None:
            raise ValueError(f"camera mount body does not exist: {mount['body']}")
        pos = np.asarray(mount["position_body_m"], float)
        axes = np.asarray(mount["xyaxes_body"], float)
        fovy = float(camera["fovy_deg"])
        if pos.shape != (3,) or axes.shape != (6,) or not np.isfinite(np.r_[pos, axes, fovy]).all():
            raise ValueError("camera mount must contain finite position, axes and fovy")
        if not 0 < fovy < 180 or not np.allclose(np.linalg.norm(axes.reshape(2, 3), axis=1), 1, atol=1e-5) or abs(axes[:3] @ axes[3:]) > 1e-5:
            raise ValueError("camera axes must be orthonormal and fovy within (0,180)")
        ET.SubElement(body, "camera", name=camera["name"],
                      pos=" ".join(map(str, pos)), xyaxes=" ".join(map(str, axes)), fovy=str(fovy))
