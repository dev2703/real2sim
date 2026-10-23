"""Explicit extra-camera mounts shared by both robot scenes.

Local camera axes follow MuJoCo (x right, y up, camera looks along -z).
Measured transforms are required for physical acceptance. An explicit nominal CAD
camera descriptor is available for provisional experiments; it is never enabled by default.
"""

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def nominal_so101_wrist_camera(mjcf: str | Path) -> dict:
    """Expose the model's existing wrist optical mount as an opt-in image input.

    The pinned SO-101 includes both mount CAD and ``wrist_cam``. Copy the compiled
    local camera transform, including quaternion normalization, rather than inventing
    an optical center from a mesh. Model optics are a hypothesis, not measurements
    of the recording camera's intrinsics, crop, distortion or installed pose.
    """
    import mujoco

    path = Path(mjcf)
    model = mujoco.MjModel.from_xml_path(str(path))
    camera = model.camera("wrist_cam")
    body = model.body(int(camera.bodyid[0]))
    if body.name != "camera_mount" or int(camera.mode[0]) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
        raise ValueError("nominal wrist camera requires fixed wrist_cam on camera_mount")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, camera.quat)
    axes = rotation.reshape(3, 3)[:, :2].T.ravel()
    return {
        "key": "observation.images.top", "name": "top_nominal_cad",
        "projection_model": "fovy", "fovy_deg": float(camera.fovy[0]),
        "mounts": {"so101": {
            "body": body.name, "position_body_m": camera.pos.tolist(),
            "xyaxes_body": axes.tolist(),
        }},
        "physical_calibration_accepted": False,
        "provenance": {
            "kind": "NOMINAL_CAD_OPTICS_NOT_PHYSICALLY_VERIFIED",
            "source_model": str(path),
            "source_model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_camera": "wrist_cam",
            "scope": "Explicit provisional two-view experiment. Recorded top camera correspondence, intrinsics/distortion and installed optical transform remain unverified.",
        },
    }


def camera_projection_attributes(camera: dict, render: dict | None = None) -> dict[str, str]:
    """Opt-in source-resolution OpenCV pinhole intrinsics; legacy FOV stays unchanged.

    MuJoCo principalpixel shifts the projection plane, opposite to OpenCV's optical-axis
    pixel offset. Distorted source images require explicit rectification outside this helper.
    """
    model = camera.get("projection_model", "fovy")
    if model == "fovy":
        return {"fovy": str(camera["fovy_deg"])}
    if model != "opencv_pinhole":
        raise ValueError(f"unsupported camera projection_model: {model}")
    width, height = camera["width"], camera["height"]
    if (not isinstance(width, int) or not isinstance(height, int)
            or isinstance(width, bool) or isinstance(height, bool) or min(width, height) <= 0):
        raise ValueError("pinhole source width/height must be positive integers")
    intrinsic = camera["intrinsics_px"]
    fx, fy, cx, cy = (float(intrinsic[k]) for k in ("fx", "fy", "cx", "cy"))
    if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ValueError("pinhole intrinsics must be finite with positive focal lengths")
    distortion = np.asarray(camera.get("distortion_coefficients"), dtype=float)
    if distortion.shape not in {(4,), (5,), (8,), (12,), (14,)} or not np.isfinite(distortion).all():
        raise ValueError("explicit finite OpenCV distortion coefficients required for pinhole rendering")
    if np.any(distortion != 0):
        raise ValueError("nonzero source distortion requires rectified images and matching rectified intrinsics")
    if render and not np.isclose(width / height, render["width"] / render["height"], atol=1e-9, rtol=0):
        raise ValueError("source/render aspect ratio differs; explicit image geometry is required")
    return {
        "resolution": f"{width} {height}", "sensorsize": "1 1",
        "focalpixel": f"{fx:.12g} {fy:.12g}",
        "principalpixel": f"{width / 2 - cx:.12g} {height / 2 - cy:.12g}",
    }


def camera_intrinsics_px(model, camera_id: int, width: int, height: int) -> np.ndarray:
    """Read OpenCV intrinsics from the compiled camera at the requested image size."""
    if min(width, height) <= 0:
        raise ValueError("camera image dimensions must be positive")
    sensor = model.cam_sensorsize[camera_id]
    if np.all(sensor > 0):
        intrinsic = model.cam_intrinsic[camera_id]
        fx, fy = intrinsic[:2] * [width, height] / sensor
        cx, cy = np.array([width, height]) / 2 - intrinsic[2:] * [width, height] / sensor
    else:
        fx = fy = height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2))
        cx, cy = width / 2, height / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


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


def add_link_cameras(root: ET.Element, task: dict, robot: str, render: dict | None = None) -> None:
    camera_bindings(task, robot)
    for camera in task.get("additional_cameras", []):
        mount = camera["mounts"][robot]
        body = root.find(f".//body[@name='{mount['body']}']")
        if body is None:
            raise ValueError(f"camera mount body does not exist: {mount['body']}")
        pos = np.asarray(mount["position_body_m"], float)
        axes = np.asarray(mount["xyaxes_body"], float)
        projection = camera_projection_attributes(camera, render)
        if pos.shape != (3,) or axes.shape != (6,) or not np.isfinite(np.r_[pos, axes]).all():
            raise ValueError("camera mount must contain finite position and axes")
        if "fovy" in projection and not 0 < float(projection["fovy"]) < 180:
            raise ValueError("camera fovy must be within (0,180)")
        if not np.allclose(np.linalg.norm(axes.reshape(2, 3), axis=1), 1, atol=1e-5) or abs(axes[:3] @ axes[3:]) > 1e-5:
            raise ValueError("camera axes must be orthonormal")
        ET.SubElement(body, "camera", name=camera["name"],
                      pos=" ".join(map(str, pos)), xyaxes=" ".join(map(str, axes)), **projection)
