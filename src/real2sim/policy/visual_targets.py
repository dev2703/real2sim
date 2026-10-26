"""RGB object targets for a calibrated tabletop controller.

This deliberately small colour/shape frontend is intended for the existing
yellow tennis ball and grey bowl scene. It reads no simulator object state.
Camera calibration, object sizes and support/rim heights are explicit inputs;
using nominal values does not establish physical calibration. Missing, occluded
or ambiguous detections return ``None`` rather than a configured object centre.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from real2sim.geometry.image_measurements import fit_plane_circle


@dataclass(frozen=True)
class VisualTarget:
    """A colour-and-shape measurement in the supplied task coordinate frame."""

    center_task: np.ndarray
    pixel_center: np.ndarray
    area_px: float
    fit_error_m: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "center_task": self.center_task.tolist(),
            "pixel_center": self.pixel_center.tolist(),
            "area_px": self.area_px,
            "fit_error_m": self.fit_error_m,
            "confidence": self.confidence,
        }


def camera_pose_task(camera_position_world, camera_rotation_mujoco, task_offset) -> np.ndarray:
    """Convert a supplied MuJoCo camera pose to task-from-OpenCV-camera axes.

    Only camera pose is used: no depth, segmentation render or object pose is an
    input. OpenCV axes are right/down/forward; MuJoCo axes are right/up/backward.
    """
    position = np.asarray(camera_position_world, dtype=float)
    offset = np.asarray(task_offset, dtype=float)
    rotation = np.asarray(camera_rotation_mujoco, dtype=float).reshape(3, 3)
    if position.shape != (3,) or offset.shape != (3,):
        raise ValueError("camera position and task offset must have three coordinates")
    pose = np.eye(4)
    pose[:3, :3] = rotation @ np.diag([1., -1., -1.])
    pose[:3, 3] = position - offset
    _validate_camera(np.eye(3), pose)
    return pose


def _validate_camera(K, pose):
    K, pose = np.asarray(K, float), np.asarray(pose, float)
    if (K.shape != (3, 3) or pose.shape != (4, 4)
            or not np.isfinite(K).all() or not np.isfinite(pose).all()
            or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K[2], [0, 0, 1])
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-5)):
        raise ValueError("finite pinhole intrinsics and rigid task-from-camera pose required")
    return K, pose


def _rays(pixels, K, pose):
    camera = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T
    directions = camera @ pose[:3, :3].T
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _plane_points(pixels, K, pose, height):
    directions = _rays(pixels, K, pose)
    with np.errstate(divide="ignore", invalid="ignore"):
        distances = (height - pose[2, 3]) / directions[:, 2]
    if not np.isfinite(distances).all() or np.any(distances <= 0):
        raise ValueError("object plane is not in front of the camera")
    return pose[:3, 3] + directions * distances[:, None]


def _project(point, K, pose):
    camera = pose[:3, :3].T @ (point - pose[:3, 3])
    if camera[2] <= 0:
        raise ValueError("object is behind the camera")
    pixel = K @ camera
    return pixel[:2] / pixel[2]


def _contours(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    height, width = mask.shape
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 30 or area > .20 * height * width or len(contour) < 20:
            continue
        points = contour[:, 0].astype(float)
        # A clipped silhouette cannot establish a full object centre.
        if (points[:, 0].min() <= 0 or points[:, 1].min() <= 0
                or points[:, 0].max() >= width - 1 or points[:, 1].max() >= height - 1):
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        if hull_area <= 0 or area / hull_area < .90:
            continue
        yield points, area


def _in_workspace(center, workspace_xy):
    if workspace_xy is None:
        return True
    bounds = np.asarray(workspace_xy, float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError("workspace_xy must be [[xmin,ymin],[xmax,ymax]]")
    return bool(np.all(center[:2] >= bounds[0]) and np.all(center[:2] <= bounds[1]))


def _unambiguous(candidates):
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda item: item.confidence, reverse=True)
    # Similar independent matches require a better frontend, not arbitrary selection.
    if len(ranked) > 1 and ranked[1].confidence >= .85 * ranked[0].confidence:
        return None
    return ranked[0]


def locate_objects(
    rgb: np.ndarray,
    K: np.ndarray,
    T_task_camera_opencv: np.ndarray,
    *,
    ball_radius_m: float,
    ball_center_height_m: float,
    bowl_rim_height_m: float,
    bowl_outer_radius_m: float,
    workspace_xy=None,
) -> dict[str, VisualTarget | None]:
    """Detect initial tabletop ball and bowl using RGB and explicit geometry.

    ``rgb`` is native-resolution uint8 RGB, not a letterboxed policy tensor.
    Heights are in the task frame. ``workspace_xy`` bounds the task workspace,
    never an object-centred ROI. Bowl centre Z is the rim plane; callers should
    use its XY with their separately declared placement/release height.

    Ball boundaries are fitted to the sphere tangent cone at the declared
    support height, avoiding the off-axis bias of projecting its pixel centroid.
    Bowl boundaries are backprojected to the declared rim plane and checked for
    metric circularity and the expected radius. These tests reject occlusion;
    this frontend should reacquire an unobstructed frame rather than extrapolate.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("native uint8 RGB image required")
    K, pose = _validate_camera(K, T_task_camera_opencv)
    dimensions = [ball_radius_m, ball_center_height_m, bowl_rim_height_m, bowl_outer_radius_m]
    if not np.isfinite(dimensions).all() or min(ball_radius_m, bowl_outer_radius_m) <= 0:
        raise ValueError("finite heights and positive object radii required")
    # Validate bounds even when neither object is visible.
    _in_workspace(np.zeros(3), workspace_xy)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Hue range is fixed for yellow/green tennis-ball material, not its location.
    # The warm wood background is hue 20--24 in the supplied scene. Including
    # it joins an otherwise visible ball to the image border, invalidating its
    # silhouette. The tennis-ball material and the synthetic tests are >=25.
    ball_mask = ((hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 50)
                 & (hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 30))
    bowl_mask = (hsv[:, :, 1] <= 46) & (hsv[:, :, 2] >= 80)
    found: dict[str, list[VisualTarget]] = {"ball": [], "bowl": []}

    for points, area in _contours(ball_mask):
        rays = _rays(points, K, pose)
        try:
            initial = _plane_points(points.mean(axis=0)[None], K, pose, ball_center_height_m)[0]
        except ValueError:
            continue

        def tangent_residual(xy, rays=rays):
            center = np.r_[xy, ball_center_height_m] - pose[:3, 3]
            return np.linalg.norm(np.cross(rays, center), axis=1) - ball_radius_m

        fit = least_squares(tangent_residual, initial[:2], max_nfev=30)
        center = np.r_[fit.x, ball_center_height_m]
        error = float(np.sqrt(np.mean(fit.fun ** 2)))
        max_error = max(.0015, .045 * ball_radius_m)
        if not fit.success or error > max_error or not _in_workspace(center, workspace_xy):
            continue
        try:
            uv = _project(center, K, pose)
        except ValueError:
            continue
        camera_center = pose[:3, :3].T @ (center - pose[:3, 3])
        # This approximate area check has a deliberately broad perspective band;
        # the exact tangent residual above supplies the geometric fit.
        if camera_center[2] <= ball_radius_m:
            continue
        projected_area = np.pi * K[0, 0] * K[1, 1] * ball_radius_m ** 2 / camera_center[2] ** 2
        ratio = area / projected_area
        if not .75 <= ratio <= 1.35:
            continue
        confidence = float(np.exp(-error / max_error) * min(ratio, 1 / ratio))
        found["ball"].append(VisualTarget(center, uv, area, error, confidence))

    for points, area in _contours(bowl_mask):
        try:
            plane = _plane_points(points, K, pose, bowl_rim_height_m)
            fit = fit_plane_circle(plane)
        except ValueError:
            continue
        center = np.r_[fit["center_plane_xy_m"], bowl_rim_height_m]
        ratio = fit["radius_m"] / bowl_outer_radius_m
        error = fit["circle_rms_m"]
        # A shallow bowl's external silhouette includes a small visible sidewall
        # below the rim plane. Retain a metric residual gate without requiring
        # its full outline to be an exact planar circle.
        max_error = max(.002, .08 * bowl_outer_radius_m)
        if (not .85 <= ratio <= 1.15 or error > max_error
                or fit["second_moment_axis_ratio"] < .75
                or not _in_workspace(center, workspace_xy)):
            continue
        uv = _project(center, K, pose)
        confidence = float(np.exp(-error / max_error) * min(ratio, 1 / ratio))
        found["bowl"].append(VisualTarget(center, uv, area, error, confidence))

    return {name: _unambiguous(candidates) for name, candidates in found.items()}
