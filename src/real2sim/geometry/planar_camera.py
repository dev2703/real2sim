"""Diagnostic camera initialization from an independently sized rectangular mat.

Four corners can constrain focal length under restrictive pinhole assumptions, but
cannot verify distortion, the principal point, the camera side of the plane or a
robot-to-mat registration. A low fitting residual is not camera calibration.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
from scipy.optimize import least_squares

from real2sim.geometry.registration import _identifiability, _intrinsics, _transform


def _points(value, dimension, name):
    value = np.asarray(value, np.float64)
    if value.ndim != 2 or value.shape[1] != dimension or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite N-by-{dimension} array")
    return value


def _project(points, x, center):
    K = np.array([[x[0], 0, center[0]], [0, x[0], center[1]], [0, 0, 1.]])
    return cv2.projectPoints(points, x[1:4], x[4:7], K, None)[0].reshape(-1, 2)


def _homography_focal(H, center):
    """Solve orthogonality and equal column norms for inverse focal squared."""
    a = H[0, :2] - center[0] * H[2, :2]
    b = H[1, :2] - center[1] * H[2, :2]
    z = H[2, :2]
    A = np.array([a[0] * a[1] + b[0] * b[1],
                  a[0] ** 2 + b[0] ** 2 - a[1] ** 2 - b[1] ** 2])
    B = -np.array([z[0] * z[1], z[0] ** 2 - z[1] ** 2])
    denominator = float(A @ A)
    if denominator < 1e-24:
        return None
    inverse_square = float(A @ B) / denominator
    if not np.isfinite(inverse_square) or inverse_square <= 0:
        return None
    focal = float(np.sqrt(1 / inverse_square))
    return focal if np.isfinite(focal) else None


def fit_planar_camera(
    corners_px, *, source_size_wh, side_lengths_m=(.305, .406),
    focal_bounds_px=None, sigma_px=1.0,
    verification_plane_points_m=None, verification_pixels_px=None,
) -> dict:
    """Fit a diagnostic square-pixel camera from four ordered source-image corners.

    Mat coordinates are ``(0,0,0), (width,0,0), (width,height,0), (0,height,0)``.
    Supply corresponding corners in that order; arbitrary clockwise sorting does
    not establish the physical mat axes. ``source_size_wh`` is the full source
    image size, with principal point assumed at ``(width/2, height/2)``. Cropped or
    resized coordinates must first be mapped back to that image.

    Only these four fitting corners determine initialization, bounds and fitting.
    Optional independent verification points (in mat coordinates, possibly off
    its plane) are scored afterwards and cannot select the solution. Focal bounds
    default to [0.2, 5] times the largest source dimension. No robot FK is used.
    """
    pixels = _points(corners_px, 2, "mat corners")
    if pixels.shape != (4, 2):
        raise ValueError("exactly four ordered mat corners are required")
    size, lengths = np.asarray(source_size_wh, float), np.asarray(side_lengths_m, float)
    if (size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0)
            or np.any(size != np.floor(size))):
        raise ValueError("source_size_wh must contain positive integer image dimensions")
    if lengths.shape != (2,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("mat side lengths must be positive finite metres")
    if not np.isscalar(sigma_px) or not np.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError("sigma_px must be positive and finite")
    if np.any(pixels < 0) or np.any(pixels > size):
        raise ValueError("mat corners must use full source-image coordinates")
    edges = np.roll(pixels, -1, axis=0) - pixels
    turns = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    singular = np.linalg.svd(pixels - pixels.mean(axis=0), compute_uv=False)
    if (singular[0] <= 1e-8 or singular[-1] / singular[0] < 1e-5
            or not (np.all(turns > 1e-5) or np.all(turns < -1e-5))):
        raise ValueError("mat corners are degenerate or not in convex boundary order")
    bounds = np.asarray(focal_bounds_px if focal_bounds_px is not None else
                        np.array([.2, 5.]) * size.max(), float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or bounds[0] <= 0 or bounds[0] >= bounds[1]):
        raise ValueError("focal bounds must be positive, finite and increasing")
    width, height = lengths
    points = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0.]])
    center = size / 2
    H, _ = cv2.findHomography(points[:, :2], pixels, method=0)
    if H is None or not np.isfinite(H).all():
        raise ValueError("mat homography initialization failed")
    derived_focal = _homography_focal(H, center)
    focals = list(np.geomspace(bounds[0] * (1 + 1e-6), bounds[1] * (1 - 1e-6), 5))
    if derived_focal is not None and bounds[0] < derived_focal < bounds[1]:
        focals.append(derived_focal)

    def residual(x):
        return ((_project(points, x, center) - pixels) / sigma_px).ravel()

    lower = np.r_[bounds[0], np.full(5, -np.inf), 1e-6]
    upper = np.r_[bounds[1], np.full(6, np.inf)]
    candidates = []
    starts = 0
    for focal in focals:
        K = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1.]])
        poses = cv2.solvePnPGeneric(points, pixels, K, None, flags=cv2.SOLVEPNP_IPPE)
        if not poses[0]:
            continue
        for rvec, tvec in zip(poses[1], poses[2], strict=True):
            x0 = np.r_[focal, rvec.ravel(), tvec.ravel()]
            if not np.isfinite(x0).all() or x0[-1] <= lower[-1]:
                continue
            starts += 1
            result = least_squares(
                residual, x0, bounds=(lower, upper), loss="linear", max_nfev=1000,
                x_scale=np.r_[focal, np.ones(3), np.full(3, max(lengths))],
                ftol=1e-12, xtol=1e-12, gtol=1e-12,
            )
            R = cv2.Rodrigues(result.x[1:4])[0]
            depth = (points @ R.T + result.x[4:7])[:, 2]
            if np.isfinite(result.x).all() and np.isfinite(result.fun).all() and np.all(depth > 1e-6):
                candidates.append(result)
    if not candidates:
        raise ValueError("no finite positive-depth camera solution; inspect mat corner ordering")
    best = min(candidates, key=lambda candidate: float(candidate.fun @ candidate.fun))
    x = best.x
    K = np.array([[x[0], 0, center[0]], [0, x[0], center[1]], [0, 0, 1.]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(x[1:4])[0], x[4:7]
    errors = np.linalg.norm(_project(points, x, center) - pixels, axis=1)
    ident = _identifiability(residual, x, np.r_[x[0], np.ones(3), np.full(3, max(lengths))])
    ident["parameter_order"] = ["focal_px", "rotation_x_rad", "rotation_y_rad", "rotation_z_rad",
                                "translation_x_m", "translation_y_m", "translation_z_m"]
    ident["ill_conditioned"] = not ident["full_rank"] or ident["scaled_condition_number"] > 1e6
    ident["uncertainty_note"] = (
        "Local data-only linear uncertainty for supplied sigma_px, conditional on exact mat dimensions, "
        "centered principal point, square pixels, zero distortion and this solution basin. "
        "It excludes systematic errors and planar ambiguity."
    )
    near_best = [c for c in candidates if float(c.fun @ c.fun) <= float(best.fun @ best.fun) + 1.]
    report = {
        "kind": "DIAGNOSTIC_PLANAR_CAMERA_CANDIDATE_NOT_ADOPTED", "automatically_adopted": False,
        "K": K.tolist(), "T_optical_plane": T.tolist(), "focal_px": float(x[0]),
        "focal_bounds_px": bounds.tolist(), "focal_at_bound": bool(
            min(abs(x[0] - bounds)) <= (bounds[1] - bounds[0]) * 1e-6),
        "homography_focal_initialization_px": derived_focal,
        "fit_points_plane_m": points.tolist(), "fit_source_pixels": pixels.tolist(),
        "source_size_wh": size.astype(int).tolist(), "side_lengths_m": lengths.tolist(),
        "per_corner_error_px": errors.tolist(), "fit_rms_px": float(np.sqrt(np.mean(errors**2))),
        "fit_p95_px": float(np.percentile(errors, 95)), "sigma_px": float(sigma_px),
        "optimizer_converged": bool(best.success), "optimizer_message": best.message,
        "deterministic_starts_attempted": starts, "positive_depth_candidates": len(candidates),
        "near_optimal_focal_range_px": [float(min(c.x[0] for c in near_best)),
                                        float(max(c.x[0] for c in near_best))],
        "identifiability": ident, "planar_ambiguity_requires_independent_check": True,
        "assumptions": ["Square pixels and zero skew", "Principal point at half full source width/height",
                        "Zero lens distortion", "All four corners on one rigid plane",
                        "Known rectangular side lengths and ordered physical corners",
                        "Source-image coordinates before any crop or resize"],
        "limitations": ["Four fitting corners are not independent calibration verification",
                        "Mat frame is not a registered robot base or task frame",
                        "Off-plane landmarks or independently known camera pose are required to resolve planar ambiguity"],
        "fit_input_sha256": hashlib.sha256(np.r_[pixels.ravel(), size, lengths, bounds, sigma_px].tobytes()).hexdigest(),
        "verification": None,
    }
    if (verification_plane_points_m is None) != (verification_pixels_px is None):
        raise ValueError("both independent verification points and pixels must be supplied")
    if verification_plane_points_m is not None:
        vp = _points(verification_plane_points_m, 3, "verification points")
        vi = _points(verification_pixels_px, 2, "verification pixels")
        if len(vp) == 0 or len(vp) != len(vi):
            raise ValueError("verification correspondences must have matching nonzero lengths")
        ve = np.linalg.norm(_project(vp, x, center) - vi, axis=1)
        report["verification"] = {
            "count": len(vp), "per_point_error_px": ve.tolist(),
            "rms_px": float(np.sqrt(np.mean(ve**2))), "p95_px": float(np.percentile(ve, 95)),
            "points_behind_camera": int(np.sum((vp @ T[:3, :3].T + T[:3, 3])[:, 2] <= 0)),
            "used_for_fitting_or_solution_selection": False,
        }
    return report


def backproject_to_plane(pixels_px, *, K, T_optical_plane, plane_z_m=0.) -> np.ndarray:
    """Intersect source-image rays with a constant-z plane in metric mat coordinates.

    This helper assumes the same zero-distortion pinhole model as the fit. Refuse
    parallel rays or intersections behind the camera instead of returning NaNs.
    """
    pixels = _points(pixels_px, 2, "source pixels")
    K, T = _intrinsics(K), _transform(T_optical_plane)
    if not np.isscalar(plane_z_m) or not np.isfinite(plane_z_m):
        raise ValueError("plane_z_m must be finite")
    origin = -T[:3, :3].T @ T[:3, 3]
    optical_rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T
    rays = optical_rays @ T[:3, :3]
    if np.any(np.abs(rays[:, 2]) <= 1e-10 * np.linalg.norm(rays, axis=1)):
        raise ValueError("a camera ray is parallel to the requested plane")
    distance = (plane_z_m - origin[2]) / rays[:, 2]
    if np.any(distance <= 0):
        raise ValueError("a plane intersection is behind the camera")
    result = origin + distance[:, None] * rays
    if not np.isfinite(result).all():
        raise ValueError("nonfinite camera-plane intersection")
    return result
