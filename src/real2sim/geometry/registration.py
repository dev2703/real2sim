"""Separate fixed-camera registration from robot joint-zero calibration.

The camera fit needs independently measured task-frame points and fixed intrinsics.
Robot FK, fitted object positions, and an assumed held-ball/tool coincidence are not
independent camera anchors. The joint fit subsequently holds that camera fixed.
Neither function changes a simulator configuration or accepts a calibration.

Groups are whole episodes (or independent calibration captures), declared before
fitting. Only fit groups enter optimization, initialization and parameter bounds.
Verification groups only score the result. Keep reserved policy-evaluation episodes
out of both groups. Pixel uncertainty is annotation uncertainty, not metric accuracy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares


def _finite(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return value


def _intrinsics(K):
    K = _finite(K, (3, 3), "intrinsics")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("invalid pinhole intrinsics")
    if K[0, 1] != 0 or K[1, 0] != 0:
        raise ValueError("OpenCV projection requires zero skew")
    return K


def _transform(T):
    T = _finite(T, (4, 4), "T_optical_task")
    R = T[:3, :3]
    if (not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-7)):
        raise ValueError("T_optical_task must be a proper rigid transform")
    return T


def _split(groups, fit_groups, verification_groups, reserved_groups):
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("one group ID is required per correspondence")
    fit_groups, verification_groups = set(fit_groups), set(verification_groups)
    if not fit_groups or not verification_groups or fit_groups & verification_groups:
        raise ValueError("nonempty, disjoint fit and verification groups are required")
    if (fit_groups | verification_groups) & set(reserved_groups):
        raise ValueError("reserved evaluation groups cannot enter calibration")
    if set(groups.tolist()) != fit_groups | verification_groups:
        raise ValueError("every supplied group must be assigned exactly once")
    return np.isin(groups, list(fit_groups)), np.isin(groups, list(verification_groups))


def _observations(pixels, groups, fit_groups, verification_groups, reserved_groups, sigma_px):
    fit, verify = _split(groups, fit_groups, verification_groups, reserved_groups)
    pixels = _finite(pixels, (len(fit), 2), "source-image pixels")
    sigma = np.broadcast_to(np.asarray(sigma_px, float), (len(fit),)).copy()
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("pixel uncertainty must be positive and finite")
    return pixels, fit, verify, sigma


def _project(points, rvec, tvec, K, distortion):
    return cv2.projectPoints(
        np.asarray(points, np.float64), rvec, tvec, K, distortion,
    )[0].reshape(-1, 2)


def _errors(predicted, observed, mask):
    err = np.linalg.norm(predicted[mask] - observed[mask], axis=1)
    return {
        "count": int(len(err)), "rms_px": float(np.sqrt(np.mean(err**2))),
        "p95_px": float(np.percentile(err, 95)), "max_px": float(np.max(err)),
    }


def _identifiability(residual: Callable, x: np.ndarray, scales: np.ndarray):
    """Data-only finite-difference Jacobian; no regularizer can manufacture rank.

    Covariance is a local linear approximation conditional on fixed geometry and
    camera intrinsics, independent pixel noise, and the chosen solution basin. It
    does not include systematic error or resolve alternative planar-PnP solutions.
    """
    columns = []
    for i, scale in enumerate(scales):
        step = np.zeros_like(x)
        step[i] = scale * 1e-5
        columns.append((residual(x + step) - residual(x - step)) / (2 * step[i]))
    J = np.column_stack(columns)
    _, singular, vt = np.linalg.svd(J * scales, full_matrices=False)
    tolerance = max(float(singular[0]), 1.0) * 1e-7
    rank = int((singular > tolerance).sum())
    identifiable = rank == len(x)
    # Avoid squaring the condition number with an inverse of J.T @ J.
    std = scales * np.sqrt(np.sum((vt.T / singular) ** 2, axis=1)) if identifiable else None
    return {
        "rank": rank, "parameter_count": len(x), "full_rank": identifiable,
        "parameter_scales": scales.tolist(), "scaled_singular_values": singular.tolist(),
        "scaled_condition_number": float(singular[0] / singular[-1]) if identifiable else None,
        "conditional_parameter_std": None if std is None else std.tolist(),
        "uncertainty_note": "Local data-only linear approximation; ignores systematic geometry, intrinsics and timing error, and solution ambiguity.",
    }


def _report(predicted, pixels, groups, fit, verify, result, identifiability, arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return {
        "kind": "DIAGNOSTIC_REGISTRATION_CANDIDATE_NOT_ADOPTED",
        "input_arrays_sha256": digest.hexdigest(),
        "fit_groups": sorted(set(np.asarray(groups)[fit].tolist())),
        "verification_groups": sorted(set(np.asarray(groups)[verify].tolist())),
        "fit": _errors(predicted, pixels, fit),
        "verification": _errors(predicted, pixels, verify),
        "per_row_error_px": np.linalg.norm(predicted - pixels, axis=1).tolist(),
        "optimizer_converged": bool(result.success),
        "optimizer_message": result.message,
        "identifiability": identifiability,
        "automatically_adopted": False,
    }


def fit_camera_registration(
    points_task_m, pixels, groups, *, K, fit_groups: Sequence, verification_groups: Sequence,
    geometry_provenance: str, independent_geometry: bool, reserved_groups: Sequence = (),
    sigma_px=1.0, distortion=None, initial_T_optical_task=None,
) -> dict:
    """Fit optical-from-task SE(3), using measured anchors and fixed intrinsics.

    Source coordinates must be independent measurements in the declared task frame.
    Focal length, robot joints, table height and object locations are never fitted
    alongside this transform. Coplanar anchors are allowed but explicitly flagged;
    require off-plane verification or an independent camera-pose measurement before
    choosing between ambiguous planar solutions.
    """
    if not independent_geometry or not geometry_provenance.strip():
        raise ValueError("independent task-frame geometry and its provenance are required")
    pixels, fit, verify, sigma = _observations(
        pixels, groups, fit_groups, verification_groups, reserved_groups, sigma_px)
    points = _finite(points_task_m, (len(fit), 3), "task-frame points")
    K = _intrinsics(K)
    if fit.sum() < 6:
        raise ValueError("at least six fitting correspondences are required")
    geometry_rank = np.linalg.matrix_rank(points[fit] - points[fit].mean(axis=0))
    if geometry_rank < 2:
        raise ValueError("camera anchors are collinear; camera registration is underconstrained")
    if initial_T_optical_task is None:
        ok, rvec, tvec = cv2.solvePnP(points[fit], pixels[fit], K, distortion)
        if not ok:
            raise ValueError("camera pose initialization failed")
        x0 = np.r_[rvec.ravel(), tvec.ravel()]
    else:
        T = _transform(initial_T_optical_task)
        x0 = np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]]

    def residual(x):
        predicted = _project(points[fit], x[:3], x[3:], K, distortion)
        return ((predicted - pixels[fit]) / sigma[fit, None]).ravel()

    result = least_squares(residual, x0, loss="soft_l1", f_scale=2, max_nfev=1000)
    R = cv2.Rodrigues(result.x[:3])[0]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, result.x[3:]
    predicted = _project(points, result.x[:3], result.x[3:], K, distortion)
    ident = _identifiability(residual, result.x, np.full(6, .1))
    report = _report(predicted, pixels, groups, fit, verify, result, ident, [points, pixels, K, sigma])
    report.update({
        "T_optical_task": T.tolist(), "geometry_provenance": geometry_provenance,
        "fitting_anchor_spatial_rank": int(geometry_rank),
        "planar_ambiguity_requires_independent_check": bool(geometry_rank == 2),
        "points_behind_camera": int(np.sum((points @ R.T + T[:3, 3])[:, 2] <= 0)),
        "intrinsics_fitted": False,
    })
    return report


def fit_joint_zero_offsets(
    q_nominal_rad, pixels, groups, points_for_q: Callable, *, K, T_optical_task,
    joint_limits_rad, offset_bounds_rad, fit_groups: Sequence, verification_groups: Sequence,
    reserved_groups: Sequence = (), sigma_px=1.0, distortion=None, initial_offsets_rad=None,
) -> dict:
    """Fit joint zeros with fixed camera, link dimensions and known robot landmarks.

    ``points_for_q(q, row_indices)`` returns one task-frame landmark per row. It must
    evaluate the supplied joint coordinates exactly, without limit clipping or an
    IK solve. Nominal coordinates already include known signs/units. Bounds are in
    radians; the feasible intersection uses fitting states only. Verification-state
    limit violations are reported, never concealed or fed back into the fit.

    Landmark offsets/shape and camera extrinsics are fixed. A camera fitted from the
    same robot FK cannot independently establish joint-zero correctness. Inspect
    weak Jacobian directions before treating a low pixel residual as calibration.
    """
    pixels, fit, verify, sigma = _observations(
        pixels, groups, fit_groups, verification_groups, reserved_groups, sigma_px)
    q = np.asarray(q_nominal_rad, np.float64)
    if q.ndim != 2 or len(q) != len(fit) or not np.isfinite(q).all():
        raise ValueError("joint coordinates must be a finite rows-by-joints array")
    n = q.shape[1]
    limits = _finite(joint_limits_rad, (n, 2), "joint limits")
    bounds = _finite(offset_bounds_rad, (n, 2), "offset bounds")
    if np.any(limits[:, 0] >= limits[:, 1]) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("lower joint/offset bounds must be less than upper bounds")
    lower = np.maximum(bounds[:, 0], np.max(limits[:, 0] - q[fit], axis=0))
    upper = np.minimum(bounds[:, 1], np.min(limits[:, 1] - q[fit], axis=0))
    if np.any(lower >= upper):
        raise ValueError("no feasible zero offsets satisfy the fitting poses without clipping")
    x0 = (lower + upper) / 2 if initial_offsets_rad is None else _finite(
        initial_offsets_rad, (n,), "initial offsets")
    if np.any(x0 < lower) or np.any(x0 > upper):
        raise ValueError("initial offsets are outside feasible bounds; they will not be clipped")
    K, T = _intrinsics(K), _transform(T_optical_task)
    rvec, tvec = cv2.Rodrigues(T[:3, :3])[0], T[:3, 3]

    def project(x, indices):
        points = _finite(points_for_q(q[indices] + x, indices), (len(indices), 3), "FK landmarks")
        return _project(points, rvec, tvec, K, distortion)

    fit_indices = np.flatnonzero(fit)

    def residual(x):
        return ((project(x, fit_indices) - pixels[fit]) / sigma[fit, None]).ravel()

    result = least_squares(residual, x0, bounds=(lower, upper), loss="soft_l1",
                           f_scale=2, max_nfev=1000, x_scale=np.full(n, .1))
    predicted = project(result.x, np.arange(len(q)))
    ident = _identifiability(residual, result.x, np.full(n, .1))
    report = _report(predicted, pixels, groups, fit, verify, result, ident,
                     [q, pixels, K, T, sigma, limits, bounds])
    mapped = q + result.x
    violation = np.maximum(np.maximum(limits[:, 0] - mapped, mapped - limits[:, 1]), 0)
    report.update({
        "joint_zero_offsets_rad": result.x.tolist(),
        "initial_offsets_rad": x0.tolist(),
        "effective_offset_bounds_rad": np.column_stack([lower, upper]).tolist(),
        "fitted_offsets_at_bound": np.isclose(result.x, lower, atol=1e-6).tolist(),
        "fit_joint_limit_violation_rows": int(np.any(violation[fit] > 1e-10, axis=1).sum()),
        "verification_joint_limit_violation_rows": int(np.any(violation[verify] > 1e-10, axis=1).sum()),
        "verification_max_joint_limit_violation_rad": float(violation[verify].max()),
        "reference_clipping_applied": False, "camera_fitted": False,
    })
    report["fitted_offsets_at_bound"] = (
        np.isclose(result.x, lower, atol=1e-6) | np.isclose(result.x, upper, atol=1e-6)).tolist()
    return report
