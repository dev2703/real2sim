"""Measured checkerboard optics, separated from robot/base registration.

Whole captures define the fit/verification split. Verification estimates only a
per-view board pose from a fixed corner subset, then scores the remaining corners.
That nuisance-pose fit is not independent metric camera/base verification.
"""

from __future__ import annotations

import cv2
import numpy as np


def board_points(inner_corner_count_xy, measured_pitch_xy_m):
    counts = np.asarray(inner_corner_count_xy)
    pitch = np.asarray(measured_pitch_xy_m, dtype=np.float64)
    if (counts.shape != (2,) or counts.dtype.kind not in "iu" or np.any(counts < 4)
            or np.any(counts > 30)):
        raise ValueError("checkerboard requires integer inner-corner counts between 4 and 30")
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("actual measured x/y pitch must be positive finite metres")
    x, y = np.meshgrid(np.arange(counts[0]), np.arange(counts[1]))
    return np.c_[x.ravel() * pitch[0], y.ravel() * pitch[1], np.zeros(x.size)]


def verification_pose_indices(inner_corner_count_xy):
    """Eight predeclared outer/edge corners; no observed pixels select these."""
    nx, ny = map(int, inner_corner_count_xy)
    if nx < 4 or ny < 4:
        raise ValueError("at least 4x4 inner corners required")
    return np.array(sorted({0, nx - 1, (ny - 1) * nx, ny * nx - 1,
                            (nx - 1) // 2, (ny - 1) * nx + (nx - 1) // 2,
                            ((ny - 1) // 2) * nx, ((ny - 1) // 2) * nx + nx - 1}))


def detect_checkerboard(image_bgr, inner_corner_count_xy):
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("native uint8 BGR image required")
    # Validate counts without substituting design dimensions for physical geometry.
    board_points(inner_corner_count_xy, [1., 1.])
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(
        gray, tuple(map(int, inner_corner_count_xy)),
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    return corners.reshape(-1, 2).astype(np.float64) if ok else None


def _project(points, rvec, tvec, K, distortion):
    return cv2.projectPoints(points, rvec, tvec, K, distortion)[0].reshape(-1, 2)


def _errors(observed, predicted):
    lengths = np.linalg.norm(np.asarray(observed) - predicted, axis=1)
    return {"n": len(lengths), "rms_px": float(np.sqrt(np.mean(lengths**2))),
            "p95_px": float(np.percentile(lengths, 95)), "max_px": float(lengths.max())}


def _homography_diversity(points, views, size):
    """Data-only planar intrinsic constraints; duplicate/frontoparallel views fail."""
    xy = points[:, :2].copy()
    xy = (xy - xy.mean(0)) / np.ptp(xy, axis=0).max()
    rows = []
    for pixels in views:
        normalized = (pixels - size / 2) / max(size)
        H, _ = cv2.findHomography(xy, normalized, method=0)
        if H is None:
            raise ValueError("checkerboard homography failed")

        def v(i, j, homography=H):
            a, b = homography[:, i], homography[:, j]
            return np.array([a[0]*b[0], a[0]*b[1]+a[1]*b[0], a[1]*b[1],
                             a[2]*b[0]+a[0]*b[2], a[2]*b[1]+a[1]*b[2], a[2]*b[2]])
        for row in (v(0, 1), v(0, 0) - v(1, 1)):
            norm = np.linalg.norm(row)
            if norm < 1e-12:
                raise ValueError("degenerate checkerboard homography")
            rows.append(row / norm)
    singular = np.linalg.svd(rows, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-7))
    # The ideal homogeneous intrinsic system has rank five, with one scale null.
    if rank < 5:
        raise ValueError("insufficient fit-view tilt diversity: planar intrinsic constraints are rank deficient")
    return {"normalized_constraint_singular_values": singular.tolist(), "constraint_rank": rank,
            "required_non_scale_rank": 5,
            "non_scale_condition_number": float(singular[0] / singular[-2]),
            "interpretation": "Numeric homography diversity only; radial distortion, printed-target errors and weak image coverage require separate inspection."}


def _verification_pose(points, pixels, indices, K, distortion):
    """Choose a positive-depth planar pose using only the fixed pose corners."""
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        points[indices], pixels[indices], K, distortion, flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        raise ValueError("verification board-pose estimation failed")
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        depth = (points @ cv2.Rodrigues(rvec)[0].T + tvec.ravel())[:, 2]
        if np.isfinite(depth).all() and np.all(depth > 0):
            predicted = _project(points[indices], rvec, tvec, K, distortion)
            candidates.append((_errors(pixels[indices], predicted)["rms_px"], rvec, tvec, depth))
    if not candidates:
        raise ValueError("all verification board poses put a landmark behind the camera")
    _, rvec, tvec, depth = min(candidates, key=lambda c: c[0])
    return rvec, tvec, depth, len(candidates)


def fit_checkerboard_optics(views, groups, roles, *, source_image_size_wh,
                            inner_corner_count_xy, measured_pitch_xy_m,
                            verification_working_scale_px=3.):
    """Fit pinhole + k1/k2/p1/p2/k3, with all verification views excluded.

    Returns a diagnostic optical candidate only. Pixels must be native-resolution
    and measured target pitch must come from the physical printed/mounted board.
    This function does not prove those provenance assertions.
    """
    points = board_points(inner_corner_count_xy, measured_pitch_xy_m)
    size = np.asarray(source_image_size_wh)
    if size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size < 100):
        raise ValueError("integer native source image width/height of at least 100 pixels required")
    views = np.asarray(views, np.float64)
    if (views.ndim != 3 or views.shape[1:] != (len(points), 2)
            or not np.isfinite(views).all() or np.any(views < 0) or np.any(views >= size)):
        raise ValueError("all views require finite native pixels for the complete checkerboard")
    if len(groups) != len(views) or len(roles) != len(views):
        raise ValueError("one whole capture group and role per view required")
    if any(not isinstance(g, str) or not g.strip() for g in groups):
        raise ValueError("nonempty named capture groups required")
    if set(roles) != {"fit", "verification"}:
        raise ValueError("nonempty fit and verification roles required")
    assignment = {}
    for group, role in zip(groups, roles, strict=True):
        if group in assignment and assignment[group] != role:
            raise ValueError("whole capture groups must not cross the fit/verification split")
        assignment[group] = role
    fit = np.asarray(roles) == "fit"
    if fit.sum() < 6 or len({g for g, r in zip(groups, roles, strict=True) if r == "fit"}) < 3:
        raise ValueError("at least six fit views from three whole capture groups required")
    if (~fit).sum() < 2 or len({g for g, r in zip(groups, roles, strict=True) if r == "verification"}) < 2:
        raise ValueError("at least two independent verification capture groups required")
    if not np.isfinite(verification_working_scale_px) or verification_working_scale_px <= 0:
        raise ValueError("positive finite predeclared pixel working scale required")
    diversity = _homography_diversity(points, views[fit], size)
    fit_pixels = views[fit].reshape(-1, 2)
    coverage_xy = np.ptp(fit_pixels, axis=0) / size
    if np.any(coverage_xy < .2):
        raise ValueError("fit checkerboard corners cover less than 20% of an image axis")
    result = cv2.calibrateCameraExtended(
        [points.astype(np.float32)] * int(fit.sum()),
        [pixels.astype(np.float32) for pixels in views[fit]], tuple(map(int, size)),
        None, None, flags=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-10),
    )
    rms, K, distortion, rvecs, tvecs, intrinsic_std, _, _ = result
    if not np.isfinite(K).all() or not np.isfinite(distortion).all() or not np.isfinite(rms):
        raise ValueError("nonfinite optical calibration result")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("nonpositive fitted focal length")
    pose_indices = verification_pose_indices(inner_corner_count_xy)
    heldout_indices = np.setdiff1d(np.arange(len(points)), pose_indices)
    rows, fit_poses, normal_vectors = [], iter(zip(rvecs, tvecs, strict=True)), []
    for index, (pixels, group, role) in enumerate(zip(views, groups, roles, strict=True)):
        if role == "fit":
            rvec, tvec = next(fit_poses)
            rotation = cv2.Rodrigues(rvec)[0]
            depth = (points @ rotation.T + tvec.ravel())[:, 2]
            normal_vectors.append(rotation[:, 2])
            if np.any(depth <= 0):
                raise ValueError("fitted board pose has nonpositive landmark depth")
            metrics = {"all_corner_error": _errors(pixels, _project(points, rvec, tvec, K, distortion))}
        else:
            rvec, tvec, depth, candidate_count = _verification_pose(points, pixels, pose_indices, K, distortion)
            predicted = _project(points, rvec, tvec, K, distortion)
            metrics = {"pose_corner_error": _errors(pixels[pose_indices], predicted[pose_indices]),
                       "heldout_corner_error": _errors(pixels[heldout_indices], predicted[heldout_indices]),
                       "positive_depth_pose_candidates": candidate_count}
        rows.append({"view_index": index, "capture_group": group, "role": role,
                     "rvec_target_to_optical": rvec.ravel().tolist(),
                     "tvec_target_to_optical_m": tvec.ravel().tolist(),
                     "minimum_landmark_depth_m": float(depth.min()), **metrics})
    normals = np.asarray(normal_vectors)
    max_tilt = float(np.rad2deg(np.arccos(np.clip(normals @ normals.T, -1, 1))).max())
    if max_tilt < 5.:
        raise ValueError("fit poses have less than five degrees of board-normal variation")
    verification = [r for r in rows if r["role"] == "verification"]
    return {"schema": "checkerboard_optical_candidate_v1", "source_image_size_wh": size.tolist(),
            "inner_corner_count_xy": list(map(int, inner_corner_count_xy)),
            "measured_pitch_xy_m": list(map(float, measured_pitch_xy_m)),
            "K_source_pixels": K.tolist(), "distortion_coefficients": distortion.ravel().tolist(),
            "distortion_model": "OpenCV pinhole k1,k2,p1,p2,k3; zero skew",
            "fit_rms_px": float(rms), "conditional_intrinsic_std": intrinsic_std.ravel().tolist(),
            "conditional_std_note": "Local OpenCV uncertainty conditional on exact target geometry, model, correspondences and independent pixel noise; not physical metrology or historical applicability.",
            "fit_corner_image_span_fraction_xy": coverage_xy.tolist(),
            "fit_board_normal_max_separation_deg": max_tilt, "homography_diversity": diversity,
            "verification_pose_corner_indices": pose_indices.tolist(),
            "verification_heldout_corner_indices": heldout_indices.tolist(),
            "verification_working_scale_px": float(verification_working_scale_px),
            "verification_within_declared_working_scale": bool(all(
                r["heldout_corner_error"]["p95_px"] <= verification_working_scale_px for r in verification)),
            "views": rows, "physical_calibration_accepted": False,
            "base_registration_verified": False, "joint_zeros_verified": False,
            "gripper_registration_verified": False,
            "limitations": ["Verification estimates an image-specific nuisance board pose from a fixed corner subset. Scored heldout corners never enter that pose or intrinsic fitting, but their image/target errors may be correlated.",
                "Fresh board captures do not independently measure camera-to-robot-base registration, historical recording optics, joint signs/zeros, gripper geometry or loaded contact response.",
                "A low residual cannot certify the printed pitch, rigid flatness, distortion model or manually stated capture provenance.",
                "Checkerboard orientation may be reversed by detection. Intrinsics are invariant to that convention; independent named base registration must resolve the physical target origin/axes."]}
