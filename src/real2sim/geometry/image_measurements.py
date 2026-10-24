"""Source-pixel shape measurements; no claim of metric calibration or object truth."""

from __future__ import annotations

import cv2
import numpy as np


def ellipse_points(center, diameters, angle_deg, count=120):
    theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
    local = np.c_[np.cos(theta) * diameters[0] / 2, np.sin(theta) * diameters[1] / 2]
    angle = np.radians(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return local @ rotation.T + np.asarray(center)


def bowl_rim_candidates(rgb: np.ndarray, box) -> list[dict]:
    """Fit observed near-complete rim arcs, retaining inner/outer ambiguity.

    The semantic model supplies the ROI, not the rim geometry. A fit must cover
    >=80% of ellipse angles and agree to1.5px RMS; fitting an occluded short arc
    cannot manufacture a complete rim. Thresholds are fixed before metric fitting.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("source uint8 RGB required")
    x0, y0, x1, y1 = np.asarray(box, float)
    x0, y0 = max(0, int(x0) - 8), max(0, int(y0) - 8)
    x1, y1 = min(rgb.shape[1], int(x1) + 8), min(rgb.shape[0], int(y1) + 8)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("nonempty bowl ROI required")
    gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 25, 70)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        if len(contour) < 70:
            continue
        (u, v), (a, b), angle = cv2.fitEllipse(contour)
        if not (.65 * (x1 - x0) < max(a, b) < 1.15 * (x1 - x0)
                and .28 < min(a, b) / max(a, b) < .95
                and .12 * (y1 - y0) < v < .7 * (y1 - y0)):
            continue
        theta = np.radians(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        normalized = ((contour[:, 0] - [u, v]) @ rotation) / [a / 2, b / 2]
        error = (np.linalg.norm(normalized, axis=1) - 1) * min(a, b) / 2
        bins = (np.floor((np.arctan2(normalized[:, 1], normalized[:, 0]) + np.pi) / (2 * np.pi) * 36).astype(int) % 36)
        coverage = len(np.unique(bins)) / 36
        rms = float(np.sqrt(np.mean(error ** 2)))
        if coverage < .8 or rms > 1.5:
            continue
        candidates.append({"center_source_px": [u + x0, v + y0], "diameters_px": [a, b],
                           "angle_deg": angle, "contour_points": len(contour),
                           "angular_coverage_fraction": coverage, "radial_rms_px": rms})
    return sorted(candidates, key=lambda c: (max(c["diameters_px"]), c["radial_rms_px"]))


def fit_plane_circle(points_xyz: np.ndarray) -> dict:
    """Algebraic circle fit in a declared horizontal plane; report anisotropy too."""
    points = np.asarray(points_xyz, float)
    if points.ndim != 2 or points.shape[1] not in (2, 3) or len(points) < 8 or not np.isfinite(points).all():
        raise ValueError("at least8 finite plane points required")
    xy = points[:, :2]
    design = np.c_[2 * xy, np.ones(len(xy))]
    x, _, rank, _ = np.linalg.lstsq(design, np.sum(xy ** 2, axis=1), rcond=None)
    if rank != 3:
        raise ValueError("noncollinear plane points required")
    radius = float(np.sqrt(x[2] + x[:2] @ x[:2]))
    residual = np.linalg.norm(xy - x[:2], axis=1) - radius
    axes = np.sqrt(2 * np.linalg.eigvalsh(np.cov(xy.T, bias=True)))
    return {"center_plane_xy_m": x[:2].tolist(), "radius_m": radius,
            "diameter_m": 2 * radius, "circle_rms_m": float(np.sqrt(np.mean(residual ** 2))),
            "second_moment_axis_ratio": float(axes[0] / axes[-1])}
