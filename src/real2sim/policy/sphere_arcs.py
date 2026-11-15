"""Robust known-radius sphere fits from well distributed RGB tangent arcs.

The fit has no table, tool or temporal prior. Occluding edges are outliers, not
sphere boundaries. A wide angular baseline is required to constrain monocular
depth; half disks and small visible caps intentionally supply no measurement.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def fit_sphere_arcs(points, K, radius):
    """Return camera-frame centre, RMS and support, or reject a weak arc fit.

    Pixel thresholds account for raster contour uncertainty. Three unit tangent
    rays define a plane on the unit sphere; its normal and offset determine the
    known-radius sphere. Fixed hypotheses make RANSAC deterministic.
    """
    points = np.asarray(points, float)
    sample = points[np.linspace(0, len(points) - 1, min(96, len(points))).astype(int)]
    focal = float(min(K[0, 0], K[1, 1]))

    def rays_from(center):
        radial = sample - center
        boundary = sample + .5 * radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
        rays = np.c_[boundary, np.ones(len(boundary))] @ np.linalg.inv(K).T
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    rays = rays_from(sample.mean(axis=0))
    triples = np.random.default_rng(0).integers(0, len(rays), size=(384, 3))
    a, b, c = (rays[triples[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(normal_length, 1e-15)
    normals *= np.where(normals[:, 2:] < 0, -1., 1.)
    cosine = np.sum(normals * a, axis=1)
    sine = np.sqrt(np.maximum(1 - cosine ** 2, 1e-15))
    centers = normals * (radius / sine)[:, None]
    # Express tangent error in pixels, so tolerance follows actual resolution.
    errors = np.abs(np.linalg.norm(np.cross(rays[None], centers[:, None]), axis=2) - radius)
    errors *= (focal / np.maximum(centers[:, 2], radius))[:, None]
    counts = np.sum(errors <= .8, axis=1)
    counts[(cosine <= 0) | (centers[:, 2] <= radius)
           | (normal_length[:, 0] < 1e-10)
           | (focal * radius / np.maximum(centers[:, 2], radius) < 8)] = 0
    best = int(np.argmax(counts))
    if counts[best] < .70 * len(sample):
        return None
    center = centers[best]
    inliers = errors[best] <= .8
    for _ in range(3):
        pixel = (K @ center)[:2] / center[2]
        rays = rays_from(pixel)
        selected = rays[inliers]
        fit = least_squares(lambda x, selected=selected: np.linalg.norm(np.cross(selected, x), axis=1) - radius,
                            center, max_nfev=40)
        center = fit.x
        if not fit.success or center[2] <= radius:
            return None
        error_px = (np.linalg.norm(np.cross(rays, center), axis=1) - radius) * focal / center[2]
        inliers = np.abs(error_px) <= .8
        if np.count_nonzero(inliers) < .70 * len(sample):
            return None
    # Require at least 240 degrees of tangent directions. This rejects the
    # ill-conditioned fits produced by a half sphere or a small visible cap.
    axis = center / np.linalg.norm(center)
    u = np.cross(axis, [0., 1., 0.])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.sort(np.arctan2(rays[inliers] @ v, rays[inliers] @ u))
    largest_gap = np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi]))
    if largest_gap > np.deg2rad(120):
        return None
    # Visible foreground must fit inside the recovered sphere. Reject fits to
    # an interior arc of a larger blob, including merged same-colour objects.
    if np.max(error_px) > 1.5:
        return None
    rms = float(np.sqrt(np.mean((error_px[inliers] * center[2] / focal) ** 2)))
    return center, rms, float(np.mean(inliers))
