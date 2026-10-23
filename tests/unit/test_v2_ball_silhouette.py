"""Independent ray geometry checks for sphere scale and support height."""

import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "ball_silhouette", Path(__file__).resolve().parents[2] / "scripts/diag_v2_ball_support.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def tangent_pixels(center, radius, intrinsic):
    normal = center / np.linalg.norm(center)
    first = np.cross(normal, [0, 1, 0])
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    angle = np.arcsin(radius / np.linalg.norm(center))
    phase = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    rays = np.cos(angle) * normal + np.sin(angle) * (
        np.cos(phase)[:, None] * first + np.sin(phase)[:, None] * second)
    projected = rays @ intrinsic.T
    return projected[:, :2] / projected[:, 2:]


def test_exact_off_axis_sphere_recovery_and_radius_scale_gauge():
    intrinsic = np.array([[1371., 0, 640.], [0, 1371., 360.], [0, 0, 1.]])
    center, radius = np.array([.17, -.09, .85]), .0335
    pixels = tangent_pixels(center, radius, intrinsic)
    recovered, score = MODULE.sphere_center(pixels, intrinsic, radius)
    np.testing.assert_allclose(recovered, center, atol=1e-8)
    assert score["rms_equivalent_pixels"] < 1e-7
    scaled, _ = MODULE.sphere_center(pixels, intrinsic, radius * 1.05)
    np.testing.assert_allclose(scaled, center * 1.05, atol=1e-8)
    np.testing.assert_allclose(tangent_pixels(scaled, radius * 1.05, intrinsic), pixels, atol=1e-8)


def test_support_height_uses_native_table_plane_not_the_mat_surface():
    transform = np.eye(4)
    center = np.array([.2, .1, .05])
    native, height = MODULE.support_height(center, transform, .0335, table_z=-.0024)
    np.testing.assert_allclose(native, center)
    np.testing.assert_allclose(height, .0189)
    # A3mm mat is part of the total tabletop support height, not a shifted native table.
    np.testing.assert_allclose(height - .003, .0159)
