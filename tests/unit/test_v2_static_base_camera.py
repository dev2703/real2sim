"""Coordinate and independent image-prediction checks for the camera diagnostic."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_diagnostic():
    path = Path(__file__).resolve().parents[2] / "scripts/diag_v2_static_base_camera.py"
    spec = importlib.util.spec_from_file_location("static_base_camera", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_up_maps_above_mat_without_reflection():
    diagnostic = load_diagnostic()
    transform = diagnostic.base_transform([-.1, .14, .2])
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-14)
    # Native base support at -2.4mm meets the mat; its top plane is17.5mm above it.
    np.testing.assert_allclose((transform @ [0, 0, -.0024, 1])[2], 0., atol=1e-14)
    np.testing.assert_allclose((transform @ [0, 0, .0151, 1])[2], -.0175, atol=1e-14)


def test_similarity_predicts_a_region_excluded_from_fitting():
    diagnostic = load_diagnostic()
    matrix = np.array([[.973, .001, 17.], [-.001, .973, 11.]])
    points = np.array([[60, 460], [90, 500], [140, 450], [550, 90], [620, 100], [690, 110.]])
    targets = diagnostic.transform_pixels(points, matrix)
    matches = [{"reference_px": p.tolist(), "target_px": q.tolist(),
                "region": "cabinet" if i > 2 else "clamp"}
               for i, (p, q) in enumerate(zip(points, targets, strict=True))]
    # A deliberately false correspondence in the withheld foot cannot influence fit.
    rows = [{"target_episode": 7, "matches": matches + [
        {"reference_px": [200., 350.], "target_px": [900., 600.], "region": "foot"}]}]
    fitted, info = diagnostic.fit_image_similarity(rows, excluded_region="foot")
    np.testing.assert_allclose(fitted, matrix, atol=1e-5)
    assert info["match_count"] == 6
    new_points = np.array([[180, 350], [300, 380.]])
    np.testing.assert_allclose(diagnostic.transform_pixels(new_points, fitted),
                               diagnostic.transform_pixels(new_points, matrix), atol=1e-4)


def test_all_match_score_keeps_repeated_pattern_outliers():
    diagnostic = load_diagnostic()
    rows = [{"matches": [
        {"reference_px": [0., 0.], "target_px": [1., 0.], "similarity_inlier": True},
        {"reference_px": [0., 0.], "target_px": [30., 0.], "similarity_inlier": False},
    ]}]
    scores = diagnostic.score_image_similarity(rows, np.eye(3)[:2])
    assert scores["all_descriptor_matches"]["maximum"] == 30.
    assert scores["all_descriptor_matches"]["count"] == 2
    assert scores["previous_per_frame_ransac_inliers"]["count"] == 1


def test_image_similarity_camera_decomposition_preserves_projection():
    diagnostic = load_diagnostic()
    theta, scale = .02, .973
    c, s = np.cos(theta), np.sin(theta)
    matrix = np.array([[scale * c, -scale * s, 17.], [scale * s, scale * c, 11.]])
    intrinsic = np.array([[1371., 0, 640.], [0, 1371., 360.], [0, 0, 1.]])
    pose = np.eye(4)
    pose[:3, 3] = [.1, -.2, 1.]
    K_late, T_late = diagnostic.camera_after_image_similarity(intrinsic, pose, matrix)
    points = np.array([[.1, .2, 0.], [.4, -.1, .2], [-.3, .5, .8]])

    def project(K, T):
        camera = points @ T[:3, :3].T + T[:3, 3]
        pixels = camera @ K.T
        return pixels[:, :2] / pixels[:, 2:]

    np.testing.assert_allclose(project(K_late, T_late),
                               diagnostic.transform_pixels(project(intrinsic, pose), matrix),
                               atol=1e-10)
    assert K_late[0, 1] == K_late[1, 0] == 0.
    with pytest.raises(ValueError, match="similarity"):
        diagnostic.camera_after_image_similarity(intrinsic, pose, [[.9, 0, 0], [0, 1.1, 0]])
