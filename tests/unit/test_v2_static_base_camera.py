"""Coordinate and independent image-prediction checks for the camera diagnostic."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim.sim.cameras import camera_projection_attributes


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


@pytest.mark.parametrize("regime", ["early", "late"])
def test_candidate_camera_compiled_frustum_matches_nonplanar_points(regime):
    diagnostic = load_diagnostic()
    # Synthetic inputs keep normal unit tests independent of generated experiment reports.
    focal, cx, cy = (1371., 640., 360.) if regime == "early" else (1334., 640.17, 360.94)
    K = np.array([[focal, 0., cx], [0., focal, cy], [0., 0., 1.]])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [.3, -.4, .2]).as_matrix()
    transform[:3, 3] = [.1, -.2, .8]
    camera = diagnostic.camera_descriptor(K, transform)
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "camera", name="candidate",
                  pos=" ".join(map(str, camera["approx_position_task"])),
                  xyaxes=" ".join(map(str, camera["xyaxes_task"])),
                  **camera_projection_attributes(camera))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root).decode())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    view, scene = mujoco.MjvCamera(), mujoco.MjvScene(model, maxgeom=10)
    view.type, view.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, 0
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, view,
                          mujoco.mjtCatBit.mjCAT_ALL, scene)
    frustum = scene.camera[0]
    optical = np.array([[-.15, -.12, .6], [.2, .1, .9], [0, .15, 1.2], [.05, -.05, .75]])
    world_points = (optical - transform[:3, 3]) @ transform[:3, :3]
    compiled_local = (world_points - data.cam_xpos[0]) @ data.cam_xmat[0].reshape(3, 3)
    plane = compiled_local[:, :2] * frustum.frustum_near / -compiled_local[:, 2:]
    u = 1280 * (.5 + (plane[:, 0] - frustum.frustum_center) / (2 * frustum.frustum_width))
    v = 720 * (frustum.frustum_top - plane[:, 1]) / (frustum.frustum_top - frustum.frustum_bottom)
    expected = optical @ K.T
    np.testing.assert_allclose(np.c_[u, v], expected[:, :2] / expected[:, 2:], atol=1e-4)
