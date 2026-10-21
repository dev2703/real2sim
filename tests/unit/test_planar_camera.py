import json

import cv2
import numpy as np
import pytest

from real2sim.geometry.planar_camera import backproject_to_plane, fit_planar_camera


def scene(rotation=(.55, -.35, .15), focal=1100.):
    points = np.array([[0, 0, 0], [.305, 0, 0], [.305, .406, 0], [0, .406, 0.]])
    K = np.array([[focal, 0, 640.], [0, focal, 360.], [0, 0, 1.]])
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array(rotation, float))[0]
    T[:3, 3] = [-.13, -.16, .9]
    pixels = cv2.projectPoints(points, np.array(rotation, float), T[:3, 3], K, None)[0].reshape(-1, 2)
    return points, pixels, K, T


def test_tilted_metric_rectangle_recovers_focal_pose_and_backprojection():
    points, pixels, K, T = scene()
    result = fit_planar_camera(pixels, source_size_wh=(1280, 720))
    assert result["focal_px"] == pytest.approx(K[0, 0], abs=1e-4)
    np.testing.assert_allclose(result["T_optical_plane"], T, atol=1e-7)
    assert result["fit_rms_px"] < 1e-8
    assert result["identifiability"]["rank"] == 7
    assert result["identifiability"]["conditional_parameter_std"][0] > 0
    assert result["planar_ambiguity_requires_independent_check"]
    assert not result["automatically_adopted"]
    xyz = backproject_to_plane(pixels, K=result["K"], T_optical_plane=result["T_optical_plane"])
    np.testing.assert_allclose(xyz, points, atol=1e-8)
    elevated = points + [0, 0, .05]
    elevated_pixels = cv2.projectPoints(elevated, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, None)[0].reshape(-1, 2)
    np.testing.assert_allclose(backproject_to_plane(
        elevated_pixels, K=K, T_optical_plane=T, plane_z_m=.05), elevated, atol=1e-10)
    json.dumps(result, allow_nan=False)


def test_noisy_fit_is_finite_and_verification_cannot_change_solution():
    points, pixels, _, _ = scene()
    noisy = pixels + np.random.default_rng(0).normal(0, .3, pixels.shape)
    a = fit_planar_camera(noisy, source_size_wh=(1280, 720), sigma_px=.3,
                          verification_plane_points_m=points, verification_pixels_px=pixels)
    b = fit_planar_camera(noisy, source_size_wh=(1280, 720), sigma_px=.3,
                          verification_plane_points_m=points, verification_pixels_px=pixels + 80)
    np.testing.assert_array_equal(a["T_optical_plane"], b["T_optical_plane"])
    assert a["focal_px"] == b["focal_px"]
    assert abs(a["focal_px"] - 1100) < 50
    assert a["fit_rms_px"] < 1
    assert b["verification"]["p95_px"] > a["verification"]["p95_px"] + 100
    assert not b["verification"]["used_for_fitting_or_solution_selection"]
    assert a["fit_input_sha256"] == b["fit_input_sha256"]
    json.dumps(a, allow_nan=False)


def test_frontoparallel_focal_is_flagged_as_unidentifiable():
    _, pixels, _, _ = scene(rotation=(0, 0, 0))
    result = fit_planar_camera(pixels, source_size_wh=(1280, 720))
    assert result["identifiability"]["rank"] < 7
    assert result["identifiability"]["ill_conditioned"]
    assert result["identifiability"]["conditional_parameter_std"] is None
    assert result["fit_rms_px"] < 1e-6
    assert result["near_optimal_focal_range_px"][1] > result["near_optimal_focal_range_px"][0] * 2
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("pixels", [
    [[100, 100], [200, 100], [300, 100], [400, 100]],
    [[100, 100], [200, 200], [200, 100], [100, 200]],
    [[100, 100], [100, 100], [200, 200], [100, 200]],
])
def test_bad_corner_geometry_is_refused(pixels):
    with pytest.raises(ValueError, match="degenerate|boundary"):
        fit_planar_camera(pixels, source_size_wh=(1280, 720))


def test_backprojection_rejects_parallel_and_behind_camera():
    T = np.eye(4)
    T[:3, 3] = [0, 0, -1]
    with pytest.raises(ValueError, match="behind"):
        backproject_to_plane([[0, 0]], K=np.eye(3), T_optical_plane=T)
    T[:3, :3] = cv2.Rodrigues(np.array([np.pi / 2, 0, 0]))[0]
    with pytest.raises(ValueError, match="parallel"):
        backproject_to_plane([[0, 0]], K=np.eye(3), T_optical_plane=T)


def test_focal_bound_does_not_silently_pass_as_calibrated():
    _, pixels, _, _ = scene()
    result = fit_planar_camera(pixels, source_size_wh=(1280, 720), focal_bounds_px=(400, 800))
    assert 400 <= result["focal_px"] <= 800
    assert result["focal_at_bound"]
    assert result["fit_rms_px"] > 1
    assert not result["automatically_adopted"]
