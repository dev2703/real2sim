"""Known optics, withheld corners, duplicate poses, and absent measured evidence."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from real2sim.geometry.checkerboard import (
    board_points,
    detect_checkerboard,
    fit_checkerboard_optics,
    verification_pose_indices,
)


def observations(pitch=(.025, .0245)):
    points = board_points([8, 6], pitch)
    K = np.array([[1050., 0., 640.], [0., 1020., 360.], [0., 0., 1.]])
    distortion = np.array([-.06, .02, .001, -.002, .005])
    poses = [(-.35, .20, -.10, -.12, -.08, .65),
             (.25, -.30, .15, -.05, -.03, .72),
             (.10, .35, -.15, -.13, -.05, .80),
             (-.30, -.20, .10, -.03, -.10, .70),
             (.30, .15, .05, -.15, .00, .76),
             (-.15, .40, -.15, -.02, -.10, .82),
             (.30, -.15, -.10, -.10, -.02, .72),
             (-.20, .25, .20, -.08, -.08, .76)]
    views = np.asarray([cv2.projectPoints(points, np.array(p[:3]), np.array(p[3:]), K, distortion)[0].reshape(-1, 2)
                        for p in poses])
    groups = ["fit1", "fit1", "fit2", "fit2", "fit3", "fit3", "verify1", "verify2"]
    roles = ["fit"] * 6 + ["verification"] * 2
    return views, groups, roles, K, pitch


def fit(views, groups, roles, pitch):
    return fit_checkerboard_optics(views, groups, roles, source_image_size_wh=[1280, 720],
                                  inner_corner_count_xy=[8, 6], measured_pitch_xy_m=pitch)


def test_known_intrinsics_and_anisotropic_measured_print_pitch():
    views, groups, roles, K, pitch = observations()
    result = fit(views, groups, roles, pitch)
    np.testing.assert_allclose(result["K_source_pixels"], K, atol=.02)
    assert result["fit_rms_px"] < .001
    assert result["verification_within_declared_working_scale"]
    assert min(v["minimum_landmark_depth_m"] for v in result["views"]) > 0
    assert all(v["heldout_corner_error"]["p95_px"] < .01
               for v in result["views"] if v["role"] == "verification")
    assert not result["physical_calibration_accepted"]
    assert not result["base_registration_verified"]
    assert not result["joint_zeros_verified"]
    assert not result["gripper_registration_verified"]


def test_verification_pixels_cannot_change_intrinsics_or_withheld_corner_pose():
    views, groups, roles, _, pitch = observations()
    reference = fit(views, groups, roles, pitch)
    changed = views.copy()
    pose = verification_pose_indices([8, 6])
    withheld = np.setdiff1d(np.arange(48), pose)
    changed[-1, withheld] += [18., -12.]
    result = fit(changed, groups, roles, pitch)
    np.testing.assert_array_equal(result["K_source_pixels"], reference["K_source_pixels"])
    np.testing.assert_array_equal(result["distortion_coefficients"], reference["distortion_coefficients"])
    np.testing.assert_array_equal(result["views"][-1]["rvec_target_to_optical"],
                                  reference["views"][-1]["rvec_target_to_optical"])
    np.testing.assert_array_equal(result["views"][-1]["tvec_target_to_optical_m"],
                                  reference["views"][-1]["tvec_target_to_optical_m"])
    assert result["views"][-1]["heldout_corner_error"]["p95_px"] > 20
    assert not result["verification_within_declared_working_scale"]


def test_repeated_fit_views_do_not_gain_diversity_from_verification():
    views, groups, roles, _, pitch = observations()
    views[:6] = views[0]
    with pytest.raises(ValueError, match="tilt diversity"):
        fit(views, groups, roles, pitch)


def test_capture_groups_and_native_pixel_bounds_are_strict():
    views, groups, roles, _, pitch = observations()
    groups[-1] = groups[0]
    with pytest.raises(ValueError, match="must not cross"):
        fit(views, groups, roles, pitch)
    groups[-1] = "verify2"
    views[-1, 0, 0] = 1280
    with pytest.raises(ValueError, match="native pixels"):
        fit(views, groups, roles, pitch)


def test_detector_uses_complete_grid_on_original_pixels():
    tile = np.array([[0, 255], [255, 0]], np.uint8)
    board = np.tile(tile, (4, 5))[:7, :9]
    board = np.repeat(np.repeat(board, 35, 0), 35, 1)
    gray = np.full((350, 430), 255, np.uint8)
    gray[45:290, 50:365] = board
    points = detect_checkerboard(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), [8, 6])
    assert points.shape == (48, 2)
    # Ordering may flip; physical source pixels and scale must remain unchanged.
    np.testing.assert_allclose(np.sort(np.unique(np.round(points[:, 0]))),
                               np.arange(85, 331, 35), atol=1)
    assert detect_checkerboard(np.full((350, 430, 3), 255, np.uint8), [8, 6]) is None


def load_cli():
    path = Path(__file__).resolve().parents[2] / "scripts/calibrate_v2_capture.py"
    spec = importlib.util.spec_from_file_location("capture_optics_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_design_pitch_is_never_a_measurement_and_incomplete_fit_does_not_decode(tmp_path, monkeypatch):
    cli = load_cli()
    session = {"schema": "v2_calibration_capture_plan_v1", "status": "planned_not_measured",
               "target": {}, "camera_records": [{"camera_key": "observation.images.iphone",
                   "fit_capture_groups": ["fit1", "fit2", "fit3"],
                   "verification_capture_groups": ["verify1", "verify2"]}],
               "collection_plan": {"design": {"square_pitch_design_m": .025}}}
    path = tmp_path / "session.json"
    path.write_text(json.dumps(session))
    readiness, request = cli.inspect_inputs(path)
    assert not readiness["ready_for_optics_fit"]
    assert not readiness["design_dimensions_used_as_measurements"]
    assert any("square_pitch_measured_xy_m" in text for text in readiness["missing_inputs"])
    assert request["target"] == {}
    monkeypatch.setattr(cli, "_decode_capture", lambda row: pytest.fail("incomplete inputs decoded"))
    out = tmp_path / "optics"
    with pytest.raises(ValueError, match="inputs incomplete"):
        cli.fit_capture(path, out)
    assert not out.exists()
