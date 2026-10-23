"""Diagnostic safeguards: observation separation, gauge freedom, and plane frames."""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


def diagnostic(name):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


@pytest.fixture
def wrist():
    module = diagnostic("diag_v2_wrist_zeros")
    rng = np.random.default_rng(4)
    states = np.c_[rng.uniform(-35, 35, (10, 5)), np.full(10, 50.)]
    arrays = {"episode_index": np.zeros(10, int), "frame_index": np.arange(10),
              "joint_state": states}
    task = {"robot_so101": {"real_to_sim": {"sign": [1] * 5, "offset_deg": [0] * 5},
                              "gripper": {"sim_range_rad": [-.1745, 1.7454]}}}
    transform = np.eye(4)
    transform[:3, 3] = [0, 0, 1.]
    camera = {"base_foot": {"candidate_K_early": [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]],
                             "candidate_T_optical_base_early": transform},
              "image_similarity_models": {"without_fixed_base_foot": {
                  "matrix_reference_to_late": np.eye(3)[:2]}}}
    captures = [{"episode": 0, "frame": i, "split": "train" if i < 8 else "val",
                 "lens": [0., 0.], "horn": [0., 0.]} for i in range(10)]
    return module, module.WristFit(arrays, task, camera, np.zeros(3), captures), states


def test_validation_pixels_cannot_change_wrist_fit_objective(wrist):
    _, fit, _ = wrist
    expected = fit.residuals([2., -3.])
    for capture in fit.captures:
        if capture["split"] == "val":
            capture["lens"] = [1e6, -1e6]
            capture["horn"] = [-1e6, 1e6]
    np.testing.assert_array_equal(fit.residuals([2., -3.]), expected)
    assert max(r["error_px"] for r in fit.rows([2., -3.]) if r["split"] == "val") > 1e6


def test_free_downstream_point_has_unidentifiable_roll_gauge(wrist):
    module, fit, _ = wrist

    def free_point(x):
        return fit.residuals(x[:2], lens_offset_mm=x[2:], landmarks=("lens",))

    identified = module.identifiability(free_point, np.array([2., -3., 1., 2., 3.]))
    assert identified["rank"] == 4
    assert identified["parameters"] == 5
    assert identified["conditional_one_sigma"] is None
    assert identified["singular_values_after_8px_weighting"][-1] < 1e-8
    known = module.identifiability(lambda x: fit.residuals(x, landmarks=("lens",)),
                                  np.array([2., -3.]))
    assert known["rank"] == 2


def test_limit_check_keeps_unclipped_coordinates(wrist):
    module, fit, states = wrist
    modified = states.copy()
    modified[:, 4] = 720.
    result = module.native_limit_check(fit, modified, np.zeros(5))
    assert result["per_joint_violating_ticks"]["wrist_roll"] == len(states)
    assert result["mapped_range_deg"][4] == pytest.approx([720., 720.])
    assert result["used_in_fitting"] is False
    np.testing.assert_array_equal(modified[:, 4], 720.)


def test_mat_thickness_preserves_table_support_plane():
    module = diagnostic("diag_v2_camera_uncertainty")
    transform = module.base_in_mat([.1, -.2, .3], .003)
    # Both planes use the downward-positive mat frame. A tabletop below the mat
    # must be +3mm, not -3mm; native support is 2.4mm below native origin.
    assert (transform @ [0, 0, -.0024, 1])[2] == pytest.approx(.003)
    assert (transform @ [0, 0, .0151, 1])[2] == pytest.approx(.003 - .0175)


def test_conditional_intrinsics_fit_recovers_noncentral_rectangular_pixels():
    module = diagnostic("diag_v2_camera_uncertainty")
    K = np.array([[1300., 0., 650.], [0., 1313., 350.], [0., 0., 1.]])
    rvec, translation = np.array([.45, -.2, .1]), np.array([-.1, -.15, .9])
    points = np.array([[0, 0, 0], [.305, 0, 0], [.305, .406, 0], [0, .406, 0]])
    distortion = np.array([.05, 0., 0., 0., 0.])
    corners = cv2.projectPoints(points, rvec, translation, K, distortion)[0].reshape(-1, 2)
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = translation
    fit = module.fit_with_intrinsic_assumptions(corners, [.305, .406],
        {"focal_px": 1320., "T_optical_plane": transform}, center=(650., 350.),
        aspect=1.01, radial=.05)
    assert fit["mat_fit_rms_px"] < 1e-7
    np.testing.assert_allclose(fit["K"], K, atol=1e-4)
    np.testing.assert_allclose(fit["T_optical_plane"], transform, atol=1e-7)
