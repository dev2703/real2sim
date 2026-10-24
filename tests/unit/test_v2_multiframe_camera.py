"""Static-camera fits retain whole-frame verification and planar gauges."""

import copy
import json

import cv2
import numpy as np
import pytest

from real2sim.data.raw import sha256_file
from scripts import diag_v2_multiframe_camera as diagnostic
from scripts.diag_v2_model_base_silhouette import compare_masks
from scripts.extract_v2_model_mat_edges import source_edges


def synthetic_camera():
    points = diagnostic.mat_points([.305, .406])
    K = np.array([[1100., 0, 640.], [0, 1100., 360.], [0, 0, 1.]])
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array([.55, -.35, .15]))[0]
    T[:3, 3] = [-.13, -.16, .9]
    return points, K, T


def test_repeating_one_plane_does_not_identify_free_intrinsics_or_distortion():
    points, K, T = synthetic_camera()
    for model, expected_parameters in (("free_intrinsics", 10), ("free_intrinsics_radial2", 12)):
        once = diagnostic.intrinsic_rank(points, K, T, model=model, repeats=1)
        repeated = diagnostic.intrinsic_rank(points, K, T, model=model, repeats=6)
        assert once["parameter_count"] == expected_parameters
        assert once["rank"] <= 8
        assert repeated["rank"] == once["rank"]
        assert not repeated["full_rank"]
        assert len(repeated["scaled_null_directions"]) >= expected_parameters - 8
        assert repeated["conditional_parameter_std"] is None


def test_centered_square_tilted_rectangle_is_only_conditionally_identifiable():
    points, K, T = synthetic_camera()
    result = diagnostic.intrinsic_rank(points, K, T)
    assert result["rank"] == 7
    assert result["full_rank"]


def test_edge_intersections_recover_virtual_corners_and_reject_unsupported_edges():
    samples = {"top": [[20, 10], [40, 10], [60, 10]],
               "right": [[80, 20], [80, 40], [80, 60]],
               "bottom": [[20, 70], [40, 70], [60, 70]],
               "left": [[10, 20], [10, 40], [10, 60]]}
    corners, report = diagnostic.edge_intersections(samples)
    np.testing.assert_allclose(corners, [[10, 10], [80, 10], [80, 70], [10, 70]], atol=1e-12)
    assert report["top"]["orthogonal_rms_px"] < 1e-12
    samples["left"] = [[10, 20], [10, 20], [10, 20]]
    with pytest.raises(ValueError, match="span"):
        diagnostic.edge_intersections(samples)


def test_verification_pixels_do_not_select_solution_or_sensitivity_envelope():
    points, K, T = synthetic_camera()
    pixels = diagnostic.project(points, K, T)
    fit = {"episode": 0, "frame": 0, "role": "fit", "sigma_px": 2., "corners": pixels}
    verification = {"episode": 2, "frame": 1, "role": "verification", "sigma_px": 2., "corners": pixels}
    a = diagnostic.fit_regime([fit, verification], [.305, .406], [1280, 720])
    b = diagnostic.fit_regime([fit, {**verification, "corners": pixels + 30}], [.305, .406], [1280, 720])
    assert a["baseline"] == b["baseline"]
    assert a["sensitivity"] == b["sensitivity"]
    assert a["per_frame_scores"][1]["rms_px"] < 1e-6
    assert b["per_frame_scores"][1]["rms_px"] > 40
    assert not a["robot_base_registration_established"]
    json.dumps(a, allow_nan=False)


def annotation_fixture(tmp_path):
    _, K, T = synthetic_camera()
    image = tmp_path / "source.png"
    cv2.imwrite(str(image), np.zeros((720, 1280, 3), np.uint8))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": [0, 2], "val": [9], "reserved": [1]}))
    selected = {"episode": 0, "frame": 4, "path": str(image), "sha256": sha256_file(image),
                "camera_key": "observation.images.iphone", "camera_regime": "early", "role": "verification"}
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"split_sha256": sha256_file(split), "images": [selected]}))
    row = {key: value for key, value in selected.items() if key != "camera_key"}
    row.update(sigma_px=3., mat_corners_source_px=diagnostic.project(diagnostic.mat_points([.305, .406]), K, T).tolist())
    document = {"schema": "v2_multiframe_static_annotations_v1", "source_size_wh": [1280, 720],
                "mat_side_lengths_m": [.305, .406], "split_path": str(split),
                "selection_path": str(selection), "selection_sha256": sha256_file(selection), "frames": [row]}
    return document


def test_random_selection_roles_cannot_be_changed_after_viewing(tmp_path):
    document = annotation_fixture(tmp_path)
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(document))
    diagnostic.load_annotations(path)
    document["frames"][0]["role"] = "fit"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="random selection"):
        diagnostic.load_annotations(path)


def test_reserved_images_and_reassigned_camera_stream_are_refused(tmp_path):
    document = annotation_fixture(tmp_path)
    document.pop("selection_path")
    document.pop("selection_sha256")
    path = tmp_path / "annotations.json"
    bad = copy.deepcopy(document)
    bad["frames"][0]["episode"] = 1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="reserved"):
        diagnostic.load_annotations(path)
    bad = copy.deepcopy(document)
    bad["frames"][0]["camera"] = "observation.images.top"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="moving wrist"):
        diagnostic.load_annotations(path)


def test_uncapped_bidirectional_contour_scores_retain_extra_model_components():
    mask = np.zeros((120, 160), bool)
    mask[30:80, 40:100] = True
    exact = compare_masks(mask, mask)
    assert exact["mask_iou"] == 1
    assert exact["symmetric_uncapped"]["rms_px"] == 0
    shifted = np.roll(mask, 20, axis=1)
    score = compare_masks(mask, shifted)
    assert 0 < score["mask_iou"] < 1
    assert score["symmetric_uncapped"]["max_px"] == pytest.approx(20)
    extra = mask.copy()
    extra[95:105, 120:140] = True
    assert compare_masks(extra, mask)["observed_to_nominal"]["max_px"] > 30


def test_model_cut_around_white_gripper_does_not_become_physical_top_edge():
    image = np.full((300, 380, 3), 200, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    quad = np.array([[60, 40], [300, 40], [340, 260], [30, 260]])
    cv2.fillConvexPoly(mask, quad, 1)
    image[mask.astype(bool)] = 40
    # A longer interior white-gripper cut must not replace the shorter, visible
    # outer top pieces at y40. This is synthetic geometry, not a measured image.
    mask[20:80, 110:250] = 0
    image[20:80, 110:250] = 230
    samples, info = source_edges(mask.astype(bool), image)
    top = np.asarray(samples["top"])
    assert np.max(np.abs(top[:, 1] - 40)) < 2
    assert info["top"]["visible_span_px"] > 80


def test_model_mask_alone_cannot_override_missing_raw_image_contrast():
    mask = np.zeros((300, 380), np.uint8)
    cv2.fillConvexPoly(mask, np.array([[60, 40], [300, 40], [340, 260], [30, 260]]), 1)
    with pytest.raises(ValueError, match="contrast"):
        source_edges(mask.astype(bool), np.full((300, 380, 3), 100, np.uint8))
