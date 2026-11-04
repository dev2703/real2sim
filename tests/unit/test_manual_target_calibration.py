"""Synthetic arbitrary planar marks verify optics, provenance and split isolation."""

import copy
import json

import cv2
import numpy as np
import pytest

from real2sim.data.raw import sha256_file
from scripts.calibrate_v2_manual_target import fit_capture, inspect_inputs


@pytest.fixture
def manual_input(tmp_path):
    x, y = np.meshgrid(np.arange(4) * .04, np.arange(3) * .04)
    points = np.c_[x.ravel(), y.ravel(), np.zeros(12)]
    K = np.array([[900., 0., 640.], [0., 880., 360.], [0., 0., 1.]])
    distortion = np.array([-.03, .01, .0004, -.0003, 0.])
    rotations = [[.1, .2, .03], [-.3, .1, -.1], [.3, -.25, .05],
                 [-.1, -.3, .2], [.4, .1, -.2], [-.35, -.2, .15],
                 [.2, .35, -.1], [-.2, -.25, -.1]]
    translations = [[-.11, -.07, .34], [-.02, -.08, .4], [-.12, .01, .35],
                    [0., -.01, .4], [-.04, -.04, .32], [-.11, -.04, .36],
                    [-.08, -.04, .37], [-.04, -.03, .34]]
    captures = []
    for i, (rotation, translation) in enumerate(zip(rotations, translations, strict=True)):
        pixels = cv2.projectPoints(points, np.asarray(rotation), np.asarray(translation), K, distortion)[0].reshape(-1, 2)
        image = np.full((720, 1280, 3), 245, np.uint8)
        for pixel in pixels:
            cv2.drawMarker(image, tuple(np.rint(pixel).astype(int)), (0, 0, 0), cv2.MARKER_CROSS, 10, 1)
        cv2.putText(image, f"synthetic view {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        path = tmp_path / f"view_{i}.png"
        assert cv2.imwrite(str(path), image)
        role = "fit" if i < 6 else "verification"
        group = f"fit_{i // 2}" if i < 6 else f"verify_{i - 6}"
        captures.append({"capture_group": group, "role": role,
            "source_frame_reference": f"synthetic-view-{i}", "source_pixels_unresized": True,
            "source_image_file": path.name, "source_image_sha256": sha256_file(path),
            "points_source_px": [{"id": f"p{j+1:02d}", "pixel_xy": value.tolist()} for j, value in enumerate(pixels)]})
    document = {"schema": "manual_metric_target_capture_input_v1", "status": "captured_measured",
        "camera_key": "observation.images.iphone", "source_image_size_wh": [1280, 720],
        "source_image_size_provenance": "Synthetic fixture image size",
        "capture_settings_provenance": "Synthetic known camera; no hardware used",
        "pixel_annotation_uncertainty_px": .1, "verification_working_scale_px": 3.,
        "target": {"geometry_evidence": "synthetic_test", "coordinate_frame_definition": "XY plane, z=0",
            "measurement_method": "Known generated synthetic coordinates; not physical measurements",
            "dimensional_uncertainty_m": .0001, "flat_rigid_backing": "Synthetic plane",
            "flatness_evidence": "Synthetic all-z-zero points",
            "points": [{"id": f"p{i+1:02d}", "point_target_m": value.tolist()} for i, value in enumerate(points)]},
        "split": {"fit_groups": [f"fit_{i}" for i in range(3)], "verification_groups": ["verify_0", "verify_1"]},
        "verification_pose_point_ids": ["p01", "p04", "p09", "p12"], "captures": captures}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(document))
    return path, document, K


def save(path, document):
    path.write_text(json.dumps(document))


def test_known_multi_pose_camera_recovered_with_manual_correspondences(manual_input, tmp_path):
    path, _, K = manual_input
    ready, _ = inspect_inputs(path)
    assert ready["ready_for_optics_fit"], ready
    result = fit_capture(path, tmp_path / "fit")
    np.testing.assert_allclose(result["K_source_pixels"], K, atol=.05, rtol=0)
    assert result["fit_rms_px"] < .001
    assert result["verification_within_declared_working_scale"]
    assert result["physical_calibration_accepted"] is False
    assert result["base_registration_verified"] is False
    assert result["changed_inputs"] == []
    assert len(result["verification_heldout_point_indices"]) == 8
    assert all(row["minimum_point_depth_m"] > 0 for row in result["views"])


def test_heldout_scored_points_cannot_change_intrinsics_or_verification_pose(manual_input, tmp_path):
    path, document, _ = manual_input
    original = fit_capture(path, tmp_path / "original")
    changed = copy.deepcopy(document)
    excluded = set(changed["verification_pose_point_ids"])
    for row in changed["captures"]:
        if row["role"] == "verification":
            for point in row["points_source_px"]:
                if point["id"] not in excluded:
                    point["pixel_xy"][0] += 15.
    second = tmp_path / "changed.json"
    save(second, changed)
    altered = fit_capture(second, tmp_path / "altered")
    np.testing.assert_array_equal(original["K_source_pixels"], altered["K_source_pixels"])
    np.testing.assert_array_equal(original["distortion_coefficients"], altered["distortion_coefficients"])
    assert not altered["verification_within_declared_working_scale"]
    for first, second in zip(original["views"][6:], altered["views"][6:], strict=True):
        np.testing.assert_array_equal(first["rvec_target_to_optical"], second["rvec_target_to_optical"])
        np.testing.assert_array_equal(first["tvec_target_to_optical_m"], second["tvec_target_to_optical_m"])


def test_missing_measurements_never_fall_back_to_design_pitch(manual_input, tmp_path):
    path, document, _ = manual_input
    document["status"] = "planned_not_measured"
    document["target"]["design_pitch_m"] = .04
    for point in document["target"]["points"]:
        point["point_target_m"] = [None, None, None]
    save(path, document)
    ready, _ = inspect_inputs(path)
    assert not ready["ready_for_optics_fit"]
    assert ready["measured_points"] == 0
    assert ready["design_coordinates_used_as_measurements"] is False
    with pytest.raises(ValueError, match="incomplete"):
        fit_capture(path, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize("copied_pixels", [False, True])
def test_source_identity_and_decoded_content_duplicates_rejected(manual_input, tmp_path, copied_pixels):
    path, document, _ = manual_input
    first, last = document["captures"][0], document["captures"][-1]
    if copied_pixels:
        # Distinct file bytes can still encode an identical frame.
        image = cv2.imread(str(tmp_path / first["source_image_file"]))
        new_path = tmp_path / "reencoded.png"
        cv2.imwrite(str(new_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        last["source_image_file"], last["source_image_sha256"] = new_path.name, sha256_file(new_path)
    else:
        last["source_image_file"], last["source_image_sha256"] = first["source_image_file"], first["source_image_sha256"]
    save(path, document)
    if copied_pixels:
        assert inspect_inputs(path)[0]["ready_for_optics_fit"]
        with pytest.raises(ValueError, match="duplicates"):
            fit_capture(path, tmp_path / "duplicate")
    else:
        assert not inspect_inputs(path)[0]["ready_for_optics_fit"]


def test_no_new_intrinsic_information_from_repeating_one_plane_pose(manual_input, tmp_path):
    path, document, _ = manual_input
    for capture in document["captures"]:
        capture["points_source_px"] = copy.deepcopy(document["captures"][0]["points_source_px"])
    save(path, document)
    with pytest.raises(ValueError, match="rank deficient|normal variation"):
        fit_capture(path, tmp_path / "no_tilt")


def test_invalid_pixel_group_and_degenerate_target_rejected(manual_input):
    path, document, _ = manual_input
    document["captures"][0]["points_source_px"][0]["pixel_xy"] = [-1., 10.]
    document["split"]["verification_groups"].append("fit_0")
    for i, point in enumerate(document["target"]["points"]):
        point["point_target_m"] = [i * .01, 0., 0.]
    save(path, document)
    ready, _ = inspect_inputs(path)
    assert not ready["ready_for_optics_fit"]
    assert any("disjoint" in message for message in ready["invalid_inputs"])
    assert any("collinear" in message for message in ready["invalid_inputs"])
    assert any("outside" in message for message in ready["invalid_inputs"])
