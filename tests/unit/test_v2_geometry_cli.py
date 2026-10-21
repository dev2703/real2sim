"""Synthetic regression checks; none of these fixtures are physical calibration evidence."""

import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from real2sim.config import REPO_ROOT
from scripts.fit_v2_geometry import fit_document, main, native_landmarks


def synthetic_document():
    K = np.array([[900., 0, 640.], [0, 900., 360.], [0, 0, 1.]])
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array([.15, -.1, .2]))[0]
    T[:3, 3] = [.04, -.02, 1.3]
    points = np.random.default_rng(4).uniform([-.2, -.2, 0], [.2, .2, .3], (24, 3))
    pixels = cv2.projectPoints(points, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, None)[0].reshape(-1, 2)
    groups = ["fit-a"] * 8 + ["fit-b"] * 8 + ["verify-c"] * 8
    rows = [{"capture_group": group, "source_frame_reference": f"synthetic:{i}",
             "landmark_id": f"synthetic-marker-{i}", "point_task_m": point.tolist(),
             "source_pixel_xy": pixel.tolist(), "sigma_px": .5}
            for i, (group, point, pixel) in enumerate(zip(groups, points, pixels, strict=True))]
    document = {
        "schema": "v2_independent_geometry_v1", "evidence_kind": "synthetic_test",
        "capture_setup_provenance": "Synthetic test; no physical measurements",
        "camera": {"role": "iphone_fixed", "K_source_pixels": K.tolist(),
                   "source_image_size_wh": [1280, 720], "distortion_coefficients": [0.] * 5,
                   "intrinsics_provenance": "Synthetic known intrinsics"},
        "camera_registration": {
            "independent_geometry": True, "geometry_provenance": "Synthetic independently known XYZ",
            "split": {"fit_groups": ["fit-a", "fit-b"], "verification_groups": ["verify-c"],
                      "reserved_groups": ["reserved-d"]}, "observations": rows,
        },
    }
    return document, T


def test_camera_cli_writes_hashed_explicitly_synthetic_candidate_and_refuses_overwrite(tmp_path):
    document, T = synthetic_document()
    source, output = tmp_path / "measurements.json", tmp_path / "candidate.json"
    source.write_text(json.dumps(document))
    original = source.read_bytes()
    assert main(["camera", "--input", str(source), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    np.testing.assert_allclose(report["T_optical_task"], T, atol=1e-8)
    assert report["verification"]["p95_px"] < 1e-6
    assert report["evidence_kind"] == "synthetic_test"
    assert report["provenance"]["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert "src/real2sim/geometry/registration.py" in report["provenance"]["dependencies_sha256"]
    assert not report["physical_acceptance_established"]
    assert not report["automatically_adopted"]
    assert source.read_bytes() == original
    with pytest.raises(SystemExit, match="2"):
        main(["camera", "--input", str(source), "--output", str(output)])


def test_verification_pixels_do_not_refit_camera():
    document, _ = synthetic_document()
    original = fit_document(document, "camera")
    for row in document["camera_registration"]["observations"][-8:]:
        row["source_pixel_xy"][0] += 50
    changed = fit_document(document, "camera")
    np.testing.assert_array_equal(changed["T_optical_task"], original["T_optical_task"])
    assert changed["verification"]["p95_px"] > 49.9


@pytest.mark.parametrize("mode,factory,section", [
    ("camera", lambda: synthetic_document(), "camera_registration"),
    ("joint-zero", lambda: joint_document(), "joint_zero_registration"),
    ("wrist", lambda: wrist_document(), "wrist_registration"),
])
def test_frame_cannot_be_relabeled_into_verification(mode, factory, section):
    document, _ = factory()
    rows = document[section]["observations"]
    rows[-1]["source_frame_reference"] = " " + rows[0]["source_frame_reference"] + " "
    with pytest.raises(ValueError, match="different capture_group"):
        fit_document(document, mode)


def test_duplicate_landmark_observations_cannot_inflate_camera_evidence():
    document, _ = synthetic_document()
    rows = document["camera_registration"]["observations"]
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate landmark_id"):
        fit_document(document, "camera")


def test_distinct_landmarks_in_shared_source_frames_are_allowed():
    document, T = synthetic_document()
    for row in document["camera_registration"]["observations"]:
        row["source_frame_reference"] = f"synthetic-frame:{row['capture_group']}"
    report = fit_document(document, "camera")
    np.testing.assert_allclose(report["T_optical_task"], T, atol=1e-8)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["camera"].update(K_source_pixels=None), "missing measured input: K_source_pixels"),
    (lambda d: d["camera"].update(distortion_coefficients=None), "distortion_coefficients"),
    (lambda d: d["camera"].update(role="wrist"), "fixed-camera"),
    (lambda d: d["camera_registration"].update(independent_geometry=False), "independent"),
    (lambda d: d["camera_registration"]["split"].update(reserved_groups=["verify-c"]), "reserved"),
    (lambda d: d["camera_registration"]["observations"][0].update(source_pixel_xy=[1280, 10]), "outside"),
    (lambda d: d["camera_registration"]["observations"][0].update(source_frame_reference=None), "source_frame_reference"),
])
def test_unmeasured_circular_reserved_or_wrong_image_inputs_rejected(mutate, match):
    document, _ = synthetic_document()
    mutate(document)
    with pytest.raises(ValueError, match=match):
        fit_document(document, "camera")


def test_empty_template_cannot_be_mistaken_for_measurement():
    template = json.loads((REPO_ROOT / "configs/v2/geometry_landmarks.template.json").read_text())
    with pytest.raises(ValueError, match="missing measured input"):
        fit_document(template, "camera")


def joint_document():
    document, _ = synthetic_document()
    rows = copy.deepcopy(document["camera_registration"]["observations"])
    q = np.random.default_rng(8).uniform(-.6, .6, (len(rows), 5))
    for row, joints in zip(rows, q, strict=True):
        row.update(q_nominal_rad=joints.tolist(), gripper_joint_rad=.2,
                   body_name="gripper", point_body_m=[.02, .01, -.09])
    T = np.eye(4)
    T[2, 3] = 1.
    section = {
        "camera_independent_of_robot_fk": True,
        "camera_registration_provenance": "Synthetic independent camera",
        "T_optical_task": T.tolist(), "T_task_mjcf": np.eye(4).tolist(),
        "base_registration_provenance": "Synthetic fixed base",
        "landmark_geometry_provenance": "Synthetic known marker",
        "joint_units_and_signs_provenance": "Synthetic nominal radians, known positive signs",
        "offset_bounds_rad": [[-.2, .2]] * 5,
        "split": document["camera_registration"]["split"], "observations": rows,
    }
    points_for_q, _ = native_landmarks(section, rows)
    offsets = np.deg2rad([2., -3., 4., -5., 3.])
    points = points_for_q(q + offsets, np.arange(len(rows)))
    pixels = cv2.projectPoints(points, np.zeros(3), T[:3, 3],
                              np.array(document["camera"]["K_source_pixels"]), None)[0].reshape(-1, 2)
    for row, pixel in zip(rows, pixels, strict=True):
        row["source_pixel_xy"] = pixel.tolist()
    document["joint_zero_registration"] = section
    return document, offsets


def test_native_joint_fit_recovers_offsets_and_keeps_verification_outside_fitting():
    document, offsets = joint_document()
    report = fit_document(document, "joint-zero")
    np.testing.assert_allclose(report["joint_zero_offsets_rad"], offsets, atol=1e-6)
    assert report["identifiability"]["full_rank"]
    assert report["verification"]["p95_px"] < 1e-5
    assert not report["reference_clipping_applied"]
    assert not report["camera_fitted"]
    for row in document["joint_zero_registration"]["observations"][-8:]:
        row["q_nominal_rad"][0] += 4
    changed = fit_document(document, "joint-zero")
    np.testing.assert_array_equal(changed["joint_zero_offsets_rad"], report["joint_zero_offsets_rad"])
    assert changed["verification_joint_limit_violation_rows"] == 8


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["joint_zero_registration"].update(camera_independent_of_robot_fk=False), "independently"),
    (lambda d: d["joint_zero_registration"].update(T_task_mjcf=None), "T_task_mjcf"),
    (lambda d: d["joint_zero_registration"]["observations"][0].update(body_name="imaginary_body"), "unknown"),
])
def test_joint_fit_requires_fixed_independent_geometry(mutate, match):
    document, _ = joint_document()
    mutate(document)
    with pytest.raises(ValueError, match=match):
        fit_document(document, "joint-zero")


@pytest.mark.parametrize("field", ["q_nominal_rad", "gripper_joint_rad"])
def test_one_joint_source_frame_cannot_claim_different_recorded_states(field):
    document, _ = joint_document()
    rows = document["joint_zero_registration"]["observations"]
    extra = copy.deepcopy(rows[0])
    extra["landmark_id"] = "another-landmark-in-the-same-frame"
    if field == "q_nominal_rad":
        extra[field][0] += .01
    else:
        extra[field] += .01
    rows.append(extra)
    with pytest.raises(ValueError, match="readings must stay identical"):
        fit_document(document, "joint-zero")


@pytest.mark.parametrize("field,value", [
    ("body_name", "wrist"), ("point_body_m", [.03, .01, -.09]),
])
def test_joint_landmark_geometry_cannot_change_between_source_frames(field, value):
    document, _ = joint_document()
    rows = document["joint_zero_registration"]["observations"]
    rows[1]["landmark_id"] = rows[0]["landmark_id"]
    rows[1][field] = value
    with pytest.raises(ValueError, match="must retain its body_name and point_body_m"):
        fit_document(document, "joint-zero")


def test_joint_landmark_can_repeat_across_varied_poses_in_one_capture_group():
    document, offsets = joint_document()
    for row in document["joint_zero_registration"]["observations"]:
        row["landmark_id"] = "one-physical-marker-on-gripper"
    report = fit_document(document, "joint-zero")
    np.testing.assert_allclose(report["joint_zero_offsets_rad"], offsets, atol=1e-6)


def wrist_document():
    document, _ = synthetic_document()
    document["wrist_camera"] = copy.deepcopy(document["camera"])
    document["wrist_camera"]["role"] = "wrist_link"
    K = np.asarray(document["wrist_camera"]["K_source_pixels"])
    T_optical_link = np.eye(4)
    T_optical_link[:3, :3] = cv2.Rodrigues(np.array([.15, -.05, .2]))[0]
    T_optical_link[:3, 3] = [.01, -.02, .7]
    rows = copy.deepcopy(document["camera_registration"]["observations"])
    for index, row in enumerate(rows):
        capture = index // 8
        T_task_link = np.eye(4)
        T_task_link[:3, :3] = cv2.Rodrigues(np.array([.12, -.06, .08]) * capture)[0]
        T_task_link[:3, 3] = [.02 * capture, -.02 * capture, .04 * capture]
        row["T_task_link"] = T_task_link.tolist()
        point = T_task_link[:3, :3].T @ (np.asarray(row["point_task_m"]) - T_task_link[:3, 3])
        row["source_pixel_xy"] = cv2.projectPoints(
            point.reshape(1, 3), cv2.Rodrigues(T_optical_link[:3, :3])[0],
            T_optical_link[:3, 3], K, None)[0].reshape(2).tolist()
    document["wrist_registration"] = {
        "independent_geometry": True,
        "geometry_provenance": "Synthetic fixed independent world landmarks",
        "link_name": "gripper", "link_pose_source": "independent_measurement",
        "link_pose_provenance": "Synthetic known varied link poses",
        "link_poses_independently_verified": True,
        "split": document["camera_registration"]["split"], "observations": rows,
    }
    return document, T_optical_link


def test_wrist_mount_recovers_varied_verified_poses_and_keeps_verification_outside_fit(tmp_path):
    document, T = wrist_document()
    source, output = tmp_path / "wrist_measurements.json", tmp_path / "wrist_candidate.json"
    source.write_text(json.dumps(document))
    assert main(["wrist", "--input", str(source), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    np.testing.assert_allclose(report["T_optical_link"], T, atol=1e-8)
    np.testing.assert_allclose(report["T_link_optical"], np.linalg.inv(T), atol=1e-8)
    assert "T_optical_task" not in report
    assert report["verification"]["p95_px"] < 1e-6
    assert report["identifiability"]["full_rank"]
    assert report["transformed_anchor_frame"] == "link"
    assert not report["link_poses_fitted"]
    assert not report["arm_mapping_fitted"]
    assert not report["automatically_adopted"]
    for row in document["wrist_registration"]["observations"][-8:]:
        row["source_pixel_xy"][0] += 50
        row["T_task_link"][0][3] += .01
    changed = fit_document(document, "wrist")
    np.testing.assert_array_equal(changed["T_optical_link"], report["T_optical_link"])
    assert changed["verification"]["p95_px"] > 50


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["wrist_registration"].update(link_poses_independently_verified=False), "independently verified"),
    (lambda d: d["wrist_registration"].update(link_pose_provenance=None), "link_pose_provenance"),
    (lambda d: d["wrist_registration"].update(link_pose_source="unverified_fk"), "link_pose_source"),
    (lambda d: d["wrist_registration"].update(link_pose_source="verified_fk", arm_mapping_verified=False), "arm_mapping_verified"),
    (lambda d: d["wrist_registration"].update(link_pose_source="verified_fk", arm_mapping_verified=True,
                                           base_registration_verified=False), "base_registration_verified"),
    (lambda d: d["wrist_registration"]["observations"][0]["T_task_link"][0].__setitem__(3, .2), "identical within"),
    (lambda d: d["wrist_registration"]["observations"][0].update(T_task_link=None), "T_task_link"),
    (lambda d: d["wrist_registration"]["split"].update(reserved_groups=["verify-c"]), "reserved"),
    (lambda d: d["wrist_camera"].update(role="iphone_fixed"), "wrist_link"),
])
def test_wrist_rejects_unverified_inconsistent_or_reserved_poses(mutate, match):
    document, _ = wrist_document()
    mutate(document)
    with pytest.raises(ValueError, match=match):
        fit_document(document, "wrist")


def test_wrist_allows_fk_only_after_arm_and_base_verification():
    document, T = wrist_document()
    document["wrist_registration"].update(
        link_pose_source="verified_fk", arm_mapping_verified=True,
        base_registration_verified=True, arm_and_base_verification_provenance="Synthetic independent checks")
    report = fit_document(document, "wrist")
    np.testing.assert_allclose(report["T_optical_link"], T, atol=1e-8)
    assert report["link_pose_source"] == "verified_fk"
