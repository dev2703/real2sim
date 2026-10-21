"""Fit diagnostic geometry candidates from independently measured, held-out landmarks.

Usage: python scripts/fit_v2_geometry.py camera|joint-zero|wrist --input measurements.json
       --output outputs/v2/registration_candidate.json

Copy configs/v2/geometry_landmarks.template.json and fill actual measurements first.
Transforms use column vectors: T_optical_task maps task XYZ to optical XYZ;
T_task_mjcf maps the pinned native MJCF world to the measured task frame. Camera
intrinsics refer to unresized source images, with explicitly supplied distortion.
Wrist mode fixes independently verified T_task_link poses per whole capture and
fits only the rigid optical-from-link mount; it never estimates arm calibration.
No simulator config is modified, and a candidate is never automatically accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mujoco
import numpy as np

from real2sim.config import REPO_ROOT
from real2sim.geometry.registration import fit_camera_registration, fit_joint_zero_offsets
from real2sim.geometry.so101_fk import ARM_JOINTS, DEFAULT_MJCF


def required(data, key):
    value = data.get(key)
    if value is None:
        raise ValueError(f"missing measured input: {key}")
    return value


def provenance(data, key):
    value = required(data, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must describe the independent measurement and its source")
    return value


def finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def rigid(value, name):
    T = finite(value, (4, 4), name)
    R = T[:3, :3]
    if (not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-7)):
        raise ValueError(f"{name} must be a proper rigid transform")
    return T


def native_landmarks(section, rows):
    """Exact native FK; joint limits constrain fitting, never alter reference states."""
    provenance(section, "base_registration_provenance")
    provenance(section, "landmark_geometry_provenance")
    provenance(section, "joint_units_and_signs_provenance")
    T = rigid(required(section, "T_task_mjcf"), "T_task_mjcf")
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MJCF))
    data = mujoco.MjData(model)
    ids = [model.joint(name).id for name in ARM_JOINTS]
    addresses = model.jnt_qposadr[ids]
    grip_address = model.joint("gripper").qposadr[0]
    body_ids, offsets, grips = [], [], []
    for row in rows:
        name = required(row, "body_name")
        try:
            body = model.body(name).id
        except KeyError as exc:
            raise ValueError(f"unknown native MJCF body landmark: {name}") from exc
        if body == 0:
            raise ValueError("joint-zero landmarks must be on a named robot body")
        body_ids.append(body)
        offsets.append(finite(required(row, "point_body_m"), (3,), "point_body_m"))
        grips.append(float(required(row, "gripper_joint_rad")))
    if not np.isfinite(grips).all():
        raise ValueError("gripper_joint_rad must be finite; do not infer its calibration")

    def points_for_q(q, indices):
        points = []
        for joints, index in zip(q, indices, strict=True):
            mujoco.mj_resetData(model, data)
            data.qpos[addresses] = joints
            data.qpos[grip_address] = grips[index]
            mujoco.mj_kinematics(model, data)
            body = body_ids[index]
            point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ offsets[index]
            points.append(T[:3, :3] @ point + T[:3, 3])
        return np.asarray(points)

    return points_for_q, model.jnt_range[ids].copy()


def wrist_landmarks(section, rows):
    """Convert independent task landmarks with separately verified, fixed link poses."""
    provenance(section, "link_name")
    provenance(section, "link_pose_provenance")
    if required(section, "link_poses_independently_verified") is not True:
        raise ValueError("wrist fitting requires independently verified link poses")
    source = required(section, "link_pose_source")
    if source not in {"independent_measurement", "verified_fk"}:
        raise ValueError("link_pose_source must be independent_measurement or verified_fk")
    if source == "verified_fk":
        for key in ("arm_mapping_verified", "base_registration_verified"):
            if required(section, key) is not True:
                raise ValueError(f"verified_fk requires {key} before wrist fitting")
        provenance(section, "arm_and_base_verification_provenance")
    captures, points = {}, []
    for row in rows:
        T = rigid(required(row, "T_task_link"), "T_task_link")
        group = row["capture_group"]
        if group in captures and not np.allclose(T, captures[group], rtol=0, atol=1e-10):
            raise ValueError("T_task_link must stay identical within each capture_group")
        captures[group] = T
        point = finite(required(row, "point_task_m"), (3,), "point_task_m")
        points.append(T[:3, :3].T @ (point - T[:3, 3]))
    return np.asarray(points)


def fit_document(document, mode):
    sections = {"camera": "camera_registration", "joint-zero": "joint_zero_registration",
                "wrist": "wrist_registration"}
    if mode not in sections:
        raise ValueError(f"unknown mode: {mode}")
    if document.get("schema") != "v2_independent_geometry_v1":
        raise ValueError("schema must be v2_independent_geometry_v1")
    evidence = required(document, "evidence_kind")
    if evidence not in {"measured", "synthetic_test"}:
        raise ValueError("evidence_kind must be measured or synthetic_test")
    provenance(document, "capture_setup_provenance")
    camera = required(document, "wrist_camera" if mode == "wrist" else "camera")
    role = "wrist_link" if mode == "wrist" else "iphone_fixed"
    if camera.get("role") != role:
        raise ValueError(f"this {'wrist' if mode == 'wrist' else 'fixed-camera'} registration requires camera.role={role}")
    provenance(camera, "intrinsics_provenance")
    K = finite(required(camera, "K_source_pixels"), (3, 3), "K_source_pixels")
    size = finite(required(camera, "source_image_size_wh"), (2,), "source_image_size_wh")
    if np.any(size <= 0) or not np.array_equal(size, np.floor(size)):
        raise ValueError("source_image_size_wh must contain positive integer dimensions")
    distortion = np.asarray(required(camera, "distortion_coefficients"), dtype=float)
    if distortion.shape not in {(4,), (5,), (8,), (12,), (14,)} or not np.isfinite(distortion).all():
        raise ValueError("distortion_coefficients must explicitly contain 4, 5, 8, 12 or 14 finite coefficients")
    section = required(document, sections[mode])
    split = required(section, "split")
    rows = required(section, "observations")
    if not isinstance(rows, list) or not rows:
        raise ValueError("observations must contain measured fit and verification correspondences")
    for row in rows:
        provenance(row, "source_frame_reference")
        provenance(row, "landmark_id")
    pixels = finite([required(row, "source_pixel_xy") for row in rows], (len(rows), 2), "source_pixel_xy")
    if np.any(pixels < 0) or np.any(pixels >= size):
        raise ValueError("source pixels fall outside the declared unresized image dimensions")
    groups = [required(row, "capture_group") for row in rows]
    if not all(isinstance(group, str) and group.strip() for group in groups):
        raise ValueError("capture_group must be a nonempty string identifying a whole capture")
    common = dict(
        K=K, distortion=distortion, pixels=pixels, groups=groups,
        sigma_px=[required(row, "sigma_px") for row in rows],
        fit_groups=required(split, "fit_groups"),
        verification_groups=required(split, "verification_groups"),
        reserved_groups=required(split, "reserved_groups"),
    )
    if mode == "camera":
        report = fit_camera_registration(
            [required(row, "point_task_m") for row in rows], **common,
            geometry_provenance=provenance(section, "geometry_provenance"),
            independent_geometry=required(section, "independent_geometry") is True,
        )
    elif mode == "joint-zero":
        provenance(section, "camera_registration_provenance")
        if required(section, "camera_independent_of_robot_fk") is not True:
            raise ValueError("joint-zero fitting requires a camera independently registered without robot FK")
        points, limits = native_landmarks(section, rows)
        q = finite([required(row, "q_nominal_rad") for row in rows], (len(rows), 5), "q_nominal_rad")
        report = fit_joint_zero_offsets(
            q, points_for_q=points, **common,
            T_optical_task=required(section, "T_optical_task"),
            joint_limits_rad=limits,
            offset_bounds_rad=required(section, "offset_bounds_rad"),
            initial_offsets_rad=section.get("initial_offsets_rad"),
        )
        report["arm_joint_order"] = list(ARM_JOINTS)
        report["fixed_native_joint_limits_rad"] = limits.tolist()
    elif mode == "wrist":
        points = wrist_landmarks(section, rows)
        report = fit_camera_registration(
            points, **common,
            geometry_provenance=provenance(section, "geometry_provenance"),
            independent_geometry=required(section, "independent_geometry") is True,
        )
        T = np.asarray(report.pop("T_optical_task"))
        report.update({
            "T_optical_link": T.tolist(), "T_link_optical": np.linalg.inv(T).tolist(),
            "link_name": section["link_name"], "link_pose_source": section["link_pose_source"],
            "link_poses_fitted": False, "arm_mapping_fitted": False,
            "link_pose_provenance": section["link_pose_provenance"],
            "transformed_anchor_frame": "link",
        })
    report.update({
        "evidence_kind": evidence, "mode": mode,
        "physical_acceptance_established": False,
        "limitations": [
            "This fit does not establish metric tabletop object error, contact correctness or replay success.",
            "Independent measurement provenance is operator supplied; the fitter cannot verify its truth.",
            "Low pixel error alone does not prove geometry, especially with weak or planar anchors.",
            "Joint offsets are conditional on fixed camera, base registration, signs and landmark dimensions.",
            "Wrist registration is conditional on independently verified link poses; low residual cannot validate unverified arm FK.",
        ],
    })
    return report


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_hashes(mode):
    files = [Path(__file__).resolve(), REPO_ROOT / "src/real2sim/geometry/registration.py"]
    if mode == "joint-zero":
        files.append(REPO_ROOT / "src/real2sim/geometry/so101_fk.py")
        # Conservatively include the whole pinned native model asset directory.
        files.extend(path for path in DEFAULT_MJCF.parent.rglob("*") if path.is_file())
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in sorted(set(files))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["camera", "joint-zero", "wrist"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output already exists; choose a new versioned candidate filename")
        payload = args.input.read_bytes()
        report = fit_document(json.loads(payload), args.mode)
        report["provenance"] = {
            "input_path": str(args.input.resolve()),
            "input_sha256": hashlib.sha256(payload).hexdigest(),
            "dependencies_sha256": dependency_hashes(args.mode),
            "package_versions": {name: importlib.metadata.version(name)
                                 for name in ("numpy", "scipy", "mujoco", "opencv-python-headless")},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"candidate": str(args.output), "evidence_kind": report["evidence_kind"],
                      "verification_p95_px": report["verification"]["p95_px"],
                      "automatically_adopted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
