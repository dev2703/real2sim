"""Fit optical candidates from manually identified, measured planar target points.

Accepts numbered marks on flat rigid-backed paper/cardboard, not checkerboard
detections. Individual measured coordinates are required; design pitch is never
substituted. This fits camera optics only, not robot/base/joint registration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from real2sim.data.raw import sha256_file
from real2sim.geometry import checkerboard

if __package__:
    from scripts.calibrate_v2_capture import _decode_capture
else:
    from calibrate_v2_capture import _decode_capture


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def text(value):
    return isinstance(value, str) and bool(value.strip())


def positive(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value) and value > 0


def finite_coordinates(value, count):
    try:
        array = np.asarray(value, float)
        return array.shape == (count,) and np.isfinite(array).all()
    except (TypeError, ValueError):
        return False


def inspect_inputs(input_path):
    input_path = input_path.resolve()
    document = json.loads(input_path.read_text())
    if document.get("schema") != "manual_metric_target_capture_input_v1":
        raise ValueError("manual_metric_target_capture_input_v1 document required")
    missing, invalid = [], []
    dependencies = {str(input_path): sha256_file(input_path)}
    if document.get("status") != "captured_measured":
        missing.append("status: captured_measured requires actual source captures and individually measured target points")
    for key in ("camera_key", "source_image_size_provenance", "capture_settings_provenance"):
        if not text(document.get(key)):
            missing.append(key)
    size = document.get("source_image_size_wh")
    size_ok = isinstance(size, list) and len(size) == 2 and all(type(v) is int and v >= 100 for v in size)
    if not size_ok:
        missing.append("source_image_size_wh: measured native integer width/height, each >=100")
    for key in ("pixel_annotation_uncertainty_px", "verification_working_scale_px"):
        if not positive(document.get(key)):
            missing.append(f"{key}: positive finite declared value required")
    target = document.get("target") or {}
    if target.get("geometry_evidence") not in {"measured", "synthetic_test"}:
        missing.append("target.geometry_evidence: measured (or synthetic_test for explicit algorithm tests)")
    for key in ("coordinate_frame_definition", "measurement_method", "flat_rigid_backing", "flatness_evidence"):
        if not text(target.get(key)):
            missing.append(f"target.{key}")
    if not positive(target.get("dimensional_uncertainty_m")):
        missing.append("target.dimensional_uncertainty_m: positive actual measurement uncertainty required")
    points, point_ids = [], []
    for index, row in enumerate(target.get("points") or []):
        name, xyz = row.get("id"), row.get("point_target_m")
        if not text(name) or name in point_ids:
            invalid.append(f"target.points[{index}]: distinct nonempty point ID required")
        elif not finite_coordinates(xyz, 3):
            missing.append(f"target.points[{index}].point_target_m: individually measured finite XYZ required; design spacing is insufficient")
        else:
            point_ids.append(name)
            points.append(xyz)
    if len(points) < 6:
        missing.append("target.points: at least six complete measured points required")
    if len(points) >= 6:
        points = np.asarray(points, float)
        if np.max(abs(points[:, 2])) > 1e-9:
            invalid.append("target coordinates must express its own flat XY plane with z=0")
        if np.linalg.matrix_rank(points[:, :2] - points[:, :2].mean(0)) < 2:
            invalid.append("target points are collinear or coincident")
        if len(np.unique(points, axis=0)) != len(points):
            invalid.append("target point coordinates are duplicated")
    pose_ids = document.get("verification_pose_point_ids") or []
    if not isinstance(pose_ids, list) or len(pose_ids) < 4 or len(set(pose_ids)) != len(pose_ids):
        missing.append("verification_pose_point_ids: at least four distinct predeclared IDs required")
        pose_ids = []
    if pose_ids and all(name in point_ids for name in pose_ids):
        indices = [point_ids.index(name) for name in pose_ids]
        selected = np.asarray(points)[indices, :2]
        if np.linalg.matrix_rank(selected - selected.mean(0)) < 2:
            invalid.append("verification pose points must span the target plane")
        if len(points) - len(pose_ids) < 2:
            invalid.append("verification requires at least two scored points excluded from pose fitting")
    elif pose_ids:
        missing.append("verification_pose_point_ids: every named pose point needs measured target coordinates")
    split = document.get("split") or {}
    fit_groups, verify_groups = split.get("fit_groups") or [], split.get("verification_groups") or []
    if len(fit_groups) < 3 or not all(text(group) for group in fit_groups):
        missing.append("split.fit_groups: at least three predeclared whole groups required")
    if len(verify_groups) < 2 or not all(text(group) for group in verify_groups):
        missing.append("split.verification_groups: at least two independent whole groups required")
    if len(set(fit_groups)) != len(fit_groups) or len(set(verify_groups)) != len(verify_groups) or set(fit_groups) & set(verify_groups):
        invalid.append("whole fit/verification group assignments must be unique and disjoint")
    normalized, identities, references = [], set(), set()
    for index, row in enumerate(document.get("captures") or []):
        label = f"captures[{index}]"
        group, role = row.get("capture_group"), row.get("role")
        if role not in {"fit", "verification"} or group not in (fit_groups if role == "fit" else verify_groups):
            invalid.append(f"{label}: group/role differs from predeclared split")
        reference = row.get("source_frame_reference")
        if not text(reference):
            missing.append(f"{label}.source_frame_reference")
        elif reference in references:
            invalid.append(f"{label}: source frame reference duplicated")
        references.add(reference)
        if row.get("source_pixels_unresized") is not True:
            missing.append(f"{label}.source_pixels_unresized: explicitly confirm native pixel coordinates")
        image, video = row.get("source_image_file"), row.get("source_video_file")
        if bool(image) == bool(video):
            missing.append(f"{label}: supply exactly one original image or video/frame")
            continue
        kind, value = ("image", image) if image else ("video", video)
        if not text(value):
            invalid.append(f"{label}: source path must be a nonempty string")
            continue
        source = Path(value)
        source = source.resolve() if source.is_absolute() else (input_path.parent / source).resolve()
        expected_hash = row.get(f"source_{kind}_sha256")
        if not text(expected_hash):
            missing.append(f"{label}.source_{kind}_sha256")
        if not source.is_file():
            missing.append(f"{label}: source file missing: {source}")
            continue
        actual_hash = sha256_file(source)
        dependencies[str(source)] = actual_hash
        if expected_hash != actual_hash:
            invalid.append(f"{label}: source file checksum mismatch")
        frame = row.get("source_frame_index")
        if kind == "video" and (type(frame) is not int or frame < 0):
            missing.append(f"{label}.source_frame_index: nonnegative presentation-order index required")
            continue
        identity = kind, actual_hash, frame if kind == "video" else None
        if identity in identities:
            invalid.append(f"{label}: duplicate source file/frame content is not an independent view")
        identities.add(identity)
        if not point_ids:
            missing.append(f"{label}: target point measurements are needed before pixel correspondences can be checked")
            continue
        pixels = row.get("points_source_px") or []
        pixel_ids = [r.get("id") for r in pixels]
        if len(set(pixel_ids)) != len(pixel_ids) or set(pixel_ids) != set(point_ids):
            missing.append(f"{label}.points_source_px: annotate every measured ID exactly once")
            continue
        by_id = {r["id"]: r.get("pixel_xy") for r in pixels}
        if not all(finite_coordinates(by_id[name], 2) for name in point_ids):
            missing.append(f"{label}.points_source_px: finite manually observed native pixels required")
            continue
        ordered = np.asarray([by_id[name] for name in point_ids], float)
        if size_ok and (np.any(ordered < 0) or np.any(ordered >= size)):
            invalid.append(f"{label}: manually annotated pixel falls outside the source image")
        normalized.append({**row, "source_kind": kind, "source_path": str(source),
                           "actual_source_sha256": actual_hash, "ordered_pixels": ordered.tolist()})
    fit_count = sum(row["role"] == "fit" for row in normalized)
    verify_count = sum(row["role"] == "verification" for row in normalized)
    if fit_count < 6:
        missing.append(f"captures: six complete fitting views required; currently {fit_count}")
    if verify_count < 2:
        missing.append(f"captures: two complete independent verification views required; currently {verify_count}")
    used = {row.get("capture_group") for row in normalized}
    if set(fit_groups + verify_groups) - used:
        missing.append("every predeclared group needs at least one complete source view")
    readiness = {"schema": "manual_metric_target_readiness_v1", "ready_for_optics_fit": not missing and not invalid,
                 "missing_inputs": missing, "invalid_inputs": invalid,
                 "fit_views": fit_count, "verification_views": verify_count,
                 "measured_points": len(points), "source_sha256": dependencies,
                 "optimization_performed": False, "physical_calibration_accepted": False,
                 "design_coordinates_used_as_measurements": False}
    return readiness, {"document": document, "points": np.asarray(points).tolist(), "point_ids": point_ids,
                       "captures": normalized, "pose_ids": pose_ids}


def fit_optics(points, views, groups, roles, *, size, pose_indices, gate):
    points, views, size = np.asarray(points, float), np.asarray(views, float), np.asarray(size, int)
    fit = np.asarray(roles) == "fit"
    diversity = checkerboard._homography_diversity(points, views[fit], size)
    coverage = np.ptp(views[fit].reshape(-1, 2), axis=0) / size
    if np.any(coverage < .2):
        raise ValueError("fitting points cover less than 20% of an image axis")
    rms, K, distortion, rvecs, tvecs, intrinsic_std, _, _ = cv2.calibrateCameraExtended(
        [points.astype(np.float32)] * int(fit.sum()),
        [view.astype(np.float32) for view in views[fit]], tuple(map(int, size)), None, None,
        flags=0, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-10))
    if not all(np.isfinite(value).all() for value in (K, distortion, np.asarray(rms))) or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("nonfinite calibration or nonpositive focal length")
    heldout = np.setdiff1d(np.arange(len(points)), pose_indices)
    poses = iter(zip(rvecs, tvecs, strict=True))
    records, normals = [], []
    for index, (pixels, group, role) in enumerate(zip(views, groups, roles, strict=True)):
        if role == "fit":
            rvec, tvec = next(poses)
            rotation = cv2.Rodrigues(rvec)[0]
            normals.append(rotation[:, 2])
            depth = (points @ rotation.T + tvec.ravel())[:, 2]
            metrics = {"all_point_error": checkerboard._errors(pixels, checkerboard._project(points, rvec, tvec, K, distortion))}
        else:
            rvec, tvec, depth, candidates = checkerboard._verification_pose(points, pixels, pose_indices, K, distortion)
            prediction = checkerboard._project(points, rvec, tvec, K, distortion)
            metrics = {"pose_point_error": checkerboard._errors(pixels[pose_indices], prediction[pose_indices]),
                       "heldout_point_error": checkerboard._errors(pixels[heldout], prediction[heldout]),
                       "positive_depth_pose_candidates": candidates}
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError("fitted target pose has nonpositive point depth")
        records.append({"view_index": index, "capture_group": group, "role": role,
                        "rvec_target_to_optical": rvec.ravel().tolist(), "tvec_target_to_optical_m": tvec.ravel().tolist(),
                        "minimum_point_depth_m": float(depth.min()), **metrics})
    normals = np.asarray(normals)
    normal_span = float(np.rad2deg(np.arccos(np.clip(normals @ normals.T, -1, 1))).max())
    if normal_span < 5:
        raise ValueError("fitting views have less than five degrees of independent plane-normal variation")
    return {"schema": "manual_metric_target_optical_candidate_v1", "K_source_pixels": K.tolist(),
            "distortion_coefficients": distortion.ravel().tolist(), "fit_rms_px": float(rms),
            "conditional_intrinsic_std": intrinsic_std.ravel().tolist(),
            "source_image_size_wh": size.tolist(), "homography_diversity": diversity,
            "fit_image_span_fraction_xy": coverage.tolist(), "fit_plane_normal_max_separation_deg": normal_span,
            "verification_pose_point_indices": list(map(int, pose_indices)),
            "verification_heldout_point_indices": heldout.tolist(), "verification_working_scale_px": gate,
            "verification_within_declared_working_scale": all(row["heldout_point_error"]["p95_px"] <= gate for row in records if row["role"] == "verification"),
            "views": records, "physical_calibration_accepted": False,
            "base_registration_verified": False, "joint_zeros_verified": False, "gripper_registration_verified": False,
            "distortion_model": "OpenCV pinhole k1,k2,p1,p2,k3; zero skew",
            "uncertainty_note": "Target-coordinate and pixel annotation uncertainties are declared metadata, not propagated metrology confidence intervals. OpenCV intrinsic std is conditional on exact supplied target geometry and model.",
            "limitations": ["Verification fits each view's nuisance target pose using only predeclared pose point IDs, then scores the remaining points. Verification pixels never fit intrinsics or distortion.",
                "A single stationary mat supplies a homography, not independently measured full intrinsics/distortion. This fit requires several known-plane orientations and useful image coverage.",
                "An optics candidate, even passing the declared pixel gate, does not establish base registration, joint signs/zeros, jaw geometry, physical transfer or applicability to older footage."]}


def fit_capture(input_path, out):
    if out.exists():
        raise FileExistsError(out)
    readiness, request = inspect_inputs(input_path)
    if not readiness["ready_for_optics_fit"]:
        raise ValueError("manual calibration inputs incomplete: " + "; ".join(readiness["missing_inputs"] + readiness["invalid_inputs"]))
    out.mkdir(parents=True)
    document = request["document"]
    dependencies = {**readiness["source_sha256"], str(Path(__file__).resolve()): sha256_file(Path(__file__)),
                    str(Path(checkerboard.__file__).resolve()): sha256_file(Path(checkerboard.__file__)),
                    str(Path(_decode_capture.__code__.co_filename).resolve()): sha256_file(Path(_decode_capture.__code__.co_filename))}
    write(out / "frozen_input.json", document)
    write(out / "plan.json", {"source_sha256": dependencies, "no_intrinsic_fitting_on_verification": True,
                              "physical_calibration_accepted": False})
    (out / "executed_fitter.py").write_bytes(Path(__file__).read_bytes())
    identities, decoded = set(), []
    try:
        for row in request["captures"]:
            image, timestamp = _decode_capture(row)
            if list(image.shape[1::-1]) != document["source_image_size_wh"]:
                raise ValueError("decoded source dimensions differ from the declared original image size")
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            if pixel_hash in identities:
                raise ValueError("decoded source frame content duplicates an earlier view")
            if row.get("decoded_frame_sha256") is not None and row["decoded_frame_sha256"] != pixel_hash:
                raise ValueError("decoded frame pixel checksum mismatch")
            identities.add(pixel_hash)
            decoded.append({"source_frame_reference": row["source_frame_reference"], "source_path": row["source_path"],
                            "source_frame_index": row.get("source_frame_index"), "source_file_sha256": row["actual_source_sha256"],
                            "decoded_frame_sha256": pixel_hash, "container_pts_s": timestamp})
        captures = request["captures"]
        result = fit_optics(request["points"], [row["ordered_pixels"] for row in captures],
            [row["capture_group"] for row in captures], [row["role"] for row in captures],
            size=document["source_image_size_wh"],
            pose_indices=np.array([request["point_ids"].index(name) for name in request["pose_ids"]]),
            gate=float(document["verification_working_scale_px"]))
        changed = [path for path, digest in dependencies.items() if sha256_file(Path(path)) != digest]
        if changed:
            raise ValueError("frozen inputs changed during calibration: " + ", ".join(changed))
        result.update(status="complete_candidate_not_accepted", point_ids=request["point_ids"],
            target_geometry=document["target"], pixel_annotation_uncertainty_px=document["pixel_annotation_uncertainty_px"],
            camera_key=document["camera_key"], source_captures=decoded,
            source_sha256_before=dependencies, source_sha256_after=dependencies, changed_inputs=[])
    except Exception as exc:
        write(out / "report.json", {"schema": "manual_metric_target_failed_v1", "status": "failed",
            "failure": f"{type(exc).__name__}: {exc}", "physical_calibration_accepted": False, "decoded_views": decoded})
        raise
    write(out / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "fit"):
        child = sub.add_parser(command)
        child.add_argument("--input", type=Path, required=True)
        child.add_argument("--out", type=Path, required=command == "fit")
    args = parser.parse_args()
    if args.command == "inspect":
        report = inspect_inputs(args.input)[0]
        if args.out:
            if args.out.exists():
                raise FileExistsError(args.out)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write(args.out, report)
    else:
        report = fit_capture(args.input, args.out)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
