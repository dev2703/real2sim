"""Inspect measured capture readiness, then fit diagnostic checkerboard optics.

`inspect` also accepts the existing session plan and reports concrete missing
inputs. `fit` requires a completed checkerboard_capture_input_v1 document.
No design pitch is promoted to a measurement and no robot/base transform is fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np

from real2sim.data.raw import sha256_file
from real2sim.geometry.checkerboard import detect_checkerboard, fit_checkerboard_optics


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _positive_pair(value, integers=False):
    if not isinstance(value, list) or len(value) != 2:
        return False
    if integers and any(type(x) is not int for x in value):
        return False
    try:
        a = np.asarray(value, float)
        return bool(np.isfinite(a).all() and np.all(a > 0))
    except (TypeError, ValueError):
        return False


def _path(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _write(path, document):
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


def inspect_inputs(input_path):
    """Do not decode images or optimize; report absent evidence explicitly."""
    input_path = input_path.resolve()
    document = json.loads(input_path.read_text())
    missing, invalid, sources = [], [], {str(input_path): sha256_file(input_path)}
    if document.get("schema") == "v2_calibration_capture_plan_v1":
        session, session_path = document, input_path
        captures = []
        camera_key = "observation.images.iphone"
        missing.append("optics input: copy checkerboard_capture_input.template.json, link this measured session and declare actual source captures")
    elif document.get("schema") == "checkerboard_capture_input_v1":
        session_path = None
        session = {}
        camera_key, captures = document.get("camera_key"), document.get("captures", [])
        if not _text(document.get("capture_session_file")):
            missing.append("capture_session_file: completed physical session log")
        else:
            session_path = _path(input_path.parent, document["capture_session_file"])
            if not session_path.is_file():
                missing.append(f"capture_session_file: file does not exist: {session_path}")
            else:
                session = json.loads(session_path.read_text())
                sources[str(session_path)] = sha256_file(session_path)
        if not _text(camera_key):
            missing.append("camera_key: choose one camera from the measured session")
    else:
        raise ValueError("requires checkerboard_capture_input_v1 or v2_calibration_capture_plan_v1")
    target = session.get("target", {})
    if session.get("status") != "captured_measured":
        missing.append("session.status: set captured_measured only after recording and measuring the physical target")
    if target.get("kind") != "checkerboard":
        missing.append("target.kind: actual checkerboard target required")
    counts = target.get("checkerboard_inner_corner_count_xy")
    if not _positive_pair(counts, integers=True) or any(x < 4 or x > 30 for x in (counts or [])):
        missing.append("target.checkerboard_inner_corner_count_xy: actual integer counts, e.g. [8,6]")
    pitch = target.get("square_pitch_measured_xy_m")
    if not _positive_pair(pitch):
        missing.append("target.square_pitch_measured_xy_m: physically measured horizontal and vertical pitch in metres; design pitch is insufficient")
    if not _text(target.get("dimensional_measurement_method")):
        missing.append("target.dimensional_measurement_method: method and measurement evidence")
    uncertainty = target.get("dimensional_uncertainty_m")
    if not isinstance(uncertainty, (int, float)) or isinstance(uncertainty, bool) or not np.isfinite(uncertainty) or uncertainty <= 0:
        missing.append("target.dimensional_uncertainty_m: positive measured dimensional uncertainty")
    elif _positive_pair(pitch) and uncertainty >= min(pitch):
        invalid.append("target dimensional uncertainty is at least one printed pitch")
    if not _text(target.get("rigid_material_or_mount")):
        missing.append("target.rigid_material_or_mount: actual flat rigid target/backing")
    camera_matches = [c for c in session.get("camera_records", []) if c.get("camera_key") == camera_key]
    camera = camera_matches[0] if len(camera_matches) == 1 else {}
    if len(camera_matches) != 1:
        missing.append("camera_records: one record matching camera_key")
    size = camera.get("source_image_size_wh")
    if not _positive_pair(size, integers=True) or any(x < 100 for x in (size or [])):
        missing.append("camera.source_image_size_wh: actual unresized source resolution")
    if not _text(camera.get("capture_settings_provenance")):
        missing.append("camera.capture_settings_provenance: actual camera/lens/crop/zoom/focus/stabilization settings or unavailable settings explicitly recorded")
    fit_groups = camera.get("fit_capture_groups") or []
    verification_groups = camera.get("verification_capture_groups") or []
    if (not isinstance(fit_groups, list) or not isinstance(verification_groups, list)
            or any(not _text(g) for g in [*fit_groups, *verification_groups])):
        invalid.append("capture split must contain nonempty string group IDs")
        fit_groups, verification_groups = [], []
    if (len(set(fit_groups)) != len(fit_groups) or len(set(verification_groups)) != len(verification_groups)
            or set(fit_groups) & set(verification_groups)):
        invalid.append("fit/verification groups must be unique and disjoint")
    if len(fit_groups) < 3:
        missing.append("camera.fit_capture_groups: at least three whole capture groups")
    if len(verification_groups) < 2:
        missing.append("camera.verification_capture_groups: at least two whole capture groups")
    if not isinstance(captures, list):
        invalid.append("captures must be a list")
        captures = []
    normalized, identities, references, fit_count, verification_count = [], set(), set(), 0, 0
    for index, row in enumerate(captures):
        label = f"captures[{index}]"
        if not isinstance(row, dict):
            invalid.append(f"{label}: must be an object")
            continue
        group, role = row.get("capture_group"), row.get("role")
        if not _text(group) or role not in {"fit", "verification"}:
            invalid.append(f"{label}: named capture_group and fit/verification role required")
        expected = fit_groups if role == "fit" else verification_groups
        if group not in expected:
            invalid.append(f"{label}: group/role not in the predeclared camera split")
        reference = row.get("source_frame_reference")
        if not _text(reference):
            missing.append(f"{label}.source_frame_reference: original capture/frame identity")
        elif reference in references:
            invalid.append(f"{label}: source frame reference duplicated across captures")
        else:
            references.add(reference)
        if row.get("source_pixels_unresized") is not True:
            missing.append(f"{label}.source_pixels_unresized: original source pixels must be explicitly confirmed")
        image_file, video_file = row.get("source_image_file"), row.get("source_video_file")
        if bool(image_file) == bool(video_file):
            missing.append(f"{label}: supply exactly one native source image or source video/frame index")
            continue
        kind, value = ("image", image_file) if image_file else ("video", video_file)
        if not _text(value):
            invalid.append(f"{label}: source path must be a nonempty string")
            continue
        path = _path(input_path.parent, value)
        digest = row.get(f"source_{kind}_sha256")
        if not _text(digest):
            missing.append(f"{label}.source_{kind}_sha256: hash the retained original source")
        if not path.is_file():
            missing.append(f"{label}: source file does not exist: {path}")
            continue
        actual = sha256_file(path)
        sources[str(path)] = actual
        if digest is not None and digest != actual:
            invalid.append(f"{label}: source checksum mismatch")
        frame = row.get("source_frame_index")
        if kind == "video" and (type(frame) is not int or frame < 0):
            missing.append(f"{label}.source_frame_index: nonnegative presentation-order frame index")
            continue
        # Byte-identical image copies are not independent views; same video frame
        # cannot cross roles by using another path to the identical source video.
        identity = (kind, actual, frame if kind == "video" else None)
        if identity in identities:
            invalid.append(f"{label}: duplicated source frame/content is not an independent view")
        identities.add(identity)
        normalized.append({**row, "source_kind": kind, "source_path": str(path), "actual_source_sha256": actual})
        fit_count += role == "fit"
        verification_count += role == "verification"
    if fit_count < 6:
        missing.append(f"captures: at least six fitting views required; {fit_count} source-backed views declared")
    if verification_count < 2:
        missing.append(f"captures: at least two verification views required; {verification_count} source-backed views declared")
    if captures:
        observed_groups = {r.get("capture_group") for r in captures if isinstance(r, dict)}
        unused = (set(fit_groups) | set(verification_groups)) - observed_groups
        if unused:
            missing.append("captures missing for predeclared groups: " + ", ".join(sorted(unused)))
    readiness = {"schema": "checkerboard_capture_readiness_v1", "ready_for_optics_fit": not missing and not invalid,
                 "physical_calibration_accepted": False, "base_registration_verified": False,
                 "joint_zeros_verified": False, "gripper_registration_verified": False,
                 "missing_inputs": missing, "invalid_inputs": invalid,
                 "camera_key": camera_key, "fit_source_views": fit_count,
                 "verification_source_views": verification_count, "source_sha256": sources,
                 "optimization_performed": False,
                 "design_dimensions_used_as_measurements": False,
                 "next_stage": "Populate measured target/session fields and predeclared native source captures, then run fit. Independent camera/base, joint and jaw/contact checks follow optical fitting."}
    request = {"input": document, "session": session, "session_path": str(session_path) if session_path else None,
               "camera": camera, "target": target, "captures": normalized}
    return readiness, request


def _decode_capture(row):
    path = Path(row["source_path"])
    if row["source_kind"] == "image":
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode source image: {path}")
        return image, None
    selected = row["source_frame_index"]
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index == selected:
                timestamp = float(frame.pts * stream.time_base) if frame.pts is not None else None
                return frame.to_ndarray(format="bgr24"), timestamp
    raise ValueError(f"source video has no frame {selected}: {path}")


def fit_capture(input_path, out):
    if out.exists():
        raise FileExistsError(out)
    readiness, request = inspect_inputs(input_path)
    if not readiness["ready_for_optics_fit"]:
        raise ValueError("calibration inputs incomplete: " + "; ".join(readiness["missing_inputs"] + readiness["invalid_inputs"]))
    out.mkdir(parents=True)
    (out / "source_frames").mkdir()
    dependencies = dict(readiness["source_sha256"])
    dependencies[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    module = Path(__file__).resolve().parents[1] / "src/real2sim/geometry/checkerboard.py"
    dependencies[str(module)] = sha256_file(module)
    working_scale = request["input"].get("verification_working_scale_px", 3.)
    if not isinstance(working_scale, (int, float)) or isinstance(working_scale, bool) or not np.isfinite(working_scale) or working_scale <= 0:
        raise ValueError("verification_working_scale_px must be positive finite pixels")
    plan = {"schema": "checkerboard_optics_run_plan_v1", "input": request["input"],
            "session": request["session"], "camera": request["camera"],
            "captures": request["captures"], "source_sha256": dependencies,
            "verification_working_scale_px": working_scale,
            "detection": "OpenCV findChessboardCornersSB EXHAUSTIVE|ACCURACY|NORMALIZE_IMAGE on native uint8 grayscale",
            "opencv_version": cv2.__version__, "physical_calibration_accepted": False,
            "failure_policy": "Retain every declared view and every failure; no source/group/threshold replacement or silent frame exclusion."}
    _write(out / "plan.json", plan)
    (out / "executed_cli.py").write_bytes(Path(__file__).read_bytes())
    (out / "executed_checkerboard.py").write_bytes(module.read_bytes())
    detections, pixel_hashes, failures = [], set(), []
    size = request["camera"]["source_image_size_wh"]
    counts = request["target"]["checkerboard_inner_corner_count_xy"]
    for index, row in enumerate(request["captures"]):
        entry = {**row, "view_index": index}
        try:
            image, timestamp = _decode_capture(row)
            if list(image.shape[1::-1]) != size:
                raise ValueError(f"decoded native size {list(image.shape[1::-1])} differs from declared {size}")
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            if pixel_hash in pixel_hashes:
                raise ValueError("decoded image pixels duplicate an earlier capture; not an independent view")
            pixel_hashes.add(pixel_hash)
            frame_path = out / "source_frames" / f"view_{index:03d}.png"
            cv2.imwrite(str(frame_path), image)
            corners = detect_checkerboard(image, counts)
            entry.update(source_pts_s=timestamp, native_pixel_sha256=pixel_hash,
                         retained_source_frame=str(frame_path), retained_frame_sha256=sha256_file(frame_path))
            if corners is None:
                raise ValueError("complete checkerboard detection failed")
            entry.update(status="detected", corners_source_px=corners.tolist())
            preview = image.copy()
            cv2.drawChessboardCorners(preview, tuple(counts), corners.astype(np.float32).reshape(-1, 1, 2), True)
            cv2.imwrite(str(out / f"view_{index:03d}_corners.png"), preview)
        except (ValueError, OSError, av.error.FFmpegError) as exc:
            entry.update(status="failed", reason=str(exc))
            failures.append({"view_index": index, "reason": str(exc)})
        detections.append(entry)
        _write(out / "detections.json", {"plan_sha256": sha256_file(out / "plan.json"), "views": detections})
    if failures:
        report = {"schema": "checkerboard_optics_failed_v1", "physical_calibration_accepted": False,
                  "plan_sha256": sha256_file(out / "plan.json"), "failed_views": failures,
                  "optimization_performed": False, "failure_policy": plan["failure_policy"]}
    else:
        try:
            report = fit_checkerboard_optics(
                [r["corners_source_px"] for r in detections], [r["capture_group"] for r in detections],
                [r["role"] for r in detections], source_image_size_wh=size, inner_corner_count_xy=counts,
                measured_pitch_xy_m=request["target"]["square_pitch_measured_xy_m"],
                verification_working_scale_px=working_scale,
            )
            report["optimization_performed"] = True
        except (ValueError, cv2.error) as exc:
            report = {"schema": "checkerboard_optics_failed_v1", "physical_calibration_accepted": False,
                      "reason": str(exc), "optimization_performed": "may_have_started_before_quality_rejection"}
        report["plan_sha256"] = sha256_file(out / "plan.json")
    changed = [path for path, digest in dependencies.items() if sha256_file(Path(path)) != digest]
    if changed:
        raise ValueError("frozen calibration inputs changed during processing: " + ", ".join(changed))
    report["frozen_inputs_unchanged"] = True
    report["detections_sha256"] = sha256_file(out / "detections.json")
    _write(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["inspect", "fit"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "inspect":
        if args.out.exists():
            raise FileExistsError(args.out)
        report, _ = inspect_inputs(args.input)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _write(args.out, report)
        print(json.dumps(report, indent=2))
    else:
        report = fit_capture(args.input, args.out)
        print(json.dumps({key: report.get(key) for key in ("schema", "fit_rms_px", "verification_within_declared_working_scale", "physical_calibration_accepted", "failed_views", "reason")}, indent=2))


if __name__ == "__main__":
    main()
