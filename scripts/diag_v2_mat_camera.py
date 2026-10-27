"""Initialize a diagnostic v2 camera from measured mat dimensions and image edges.

This uses no robot FK, never changes scene configuration, and does not establish
physical calibration. Four fitted virtual corners are not held-out verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from real2sim.artifacts import sha256_file as sha256
from real2sim.config import REPO_ROOT, load_yaml
from real2sim.geometry.planar_camera import fit_planar_camera


def fit_edges(samples):
    lines, diagnostics = [], {}
    for name in ("top", "right", "bottom", "left"):
        points = np.asarray(samples[name], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("each edge requires at least three finite source-pixel samples")
        centre = points.mean(axis=0)
        _, singular, vt = np.linalg.svd(points - centre, full_matrices=False)
        if singular[0] < 1:
            raise ValueError("edge samples must span a visible straight segment")
        normal = vt[-1]
        line = np.r_[normal, -normal @ centre]
        lines.append(line)
        distances = points @ normal + line[2]
        diagnostics[name] = {
            "line_ax_by_c_equals_zero": line.tolist(), "sample_count": len(points),
            "orthogonal_residual_rms_px": float(np.sqrt(np.mean(distances**2))),
            "per_sample_signed_distance_px": distances.tolist(),
        }
    corners = []
    for first, second in ((3, 0), (0, 1), (1, 2), (2, 3)):
        homogeneous = np.cross(lines[first], lines[second])
        if abs(homogeneous[2]) < 1e-8:
            raise ValueError("adjacent mat edges do not have a finite intersection")
        corners.append(homogeneous[:2] / homogeneous[2])
    return np.asarray(corners), diagnostics


def camera_summary(report):
    transform = np.asarray(report["T_optical_plane"])
    position = -transform[:3, :3].T @ transform[:3, 3]
    return {
        "focal_px": report["focal_px"],
        "camera_position_in_mat_frame_m": position.tolist(),
        "absolute_camera_to_mat_plane_distance_m": float(abs(position[2])),
        "fit_rms_px": report["fit_rms_px"], "focal_at_bound": report["focal_at_bound"],
        "optimizer_converged": report["optimizer_converged"],
    }


def run(annotations_path, output_path, overlay_path):
    if output_path.exists() or overlay_path.exists():
        raise FileExistsError("use fresh output and overlay paths; existing evidence is never overwritten")
    annotations = json.loads(annotations_path.read_text())
    if annotations.get("schema") != "v2_mat_edge_annotations_v1":
        raise ValueError("unsupported annotation schema")
    source = annotations["source"]
    paths = {key: REPO_ROOT / annotations[key]["path"]
             for key in ("task_config", "split", "source_manifest")}
    paths["source"] = REPO_ROOT / source["path"]
    for key, path in paths.items():
        if sha256(path) != annotations[key]["sha256"]:
            raise ValueError(f"{key} hash differs from the reviewed annotation source")
    split = json.loads(paths["split"].read_text())
    episode = source["episode"]
    if (episode not in split["train"]
            or any(episode in rows for name, rows in split.items()
                   if name in ("val", "test", "reserved", "evaluation"))):
        raise ValueError("the reviewed source must belong exclusively to the training split")
    manifest = json.loads(paths["source_manifest"].read_text())
    matches = [row for row in manifest["source_frames"]
               if row["episode"] == episode and row["position"] == source["frame"]]
    if (len(matches) != 1 or matches[0]["sha256"] != source["sha256"]
            or (REPO_ROOT / matches[0]["path"]).resolve() != paths["source"].resolve()
            or manifest["source_camera"] != source["camera"]):
        raise ValueError("source frame is not uniquely identified by the saved extraction manifest")
    image = cv2.imread(str(paths["source"]))
    if image is None:
        raise ValueError("source image could not be decoded")
    height, width = image.shape[:2]
    if [width, height] != source["size_wh"]:
        raise ValueError("source image dimensions differ from reviewed annotations")
    dimensions = load_yaml(paths["task_config"])["table"]["mat"]["size_xy"]
    samples = annotations["edge_samples_source_px"]
    for points in samples.values():
        points = np.asarray(points, dtype=float)
        if np.any(points < 0) or np.any(points >= [width, height]):
            raise ValueError("edge samples must be inside the original source image")
    corners, edge_report = fit_edges(samples)
    options = {"source_size_wh": (width, height), "side_lengths_m": dimensions,
               "sigma_px": annotations["assumed_corner_sigma_px"]}
    fit = fit_planar_camera(corners, **options)
    sensitivity = []
    for magnitude in (1., 3.):
        for corner in range(4):
            for axis in range(2):
                for sign in (-1., 1.):
                    changed = corners.copy()
                    changed[corner, axis] += sign * magnitude
                    perturbed = fit_planar_camera(changed, **options)
                    sensitivity.append({
                        "corner_index": corner, "axis": "xy"[axis],
                        "perturbation_px": sign * magnitude, **camera_summary(perturbed),
                    })
    sensitivity_summary = {}
    for magnitude in (1., 3.):
        selected = [row for row in sensitivity if abs(row["perturbation_px"]) == magnitude]
        sensitivity_summary[str(int(magnitude))] = {
            "cases": len(selected),
            "focal_range_px": [min(row["focal_px"] for row in selected),
                               max(row["focal_px"] for row in selected)],
            "camera_plane_distance_range_m": [
                min(row["absolute_camera_to_mat_plane_distance_m"] for row in selected),
                max(row["absolute_camera_to_mat_plane_distance_m"] for row in selected)],
            "focal_bound_cases": sum(row["focal_at_bound"] for row in selected),
        }
    summary = camera_summary(fit)
    check = annotations["independent_scalar_check"]
    difference = summary["absolute_camera_to_mat_plane_distance_m"] - check["camera_height_above_table_m"]
    report = {
        "schema": "v2_measured_mat_camera_initialization_v1",
        "status": "DIAGNOSTIC_PLANAR_INITIALIZATION_NOT_CALIBRATION",
        "evidence_kind": "operator_measured_mat_dimensions_and_image_annotated_edges",
        "source": source, "mat_side_lengths_m_from_config": dimensions,
        "mat_frame": "Origin virtual top-left; x toward top-right; y toward bottom-left. Right-handed z normal. This is not the robot/task frame.",
        "virtual_corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "virtual_corners_source_px": corners.tolist(), "edge_fits": edge_report,
        "summary": summary, "planar_fit": fit,
        "sensitivity": {"method": "Move one corner coordinate at a time by +/-1 and +/-3 source pixels; deterministic sensitivity, not a statistical confidence interval.",
                        "summary_by_magnitude_px": sensitivity_summary, "cases": sensitivity},
        "independent_scalar_check": {
            **check, "signed_difference_from_operator_height_m": difference,
            "relative_difference": difference / check["camera_height_above_table_m"],
            "used_for_fitting_or_solution_selection": False,
            "note": "Distance is to the assumed mat plane; camera measurement is to tabletop. The October 13 measurement predates v2; applicability to this recording is not independently established. Mat thickness and operator reference-point uncertainty are not separately measured. One distance does not verify pose, distortion or planar ambiguity.",
        },
        "provenance": {
            "annotations_path": str(annotations_path), "annotations_sha256": sha256(annotations_path),
            "inputs": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()},
            "dependencies_sha256": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in (
                Path(__file__).resolve(), REPO_ROOT / "src/real2sim/artifacts.py", REPO_ROOT / "src/real2sim/geometry/planar_camera.py",
                REPO_ROOT / "src/real2sim/geometry/registration.py")},
            "opencv_version": cv2.__version__,
        },
        "verification_landmarks": None, "reserved_or_validation_frames_used": False,
        "robot_fk_used": False, "joint_zero_fit_performed": False,
        "physical_acceptance_established": False, "automatically_adopted": False,
        "limitations": [
            "The four corner intersections are fitted observations, not held-out verification.",
            "Outer edges and stitch/shadow boundaries have annotation ambiguity; the soft mat is not proven perfectly rectangular or planar.",
            "Square pixels, centred principal point and zero distortion are assumptions; source crop and optical settings are not independently calibrated.",
            "Measured mat dimensions constrain scale but do not locate the mat relative to the robot base.",
            "Assigning 0.305 m across the image and 0.406 m along it inherits the config's image-inferred axis hypothesis, not a separately measured axis correspondence.",
            "Only training episode 0 frame 200 is used. Apparent view changes in later recordings require separate checks.",
            "Off-plane landmarks or an independently verified camera pose remain required before calibration acceptance.",
        ],
    }
    canvas = image.copy()
    cv2.polylines(canvas, [np.rint(corners).astype(np.int32)], True, (0, 220, 255), 2)
    for points in samples.values():
        for point in points:
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 3, (50, 255, 50), -1)
    for index, corner in enumerate(corners):
        location = tuple(np.rint(corner).astype(int))
        cv2.circle(canvas, location, 5, (0, 0, 255), 2)
        cv2.putText(canvas, str(index), location, cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(overlay_path), canvas):
        raise OSError("could not write edge review overlay")
    report["edge_review_overlay"] = {"path": str(overlay_path), "sha256": sha256(overlay_path)}
    with output_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path,
                        default=Path("configs/v2/mat_camera.annotations_20261022.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/v2/mat_camera_initialization_20261022.json"))
    parser.add_argument("--overlay", type=Path,
                        default=Path("outputs/v2/mat_camera_20261022/edges.png"))
    args = parser.parse_args(argv)
    try:
        report = run(args.annotations, args.output, args.overlay)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
