"""Fit conditional cameras from newly annotated random frames and expose planar gauges.

Repeated views of one static mat improve pixel precision but do not create independent
camera poses or remove planar intrinsic ambiguity. Fits are conditional image estimates,
never metrology or production calibration. Whole verification frames do not enter fits,
initialization, parameter bounds, uncertainty scenarios, or solution selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from real2sim.data.raw import sha256_file
from real2sim.geometry.planar_camera import fit_planar_camera
from real2sim.geometry.registration import _identifiability


def mat_points(lengths):
    w, h = lengths
    return np.array([[0., 0., 0.], [w, 0., 0.], [w, h, 0.], [0., h, 0.]])


def edge_intersections(samples):
    """Virtual rectangle corners from independently annotated visible straight segments."""
    lines, diagnostics = [], {}
    for name in ("top", "right", "bottom", "left"):
        points = np.asarray(samples[name], float)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("each mat edge needs >=3 finite source-pixel samples")
        center = points.mean(0)
        _, singular, vt = np.linalg.svd(points - center, full_matrices=False)
        if singular[0] < 1:
            raise ValueError("mat edge samples must span a visible segment")
        line = np.r_[vt[-1], -vt[-1] @ center]
        lines.append(line)
        error = points @ line[:2] + line[2]
        diagnostics[name] = {"sample_count": len(points), "orthogonal_rms_px": float(np.sqrt(np.mean(error**2))),
                             "line_ax_by_c_equals_zero": line.tolist()}
    corners = []
    for a, b in ((3, 0), (0, 1), (1, 2), (2, 3)):
        point = np.cross(lines[a], lines[b])
        if abs(point[2]) < 1e-8:
            raise ValueError("adjacent edges are parallel")
        corners.append(point[:2] / point[2])
    return np.asarray(corners), diagnostics


def project(points, K, T, distortion=None):
    return cv2.projectPoints(np.asarray(points, float), cv2.Rodrigues(np.asarray(T)[:3, :3])[0],
                             np.asarray(T)[:3, 3], np.asarray(K, float), distortion)[0].reshape(-1, 2)


def intrinsic_rank(points, K, T, *, model="centered_square", repeats=1, sigma_px=3.):
    """Data-only rank, including exact null directions for unconstrained intrinsics."""
    K, T = np.asarray(K), np.asarray(T)
    pose = np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]]
    if model == "centered_square":
        x = np.r_[K[0, 0], pose]
        names = ["focal_px"]
        scales = [K[0, 0]]
    elif model in ("free_intrinsics", "free_intrinsics_radial2"):
        x = np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], pose]
        names = ["fx_px", "fy_px", "cx_px", "cy_px"]
        scales = [K[0, 0], K[1, 1], 64., 36.]
        if model == "free_intrinsics_radial2":
            x = np.r_[x, 0., 0.]
    else:
        raise ValueError(model)
    names += ["rvec_x_rad", "rvec_y_rad", "rvec_z_rad", "t_x_m", "t_y_m", "t_z_m"]
    scales += [1., 1., 1., .4, .4, .4]
    if model == "free_intrinsics_radial2":
        names += ["radial_k1", "radial_k2"]
        scales += [.1, .1]

    def residual(parameters):
        if model == "centered_square":
            intrinsic = np.array([[parameters[0], 0., K[0, 2]], [0., parameters[0], K[1, 2]], [0., 0., 1.]])
            p = parameters[1:7]
        else:
            fx, fy, cx, cy = parameters[:4]
            intrinsic = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
            p = parameters[4:10]
        distortion = np.r_[parameters[-2:], 0., 0., 0.] if model == "free_intrinsics_radial2" else None
        prediction = cv2.projectPoints(points, p[:3], p[3:], intrinsic, distortion)[0].reshape(-1, 2)
        return np.tile(prediction.ravel() / sigma_px, repeats)

    result = _identifiability(residual, x, np.asarray(scales))
    result.update(parameter_order=names, unique_metric_points=len(points), repeated_frames=repeats,
                  residual_count=2 * len(points) * repeats,
                  note="Rank is local and conditional on the chosen geometry. Repeating the same plane and pose does not add independent projective constraints. Conditional std assumes independent pixel errors; shared boundary/axis mistakes remain systematic.")
    return result


def conditional_fit(corners, lengths, baseline, *, principal_shift=(0., 0.), aspect=1., radial_k1=0.):
    """Fit the same corners under an explicit fixed alternative optical assumption."""
    points = mat_points(lengths)
    K0, T0 = np.asarray(baseline["K"]), np.asarray(baseline["T_optical_plane"])
    center = K0[:2, 2] + principal_shift
    x0 = np.r_[baseline["focal_px"], cv2.Rodrigues(T0[:3, :3])[0].ravel(), T0[:3, 3]]
    distortion = np.array([radial_k1, 0., 0., 0., 0.])

    def K(x):
        return np.array([[x[0], 0., center[0]], [0., x[0] * aspect, center[1]], [0., 0., 1.]])

    def residual(x):
        return (cv2.projectPoints(points, x[1:4], x[4:7], K(x), distortion)[0].reshape(-1, 2) - corners).ravel()

    result = least_squares(residual, x0, bounds=(np.r_[256., np.full(5, -np.inf), 1e-6],
                                               np.r_[6400., np.full(6, np.inf)]),
                           x_scale=np.r_[1400., np.ones(3), np.full(3, .4)], max_nfev=1000)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(result.x[1:4])[0], result.x[4:7]
    depth = (points @ T[:3, :3].T + T[:3, 3])[:, 2]
    position = -T[:3, :3].T @ T[:3, 3]
    return {"K": K(result.x).tolist(), "T_optical_mat": T.tolist(),
            "focal_px": float(result.x[0]), "camera_position_mat_m": position.tolist(),
            "camera_plane_distance_m": float(abs(position[2])),
            "fit_rms_px": float(np.sqrt(np.mean(residual(result.x).reshape(-1, 2)**2) * 2)),
            "optimizer_converged": bool(result.success), "all_points_positive_depth": bool(np.all(depth > 0)),
            "distortion_coefficients": distortion.tolist(),
            "focal_at_bound": bool(min(abs(result.x[0] - np.array([256., 6400.]))) < .01)}


def fit_regime(rows, lengths, source_size):
    fit = [row for row in rows if row["role"] == "fit"]
    verification = [row for row in rows if row["role"] == "verification"]
    if not fit:
        return {"status": "no_fit_frames", "verification_frame_count": len(verification)}
    weights = np.array([1 / row["sigma_px"]**2 for row in fit])
    corners = np.average([row["corners"] for row in fit], axis=0, weights=weights)
    # Do not manufacture extra independent geometry or claim subpixel truth by averaging.
    sigma = float(max(row["sigma_px"] for row in fit))
    baseline = fit_planar_camera(corners, source_size_wh=source_size, side_lengths_m=lengths, sigma_px=sigma)
    K, T = np.asarray(baseline["K"]), np.asarray(baseline["T_optical_plane"])
    camera_position = -T[:3, :3].T @ T[:3, 3]
    baseline.update(camera_position_mat_m=camera_position.tolist(),
                    camera_plane_distance_m=float(abs(camera_position[2])),
                    horizontal_fov_deg=float(np.degrees(2 * np.arctan(source_size[0] / (2 * K[0, 0])))),
                    vertical_fov_deg=float(np.degrees(2 * np.arctan(source_size[1] / (2 * K[1, 1])))))
    prediction = project(mat_points(lengths), K, T)
    scores = []
    for row in fit + verification:
        error = np.linalg.norm(prediction - row["corners"], axis=1)
        scores.append({"episode": row["episode"], "frame": row["frame"], "role": row["role"],
                       "rms_px": float(np.sqrt(np.mean(error**2))), "p95_px": float(np.percentile(error, 95)),
                       "per_corner_error_px": error.tolist()})
    cases = []
    # These scenarios use fitting-frame annotations only. They are not confidence intervals.
    for axis in range(2):
        for sign in (-1, 1):
            shift = np.zeros(2)
            shift[axis] = sign * 20.
            cases.append({"scenario": f"principal_{axis}_{sign:+d}20px",
                          **conditional_fit(corners, lengths, baseline, principal_shift=shift)})
            sides = np.asarray(lengths, float).copy()
            sides[axis] *= 1 + sign * .03
            cases.append({"scenario": f"mat_side_{axis}_{sign:+d}3percent",
                          **conditional_fit(corners, sides, baseline)})
    for sign in (-1, 1):
        cases.append({"scenario": f"pixel_aspect_{sign:+d}1percent",
                      **conditional_fit(corners, lengths, baseline, aspect=1 + sign * .01)})
        cases.append({"scenario": f"radial_k1_{sign:+d}005",
                      **conditional_fit(corners, lengths, baseline, radial_k1=sign * .05)})
    for corner in range(4):
        for axis in range(2):
            for sign in (-1, 1):
                changed = corners.copy()
                changed[corner, axis] += sign * sigma
                cases.append({"scenario": f"pooled_corner{corner}_{axis}_{sign:+d}sigma",
                              **conditional_fit(changed, lengths, baseline)})
    focal = [row["focal_px"] for row in cases]
    distance = [row["camera_plane_distance_m"] for row in cases]
    return {"status": "CONDITIONAL_IMAGE_CAMERA_ESTIMATE_NOT_CALIBRATION", "fit_frame_count": len(fit),
            "verification_frame_count": len(verification), "pooled_corners_source_px": corners.tolist(),
            "pooling": "Inverse declared variance weighted source pixels; fixed camera within declared regime. Systematic annotation errors are not reduced by frame count.",
            "baseline": baseline, "per_frame_scores": scores,
            "expanded_intrinsic_identifiability": {
                name: intrinsic_rank(mat_points(lengths), K, T, model=name, repeats=len(fit), sigma_px=sigma)
                for name in ("centered_square", "free_intrinsics", "free_intrinsics_radial2")},
            "sensitivity": {"cases": cases, "focal_range_px": [min(focal), max(focal)],
                            "camera_plane_distance_range_m": [min(distance), max(distance)],
                            "method": "Declared deterministic assumption envelope, not a confidence interval; verification pixels never fit or select scenarios."},
            "robot_base_registration_established": False,
            "verification_is_untouched_physical_calibration": False}


def load_annotations(path):
    document = json.loads(path.read_text())
    if document.get("schema") != "v2_multiframe_static_annotations_v1":
        raise ValueError("unsupported annotation schema")
    split_path = Path(document.get("split_path", "results/v2/split.json"))
    split = json.loads(split_path.read_text())
    selections = None
    if "selection_path" in document:
        selection_path = Path(document["selection_path"])
        if sha256_file(selection_path) != document["selection_sha256"]:
            raise ValueError("random source-selection checksum mismatch")
        selection = json.loads(selection_path.read_text())
        if selection.get("split_sha256") != sha256_file(split_path):
            raise ValueError("random selection uses a different episode split")
        selections = {(row["episode"], row["frame"], row["camera_key"]): row
                      for row in selection["images"]}
    size = np.asarray(document["source_size_wh"], int)
    if size.shape != (2,) or np.any(size <= 0):
        raise ValueError("source_size_wh must be positive dimensions")
    rows, identities = [], set()
    for source in document["frames"]:
        row = dict(source)
        identity = (row["episode"], row["frame"], row.get("camera", "observation.images.iphone"))
        if identity in identities:
            raise ValueError("source frame cannot be duplicated under a different role/regime")
        identities.add(identity)
        if selections is not None:
            selected = selections.get(identity)
            if (selected is None or row["path"] != selected["path"] or row["sha256"] != selected["sha256"]
                    or row["camera_regime"] != selected["camera_regime"]
                    or (row["role"] != "excluded" and row["role"] != selected["role"])):
                raise ValueError("annotations changed random selection identity, fit/verification role or regime")
        if row["episode"] in split.get("reserved", []):
            raise ValueError("reserved policy episodes cannot enter this geometry diagnostic")
        if row["role"] not in ("fit", "verification", "excluded"):
            raise ValueError("role must be fit, verification or excluded")
        if row["role"] == "fit" and row["episode"] not in split["train"]:
            raise ValueError("fitting frames must come from training episodes")
        if identity[2] != "observation.images.iphone":
            raise ValueError("this static-camera fit cannot pool a moving wrist camera")
        if row["episode"] not in split["train"] + split["val"]:
            raise ValueError("source episode is not in the declared development split")
        image_path = Path(row["path"])
        if sha256_file(image_path) != row["sha256"]:
            raise ValueError("source image hash mismatch")
        image = cv2.imread(str(image_path))
        if image is None or list(image.shape[1::-1]) != size.tolist():
            raise ValueError("source image dimensions mismatch")
        if row["role"] == "excluded":
            if not row.get("reason"):
                raise ValueError("excluded random frames require an explicit reason")
            rows.append(row)
            continue
        row["sigma_px"] = float(row["sigma_px"])
        if not np.isfinite(row["sigma_px"]) or row["sigma_px"] <= 0:
            raise ValueError("annotation sigma_px must be finite and positive")
        if "mat_edge_samples_source_px" in row:
            for points in row["mat_edge_samples_source_px"].values():
                if np.any(np.asarray(points) < 0) or np.any(np.asarray(points) > size):
                    raise ValueError("mat edge samples must remain inside the source image")
            corners, edge_report = edge_intersections(row["mat_edge_samples_source_px"])
            row["edge_fit"] = edge_report
        else:
            corners = np.asarray(row["mat_corners_source_px"], float)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError("four ordered finite mat corners are required")
        row["corners"] = corners
        rows.append(row)
    if selections is not None:
        expected = {key for key in selections if key[2] == "observation.images.iphone"}
        if identities != expected:
            raise ValueError("retain every randomly selected fixed-camera frame, using explicit exclusions where needed")
    return document, rows, split_path


def run(annotations, out):
    if out.exists():
        raise FileExistsError("choose a fresh output directory")
    document, rows, split_path = load_annotations(annotations)
    included = [row for row in rows if row["role"] != "excluded"]
    regimes = sorted({row["camera_regime"] for row in included})
    results = {name: fit_regime([row for row in included if row["camera_regime"] == name],
                               document["mat_side_lengths_m"], document["source_size_wh"]) for name in regimes}
    out.mkdir(parents=True)
    for row in included:
        report = results[row["camera_regime"]]
        if "baseline" not in report:
            continue
        fit = report["baseline"]
        predicted = project(mat_points(document["mat_side_lengths_m"]), fit["K"], fit["T_optical_plane"])
        image = cv2.imread(row["path"])
        for observed, estimate in zip(row["corners"], predicted, strict=True):
            cv2.circle(image, tuple(np.rint(observed).astype(int)), 5, (0, 200, 0), 2)
            cv2.drawMarker(image, tuple(np.rint(estimate).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.polylines(image, [np.rint(predicted).astype(np.int32)], True, (0, 0, 255), 1)
        cv2.imwrite(str(out / f"ep{row['episode']:02d}_f{row['frame']:04d}_{row['role']}.png"), image)
    report = {"schema": "v2_multiframe_static_camera_diagnostic_v1", "physical_calibration_accepted": False,
              "production_inputs_changed": False, "reserved_episodes_used": False,
              "robot_fk_used": False, "regimes": results,
              "source_annotations_sha256": sha256_file(annotations), "split_sha256": sha256_file(split_path),
              "selection": document.get("selection"),
              "random_selection": {key: document[key] for key in ("selection_path", "selection_sha256") if key in document},
              "excluded_frames": [row for row in rows if row["role"] == "excluded"],
              "source_frames": [{key: row[key] for key in ("episode", "frame", "path", "sha256", "role", "camera_regime")} for row in rows],
              "code_sha256": {str(p): sha256_file(p) for p in (Path(__file__), Path("src/real2sim/geometry/planar_camera.py"), Path("src/real2sim/geometry/registration.py"))},
              "limitations": ["Mat dimensions and corner-axis assignment are conditional operator/config inputs.",
                              "Centered principal point, square pixels and zero distortion constrain the baseline; free intrinsic models expose remaining gauges.",
                              "All camera poses are in the mat frame. Named physical static CAD correspondence is needed to register the robot base.",
                              "Repeated static views do not create calibration-board motion or establish absolute metric accuracy."]}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.annotations, args.out)
    print(json.dumps({key: {"status": value["status"], "fit_frames": value.get("fit_frame_count"),
                           "verification_frames": value.get("verification_frame_count")}
                      for key, value in result["regimes"].items()}, indent=2))
