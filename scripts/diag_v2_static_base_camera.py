"""Test image regimes and a static-foot registration independently of arm joints.

The foot correspondences are exploratory nominal-CAD hypotheses. Repeated holes,
soft-mat geometry and the assumed pinhole model prevent automatic adoption.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from real2sim.artifacts import sha256_file as sha256
from real2sim.config import load_yaml
from real2sim.geometry.planar_camera import backproject_to_plane, fit_planar_camera

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/v2/iphone_stability_20261021/report.json"
MAT = ROOT / "results/v2/mat_camera_initialization_20261022.json"
MODEL = ROOT / "assets/menagerie/robotstudio_so101/so101.xml"
ROI = np.array([[227, 338], [275, 334], [318, 345], [323, 370], [244, 388], [221, 375.]])


def summarize(errors):
    errors = np.asarray(errors, float)
    if not len(errors):
        return {"count": 0}
    return {"count": len(errors), "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)), "rms": float(np.sqrt(np.mean(errors**2))),
            "maximum": float(np.max(errors))}


def transform_pixels(points, matrix):
    points = np.asarray(points, float)
    return np.c_[points, np.ones(len(points))] @ np.asarray(matrix, float).T


def fit_image_similarity(rows, excluded_region=None):
    matches = [m for row in rows for m in row["matches"] if m["region"] != excluded_region]
    p = np.array([m["reference_px"] for m in matches])
    q = np.array([m["target_px"] for m in matches])
    if len(p) < 4:
        raise ValueError("at least four training correspondences required")
    cv2.setRNGSeed(0)
    matrix, inliers = cv2.estimateAffinePartial2D(
        p, q, method=cv2.RANSAC, ransacReprojThreshold=2., maxIters=10000,
        confidence=.999, refineIters=30,
    )
    if matrix is None:
        raise ValueError("training image similarity failed")
    return matrix, {"match_count": len(p), "ransac_inliers": int(inliers.sum()),
                    "fit_episodes": sorted({r["target_episode"] for r in rows})}


def score_image_similarity(rows, matrix, region=None):
    matches = [m for row in rows for m in row["matches"]
               if region is None or m["region"] == region]
    if not matches:
        return {"all_descriptor_matches": summarize([])}
    p = np.array([m["reference_px"] for m in matches])
    q = np.array([m["target_px"] for m in matches])
    error = np.linalg.norm(transform_pixels(p, matrix) - q, axis=1)
    prior_inliers = np.array([m.get("similarity_inlier", False) for m in matches])
    return {"all_descriptor_matches": summarize(error),
            "previous_per_frame_ransac_inliers": summarize(error[prior_inliers]),
            "prior_inlier_note": "Secondary diagnostic; the original per-frame RANSAC used each target's coordinates. All-match scores above do not select target inliers."}


def native_base_mesh():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom = next(g for g in range(model.ngeom)
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and model.mesh(model.geom_dataid[g]).name == "base_so101_v2")
    mesh = model.geom_dataid[geom]
    first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = (model.mesh_vert[first:first + count] @ data.geom_xmat[geom].reshape(3, 3).T
                + data.geom_xpos[geom])
    first, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    return np.asarray(vertices, float), np.asarray(model.mesh_face[first:first + count], int)


def foot_holes(vertices, faces, height=.0151):
    """Recover top-plane aperture centroids from exact nominal-CAD boundary loops."""
    faces = faces[np.all(np.abs(vertices[faces, 2] - height) < 1e-6, axis=1)]
    counts = Counter(tuple(sorted(pair)) for face in faces
                     for pair in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])))
    adjacent = defaultdict(list)
    for (a, b), count in counts.items():
        if count == 1:
            adjacent[a].append(b)
            adjacent[b].append(a)
    if any(len(neighbors) != 2 for neighbors in adjacent.values()):
        raise ValueError("nominal top-plane boundaries are not closed manifold loops")
    seen, rows = set(), []
    for start in adjacent:
        if start in seen:
            continue
        ring, previous, current = [start], -1, start
        while True:
            following = next(v for v in adjacent[current] if v != previous)
            if following == start:
                break
            if following in ring:
                raise ValueError("top-plane boundary does not close")
            ring.append(following)
            previous, current = current, following
        seen.update(ring)
        p = vertices[ring, :2]
        q = np.roll(p, -1, axis=0)
        cross = p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]
        area = float(cross.sum() / 2)
        center = ((p + q) * cross[:, None]).sum(axis=0) / (6 * area)
        if 5e-6 < abs(area) < 100e-6 and center[0] > -.01:
            rows.append({"native_xyz_m": [*center.tolist(), height],
                         "opening_area_m2": abs(area), "loop_indices": ring})
    return rows


def detect_holes(image, matrix=None):
    """Detect dark spots in a fixed reviewed ROI, before associating CAD points."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    dark = cv2.GaussianBlur(gray, (0, 0), 3) - cv2.GaussianBlur(gray, (0, 0), .6)
    polygon = ROI if matrix is None else transform_pixels(ROI, matrix)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    count, _, stats, centers = cv2.connectedComponentsWithStats(
        ((dark > 6) & (mask > 0)).astype(np.uint8))
    return np.array([centers[i] for i in range(1, count) if 2 <= stats[i, 4] <= 60])


def base_transform(x):
    """Native z-up base to mat frame whose z points below the table."""
    c, s = np.cos(x[2]), np.sin(x[2])
    transform = np.eye(4)
    transform[:3, :3] = [[c, s, 0], [s, -c, 0], [0, 0, -1]]
    transform[:3, 3] = [x[0], x[1], -.0024]
    return transform


def project_base(points, x, mat_fit, image_matrix=None):
    transform = np.asarray(mat_fit["T_optical_plane"]) @ base_transform(x)
    result = cv2.projectPoints(
        np.asarray(points, float), cv2.Rodrigues(transform[:3, :3])[0], transform[:3, 3],
        np.asarray(mat_fit["K"]), None)[0].reshape(-1, 2)
    return result if image_matrix is None else transform_pixels(result, image_matrix)


def camera_after_image_similarity(K, T_optical_base, image_matrix):
    """Represent a diagnostic image similarity as pinhole intrinsics and pose.

    Keep the small image-plane rotation in the optical pose, rather than hiding
    it as skew in the intrinsic matrix. This is algebraic, not physical adoption.
    """
    K, transform, matrix = (np.asarray(v, float) for v in
                            (K, T_optical_base, image_matrix))
    if not np.allclose(K[:2, :2], np.eye(2) * K[0, 0], atol=1e-8):
        raise ValueError("the image-similarity decomposition requires square pixels and zero skew")
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if scale <= 0:
        raise ValueError("image similarity has zero scale")
    rotation = matrix[:, :2] / scale
    if (not np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-8)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-8)):
        raise ValueError("image transform must be an orientation-preserving similarity")
    intrinsic = np.eye(3)
    intrinsic[0, 0] = intrinsic[1, 1] = scale * K[0, 0]
    intrinsic[:2, 2] = transform_pixels([K[:2, 2]], matrix)[0]
    optical_rotation = np.eye(4)
    optical_rotation[:2, :2] = rotation
    return intrinsic, optical_rotation @ transform


def camera_descriptor(K, T_optical_base):
    """Return an opt-in renderer descriptor in the native-base task frame."""
    K, transform = np.asarray(K, float), np.asarray(T_optical_base, float)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    position = -rotation.T @ translation
    return {
        "key": "observation.images.iphone", "projection_model": "opencv_pinhole",
        "width": 1280, "height": 720,
        "intrinsics_px": {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                          "cx": float(K[0, 2]), "cy": float(K[1, 2])},
        "distortion_coefficients": [0., 0., 0., 0., 0.],
        "approx_position_task": position.tolist(),
        "xyaxes_task": np.r_[rotation[0], -rotation[1]].tolist(),
        "look_at_task": (position + .8 * rotation[2]).tolist(),
    }


def candidate_camera_descriptors(report, report_path):
    foot = report["base_foot"]
    report_path = Path(report_path).resolve()
    return {
        "schema": "v2_camera_render_candidates_v1", "physical_calibration_accepted": False,
        "automatically_adopted": False, "task_frame": "native_so101_base",
        "distortion_status": "All five zero coefficients are assumed, not measured. Source images are not independently proven rectified.",
        "scope": "Reviewable camera-only descriptors for the selected static-foot hypothesis. A coherent scene registration is required before using these with an existing task configuration; this file does not replace a task config.",
        "source_report": str(report_path.relative_to(ROOT) if report_path.is_relative_to(ROOT)
                             else report_path),
        "source_report_sha256": sha256(report_path),
        "shared_geometry_metadata_only": {
            "assumed_table_plane_z_in_native_base_m": -.0024,
            "candidate_mat_center_in_native_base_m": foot["candidate_mat_center_in_native_base_m"],
            "assumed_mat_thickness_m": 0.,
            "physical_hole_identities_verified": False,
            "competing_alignment_basins_retained_in_source_report": True,
        },
        "candidates": {
            regime: {
                "camera": camera_descriptor(foot[f"candidate_K_{regime}"],
                                            foot[f"candidate_T_optical_base_{regime}"]),
                "K": foot[f"candidate_K_{regime}"],
                "T_optical_native_base": foot[f"candidate_T_optical_base_{regime}"],
                "recording_regime_hypothesis": "episodes0–6" if regime == "early" else "episodes7 onward",
                "physical_calibration_accepted": False,
            } for regime in ("early", "late")
        },
    }


def fit_unlabelled_foot(pixels, holes, mat_fit):
    """Enumerate both wings and rotation basins; retain ambiguity rather than adopt."""
    observed = backproject_to_plane(
        pixels, K=mat_fit["K"], T_optical_plane=mat_fit["T_optical_plane"],
        plane_z_m=-.0175)[:, :2]
    native = np.array([r["native_xyz_m"] for r in holes])
    candidates = []
    for wing in (-1, 1):
        source = native[native[:, 1] * wing > 0, :2] * [1, -1]
        for angle in np.linspace(-np.pi, np.pi, 72, endpoint=False):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s], [s, c]])
            start = np.r_[observed.mean(0) - (source @ rotation.T).mean(0), angle]

            def residual(x, source=source):
                c, s = np.cos(x[2]), np.sin(x[2])
                projected = source @ np.array([[c, s], [-s, c]]) + x[:2]
                _, nearest = cKDTree(projected).query(observed)
                return (observed - projected[nearest]).ravel()

            fitted = least_squares(residual, start, loss="soft_l1", f_scale=.0005,
                                   max_nfev=100)
            errors = np.linalg.norm(residual(fitted.x).reshape(-1, 2), axis=1)
            candidates.append({"x": fitted.x, "wing": wing,
                               "trimmed_score_m2": float(np.mean(np.minimum(errors, .003)**2)),
                               "errors": errors})
    candidates.sort(key=lambda row: row["trimmed_score_m2"])
    unique = []
    for row in candidates:
        if any(row["wing"] == prior["wing"]
               and np.linalg.norm(row["x"][:2] - prior["x"][:2]) < .003
               and abs(np.arctan2(np.sin(row["x"][2] - prior["x"][2]),
                                  np.cos(row["x"][2] - prior["x"][2]))) < .1
               for prior in unique):
            continue
        unique.append(row)
    return unique


def validate_source_rows(report, split):
    for row in report["source_frames"]:
        if row["episode"] in split["reserved"]:
            raise ValueError("reserved image in diagnostic inputs")
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"source image hash changed: {row['path']}")


def run(output):
    if output.exists():
        raise FileExistsError(output)
    cv2.setNumThreads(1)
    report = json.loads(SOURCE.read_text())
    split_path = ROOT / "results/v2/split.json"
    split = json.loads(split_path.read_text())
    validate_source_rows(report, split)
    comparisons = [r for r in report["comparisons"] if r["kind"] == "between_episodes"]
    training = [r for r in comparisons if r["partition"] == "train" and r["target_episode"] >= 7]
    validation = [r for r in comparisons if r["partition"] == "val"]
    if any(r["target_episode"] not in split["train"] for r in training):
        raise ValueError("image model fit includes a non-training capture")
    image_models = {}
    for omitted in (None, "fixed_base_foot", "lower_table_clamp", "cabinet_handle"):
        matrix, fit_info = fit_image_similarity(training, omitted)
        image_models["all_regions" if omitted is None else f"without_{omitted}"] = {
            "matrix_reference_to_late": matrix.tolist(), **fit_info,
            "validation": score_image_similarity(validation, matrix),
            "held_out_region_training": None if omitted is None else
            score_image_similarity(training, matrix, omitted),
            "held_out_region_validation": None if omitted is None else
            score_image_similarity(validation, matrix, omitted),
            "scale": float(np.hypot(matrix[0, 0], matrix[1, 0])),
            "rotation_deg": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
            "mapped_image_center_px": transform_pixels([[640, 360]], matrix)[0].tolist(),
        }
    mat = json.loads(MAT.read_text())
    fit = mat["planar_fit"]
    swapped = fit_planar_camera(mat["virtual_corners_source_px"], source_size_wh=[1280, 720],
                                side_lengths_m=[.406, .305])
    source = next(row for row in report["source_frames"]
                  if row["episode"] == 0 and row["position"] == 200)
    image = cv2.imread(str(ROOT / source["path"]))
    spots = detect_holes(image)
    vertices, faces = native_base_mesh()
    holes = foot_holes(vertices, faces)
    alternatives = fit_unlabelled_foot(spots, holes, fit)
    candidate = alternatives[0]
    x = candidate["x"]
    native = np.array([row["native_xyz_m"] for row in holes])
    native = native[native[:, 1] * candidate["wing"] > 0]
    projected = project_base(native, x, fit)
    distance, association = cKDTree(projected).query(spots)
    # Retain the entire association table, including large residuals. No spot is
    # silently treated as a known physical identity or discarded as an outlier.
    association_rows = [
        {"detected_source_px": p.tolist(), "nearest_nominal_xyz_m": native[j].tolist(),
         "distance_px": float(e)}
        for p, j, e in zip(spots, association, distance, strict=True)
    ]
    late_matrix = np.asarray(image_models["without_fixed_base_foot"]["matrix_reference_to_late"])
    validation_spots = []
    for row in report["source_frames"]:
        if row["episode"] not in split["val"] or row["position"] != 0:
            continue
        target = cv2.imread(str(ROOT / row["path"]))
        detected = detect_holes(target, late_matrix)
        prediction = transform_pixels(projected, late_matrix)
        if len(detected):
            error, _ = cKDTree(prediction).query(detected)
            within = error <= 4
        else:
            error, within = np.empty(0), np.empty(0, bool)
        validation_spots.append({
            "source": row, "detected_count": len(detected),
            "all_detected_to_nearest_predicted_hole_px": summarize(error),
            "within_4px_count": int(within.sum()),
            "within_4px_error_px": summarize(error[within]),
            "unmatched_detected_count": int((~within).sum()),
            "detected_pixels": detected.tolist(),
            "note": "Association is by nearest repeated hole, so this checks image prediction locally and cannot verify physical hole identities or distinguish grid-period aliases.",
        })
    canvas = image.copy()
    for p in projected:
        cv2.drawMarker(canvas, tuple(np.rint(p).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 7, 1)
    for p in spots:
        cv2.circle(canvas, tuple(np.rint(p).astype(int)), 3, (0, 220, 0), 1)
    output.mkdir(parents=True)
    cv2.imwrite(str(output / "foot_overlay.png"), canvas[280:420, 140:380])
    T_optical_base = np.asarray(fit["T_optical_plane"]) @ base_transform(x)
    K_late, T_optical_base_late = camera_after_image_similarity(
        fit["K"], T_optical_base, late_matrix)
    T_mat_base = base_transform(x)
    camera_in_base = -T_optical_base[:3, :3].T @ T_optical_base[:3, 3]
    center_mat = np.array([.305 / 2, .406 / 2, 0.])
    mat_center_in_base = T_mat_base[:3, :3].T @ (center_mat - T_mat_base[:3, 3])
    task_path = ROOT / "configs/task.yaml"
    legacy = load_yaml(task_path)
    result = {
        "schema": "v2_static_base_camera_diagnostic_v1",
        "status": "EXPLORATORY_CANDIDATE_NOT_PHYSICAL_CALIBRATION",
        "robot_joint_readings_used": False, "reserved_episodes_used": False,
        "production_inputs_changed": False, "automatically_adopted": False,
        "image_regime_boundary_episode": 7,
        "boundary_provenance": "Prior training-only static-image diagnostic, fixed before this validation scoring.",
        "validation_identity_model": score_image_similarity(validation, np.eye(3)[:2]),
        "image_similarity_models": image_models,
        "mat_axis_alternative": {"side_lengths_m": [.406, .305],
                                 "fit_rms_px": swapped["fit_rms_px"],
                                 "focal_px": swapped["focal_px"],
                                 "focal_at_bound": swapped["focal_at_bound"]},
        "base_foot": {
            "source": source, "detected_spots": len(spots),
            "native_top_plane_z_m": .0151, "assumed_base_origin_above_mat_plane_m": .0024,
            "mat_thickness_assumed_m": 0., "model_hole_count_both_wings": len(holes),
            "method": "Detect dark blobs using one fixed training ROI and thresholds. Compare to CAD top-plane aperture centroids; enumerate two wings and 72 yaw starts. Robust 3DOF in-plane registration, 0.5mm soft-L1 scale. Candidate ranking caps each residual at3mm, all uncapped residuals are also reported.",
            "candidate_T_mat_base": base_transform(x).tolist(),
            "candidate_T_optical_base_early": T_optical_base.tolist(),
            "candidate_K_early": fit["K"],
            "candidate_K_late": K_late.tolist(),
            "candidate_T_optical_base_late": T_optical_base_late.tolist(),
            "late_camera_note": "Exact algebraic decomposition of the training cabinet/clamp image similarity: scale/principal point enter K, small image-plane rotation enters optical pose. No warp or camera configuration is adopted.",
            "candidate_camera_position_in_native_base_m": camera_in_base.tolist(),
            "candidate_mat_center_in_native_base_m": mat_center_in_base.tolist(),
            "legacy_camera_position_task_m": legacy["camera_iphone"]["approx_position_task"],
            "legacy_mat_center_task_xy_m": legacy["table"]["mat"]["center_task_xy"],
            "comparison_note": "Legacy task/base transform is identity; candidate differences are conditional on tentative repeated-hole identities and the nominal mat camera. They are hypotheses, not physical placement corrections.",
            "xy_yaw_radians": x.tolist(), "matched_nominal_wing_sign": candidate["wing"],
            "pixel_associations": association_rows,
            "all_spot_nearest_candidate_error_px": summarize(distance),
            "all_spot_nearest_candidate_error_mm": summarize(candidate["errors"] * 1000),
            "within_2mm_count": int(np.sum(candidate["errors"] <= .002)),
            "distinct_alignment_basins": [
                {"xy_yaw_radians": a["x"].tolist(), "wing_sign": a["wing"],
                 "T_optical_base_early": (np.asarray(fit["T_optical_plane"])
                                          @ base_transform(a["x"])).tolist(),
                 "trimmed_score_mm2": a["trimmed_score_m2"] * 1e6,
                 "all_spot_nearest_error_mm": summarize(a["errors"] * 1000),
                 "within_2mm_count": int(np.sum(a["errors"] <= .002))}
                for a in alternatives[:6]],
            "separate_validation_captures": validation_spots,
            "physical_hole_identities_verified": False,
            "new_untouched_spatial_check": False,
        },
        "limitations": [
            "Mat-frame camera assumes centered principal point, square pixels, zero distortion and exact rectangular dimensions; its four corners are fit data.",
            "The same training foot image was inspected while forming the correspondence hypothesis. No withheld spatial foot subset is claimed.",
            "Unlabelled repeated holes admit yaw/translation aliases. Distinct captures cannot independently verify physical hole identities.",
            "Nominal CAD, no spacer, zero mat thickness and an exactly planar tabletop are assumptions. CAD correspondence can support an arm-independent diagnostic but is not a physically measured calibration.",
            "Static-image similarity generalization does not distinguish digital crop, focus, intrinsics, image stabilization or physical camera motion; it is not adopted as a corrective policy warp.",
            "All descriptor-match scores include repeated-pattern mismatches. Previously selected per-frame inlier scores are separately labeled and are not an untouched acceptance test.",
            "Validation dark-spot assignment uses nearest predicted holes and a stated 4px threshold. Unmatched spots remain counted; these scores are unsuitable as independent joint-zero acceptance.",
        ],
        "software_versions": {"opencv": cv2.__version__, "mujoco": mujoco.__version__,
                              "numpy": np.__version__},
        "provenance_sha256": {str(p.relative_to(ROOT)): sha256(p)
                              for p in (Path(__file__).resolve(), ROOT / "src/real2sim/artifacts.py", SOURCE, MAT, MODEL, split_path, task_path,
                                        MODEL.parent / "assets/base_so101_v2.stl")},
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/v2/static_base_camera_20261023")
    parser.add_argument("--camera-descriptors", type=Path)
    args = parser.parse_args()
    result = run(args.out)
    if args.camera_descriptors:
        args.camera_descriptors.parent.mkdir(parents=True, exist_ok=True)
        with args.camera_descriptors.open("x") as stream:
            stream.write(json.dumps(candidate_camera_descriptors(result, args.out / "report.json"), indent=2,
                                    allow_nan=False) + "\n")
    print(json.dumps({"image_models": result["image_similarity_models"],
                      "base_foot_errors_px": result["base_foot"]["all_spot_nearest_candidate_error_px"]},
                     indent=2))


if __name__ == "__main__":
    main()
