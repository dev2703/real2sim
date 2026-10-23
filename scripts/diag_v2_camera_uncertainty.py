"""Bound a conditional camera hypothesis using fixed CAD contours and input sweeps.

No arm states, validation images, reserved images, or task-success scores enter
fitting. These deterministic perturbations are sensitivity scenarios, not a
confidence distribution or a physically accepted camera calibration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "outputs/v2/static_base_camera_20261023/report.json"
MAT = ROOT / "results/v2/mat_camera_initialization_20261022.json"
# Reviewed outer boundaries in the same training image. They were never used
# by the original repeated-hole fit. No untouched image acceptance is claimed.
CONTOUR_CHECKS = {
    "upper_fixed_clamp": [[175, 230], [195, 231], [215, 231],
                          [157, 273], [159, 292], [161, 307]],
    "front_fixed_foot": [[197, 389], [218, 398], [235, 403], [265, 396],
                         [285, 390], [306, 384], [326, 378], [329, 362]],
}


def static_helpers():
    spec = importlib.util.spec_from_file_location(
        "camera_uncertainty_static", ROOT / "scripts/diag_v2_static_base_camera.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_with_intrinsic_assumptions(corners, lengths, initial, *, center=(640., 360.),
                                  aspect=1., radial=0.):
    """Fit only four training corners, conditional on supplied optical assumptions."""
    width, height = lengths
    points = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0.]])
    transform = np.asarray(initial["T_optical_plane"])
    x0 = np.r_[initial["focal_px"], cv2.Rodrigues(transform[:3, :3])[0].ravel(),
               transform[:3, 3]]

    def intrinsics(x):
        return np.array([[x[0], 0, center[0]], [0, x[0] * aspect, center[1]], [0, 0, 1.]])

    distortion = np.array([radial, 0., 0., 0., 0.])

    def residual(x):
        pixels = cv2.projectPoints(points, x[1:4], x[4:7], intrinsics(x), distortion)[0]
        return (pixels.reshape(-1, 2) - corners).ravel()

    fit = least_squares(residual, x0, bounds=(np.r_[256., np.full(5, -np.inf), 1e-6],
                                            np.r_[6400., np.full(6, np.inf)]),
                        x_scale=np.r_[1371., np.ones(3), np.full(3, .4)],
                        max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(fit.x[1:4])[0]
    result[:3, 3] = fit.x[4:7]
    depths = (points @ result[:3, :3].T + result[:3, 3])[:, 2]
    if not fit.success or np.any(depths <= 0):
        raise ValueError("conditional mat fit failed or has nonpositive depth")
    return {"K": intrinsics(fit.x), "T_optical_plane": result,
            "distortion_coefficients": distortion,
            "mat_fit_rms_px": float(np.sqrt(np.mean(residual(fit.x).reshape(-1, 2)**2) * 2))}


def base_in_mat(x, mat_thickness):
    transform = static_helpers().base_transform(x)
    # The base rests on the tabletop; the mat plane is thickness metres higher.
    transform[2, 3] += mat_thickness
    return transform


def project(points, K, transform, distortion=None):
    return cv2.projectPoints(np.asarray(points, float),
                             cv2.Rodrigues(transform[:3, :3])[0], transform[:3, 3],
                             np.asarray(K, float), distortion)[0].reshape(-1, 2)


def refit_foot(pixels, native, camera, initial, *, mat_thickness=0.):
    """Stay in the selected wing/yaw basin; keep every detected training spot."""
    K, distortion = camera["K"], camera["distortion_coefficients"]
    corrected = cv2.undistortPoints(np.asarray(pixels, float)[:, None], K, distortion,
                                   P=K).reshape(-1, 2)
    observed = static_helpers().backproject_to_plane(
        corrected, K=K, T_optical_plane=camera["T_optical_plane"],
        plane_z_m=mat_thickness - .0175)[:, :2]
    source = np.asarray(native)[:, :2] * [1., -1.]

    def residual(x):
        c, s = np.cos(x[2]), np.sin(x[2])
        predicted = source @ np.array([[c, s], [-s, c]]) + x[:2]
        _, nearest = cKDTree(predicted).query(observed)
        return (observed - predicted[nearest]).ravel()

    initial = np.asarray(initial, float)
    fit = least_squares(residual, initial, loss="soft_l1", f_scale=.0005,
                        bounds=(initial - [.03, .03, .3], initial + [.03, .03, .3]),
                        max_nfev=500)
    if not fit.success:
        raise ValueError("conditional foot fit failed")
    transform = camera["T_optical_plane"] @ base_in_mat(fit.x, mat_thickness)
    return fit.x, transform, np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)


def contour_distances(vertices, faces, K, transform, points, distortion=None):
    """Distance to the fixed-mesh outer silhouette, without using moving links."""
    pixels = project(vertices, K, transform, distortion)
    mask = np.zeros((720, 1280), np.uint8)
    for triangle in pixels[faces]:
        cv2.fillConvexPoly(mask, np.rint(triangle).astype(np.int32), 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("no fixed-base contour is visible")
    boundary = np.concatenate(contours)[:, 0]
    errors, _ = cKDTree(boundary).query(points)
    return errors, contours


def scenarios(corners):
    """Explicit one-at-a-time perturbations plus disclosed joint extremes."""
    cases = [{"name": "nominal", "family": "nominal"}]
    for corner in range(4):
        for axis in range(2):
            for sign in (-1, 1):
                changed = corners.copy()
                changed[corner, axis] += sign * 3
                cases.append({"name": f"corner{corner}_{axis}_{sign:+d}3px",
                              "family": "corner_3px", "corners": changed})
    for axis in range(2):
        for sign in (-1, 1):
            center = np.array([640., 360.])
            center[axis] += sign * 20
            cases.append({"name": f"principal{axis}_{sign:+d}20px",
                          "family": "principal_20px", "center": center})
            shift = np.zeros(2)
            shift[axis] = sign
            cases.append({"name": f"foot_shift{axis}_{sign:+d}px",
                          "family": "foot_1px", "foot_shift": shift})
    for axis in range(2):
        for sign in (-1, 1):
            lengths = np.array([.305, .406])
            lengths[axis] *= 1 + sign * .03
            cases.append({"name": f"mat_side{axis}_{sign:+d}3percent",
                          "family": "mat_side_3percent", "lengths": lengths})
    cases.append({"name": "mat_thickness_3mm", "family": "mat_thickness",
                  "mat_thickness": .003})
    for sign in (-1, 1):
        cases.append({"name": f"pixel_aspect_{sign:+d}1percent", "family": "aspect_1percent",
                      "aspect": 1 + sign * .01})
        cases.append({"name": f"radial_k1_{sign:+d}005", "family": "radial_005",
                      "radial": sign * .05})
        cases.append({"name": f"joint_extreme_{sign:+d}", "family": "joint_extremes",
                      "corners": corners + sign * np.array([[3, -3], [-3, 3], [3, 3], [-3, -3]]),
                      "center": [640 + sign * 20, 360 - sign * 20],
                      "lengths": [.305 * (1 + sign * .03), .406 * (1 - sign * .03)],
                      "foot_shift": [sign, -sign], "mat_thickness": .003 if sign > 0 else 0.})
    return cases


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def run(out):
    if out.exists():
        raise FileExistsError("use a fresh report path")
    helper = static_helpers()
    static, mat = json.loads(STATIC.read_text()), json.loads(MAT.read_text())
    split_path = ROOT / "results/v2/split.json"
    split = json.loads(split_path.read_text())
    foot = static["base_foot"]
    source = ROOT / foot["source"]["path"]
    if (foot["source"]["episode"] not in split["train"]
            or helper.sha256(source) != foot["source"]["sha256"]):
        raise ValueError("reviewed source must be hash-verified and training-only")
    image = cv2.imread(str(source))
    vertices, faces = helper.native_base_mesh()
    holes = np.array([r["native_xyz_m"] for r in helper.foot_holes(vertices, faces)])
    holes = holes[holes[:, 1] * foot["matched_nominal_wing_sign"] > 0]
    pixels = helper.detect_holes(image)
    corners = np.asarray(mat["virtual_corners_source_px"])
    groups = {name: np.asarray(points, float) for name, points in CONTOUR_CHECKS.items()}
    all_points = np.concatenate(list(groups.values()))
    K = np.asarray(mat["planar_fit"]["K"])
    basin_scores, tiles = [], []
    for index, basin in enumerate(foot["distinct_alignment_basins"]):
        transform = np.asarray(basin["T_optical_base_early"])
        error, contours = contour_distances(vertices, faces, K, transform, all_points)
        offset, group_scores = 0, {}
        for name, points in groups.items():
            group_scores[name] = helper.summarize(error[offset:offset + len(points)])
            offset += len(points)
        basin_scores.append({"original_hole_fit_rank": index, "wing": basin["wing_sign"],
                             "xy_yaw_radians": basin["xy_yaw_radians"],
                             "original_hole_trimmed_score_mm2": basin["trimmed_score_mm2"],
                             "contour_error_px": helper.summarize(error),
                             "per_group": group_scores})
        canvas = image.copy()
        cv2.drawContours(canvas, contours, -1, (0, 0, 255), 1)
        for point in all_points:
            cv2.circle(canvas, tuple(point.astype(int)), 2, (0, 255, 0), -1)
        cv2.putText(canvas, f"basin {index}", (145, 220), 0, .7, (0, 0, 255), 2)
        tiles.append(canvas[200:420, 130:400])
    late_matrix = np.asarray(static["image_similarity_models"]["without_fixed_base_foot"]
                             ["matrix_reference_to_late"])
    samples = []
    for case in scenarios(corners):
        camera = fit_with_intrinsic_assumptions(
            case.get("corners", corners), case.get("lengths", [.305, .406]),
            mat["planar_fit"], center=case.get("center", [640., 360.]),
            aspect=case.get("aspect", 1.), radial=case.get("radial", 0.))
        thickness = case.get("mat_thickness", 0.)
        x, transform, errors = refit_foot(pixels + case.get("foot_shift", [0., 0.]),
                                         holes, camera, foot["xy_yaw_radians"],
                                         mat_thickness=thickness)
        # Image similarity factorization also works for a rectangular-pixel K:
        # use RQ decomposition rather than the square-pixel helper.
        H = np.eye(3)
        H[:2] = late_matrix
        upper, orthogonal = cv2.RQDecomp3x3(H @ camera["K"])[1:3]
        late = transform.copy()
        late[:3] = orthogonal @ transform[:3]
        contour_error, _ = contour_distances(vertices, faces, camera["K"], transform,
                                             all_points, camera["distortion_coefficients"])
        samples.append({"name": case["name"], "family": case["family"],
                        "assumptions": {k: v for k, v in case.items() if k not in ("name", "family")},
                        "mat_fit_rms_px": camera["mat_fit_rms_px"],
                        "foot_nearest_error_mm": helper.summarize(errors * 1000),
                        "contour_check_error_px": helper.summarize(contour_error),
                        "xy_yaw_radians": x, "mat_thickness_m": thickness,
                        "K_early": camera["K"], "T_optical_base_early": transform,
                        "K_late": upper, "T_optical_base_late": late,
                        "distortion_coefficients": camera["distortion_coefficients"],
                        "camera_position_native_base_m": -transform[:3, :3].T @ transform[:3, 3]})
    positions = np.array([s["camera_position_native_base_m"] for s in samples])
    result = {"schema": "v2_conditional_camera_sensitivity_v1", "physical_calibration_accepted": False,
              "automatically_adopted": False, "training_only": True,
              "arm_states_used": False, "reserved_or_validation_images_used": False,
              "source": foot["source"],
              "contour_check": {"manual_source_pixels": CONTOUR_CHECKS,
                                "assumed_annotation_uncertainty_px": 3.,
                                "used_to_refit_camera_or_foot": False,
                                "note": "Occlusion-free reviewed fixed-base outer edges. The same training image was inspected previously, so this is an exploratory spatial cross-check, not untouched physical acceptance. Rasterization adds <=~1px error. The score uses all14 pixels, with no trimming.",
                                "basins": basin_scores},
              "sample_method": "Deterministic one-at-a-time and two combined perturbations. No probability distribution or confidence coverage is asserted. Every foot refit remains in the original selected wing/yaw basin and uses all21 detected spots with the original0.5mm soft-L1 loss. Contour checks never select perturbations or alter fits.",
              "scenario_ranges_are_assumptions": True,
              "camera_position_native_base_coordinate_range_m": np.c_[positions.min(0), positions.max(0)],
              "samples": samples,
              "remaining_unbounded": ["Physical CAD/printed-base mismatch and unknown point identity",
                                      "Higher distortion terms, tangential distortion and rolling shutter",
                                      "Unknown mat planarity and source exposure/state synchronization",
                                      "Image-regime transform variation beyond the fixed training similarity"],
              "provenance_sha256": {str(p.relative_to(ROOT)): helper.sha256(p)
                                    for p in (Path(__file__).resolve(), STATIC, MAT, source, split_path,
                                              ROOT / "scripts/diag_v2_static_base_camera.py",
                                              helper.MODEL.parent / "assets/base_so101_v2.stl")}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, default=json_default, allow_nan=False) + "\n")
    overlay = out.with_suffix(".jpg")
    if overlay.exists():
        raise FileExistsError(overlay)
    cv2.imwrite(str(overlay), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/v2/camera_uncertainty_20261023.json")
    result = run(parser.parse_args().out)
    print(json.dumps({"samples": len(result["samples"]),
                      "contour_basins": result["contour_check"]["basins"],
                      "position_range": result["camera_position_native_base_coordinate_range_m"]},
                     default=json_default))


if __name__ == "__main__":
    main()
