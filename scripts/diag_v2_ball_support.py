"""Infer conditional support height from sphere silhouettes, exposing scale ambiguity.

No scene or training input is modified. Camera hypotheses and nominal ball radius
are assumptions; a visually good silhouette fit is not physical height calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from real2sim.config import load_yaml
from real2sim.data.raw import sha256_file
from real2sim.geometry.planar_camera import backproject_to_plane


def sphere_center(pixels, intrinsic, radius, distortion=None):
    """Fit the exact tangent cone of a known-radius sphere in optical coordinates."""
    points, k = np.asarray(pixels, float), np.asarray(intrinsic, float)
    if radius <= 0 or not np.isfinite(radius) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("finite positive radius and Nx2 boundary required")
    if len(points) < 8 or not np.isfinite(points).all():
        raise ValueError("at least eight finite silhouette points required")
    if distortion is not None and np.any(np.asarray(distortion) != 0):
        normalized = cv2.undistortPoints(points[:, None], k, np.asarray(distortion, float))[:, 0]
        rays = np.c_[normalized, np.ones(len(points))]
    else:
        rays = np.c_[points, np.ones(len(points))] @ np.linalg.inv(k).T
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    mean = rays.mean(axis=0)
    mean /= np.linalg.norm(mean)
    angle = np.median(np.arccos(np.clip(rays @ mean, -1, 1)))
    initial = mean * radius / np.sin(max(angle, 1e-4))

    def residual(center):
        distance = np.linalg.norm(center)
        observed = np.arccos(np.clip(rays @ center / distance, -1, 1))
        return (observed - np.arcsin(np.clip(radius / distance, 0, .999999))) * k[0, 0]

    fit = least_squares(residual, initial, bounds=([-10, -10, radius * 1.01], [10, 10, 10]),
                        loss="soft_l1", f_scale=1., max_nfev=200)
    if not fit.success:
        raise RuntimeError("sphere tangent-cone fit did not converge")
    return fit.x, {"boundary_points": len(points), "rms_equivalent_pixels": float(
        np.sqrt(np.mean(residual(fit.x)**2))), "cost": float(fit.cost)}


def ball_boundary(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([20, 80, 60]), np.array([50, 255, 255]))
    mask[:100] = 0
    mask[:, :400] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    choices = []
    for contour in contours:
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        if 1000 < area < 30000 and perimeter > 0 and 4 * np.pi * area / perimeter**2 > .65:
            choices.append((area, hull))
    if not choices:
        raise ValueError("no sufficiently visible round yellow ball")
    return max(choices, key=lambda x: x[0])[1].reshape(-1, 2).astype(float)


def support_height(center_optical, transform, radius, table_z=-.0024):
    pose = np.asarray(transform)
    native = pose[:3, :3].T @ (center_optical - pose[:3, 3])
    return native, float(native[2] - table_z - radius)


def image_scene_proxies(camera_document, silhouette_document):
    """Predeclared scene hypotheses using the visible off-center support footprint."""
    candidate = camera_document["candidates"]["early"]
    k, pose = np.asarray(candidate["K"]), np.asarray(candidate["T_optical_native_base"])
    capture = next(r for r in silhouette_document["captures"]
                   if r["episode"] == 0 and r["source_frame"] == 200)
    center_native = np.asarray(capture["nominal"]["ball_center_native_m"])
    center_optical = pose[:3, :3] @ center_native + pose[:3, 3]
    pixel = k @ center_optical
    pixel = pixel[:2] / pixel[2]
    # Visible outer paper edges, read before any retention outcome. Approximate±5px.
    quad = np.array([[711., 212.], [856., 211.], [887., 337.], [713., 349.]])
    rim = np.array([[846., 487.]])  # approximate source rim center, ±8px.
    cases = []
    for height in (0., .005, .010, .020):
        footprint = backproject_to_plane(quad, K=k, T_optical_plane=pose,
                                        plane_z_m=-.0024 + height)
        center, size, yaw = cv2.minAreaRect(footprint[:, :2].astype(np.float32))
        ball = backproject_to_plane(pixel[None], K=k, T_optical_plane=pose,
                                   plane_z_m=-.0024 + height + .0335)[0]
        bowl = backproject_to_plane(rim, K=k, T_optical_plane=pose,
                                   plane_z_m=-.0024 + .030)[0]
        cases.append({"case_id": f"image_support_{round(height * 1000):02d}mm",
            "ball_xy": ball[:2].tolist(), "bowl_xy": bowl[:2].tolist(),
            "support": {"type": "box", "height": height, "size_xy": list(size),
                        "center_task_xy": list(center), "yaw_deg": yaw}})
    return {"schema": "image_based_rigid_support_scene_hypotheses_v1",
        "physical_calibration_accepted": False, "source_episode": 0, "source_frame": 200,
        "paper_quad_source_px": quad.tolist(), "paper_edge_uncertainty_assumed_px": 5.,
        "bowl_rim_center_source_px": rim.tolist(), "rim_center_uncertainty_assumed_px": 8.,
        "assumed_bowl_rim_height_m": .030, "assumed_ball_radius_m": .0335,
        "task_frame": "native_so101_base", "cases": cases,
        "limitations": "All object positions change coherently with this camera hypothesis. "
        "Rigid enclosing rectangle is a proxy for irregular/compliant paper. Heights are "
        "predeclared sensitivities, never selected by retention success. Fixed image cases "
        "with zero placement jitter do not produce independent samples by changing seed."}


def diagnose(camera_path, radius=.0335):
    evidence = json.loads(camera_path.read_text())
    samples = evidence["samples"]
    task_path = Path("configs/v2/task.exploratory_20261022.yaml")
    legacy_xy = np.asarray(load_yaml(task_path)["ball"]["start_region"]["center_task_xy"])
    rows = []
    for episode, frame in ((0, 0), (0, 200), (6, 0), (23, 0), (31, 0), (36, 0)):
        path = Path(f"outputs/v2/geometry_alignment_20261021/ep{episode:02d}_f{frame:04d}.jpg")
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        points = ball_boundary(image)
        regime = "early" if episode < 7 else "late"
        estimates, excluded = [], []
        for sample in samples:
            k = np.asarray(sample[f"K_{regime}"])
            if abs(k[0, 1]) > 1e-7:
                excluded.append(sample["name"])
                continue  # OpenCV's radial inverse does not implement skew.
            transform = np.asarray(sample[f"T_optical_base_{regime}"])
            distortion = sample.get("distortion_coefficients")
            center, score = sphere_center(points, k, radius, distortion)
            # Silhouette determines center/radius: scaling both leaves every ray tangent.
            for changed_radius in (radius - .001, radius, radius + .001):
                native, height = support_height(center * changed_radius / radius, transform,
                                                changed_radius, -.0024)
                estimates.append({"camera": sample["name"], "family": sample["family"],
                    "assumed_radius_m": changed_radius, "ball_center_native_m": native.tolist(),
                    "support_height_mm": height * 1000,
                    "support_height_above_mat_mm": 1000 * (height - sample.get("mat_thickness_m", 0)),
                    **score})
        nominal = next(r for r in estimates if r["camera"] == "nominal"
                       and r["assumed_radius_m"] == radius)
        heights = [r["support_height_mm"] for r in estimates]
        row = {"episode": episode, "source_frame": frame, "source_path": str(path),
               "source_sha256": sha256_file(path), "nominal": nominal,
               "assumption_height_range_mm": [min(heights), max(heights)],
               "excluded_camera_scenarios_for_nonzero_skew": excluded,
               "nominal_ball_xy_difference_from_provisional_task_mm": float(np.linalg.norm(
                   np.asarray(nominal["ball_center_native_m"])[:2] - legacy_xy) * 1000),
               "assumption_scenarios": estimates}
        rows.append(row)
    return {"schema": "conditional_ball_support_from_silhouette_v1",
        "physical_measurement_accepted": False, "production_inputs_changed": False,
        "radius_assumptions_m": [radius - .001, radius, radius + .001], "captures": rows,
        "method": "Exact optical ray tangency cone, yellow convex boundary, robust angular residual. "
                  "Each camera hypothesis is fixed before fitting ball center; no arm FK used.",
        "limitations": ["Camera ranges are assumed scenarios, not confidence intervals.",
            "Ball radius is nominal; sphere size and optical distance have an exact scale ambiguity.",
            "Segmentation, fuzzy tennis surface, occlusion and JPEG compression add systematic error.",
            "Rigid support is only a scene proxy; shape, thickness, compliance and friction are unmeasured.",
            "Capture frame identities use prior saved JPEG provenance; physical exposure timing is unknown.",
            "No support height is selected by simulated grasp success; no task config is adopted."],
        "input_sha256": {str(camera_path): sha256_file(camera_path),
                         str(task_path): sha256_file(task_path),
                         str(Path(__file__)): sha256_file(Path(__file__))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=Path, default=Path("results/v2/camera_uncertainty_20261023.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = diagnose(args.camera)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in report["captures"]:
        print(row["episode"], row["nominal"]["support_height_mm"], row["assumption_height_range_mm"])


if __name__ == "__main__":
    main()
