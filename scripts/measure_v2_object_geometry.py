"""Conditional object dimensions from independently image-fitted camera hypotheses.

No robot FK or rollout outcome enters these estimates. Metric scale is conditional
on supplied mat dimensions and nominal tennis-ball radius, never invented by a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from diag_v2_ball_support import sphere_center

from real2sim.data.raw import sha256_file
from real2sim.geometry.image_measurements import (
    bowl_rim_candidates,
    ellipse_points,
    fit_plane_circle,
)
from real2sim.geometry.planar_camera import backproject_to_plane

# Visual-model review of the preselected ORIGINAL captures, before fitting dimensions.
# Remaining images show ball occlusion by jaws or bowl and cannot supply a full tangent cone.
FULL_BALL_ON_PAPER = {(4, 179), (0, 98), (26, 150)}


def camera_cases(regime):
    baseline = regime["baseline"]
    cases = [{"scenario": "baseline", **baseline}, *regime["sensitivity"]["cases"]]
    result = []
    for row in cases:
        if not row.get("optimizer_converged", True) or row.get("focal_at_bound", False):
            continue
        # Exact spherical tangent implementation includes OpenCV distortion handling.
        transform = row["T_optical_plane"] if "T_optical_plane" in row else row["T_optical_mat"]
        result.append((row["scenario"], np.asarray(row["K"]), np.asarray(transform),
                       row.get("distortion_coefficients")))
    return result


def run(segmentation_path, camera_path, out):
    if out.exists():
        raise FileExistsError(out)
    segmentation = json.loads(segmentation_path.read_text())
    cameras = json.loads(camera_path.read_text())
    rows, previews = [], []
    for frame in segmentation["frames"]:
        if not frame["camera_key"].endswith("iphone"):
            continue
        path = Path(frame["path"])
        if sha256_file(path) != frame["sha256"]:
            raise ValueError("source frame checksum mismatch")
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        regime = cameras["regimes"][frame["camera_regime"]]
        cases = camera_cases(regime)
        result = {key: frame[key] for key in ("episode", "frame", "role", "partition", "camera_regime", "path", "sha256")}
        result["camera_acceptance"] = "conditional only; shared-pose held-out residuals exceed the declared3px premise"
        # Mat coordinates TL→TR then TL→BL have +z pointing away from the camera.
        result["mat_frame_convention"] = "x top-left→top-right, y top-left→bottom-left; visible objects have negative z; height above mat=-z"
        identity = frame["episode"], frame["frame"]
        if identity in FULL_BALL_ON_PAPER:
            mask = np.load(frame["mask_path"])["tennis_ball"]
            contour = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)[:, 0]
            boundary = contour[::max(1, len(contour) // 180)].astype(float)
            estimates = []
            for name, K, T, distortion in cases:
                center, score = sphere_center(boundary, K, .0335, distortion)
                for radius in (.0325, .0335, .0345):
                    optical = center * radius / .0335
                    plane = T[:3, :3].T @ (optical - T[:3, 3])
                    estimates.append({"camera_scenario": name, "radius_assumption_m": radius,
                                      "center_mat_m": plane.tolist(), "support_height_above_mat_mm": float((-plane[2] - radius) * 1000),
                                      **score})
            nominal = next(v for v in estimates if v["camera_scenario"] == "baseline" and v["radius_assumption_m"] == .0335)
            heights = [v["support_height_above_mat_mm"] for v in estimates]
            result["ball_support"] = {"visual_review": "Full visible ball resting on paper; silhouette retained fromSAM2, no robot pose used",
                "baseline": nominal, "declared_assumption_height_range_mm": [min(heights), max(heights)],
                "scenarios": estimates, "physical_height_accepted": False,
                "limitations": "Envelope is not a confidence interval. It inherits failed shared-camera verification, nominal ball-radius scale, segmentation error and mat plane uncertainty; paper thickness is not independently known."}
        else:
            result["ball_support"] = {"excluded": True, "reason": "Visual review: ball occluded by jaw or bowl, or held off support; a complete spherical tangent cone is unavailable"}
        candidates = bowl_rim_candidates(rgb, frame["objects"]["bowl"]["box_source_xyxy"])
        bowl_estimates = []
        for j, ellipse in enumerate(candidates):
            pixels = ellipse_points(ellipse["center_source_px"], ellipse["diameters_px"], ellipse["angle_deg"])
            for name, K, T, distortion in cases:
                if distortion is not None and np.any(np.asarray(distortion) != 0):
                    points = cv2.undistortPoints(pixels[:, None], K, np.asarray(distortion), P=K)[:, 0]
                else:
                    points = pixels
                for height in (.0254, .030, .040):
                    world = backproject_to_plane(points, K=K, T_optical_plane=T, plane_z_m=-height)
                    bowl_estimates.append({"ellipse_index": j, "camera_scenario": name,
                                           "rim_height_assumption_m": height, **fit_plane_circle(world)})
            cv2.ellipse(rgb, (tuple(ellipse["center_source_px"]), tuple(ellipse["diameters_px"]), ellipse["angle_deg"]), (255, 20, 20), 1)
        diameters = [v["diameter_m"] * 1000 for v in bowl_estimates]
        nominal = [v for v in bowl_estimates if v["camera_scenario"] == "baseline" and v["rim_height_assumption_m"] == .030]
        result["bowl_rim"] = {"pixel_candidates": candidates, "baseline_30mm_rim_plane": nominal,
            "declared_assumption_diameter_range_mm": [min(diameters), max(diameters)] if diameters else None,
            "scenarios": bowl_estimates, "operator_inner_plus_wall_outer_diameter_mm": 104.,
            "limitations": "Near-complete observed arcs only; inner/outer rim identity remains ambiguous. Radius and elevation of a parallel circular rim cannot both be independently recovered here. Heights and optical assumptions are declared sensitivities, not measured truth."}
        rows.append(result)
        previews.append((path.stem, rgb))
    out.mkdir(parents=True)
    for name, rgb in previews:
        cv2.imwrite(str(out / f"{name}_rim_candidates.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    report = {"schema": "conditional_source_object_geometry_v1", "physical_calibration_accepted": False,
              "robot_fk_used": False, "reserved_episodes_used": [], "production_geometry_changed": False,
              "source_sha256": {str(p): sha256_file(p) for p in (segmentation_path, camera_path, Path(__file__), Path("src/real2sim/geometry/image_measurements.py"))},
              "reviewer": "Multimodal assistant source-frame review; shape models refine source pixels; annotations are not physical metrology",
              "frames": rows,
              "wrist_camera": {"intrinsics_measured": False,
                  "reason": "The random wrist views contain cropped/occluded objects and model class confusions (ball also labeled bowl); known-radius/optical-pose ambiguities remain. Unverified detections are retained as raw predictions only."},
              "unobservable_parameters": ["absolute world table height without an independently referenced world point", "mass, friction, compliance, contact force/current response", "physical joint sign/zero and wrist optical mount separated from each other without verified robot-to-mat registration", "hidden jaw gap, exactTCP/ball load offset and full release response from these sparse unsynchronized exposure captures"]}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segmentation", type=Path, default=Path("outputs/v2/grounded_geometry_20261024/report.json"))
    p.add_argument("--camera", type=Path, default=Path("outputs/v2/model_multiframe_camera_20261024_v4/report.json"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = run(args.segmentation, args.camera, args.out)
    for frame in report["frames"]:
        support = frame["ball_support"].get("baseline", {}).get("support_height_above_mat_mm")
        diam = [round(v["diameter_m"] * 1000, 1) for v in frame["bowl_rim"]["baseline_30mm_rim_plane"]]
        print(frame["episode"], frame["frame"], "conditional support mm", support, "rim diameters mm", diam)
