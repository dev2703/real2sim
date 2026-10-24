"""Score nominal fixed-foot yaw aliases against new image-model masks.

This is a contour cross-check of pre-existing candidates, not physical identity
verification. A visible-part segmentation and an unoccluded CAD top surface have
different occlusion/thickness semantics; all uncapped errors are retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from real2sim.data.raw import sha256_file
from scripts.diag_v2_multiframe_camera import project
from scripts.diag_v2_static_base_camera import base_transform, native_base_mesh


def projected_top_surface(vertices, faces, *, K, T_optical_base, shape_hw, wing_sign):
    """Rasterize only exact nominal foot-top triangles; no renderer or model mutation."""
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    triangle = vertices[faces]
    selected = np.all(np.abs(triangle[:, :, 2] - .0151) < 1e-6, axis=1)
    selected &= np.mean(triangle[:, :, 1], axis=1) * wing_sign > 0
    if wing_sign not in (-1, 1) or not selected.any():
        raise ValueError("one nominal foot wing with exact top-surface faces is required")
    camera = vertices @ np.asarray(T_optical_base)[:3, :3].T + np.asarray(T_optical_base)[:3, 3]
    if np.any(camera[faces[selected], 2] <= 0):
        raise ValueError("projected foot is behind camera")
    pixels = project(vertices, K, T_optical_base)
    mask = np.zeros(shape_hw, np.uint8)
    for face in faces[selected]:
        cv2.fillConvexPoly(mask, np.rint(pixels[face]).astype(np.int32), 1)
    return mask.astype(bool)


def boundary(mask):
    if np.asarray(mask).ndim != 2:
        raise ValueError("binary source-resolution mask required")
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty foot segmentation or projected contour")
    # Retain every visible component instead of selecting whichever fits CAD best.
    return np.concatenate([contour.reshape(-1, 2) for contour in contours]).astype(float)


def summarize(values):
    return {"count": len(values), "mean_px": float(np.mean(values)),
            "median_px": float(np.median(values)), "p95_px": float(np.percentile(values, 95)),
            "rms_px": float(np.sqrt(np.mean(values**2))), "max_px": float(np.max(values))}


def compare_masks(observed, projected):
    if np.shape(observed) != np.shape(projected):
        raise ValueError("observed and nominal masks must share source resolution")
    a, b = boundary(observed), boundary(projected)
    ab = cKDTree(b).query(a)[0]
    ba = cKDTree(a).query(b)[0]
    observed, projected = np.asarray(observed, bool), np.asarray(projected, bool)
    return {"observed_to_nominal": summarize(ab), "nominal_to_observed": summarize(ba),
            "symmetric_uncapped": summarize(np.r_[ab, ba]),
            "mask_iou": float(np.sum(observed & projected) / np.sum(observed | projected))}


def run(mask_document, camera_path, prior_path, out):
    if out.exists():
        raise FileExistsError(out)
    document = json.loads(mask_document.read_text())
    cameras, prior = json.loads(camera_path.read_text()), json.loads(prior_path.read_text())
    if document.get("schema") != "v2_fixed_foot_model_masks_v1":
        raise ValueError("unsupported fixed-foot mask document")
    split_path = Path("results/v2/split.json")
    split = json.loads(split_path.read_text())
    vertices, faces = native_base_mesh()
    aliases = prior["base_foot"]["distinct_alignment_basins"]
    output_rows = []
    out.mkdir(parents=True)
    for row in document["frames"]:
        if row["episode"] in split["reserved"] or row["episode"] not in split["train"] + split["val"]:
            raise ValueError("only train/validation source frames may enter the contour check")
        for key in ("mask", "source"):
            if sha256_file(Path(row[f"{key}_path"])) != row[f"{key}_sha256"]:
                raise ValueError(f"changed {key} artifact")
        mask = np.load(row["mask_path"], allow_pickle=False)
        if mask.dtype != np.bool_ or mask.shape != (720, 1280):
            raise ValueError("foot masks must be boolean1280x720 source-resolution arrays")
        camera = cameras["regimes"][row["camera_regime"]]["baseline"]
        scores = []
        source = cv2.imread(row["source_path"])
        if source is None or source.shape[:2] != mask.shape:
            raise ValueError("source image shape differs from foot mask")
        for i, alias in enumerate(aliases):
            transform = np.asarray(camera["T_optical_plane"]) @ base_transform(np.asarray(alias["xy_yaw_radians"]))
            projection = projected_top_surface(vertices, faces, K=camera["K"],
                T_optical_base=transform, shape_hw=mask.shape, wing_sign=alias["wing_sign"])
            score = {"prior_basin_index": i, "xy_yaw_radians": alias["xy_yaw_radians"],
                     "wing_sign": alias["wing_sign"], **compare_masks(mask, projection)}
            scores.append(score)
            overlay = source.copy()
            for footprint, color in ((mask, (0, 220, 0)), (projection, (0, 0, 255))):
                points = boundary(footprint).astype(int)
                overlay[points[:, 1], points[:, 0]] = color
            cv2.imwrite(str(out / f"ep{row['episode']:02d}_f{row['frame']:04d}_basin{i}.png"), overlay)
        output_rows.append({**row, "scores": scores})
    report = {"schema": "v2_model_fixed_foot_contour_check_v1", "physical_calibration_accepted": False,
              "robot_joint_readings_used": False, "production_inputs_changed": False,
              "all_candidates_retained_without_adoption": True, "reserved_episodes_used": [],
              "native_foot_top_plane_z_m": .0151,
              "method": "Project the exact pinned CAD top triangles for each previously declared wing/XY/yaw basin through the new conditional mat camera. Compare every model-mask external component with uncapped bidirectional nearest-contour errors; report IoU separately. No frame pixels refit a base pose.",
              "frames": output_rows,
              "limitations": ["SAM-visible foot material is not an independently registered CAD identity.",
                  "Nominal top-plane projection omits visible side thickness, occlusion by the holder/arm/clamp and print/CAD mismatch.",
                  "Mat camera and mat/base height remain conditional assumptions. A better contour score cannot grant physical calibration or joint-zero acceptance.",
                  "Prior candidate basins came from an earlier explored training foot image. New capture scores can reject inconsistency but do not turn ambiguous matches into metrology."],
              "provenance_sha256": {str(path): sha256_file(path) for path in (mask_document, camera_path, prior_path, Path(__file__), split_path,
                  Path("assets/menagerie/robotstudio_so101/so101.xml"), Path("assets/menagerie/robotstudio_so101/assets/base_so101_v2.stl"))}}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=Path("outputs/v2/static_base_camera_20261023/report.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.masks, args.camera, args.prior, args.out)
    print(json.dumps({"frames": len(report["frames"]), "physical_calibration_accepted": False}))
