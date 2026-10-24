"""Separate static-image changes, mat-pattern motion and model-edge disagreement.

This uses independent fixed regions to estimate image similarities, then measures
the mat relative to those references. It never adopts metric camera calibration,
warps policy inputs, reads reserved images, or uses robot joint/FK labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from real2sim.data.raw import sha256_file

STATIC_POLYGONS = {
    "fixed_base_foot": [[175, 353], [205, 340], [335, 348], [350, 385], [300, 405], [190, 405]],
    "lower_table_clamp": [[58, 455], [90, 442], [150, 443], [160, 463], [144, 502], [116, 520], [84, 495]],
    "cabinet_handle": [[535, 80], [700, 80], [700, 120], [535, 120]],
}
# A sibling CAD audit associates this region with a fixed base-body motor holder.
# It is withheld from global fitting and retained as an independent image check;
# region membership alone does not establish exact physical point identities.
HOUSING_POLYGON = [[165, 238], [235, 238], [235, 317], [165, 317]]
PARAMETERS = {
    "sift_features": 5000, "sift_contrast": .015, "lowe_ratio": .7,
    "maximum_descriptor_displacement_px": 60., "forward_backward_max_px": 1.,
    "descriptor_to_lk_max_px": 4., "patch_ncc_min": .65,
    "patch_size": 21, "patch_std_min": 4., "ransac_threshold_px": 2.,
    "minimum_region_matches": 4, "minimum_global_supported_regions": 2,
    "mat_mask_erosion_px": 4,
}


def transform(points, matrix):
    points = np.asarray(points, float)
    return np.c_[points, np.ones(len(points))] @ np.asarray(matrix, float).T


def compose(a, b):
    return (np.vstack([a, [0., 0., 1.]]) @ np.vstack([b, [0., 0., 1.]]))[:2]


def inverse(matrix):
    return np.linalg.inv(np.vstack([matrix, [0., 0., 1.]]))[:2]


def mask_for(shape, polygon):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 255)
    return mask


def point_in_mask(point, mask):
    x, y = np.rint(point).astype(int)
    return bool(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x])


def patch_ncc(a, b, p, q):
    half = PARAMETERS["patch_size"] // 2 + 1
    for image, point in ((a, p), (b, q)):
        if min(point) < half or point[0] >= image.shape[1] - half or point[1] >= image.shape[0] - half:
            return None
    size = (PARAMETERS["patch_size"],) * 2
    x = cv2.getRectSubPix(a, size, tuple(map(float, p))).astype(float)
    y = cv2.getRectSubPix(b, size, tuple(map(float, q))).astype(float)
    if min(x.std(), y.std()) < PARAMETERS["patch_std_min"]:
        return None
    x, y = x - x.mean(), y - y.mean()
    return float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))


def feature_matches(reference, target, reference_masks, target_masks=None):
    """Mutual descriptors followed by initialized LK, cycle and photometric checks."""
    cv2.setRNGSeed(0)
    a, b = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in (reference, target))
    detector = cv2.SIFT_create(nfeatures=PARAMETERS["sift_features"], contrastThreshold=PARAMETERS["sift_contrast"], edgeThreshold=15)
    union = np.maximum.reduce(list(reference_masks.values()))
    ka, da = detector.detectAndCompute(a, union)
    kb, db = detector.detectAndCompute(b, None)
    if da is None or db is None:
        return []
    matcher = cv2.BFMatcher()
    forward, backward = matcher.knnMatch(da, db, k=2), matcher.knnMatch(db, da, k=2)
    reverse = {m.queryIdx: m.trainIdx for pair in backward if len(pair) == 2
               for m, n in [pair] if m.distance < PARAMETERS["lowe_ratio"] * n.distance}
    candidates, seen = [], set()
    for pair in forward:
        if len(pair) != 2:
            continue
        m, n = pair
        p, q = np.asarray(ka[m.queryIdx].pt), np.asarray(kb[m.trainIdx].pt)
        region = next((name for name, mask in reference_masks.items() if point_in_mask(p, mask)), None)
        key = tuple(np.rint(p / 2).astype(int))
        if (region is None or key in seen or m.distance >= PARAMETERS["lowe_ratio"] * n.distance
                or reverse.get(m.trainIdx) != m.queryIdx
                or np.linalg.norm(q - p) > PARAMETERS["maximum_descriptor_displacement_px"]):
            continue
        seen.add(key)
        candidates.append({"region": region, "reference_px": p.tolist(), "descriptor_target_px": q.tolist(),
                           "descriptor_distance": float(m.distance)})
    if not candidates:
        return []
    p = np.asarray([r["reference_px"] for r in candidates], np.float32).reshape(-1, 1, 2)
    q0 = np.asarray([r["descriptor_target_px"] for r in candidates], np.float32).reshape(-1, 1, 2)
    lk = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
              flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=1e-4)
    q, sf, _ = cv2.calcOpticalFlowPyrLK(a, b, p, q0.copy(), **lk)
    back, sb, _ = cv2.calcOpticalFlowPyrLK(b, a, q, p.copy(), **lk)
    for i, row in enumerate(candidates):
        query, source = q[i, 0], p[i, 0]
        fb = float(np.linalg.norm(back[i, 0] - source))
        adjustment = float(np.linalg.norm(query - q0[i, 0]))
        ncc = patch_ncc(a, b, source, query) if np.isfinite(query).all() else None
        reasons = []
        if not sf[i, 0] or not sb[i, 0] or not np.isfinite(query).all():
            reasons.append("flow_failed")
        if fb > PARAMETERS["forward_backward_max_px"]:
            reasons.append("flow_cycle")
        if adjustment > PARAMETERS["descriptor_to_lk_max_px"]:
            reasons.append("descriptor_flow_disagreement")
        if ncc is None or ncc < PARAMETERS["patch_ncc_min"]:
            reasons.append("photometric_patch")
        if target_masks is not None and not point_in_mask(query, target_masks[row["region"]]):
            reasons.append("outside_target_object_mask")
        row.update(target_px=query.astype(float).tolist(), forward_backward_error_px=fb,
                   descriptor_refinement_px=adjustment, patch_ncc=ncc,
                   accepted=not reasons, rejection_reasons=reasons)
    return candidates


def summarize_vectors(points, targets):
    if len(points) == 0:
        return {"n": 0}
    delta = np.asarray(targets) - np.asarray(points)
    lengths = np.linalg.norm(delta, axis=1)
    return {"n": len(delta), "median_dx_dy_px": np.median(delta, axis=0).tolist(),
            "median_magnitude_px": float(np.median(lengths)), "p95_magnitude_px": float(np.percentile(lengths, 95)),
            "rms_magnitude_px": float(np.sqrt(np.mean(lengths**2)))}


def fit_similarity(rows, regions=None):
    accepted = [r for r in rows if r["accepted"] and (regions is None or r["region"] in regions)]
    if len(accepted) < PARAMETERS["minimum_region_matches"]:
        return {"status": "insufficient_matches", "matches": len(accepted)}
    p, q = (np.asarray([r[key] for r in accepted], float) for key in ("reference_px", "target_px"))
    if np.prod(np.ptp(p, axis=0)) < 40:
        return {"status": "insufficient_spatial_span", "matches": len(accepted)}
    cv2.setRNGSeed(0)
    matrix, mask = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
        ransacReprojThreshold=PARAMETERS["ransac_threshold_px"], maxIters=5000, confidence=.999, refineIters=30)
    if matrix is None:
        return {"status": "fit_failed", "matches": len(accepted)}
    inliers = mask[:, 0].astype(bool)
    per_region = {}
    for region in sorted({r["region"] for r in accepted}):
        selected = np.asarray([r["region"] == region for r in accepted])
        retained = selected & inliers
        per_region[region] = {"accepted_matches": int(selected.sum()), "fit_inliers": int(retained.sum()),
            "all_match_residual": summarize_vectors(transform(p[selected], matrix), q[selected]),
            "inlier_residual": summarize_vectors(transform(p[retained], matrix), q[retained])}
    supported = sum(r["fit_inliers"] >= PARAMETERS["minimum_region_matches"] for r in per_region.values())
    return {"status": "fit", "matrix_reference_to_target": matrix.tolist(), "matches": len(accepted),
            "inliers": int(inliers.sum()), "supported_regions": supported,
            "scale": float(np.hypot(matrix[0, 0], matrix[1, 0])),
            "rotation_deg": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
            "direct_displacement": summarize_vectors(p[inliers], q[inliers]),
            "inlier_residual": summarize_vectors(transform(p[inliers], matrix), q[inliers]),
            "per_region": per_region}


def edges_to_corners(samples):
    lines = []
    for edge in ("top", "right", "bottom", "left"):
        p = np.asarray(samples[edge], float)
        center = p.mean(0)
        _, _, vt = np.linalg.svd(p - center, full_matrices=False)
        lines.append(np.r_[vt[-1], -vt[-1] @ center])
    corners = []
    for i, j in ((3, 0), (0, 1), (1, 2), (2, 3)):
        p = np.cross(lines[i], lines[j])
        corners.append(p[:2] / p[2])
    return np.asarray(corners)


def load_sources():
    segmentation_paths = [Path("outputs/v2/grounded_geometry_20261024/report.json"),
                          Path("outputs/v2/fresh_camera_confirmation_models_20261024/report.json")]
    annotation_paths = [Path("outputs/v2/model_mat_edges_20261024_v4/annotations.json"),
                        Path("outputs/v2/fresh_camera_confirmation_edges_20261024/annotations.json")]
    corners = {}
    for path in annotation_paths:
        document = json.loads(path.read_text())
        for row in document["frames"]:
            if row["role"] != "excluded":
                corners[row["episode"], row["frame"]] = edges_to_corners(row["mat_edge_samples_source_px"])
    rows = []
    for path in segmentation_paths:
        for row in json.loads(path.read_text())["frames"]:
            if row["camera_key"] == "observation.images.iphone":
                rows.append({**row, "model_report_path": str(path), "corners": corners.get((row["episode"], row["frame"]))})
    return rows, segmentation_paths + annotation_paths


def draw(reference, target, rows, path):
    image = np.hstack([reference, target])
    for row in rows:
        p = tuple(np.rint(row["reference_px"]).astype(int))
        q = np.rint(row["target_px"]).astype(int) + [reference.shape[1], 0]
        color = (50, 220, 50) if row["accepted"] else (40, 40, 230)
        cv2.circle(image, p, 2, color, 1)
        cv2.line(image, p, tuple(q), color, 1)
    cv2.imwrite(str(path), image)


def run(out):
    if out.exists():
        raise FileExistsError(out)
    sources, dependencies = load_sources()
    split_path = Path("results/v2/split.json")
    split = json.loads(split_path.read_text())
    if any(row["episode"] in split["reserved"] for row in sources):
        raise ValueError("reserved captures cannot enter this diagnostic")
    reference_source = next(row for row in sources if row["episode"] == 0 and row["frame"] == 98)
    reference = cv2.imread(reference_source["path"])
    static_masks = {k: mask_for(reference.shape, p) for k, p in STATIC_POLYGONS.items()}
    all_masks = {**static_masks, "heldout_base_housing": mask_for(reference.shape, HOUSING_POLYGON)}
    ref_mat = np.load(reference_source["mask_path"])["black_mat"]
    k = PARAMETERS["mat_mask_erosion_px"]
    ref_mat = cv2.erode(ref_mat.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    out.mkdir(parents=True)
    report = {"schema": "v2_relative_scene_motion_v1", "physical_calibration_accepted": False,
              "production_inputs_changed": False, "robot_fk_used": False, "reserved_episodes_used": [],
              "reference": {key: reference_source[key] for key in ("episode", "frame", "path", "sha256")},
              "parameters": PARAMETERS, "static_reference_polygons": STATIC_POLYGONS,
              "heldout_housing_polygon": HOUSING_POLYGON,
              "definition": "Static source-pixel similarity is a capture-coordinate diagnostic, not a physical camera pose. Mat-pattern motion is measured separately from model-derived virtual edge corners.",
              "source_sha256": {str(p): sha256_file(p) for p in [Path(__file__), split_path, *dependencies]},
              "comparisons": []}
    for source in sources:
        if source["episode"] == 0 and source["frame"] == 98:
            continue
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("source image hash mismatch")
        target = cv2.imread(source["path"])
        matches = feature_matches(reference, target, all_masks)
        global_fit = fit_similarity(matches, STATIC_POLYGONS)
        regions = {name: fit_similarity(matches, [name]) for name in all_masks}
        row = {key: source[key] for key in ("episode", "frame", "role", "partition", "path", "sha256", "camera_regime")}
        row.update(static_global=global_fit, static_regions=regions, static_matches=matches)
        target_mat = np.load(source["mask_path"])["black_mat"]
        target_mat = cv2.erode(target_mat.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        mat_matches = feature_matches(reference, target, {"mat_pattern": ref_mat}, {"mat_pattern": target_mat})
        mat_fit = fit_similarity(mat_matches)
        row.update(mat_pattern_fit=mat_fit, mat_pattern_matches=mat_matches)
        if global_fit["status"] == "fit":
            matrix = np.asarray(global_fit["matrix_reference_to_target"])
            housing = [m for m in matches if m["region"] == "heldout_base_housing" and m["accepted"]]
            row["heldout_housing_residual"] = summarize_vectors(
                transform(np.asarray([m["reference_px"] for m in housing]).reshape(-1, 2), matrix),
                np.asarray([m["target_px"] for m in housing]).reshape(-1, 2))
            mat_good = [m for m in mat_matches if m["accepted"]]
            p = np.asarray([m["reference_px"] for m in mat_good]).reshape(-1, 2)
            q = np.asarray([m["target_px"] for m in mat_good]).reshape(-1, 2)
            row["mat_pattern_relative_to_static"] = summarize_vectors(p, transform(q, inverse(matrix)))
            if source["corners"] is not None:
                row["model_corners_relative_to_static"] = summarize_vectors(reference_source["corners"], transform(source["corners"], inverse(matrix)))
            if mat_fit["status"] == "fit":
                relative = compose(inverse(matrix), np.asarray(mat_fit["matrix_reference_to_target"]))
                row["mat_relative_similarity"] = relative.tolist()
                if source["corners"] is not None:
                    row["model_corners_error_vs_direct_mat_pattern"] = summarize_vectors(
                        transform(reference_source["corners"], np.asarray(mat_fit["matrix_reference_to_target"])), source["corners"])
            row["stabilization_evidence"] = "supported" if global_fit["supported_regions"] >= PARAMETERS["minimum_global_supported_regions"] else "limited_static_spatial_support"
        report["comparisons"].append(row)
        draw(reference, target, matches, out / f"ep{source['episode']:03d}_f{source['frame']:04d}_static.jpg")
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(source["episode"], source["frame"], "static", global_fit.get("inliers"), global_fit.get("supported_regions"),
              "mat", mat_fit.get("inliers"), row.get("mat_pattern_relative_to_static"), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.out)
