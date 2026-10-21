"""Measure v2 iPhone pixel stability using fixed scene features, never joint labels.

This is a 2-D diagnostic, not camera calibration. It cannot distinguish camera motion,
capture cropping, focus changes, or other causes of changed image coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from real2sim.data.raw import RawDataset, sha256_file

# Defined on training episode 0, 1280x720 source pixels. Mat, ball, bowl, tissue,
# robot links, electronics, and loose cables are excluded from the reference mask.
# The fixed mounting foot has repeated holes, so mutual ratio matching and a robust
# similarity fit reject mismatches; all retained coordinates are saved for inspection.
REGIONS = {
    "fixed_base_foot": [[175, 353], [205, 340], [335, 348], [350, 385],
                        [300, 405], [190, 405]],
    "lower_table_clamp": [[58, 455], [90, 442], [150, 443], [160, 463],
                          [144, 502], [116, 520], [84, 495]],
    "cabinet_handle": [[535, 80], [700, 80], [700, 120], [535, 120]],
}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def region_masks(shape):
    if shape[:2] != (720, 1280):
        raise ValueError("the reviewed static masks require 1280x720 source images")
    masks = {}
    for name, polygon in REGIONS.items():
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        masks[name] = mask
    return masks


def feature_region(point, masks):
    x, y = np.rint(point).astype(int)
    for name, mask in masks.items():
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
            return name
    return None


def summarize_displacements(points, targets):
    if not len(points):
        return None
    delta = targets - points
    magnitude = np.linalg.norm(delta, axis=1)
    return {
        "count": len(points), "median_dx_dy_px": np.median(delta, axis=0).tolist(),
        "displacement_px_p50": float(np.median(magnitude)),
        "displacement_px_p95": float(np.percentile(magnitude, 95)),
    }


def match_static(reference, target, masks):
    """Mutual Lowe-ratio SIFT matches, bounded displacement, then robust similarity."""
    cv2.setRNGSeed(0)
    detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.015, edgeThreshold=15)
    k0, d0 = detector.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    k1, d1 = detector.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), None)
    if d0 is None or d1 is None:
        return {"status": "insufficient_features", "matches": []}
    matcher = cv2.BFMatcher()
    forward = matcher.knnMatch(d0, d1, k=2)
    backward = matcher.knnMatch(d1, d0, k=2)
    reverse_good = {m.queryIdx: m.trainIdx for pair in backward if len(pair) == 2
                    for m, n in [pair] if m.distance < .7 * n.distance}
    rows = []
    seen = set()
    for pair in forward:
        if len(pair) != 2:
            continue
        m, n = pair
        p, q = np.asarray(k0[m.queryIdx].pt), np.asarray(k1[m.trainIdx].pt)
        region = feature_region(p, masks)
        if (region is None or m.distance >= .7 * n.distance
                or reverse_good.get(m.trainIdx) != m.queryIdx
                or np.linalg.norm(q - p) > 60):
            continue
        # SIFT may provide several orientations at one location; count it once.
        point_key = tuple(np.round(p, 1))
        if point_key in seen:
            continue
        seen.add(point_key)
        rows.append({"region": region, "reference_px": p.tolist(), "target_px": q.tolist(),
                     "descriptor_distance": float(m.distance)})
    if len(rows) < 4:
        return {"status": "insufficient_matches", "matches": rows}
    p = np.asarray([r["reference_px"] for r in rows])
    q = np.asarray([r["target_px"] for r in rows])
    matrix, inliers = cv2.estimateAffinePartial2D(
        p, q, method=cv2.RANSAC, ransacReprojThreshold=2., maxIters=4000,
        confidence=.999, refineIters=20,
    )
    if matrix is None:
        return {"status": "similarity_fit_failed", "matches": rows}
    inliers = inliers[:, 0].astype(bool)
    residual = np.linalg.norm(p @ matrix[:, :2].T + matrix[:, 2] - q, axis=1)
    for row, keep, error in zip(rows, inliers, residual, strict=True):
        row.update(similarity_inlier=bool(keep), similarity_residual_px=float(error))
    regions = {}
    for name in masks:
        selected = inliers & np.asarray([r["region"] == name for r in rows])
        regions[name] = summarize_displacements(p[selected], q[selected])
    supported_regions = sum(v is not None and v["count"] >= 2 for v in regions.values())
    return {
        "status": "measured" if supported_regions >= 2 else "limited_spatial_support",
        "match_count": len(rows), "inlier_count": int(inliers.sum()),
        "supported_static_regions": supported_regions,
        "direct_displacement": summarize_displacements(p[inliers], q[inliers]),
        "similarity_reference_to_target": matrix.tolist(),
        "similarity_scale": float(np.hypot(matrix[0, 0], matrix[1, 0])),
        "similarity_rotation_deg": float(np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0]))),
        "similarity_residual_px_p95": float(np.percentile(residual[inliers], 95)),
        "regions": regions, "matches": rows,
    }


def draw_matches(reference, target, result, path, title):
    canvas = np.hstack([reference, target])
    width = reference.shape[1]
    for polygon in REGIONS.values():
        cv2.polylines(canvas, [np.asarray(polygon, dtype=np.int32)], True, (255, 180, 0), 2)
    for row in result["matches"]:
        a = np.rint(row["reference_px"]).astype(int)
        b = np.rint(row["target_px"]).astype(int) + [width, 0]
        color = (60, 220, 60) if row.get("similarity_inlier") else (30, 30, 230)
        cv2.line(canvas, tuple(a), tuple(b), color, 1)
        cv2.circle(canvas, tuple(a), 3, color, 1)
        cv2.circle(canvas, tuple(b), 3, color, 1)
    cv2.putText(canvas, title, (15, 700), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    cv2.imwrite(str(path), canvas)


def aggregate(rows):
    supported = [r for r in rows if r["status"] == "measured"]
    if not supported:
        return {"count": len(rows), "supported": 0}
    magnitudes = [r["direct_displacement"]["displacement_px_p50"] for r in supported]
    return {
        "count": len(rows), "supported": len(supported),
        "episode_median_displacement_px_median": float(np.median(magnitudes)),
        "episode_median_displacement_px_max": float(max(magnitudes)),
        "similarity_scale_median": float(np.median([r["similarity_scale"] for r in supported])),
        "similarity_scale_range": [min(r["similarity_scale"] for r in supported),
                                   max(r["similarity_scale"] for r in supported)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v2")
    parser.add_argument("--split", type=Path, default=Path("results/v2/split.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v2/iphone_stability_20261021"))
    parser.add_argument("--within-episodes", default="0,6,23,31,36")
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    within = [int(v) for v in args.within_episodes.split(",") if v]
    allowed = set(split["train"]) | set(split["val"])
    if 0 not in split["train"] or not set(within) <= allowed:
        raise ValueError("episode 0 must be training and within-episode checks must exclude reserved episodes")
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "source_frames"
    frame_dir.mkdir(exist_ok=True)
    dataset = RawDataset(args.dataset)
    frames, sources = {}, []
    key = "observation.images.iphone"
    for episode in sorted(allowed):
        segment = dataset.video_segment(key, episode)
        selected = set(range(0, 401, 50)) if episode in within else {0}
        for position, pts, frame in dataset.iter_frames(key, episode, stride=50):
            if position not in selected:
                if position > max(selected):
                    break
                continue
            decoded = frame.to_ndarray(format="bgr24")
            frames[episode, position] = decoded
            path = frame_dir / f"ep{episode:02d}_f{position:04d}.png"
            cv2.imwrite(str(path), decoded)
            sources.append({"episode": episode, "position": position, "pts_seconds": pts,
                            "source_video": str(segment.path),
                            "episode_video_from_to_seconds": [segment.from_ts, segment.to_ts],
                            "path": str(path), "sha256": sha256_file(path)})
    reference = frames[0, 0]
    masks = region_masks(reference.shape)
    mask_image = reference.copy()
    for name, polygon in REGIONS.items():
        cv2.polylines(mask_image, [np.asarray(polygon, dtype=np.int32)], True, (255, 180, 0), 2)
        cv2.putText(mask_image, name, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 180, 0), 1)
    cv2.imwrite(str(args.output / "reference_static_regions.png"), mask_image)
    comparisons = []
    # Train comparisons precede validation; validation never changes masks or fitting thresholds.
    for partition in ("train", "val"):
        for episode in split[partition]:
            if episode == 0:
                continue
            target = frames[episode, 0]
            row = match_static(reference, target, masks)
            row.update(kind="between_episodes", partition=partition, reference_episode=0,
                       reference_frame=0, target_episode=episode, target_frame=0)
            comparisons.append(row)
            draw_matches(reference, target, row, args.output / f"matches_ep00_to_ep{episode:02d}.jpg",
                         f"Episode 0 -> {episode} ({partition}); fixed scene features only")
    for episode in within:
        for position in range(50, 401, 50):
            if (episode, position) not in frames:
                continue
            row = match_static(frames[episode, 0], frames[episode, position], masks)
            row.update(kind="within_episode", partition="train" if episode in split["train"] else "val",
                       reference_episode=episode, reference_frame=0,
                       target_episode=episode, target_frame=position)
            comparisons.append(row)
    summary = {
        "between_train": aggregate([r for r in comparisons if r["kind"] == "between_episodes" and r["partition"] == "train"]),
        "between_validation": aggregate([r for r in comparisons if r["kind"] == "between_episodes" and r["partition"] == "val"]),
        "within_episodes": {str(episode): aggregate([r for r in comparisons if r["kind"] == "within_episode" and r["target_episode"] == episode]) for episode in within},
    }
    # Exploratory grouping based on the training-only step in static-image coordinates.
    # It is a descriptive diagnostic, never a condition label or calibration adoption.
    summary["train_episodes_before_7"] = aggregate([r for r in comparisons if r["kind"] == "between_episodes" and r["partition"] == "train" and r["target_episode"] < 7])
    summary["train_episodes_from_7"] = aggregate([r for r in comparisons if r["kind"] == "between_episodes" and r["partition"] == "train" and r["target_episode"] >= 7])
    report = {
        "schema": "v2_static_image_stability_v1", "source_camera": key,
        "status": "PIXEL_STABILITY_DIAGNOSTIC_NOT_METRIC_CALIBRATION",
        "script_sha256": sha256_file(Path(__file__)), "split_sha256": sha256_file(args.split),
        "dataset_info_sha256": sha256_file(args.dataset / "meta/info.json"),
        "opencv_version": cv2.__version__, "reference_masks_source_px": REGIONS,
        "mask_sha256": json_hash(REGIONS), "source_frames": sources,
        "method": {"features": "SIFT", "matching": "mutual Lowe ratio 0.7", "maximum_match_displacement_px": 60,
                   "fit": "RANSAC 2D similarity; threshold 2 px; seed 0", "reserved_episodes_used": False},
        "limitations": ["No intrinsic/extrinsic or metric calibration is fitted or adopted.",
                        "Changed static-image coordinates do not identify camera motion versus crop/focus/other capture changes.",
                        "Only three small static regions are compared; no claims about full-image distortion.",
                        "Repeated mounting-foot holes and illumination can yield mismatches; coordinates and inlier masks are saved.",
                        "Within-episode masks retain the same pixel locations; camera changes/occlusion may reduce support."],
        "summary": summary, "comparisons": comparisons,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Fixed iPhone image stability — 2026-10-21", "",
             "This diagnostic compares fixed mounting-foot, lower-clamp and cabinet-handle features. It excludes the mat, task objects and moving robot links; no joint readings enter the measurement. It does not establish metric camera calibration or the cause of a changed view.", "",
             f"The training recordings show a change between sampled episode 6 and episode 7: early training first frames have median static-feature displacement {summary['train_episodes_before_7']['episode_median_displacement_px_median']:.2f} px from episode 0; later training first frames have {summary['train_episodes_from_7']['episode_median_displacement_px_median']:.2f} px and median apparent scale {summary['train_episodes_from_7']['similarity_scale_median']:.5f}. Validation first frames have median displacement {summary['between_validation']['episode_median_displacement_px_median']:.2f} px. Thus one fixed pixel camera model across v2 needs checking even though the operator reports unchanged hardware setup.", "",
             "The sampled within-episode comparisons are substantially more stable. This evidence does not resolve whether capture settings, focus/cropping, physical camera motion, or another cause changed the image. Reusing a single camera registration without checking this effect could create systematic projected landmark errors. Do not warp policy inputs or alter camera calibration from this diagnostic alone.", "",
             "Reference: training episode 0, frame 0. Whole-episode split preserved; reserved evaluation recordings are excluded. Train comparisons run before the same fixed settings are checked on validation.", "",
             "| Episode | Split | Inliers | Regions | Median displacement (px) | Image similarity scale | Fit residual p95 (px) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        if row["kind"] == "between_episodes" and "direct_displacement" in row:
            lines.append(f"| {row['target_episode']} | {row['partition']} | {row['inlier_count']} | {row['supported_static_regions']} | {row['direct_displacement']['displacement_px_p50']:.2f} | {row['similarity_scale']:.5f} | {row['similarity_residual_px_p95']:.2f} |")
    lines += ["", "The similarity describes how reference image coordinates map to the target. A scale below one describes apparent image contraction; it does not prove zoom, autofocus or physical camera movement. Regional displacements and all accepted/rejected matches are in `report.json`. Blue polygons show reference regions; green/red links show robust-fit inliers/outliers.", "",
              "## Within-episode checks", "", "| Episode | Supported comparisons | Median / max per-frame median displacement (px) |", "|---|---:|---:|"]
    for episode, stats in summary["within_episodes"].items():
        lines.append(f"| {episode} | {stats['supported']}/{stats['count']} | {stats.get('episode_median_displacement_px_median', float('nan')):.2f} / {stats.get('episode_median_displacement_px_max', float('nan')):.2f} |")
    lines += ["", "Reproduce from repository root:", "", "```sh", ".venv/bin/python scripts/diag_v2_camera_stability.py", "```", "", "No production scene, recordings, normalizers or camera contracts are changed."]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
