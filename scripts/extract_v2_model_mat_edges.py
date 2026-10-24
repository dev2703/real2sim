"""Refine model-visible mat boundaries into source-pixel straight-edge evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from real2sim.data.raw import sha256_file


def line_from_points(points):
    center = np.mean(points, axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    return np.r_[vt[-1], -vt[-1] @ center]


def source_edges(mask, image):
    """Fixed geometric rules; no camera solution or held-out error selects pixels."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty mat model mask")
    points = np.concatenate([c.reshape(-1, 2) for c in contours]).astype(float)
    low, high = points.min(0), points.max(0)
    size = high - low
    center = (low + high) / 2
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 1.)
    selected, diagnostics = {}, {}
    for j, name in enumerate(("top", "right", "bottom", "left")):
        # Occlusion can remove a true corner from a convex hull; do not use the
        # resulting paper/cable corner as the orientation of the physical edge.
        bands = {"top": (points[:, 1] < low[1] + .18 * size[1]) & (points[:, 0] > low[0] + .12 * size[0]) & (points[:, 0] < high[0] - .12 * size[0]),
                 "bottom": (points[:, 1] > low[1] + .86 * size[1]) & (points[:, 0] > low[0] + .12 * size[0]) & (points[:, 0] < high[0] - .12 * size[0]),
                 "left": (points[:, 0] < low[0] + .16 * size[0]) & (points[:, 1] > low[1] + .10 * size[1]) & (points[:, 1] < high[1] - .10 * size[1]),
                 "right": (points[:, 0] > high[0] - .30 * size[0]) & (points[:, 1] > low[1] + .08 * size[1]) & (points[:, 1] < high[1] - .10 * size[1])}
        proposed = points[bands[name]]
        if name == "top" and len(proposed):
            # Upper-envelope evidence avoids a longer interior cut around a
            # white gripper. The percentile uses mask pixels, never a camera
            # fit or metric agreement with another frame.
            proposed = proposed[proposed[:, 1] <= np.percentile(proposed[:, 1], 20) + 3.]
        if len(proposed) < 12:
            raise ValueError(f"insufficient model-visible straight {name} boundary")
        if len(proposed) > 600:
            proposed = proposed[np.unique(np.linspace(0, len(proposed) - 1, 600).astype(int))]
        rng = np.random.default_rng(0)
        initial = None
        for _ in range(400):
            pair = rng.choice(len(proposed), 2, replace=False)
            vector = proposed[pair[1]] - proposed[pair[0]]
            length = np.linalg.norm(vector)
            if length < 80:
                continue
            if name in ("top", "bottom"):
                orient = abs(vector[1]) / length < .20
            elif name == "left":
                orient = abs(vector[0]) / length < .30
            else:
                orient = .05 < abs(vector[0] / max(abs(vector[1]), 1e-6)) < .80 and vector[0] * vector[1] > 0
            if not orient:
                continue
            n = np.array([-vector[1], vector[0]]) / length
            error = np.abs((proposed - proposed[pair[0]]) @ n)
            inliers = error < 3.
            span = float(np.ptp((proposed[inliers] - proposed[pair[0]]) @ (vector / length)))
            if inliers.sum() < 12 or span < 80.:
                continue
            score = (int(inliers.sum()), -float(error[inliers].sum()))
            if initial is None or score > initial[0]:
                initial = score, inliers
        if initial is None or initial[1].sum() < 12:
            raise ValueError(f"no supported model straight segment on {name}")
        candidates = proposed[initial[1]]
        line = line_from_points(candidates)
        direction = np.array([-line[1], line[0]])
        a = candidates.mean(0)
        normal = np.array([-direction[1], direction[0]])
        if normal @ (a - center) < 0:
            normal = -normal
        order = np.argsort((candidates - a) @ direction)
        candidates = candidates[order[np.unique(np.linspace(0, len(order) - 1, min(80, len(order))).astype(int))]]
        offsets = np.arange(-6., 6.1, .5)
        sample = candidates[:, None, :] + offsets[None, :, None] * normal

        def values(shift):
            sample_xy = sample + shift * normal
            return cv2.remap(gray, sample_xy[:, :, 0].astype(np.float32), sample_xy[:, :, 1].astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        contrast = values(2) - values(-2)
        # Prefer the strongest dark-mat -> light-table transition near the model boundary.
        choice = np.argmax(contrast - .5 * np.abs(offsets)[None], axis=1)
        strength = contrast[np.arange(len(candidates)), choice]
        refined = sample[np.arange(len(candidates)), choice]
        refined = refined[strength >= 12.]
        if len(refined) < 8:
            raise ValueError(f"weak raw-image contrast at {name} boundary")
        rng = np.random.default_rng(0)
        best = None
        for _ in range(160):
            pair = rng.choice(len(refined), 2, replace=False)
            v = refined[pair[1]] - refined[pair[0]]
            if np.linalg.norm(v) < 40 or abs(v @ direction) / np.linalg.norm(v) < .98:
                continue
            n = np.array([-v[1], v[0]]) / np.linalg.norm(v)
            errors = np.abs((refined - refined[pair[0]]) @ n)
            inliers = errors < 2.5
            score = (int(inliers.sum()), -float(errors[inliers].sum()))
            if best is None or score > best[0]:
                best = score, inliers
        if best is None or best[1].sum() < 8:
            raise ValueError(f"no supported straight segment on {name} boundary")
        fitted = refined[best[1]]
        line = line_from_points(fitted)
        error = fitted @ line[:2] + line[2]
        span = float(np.ptp((fitted - a) @ direction))
        if span < 80.:
            raise ValueError(f"visible {name} straight segment spans less than 80px")
        selected[name] = fitted.tolist()
        diagnostics[name] = {"model_near_edge_samples": len(candidates), "raw_contrast_passing_samples": len(refined),
                             "retained_straight_samples": len(fitted), "visible_span_px": span,
                             "orthogonal_rms_px": float(np.sqrt(np.mean(error**2))),
                             "fixed_inlier_tolerance_px": 2.5, "line_ax_by_c_equals_zero": line.tolist()}
    return selected, diagnostics


def extract(model_path, out):
    if out.exists():
        raise FileExistsError(out)
    model = json.loads(model_path.read_text())
    frames, reports, foot_rows = [], [], []
    out.mkdir(parents=True)
    for row in model["frames"]:
        if row["camera_key"] != "observation.images.iphone":
            continue
        for path, digest in ((row["path"], row["sha256"]), (row["mask_path"], row["mask_sha256"])):
            if sha256_file(Path(path)) != digest:
                raise ValueError("changed model source/mask artifact")
        source = cv2.imread(row["path"])
        arrays = np.load(row["mask_path"], allow_pickle=False)
        common = {key: row[key] for key in ("episode", "frame", "camera_regime", "role", "path", "sha256")}
        try:
            if "black_mat" not in arrays:
                raise ValueError("text detector/model did not provide a mat mask")
            samples, diagnostics = source_edges(arrays["black_mat"].astype(bool), source)
            frames.append({**common, "sigma_px": 3., "mat_edge_samples_source_px": samples})
            canvas = source.copy()
            for points in samples.values():
                for point in points:
                    cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2, (0, 200, 0), -1)
            cv2.imwrite(str(out / f"ep{row['episode']:03d}_f{row['frame']:04d}_edges.png"), canvas)
            reports.append({**common, "edges": diagnostics})
        except ValueError as exc:
            frames.append({**common, "role": "excluded", "reason": str(exc)})
            reports.append({**common, "exclusion_reason": str(exc)})
        if "foot_prompt" in arrays and arrays["foot_prompt"].any():
            foot_path = out / f"ep{row['episode']:03d}_f{row['frame']:04d}_foot.npy"
            np.save(foot_path, arrays["foot_prompt"].astype(bool), allow_pickle=False)
            foot_rows.append({**{key: row[key] for key in ("episode", "frame", "camera_regime", "role")},
                              "mask_path": str(foot_path), "mask_sha256": sha256_file(foot_path),
                              "source_path": row["path"], "source_sha256": row["sha256"],
                              "model_mask_source": row["mask_path"], "model_mask_source_sha256": row["mask_sha256"]})
    annotations = {"schema": "v2_multiframe_static_annotations_v1", "source_size_wh": [1280, 720],
                   "mat_side_lengths_m": [.305, .406], "split_path": "results/v2/split.json",
                   "selection_path": model["selection_path"], "selection_sha256": model["selection_sha256"],
                   "selection": "Predeclared source random frames and roles; pretrained model boundary proposals plus fixed raw-gradient refinement; every selected fixed-camera frame is retained or explicitly excluded.",
                   "sigma_note": "3px declared working uncertainty, not fitted/model-calibrated statistical confidence; shared boundary/axis/systematic errors remain.",
                   "model_report_path": str(model_path), "model_report_sha256": sha256_file(model_path), "frames": frames}
    (out / "annotations.json").write_text(json.dumps(annotations, indent=2, allow_nan=False) + "\n")
    (out / "extraction.json").write_text(json.dumps({"schema": "v2_model_mat_edge_extraction_v1", "frames": reports,
        "method": "Orientation-constrained straight-line consensus within fixed side bands of model mask boundary; raw-image outward dark-to-light gradients within6px refine proposals. Fixed contrast>=12, model consensus3px, raw line consensus2.5px, retained segment>=80px. Occlusion robustness was developed on training-fit frame26; no verification camera residual selects pixels.",
        "script_sha256": sha256_file(Path(__file__))}, indent=2, allow_nan=False) + "\n")
    (out / "foot_masks.json").write_text(json.dumps({"schema": "v2_fixed_foot_model_masks_v1", "frames": foot_rows,
        "limitations": "VLM-selected fixed box prompts; visible front foot can include clamp/sidewall pixels and occlusions. This is model segmentation, not named CAD physical identity."}, indent=2) + "\n")
    return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = extract(args.model, args.out)
    print(json.dumps({"frames": len(report["frames"]), "excluded": sum(r["role"] == "excluded" for r in report["frames"])}))
