"""Fixed-base image tracks and raw mat edges, independent of FK and segmentation.

Named image anchors remain CAD identity hypotheses. This diagnostic estimates 2-D
capture-coordinate motion, not an accepted metric camera or joint correction.
Selection and thresholds are persisted before decoding fresh frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from diag_v2_relative_scene_motion import (
    fit_similarity,
    inverse,
    mask_for,
    summarize_vectors,
    transform,
)

from real2sim.data.raw import RawDataset, sha256_file

REGIONS = {
    "base_foot": [[211, 356], [268, 338], [328, 347], [330, 380], [242, 399], [211, 383]],
    "base_holder": [[158, 231], [219, 231], [243, 257], [224, 278], [225, 314], [167, 314]],
    "table_clamp": [[66, 451], [135, 450], [143, 475], [127, 517], [106, 499], [104, 478], [70, 487]],
}
# The cabinet handle is occluded by the wrist in ep0/f200. It cannot provide a
# static reference; only these two visible fixed regions enter the similarity.
FIT_REGIONS = ("base_foot", "table_clamp")
ANCHORS = {"candidate_E_upper_opening": ("base_holder", [199., 249.]),
           "candidate_I_side_hole": ("base_holder", [191., 301.])}
# Raw contrast corridors chosen on the existing ep0/f200 reference. No model masks.
# Endpoints deliberately avoid rounded corners and the usual ball/bowl occlusions.
EDGES = {
    "left": {"ends": [[409., 390.], [385., 610.]], "normal": [1., 0.], "sign": -1.},
    "right": {"ends": [[926., 415.], [982., 610.]], "normal": [1., 0.], "sign": 1.},
    "top": {"ends": [[470., 214.], [640., 212.]], "normal": [0., 1.], "sign": -1.},
    "bottom": {"ends": [[440., 679.], [930., 672.]], "normal": [0., 1.], "sign": 1.},
}
PARAMETERS = {"patch_size": 25, "search_radius_px": 48, "ncc_min": .80,
              "ncc_peak_margin_min": .035, "peak_exclusion_radius_px": 8,
              "cycle_max_px": .75, "lk_template_disagreement_max_px": 2.,
              "max_features_per_region": 35, "feature_min_distance_px": 7.,
              "edge_search_radius_px": 32, "edge_contrast_min": 18.,
              "edge_min_rays": 10, "edge_min_fraction": .6,
              "edge_inlier_distance_px": 2., "motion_working_scale_px": 3.}


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_landmarks(reference):
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    rows = [{"landmark_id": name, "region": region, "reference_px": point,
             "cad_identity_verified": False} for name, (region, point) in ANCHORS.items()]
    for region, polygon in REGIONS.items():
        points = cv2.goodFeaturesToTrack(gray, PARAMETERS["max_features_per_region"], .02,
                                        PARAMETERS["feature_min_distance_px"],
                                        mask=mask_for(gray.shape, polygon), blockSize=5)
        if points is None:
            continue
        for j, point in enumerate(points[:, 0]):
            if any(np.linalg.norm(point - np.asarray(r["reference_px"])) < 5 for r in rows):
                continue
            rows.append({"landmark_id": f"{region}_{j:02d}", "region": region,
                         "reference_px": point.astype(float).tolist(), "cad_identity_verified": False})
    return rows


def track_landmarks(reference, target, seeds):
    """Direct reference matching avoids cumulative drift; reject occlusion/aliases."""
    a, b = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (reference, target))
    size, radius = PARAMETERS["patch_size"], PARAMETERS["search_radius_px"]
    half = size // 2
    rows = []
    for seed in seeds:
        p = np.asarray(seed["reference_px"], np.float32)
        center = np.rint(p).astype(int)
        x0, y0 = np.maximum(center - radius - half, 0)
        x1, y1 = np.minimum(center + radius + half + 1, [b.shape[1], b.shape[0]])
        patch = cv2.getRectSubPix(a, (size, size), tuple(p))
        scores = cv2.matchTemplate(b[y0:y1, x0:x1], patch, cv2.TM_CCOEFF_NORMED)
        _, best, _, (x, y) = cv2.minMaxLoc(scores)
        masked = scores.copy()
        yy, xx = np.indices(scores.shape)
        masked[(xx - x)**2 + (yy - y)**2 <= PARAMETERS["peak_exclusion_radius_px"]**2] = -1
        margin = float(best - masked.max())
        q0 = np.array([x0 + x + half, y0 + y + half], np.float32).reshape(1, 1, 2)
        initial = p.reshape(1, 1, 2)
        lk = dict(winSize=(25, 25), maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=1e-5)
        q, forward, _ = cv2.calcOpticalFlowPyrLK(a, b, initial, q0.copy(), **lk)
        finite_forward = q is not None and np.isfinite(q).all()
        if finite_forward:
            back, backward, _ = cv2.calcOpticalFlowPyrLK(b, a, q, initial.copy(), **lk)
        else:
            back, backward = None, None
        finite_backward = back is not None and np.isfinite(back).all()
        cycle = float(np.linalg.norm(back[0, 0] - p)) if finite_backward else None
        refinement = float(np.linalg.norm(q[0, 0] - q0[0, 0])) if finite_forward else None
        reasons = []
        if (not finite_forward or not finite_backward or forward is None or backward is None
                or not forward[0, 0] or not backward[0, 0]):
            reasons.append("flow_failed")
        if best < PARAMETERS["ncc_min"] or patch.std() < 4:
            reasons.append("poor_patch_match")
        if margin < PARAMETERS["ncc_peak_margin_min"]:
            reasons.append("ambiguous_patch")
        if cycle is not None and cycle > PARAMETERS["cycle_max_px"]:
            reasons.append("cycle_failed")
        if refinement is not None and refinement > PARAMETERS["lk_template_disagreement_max_px"]:
            reasons.append("template_flow_disagreement")
        rows.append({**seed, "target_px": q[0, 0].astype(float).tolist() if finite_forward else None,
                     "ncc": float(best), "peak_margin": margin, "cycle_px": cycle,
                     "refinement_px": refinement, "accepted": not reasons,
                     "rejection_reasons": reasons})
    return rows


def fit_edge(gray, spec, matrix):
    """Signed raw contrast scans, then robust line consensus; absence is retained."""
    ends = transform(spec["ends"], matrix)
    normal = np.asarray(matrix)[:, :2] @ np.asarray(spec["normal"])
    normal /= np.linalg.norm(normal)
    rays = np.linspace(ends[0], ends[1], 31)
    radius = PARAMETERS["edge_search_radius_px"]
    offsets = np.arange(-radius, radius + 1, dtype=float)
    locations = rays[:, None, :] + offsets[None, :, None] * normal
    def sample(displacement):
        points = locations + displacement * normal
        return cv2.remap(gray.astype(np.float32), points[:, :, 0].astype(np.float32),
                         points[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
    contrast = spec["sign"] * (sample(3.) - sample(-3.))
    peaks = contrast.argmax(1)
    strengths = contrast[np.arange(len(rays)), peaks]
    points = locations[np.arange(len(rays)), peaks]
    good = strengths >= PARAMETERS["edge_contrast_min"]
    candidate = points[good]
    if len(candidate) < PARAMETERS["edge_min_rays"]:
        return {"status": "insufficient_raw_contrast", "rays": len(rays), "candidate_rays": len(candidate)}
    # Enumerate line hypotheses deterministically; never let one occluding cable set a line.
    best = None
    direction = ends[1] - ends[0]
    direction /= np.linalg.norm(direction)
    for i in range(len(candidate)):
        for j in range(i + 1, len(candidate)):
            delta = candidate[j] - candidate[i]
            if np.linalg.norm(delta) < 60:
                continue
            tangent = delta / np.linalg.norm(delta)
            if abs(tangent @ direction) < np.cos(np.deg2rad(8)):
                continue
            n = np.array([-tangent[1], tangent[0]])
            error = np.abs((candidate - candidate[i]) @ n)
            keep = error <= PARAMETERS["edge_inlier_distance_px"]
            score = (int(keep.sum()), -float(np.minimum(error, 2).sum()))
            if best is None or score > best[0]:
                best = (score, keep)
    if best is None or best[0][0] < max(PARAMETERS["edge_min_rays"], PARAMETERS["edge_min_fraction"] * len(rays)):
        return {"status": "insufficient_line_consensus", "candidate_rays": len(candidate),
                "consensus_rays": 0 if best is None else best[0][0]}
    retained = candidate[best[1]]
    center = retained.mean(0)
    _, _, vt = np.linalg.svd(retained - center, full_matrices=False)
    n = vt[-1]
    if n @ normal < 0:
        n = -n
    line = np.r_[n, -n @ center]
    return {"status": "measured", "line": line.tolist(), "points": retained.tolist(),
            "rays": len(rays), "candidate_rays": len(candidate), "inlier_rays": len(retained),
            "rms_px": float(np.sqrt(np.mean((retained @ n + line[2])**2)))}


def compare_edges(reference_edges, target_edges, matrix):
    result = {}
    for name, ref in reference_edges.items():
        target = target_edges[name]
        if ref["status"] != "measured" or target["status"] != "measured":
            result[name] = {"status": "unmeasured"}
            continue
        stabilized = transform(target["points"], inverse(matrix))
        line = np.asarray(ref["line"])
        distances = stabilized @ line[:2] + line[2]
        result[name] = {"status": "measured", "median_signed_normal_px": float(np.median(distances)),
                        "p95_abs_normal_px": float(np.percentile(np.abs(distances), 95))}
    return result


def make_selection(raw, split, seed, excluded=()):
    if set(split["train"]) & set(split["val"]) or (set(split["train"]) | set(split["val"])) & set(split["reserved"]):
        raise ValueError("split overlap")
    rng = np.random.default_rng(seed)
    excluded = set(excluded)
    selections = []
    for partition in ("train", "val"):
        for episode in sorted(split[partition]):
            length = raw.episode_length(episode)
            # One seeded fresh frame in each third; no outcome-conditioned replacement.
            for low, high in zip(np.linspace(0, length, 4).astype(int)[:-1],
                                 np.linspace(0, length, 4).astype(int)[1:], strict=True):
                available = [f for f in range(low, high) if (episode, f) not in excluded]
                if not available:
                    raise ValueError(f"no uninspected frames in episode {episode} stratum {low}:{high}")
                selections.append({"episode": episode, "frame": int(rng.choice(available)),
                                   "partition": partition, "role": "fresh_tracking_check"})
    return selections


def residual_for(rows, region, matrix):
    selected = [r for r in rows if r["accepted"] and r["region"] == region]
    p = np.asarray([r["reference_px"] for r in selected]).reshape(-1, 2)
    q = np.asarray([r["target_px"] for r in selected]).reshape(-1, 2)
    return summarize_vectors(transform(p, matrix), q)


def run(raw_root, split_path, out, seed, exclusion_paths=()):
    if out.exists():
        raise FileExistsError(out)
    raw, split = RawDataset(raw_root), json.loads(split_path.read_text())
    reference_path = Path("outputs/v2/iphone_stability_20261021/source_frames/ep00_f0200.png")
    reference = cv2.imread(str(reference_path))
    if reference is None or reference.shape != (720, 1280, 3) or 0 not in split["train"]:
        raise ValueError("existing training reference ep0/f200 is required at source resolution")
    # Also check every previously model-measured development image, with no re-selection.
    from diag_v2_relative_scene_motion import load_sources
    previous_sources, dependencies = load_sources()
    source_lookup = {(r["episode"], r["frame"]): r for r in previous_sources}
    excluded = set(source_lookup) | {(0, 200)}
    for path in exclusion_paths:
        document = json.loads(path.read_text())
        records = document.get("selections", document.get("source_frames"))
        if records is None:
            raise ValueError(f"exclusion source lacks selections/source_frames: {path}")
        excluded.update((r["episode"], r.get("frame", r.get("position"))) for r in records)
    selections = make_selection(raw, split, seed, excluded)
    for source in previous_sources:
        if source["episode"] in split["reserved"]:
            raise ValueError("reserved source")
        selections.append({k: source[k] for k in ("episode", "frame", "partition")} | {"role": "previous_model_measurement"})
    seeds = seed_landmarks(reference)
    out.mkdir(parents=True)
    (out / "frames").mkdir()
    (out / "executed_tracker.py").write_bytes(Path(__file__).read_bytes())
    plan = {"schema": "fixed_landmark_tracks_plan_v2", "seed": seed, "parameters": PARAMETERS,
            "regions": REGIONS, "fit_regions": FIT_REGIONS, "heldout_region": "base_holder",
            "edge_corridors": EDGES, "reference": str(reference_path),
            "reference_sha256": sha256_file(reference_path), "seeds": seeds,
            "selections": selections, "reserved_episodes_used": [],
            "excluded_previously_selected_frames": [list(p) for p in sorted(excluded)],
            "freshness_scope": "New relative to the supplied selection inventories, not a claim about every historical inspection. Previously model-measured frames are scored separately.",
            "source_sha256": {str(p): sha256_file(p) for p in [Path(__file__),
                Path("scripts/diag_v2_relative_scene_motion.py"), split_path, *dependencies, *exclusion_paths]}}
    json_write(out / "plan.json", plan)
    identity = np.array([[1., 0., 0.], [0., 1., 0.]])
    reference_edges = {k: fit_edge(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), spec, identity)
                       for k, spec in EDGES.items()}
    report = {"schema": "fixed_landmark_tracks_v2", "plan_sha256": sha256_file(out / "plan.json"),
              "physical_calibration_accepted": False, "robot_fk_used": False,
              "reserved_episodes_used": [], "reference_edges": reference_edges, "comparisons": [],
              "limitations": ["2-D similarity combines camera pose, stabilization, cropping, zoom and parallax; it is not metric extrinsics.",
                  "Candidate E/I names are unverified CAD identities; image tracking cannot establish physical dimensions.",
                  "Raw edge movement relative to fixed features supports mat-relative image change but includes parallax and boundary-definition uncertainty.",
                  "Model-edge disagreement is not a pure segmentation error: virtual corners and raw visible boundary have different semantics.",
                  "Sparse temporal samples cannot exclude unobserved transient motion; failed/occluded tracks and edges are retained."]}
    report["limitations"].append("Both fitting regions are on the left of the image; there is no independently tracked background region. Extrapolating their similarity to the far mat has unmeasured parallax uncertainty.")
    key = "observation.images.iphone"
    for episode in sorted({s["episode"] for s in selections}):
        requested = {s["frame"]: s for s in selections if s["episode"] == episode}
        segment = raw.video_segment(key, episode)
        video_hash = sha256_file(segment.path)
        seen = set()
        for frame, pts, decoded in raw.iter_frames(key, episode):
            if frame not in requested:
                if frame > max(requested):
                    break
                continue
            seen.add(frame)
            source = requested[frame]
            target = decoded.to_ndarray(format="bgr24")
            path = out / "frames" / f"ep{episode:03d}_f{frame:04d}.png"
            cv2.imwrite(str(path), target)
            old = source_lookup.get((episode, frame))
            if old is not None:
                if sha256_file(Path(old["path"])) != old["sha256"]:
                    raise ValueError("prior model source image hash mismatch")
                if not np.array_equal(cv2.imread(old["path"]), target):
                    raise ValueError(f"redecoded pixels differ from model source ep{episode}/f{frame}")
            tracks = track_landmarks(reference, target, seeds)
            fit = fit_similarity(tracks, FIT_REGIONS)
            row = {**source, "path": str(path), "sha256": sha256_file(path),
                   "video": str(segment.path), "video_sha256": video_hash, "pts_s": pts,
                   "tracks": tracks, "static_fit": fit}
            if fit["status"] == "fit":
                matrix = np.asarray(fit["matrix_reference_to_target"])
                for track in tracks:
                    if track["target_px"] is not None:
                        residual = float(np.linalg.norm(transform([track["reference_px"]], matrix)[0]
                                                        - np.asarray(track["target_px"])))
                        track["static_similarity_residual_px"] = residual
                        track["supports_static_fit"] = bool(track["accepted"] and track["region"] in FIT_REGIONS
                                                           and residual <= 2.)
                row["static_support"] = "supported" if fit["supported_regions"] >= 2 else "limited"
                row["heldout_holder_residual"] = residual_for(tracks, "base_holder", matrix)
                edges = {k: fit_edge(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), spec, matrix)
                         for k, spec in EDGES.items()}
                row["raw_mat_edges"] = edges
                row["mat_relative_to_static"] = compare_edges(reference_edges, edges, matrix)
                if old is not None and old["corners"] is not None:
                    # Measure model virtual edge against raw contrast points, without a metric camera.
                    corners = old["corners"]
                    mismatch = {}
                    for name, (i, j) in {"top": (0, 1), "right": (1, 2), "bottom": (2, 3), "left": (3, 0)}.items():
                        if edges[name]["status"] != "measured":
                            continue
                        vector = corners[j] - corners[i]
                        normal = np.array([-vector[1], vector[0]]) / np.linalg.norm(vector)
                        error = (np.asarray(edges[name]["points"]) - corners[i]) @ normal
                        mismatch[name] = {"median_abs_px": float(np.median(np.abs(error))),
                                          "p95_abs_px": float(np.percentile(np.abs(error), 95))}
                    row["model_edge_vs_raw_boundary"] = mismatch
            canvas = target.copy()
            for track in tracks:
                if track["target_px"] is None:
                    continue
                p = tuple(np.rint(track["target_px"]).astype(int))
                color = (0, 220, 0) if track.get("supports_static_fit") else (
                    (255, 0, 200) if track["accepted"] and track["region"] == "base_holder" else (0, 0, 220))
                cv2.circle(canvas, p, 3, color, 1)
            cv2.putText(canvas, "green: static support | magenta: withheld holder | red: rejected", (15, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 240), 1)
            for edge in row.get("raw_mat_edges", {}).values():
                for point in edge.get("points", []):
                    cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2, (255, 180, 0), 1)
            cv2.imwrite(str(out / f"ep{episode:03d}_f{frame:04d}_tracks.jpg"), canvas)
            report["comparisons"].append(row)
        if seen != set(requested):
            raise ValueError(f"missing requested frames in episode {episode}: {set(requested) - seen}")
        json_write(out / "report.json", report)
        print(f"episode {episode}: {len(seen)} frames; {len(report['comparisons'])} total", flush=True)
    report["summary"] = {
        "frames": len(report["comparisons"]),
        "supported_static_frames": sum(r.get("static_support") == "supported" for r in report["comparisons"]),
        "fresh_frames": sum(r["role"] == "fresh_tracking_check" for r in report["comparisons"]),
        "physical_registration_gate": "not_passed_unverified_CAD_identities_and_intrinsics"}
    json_write(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("~/.cache/huggingface/lerobot/local/so101_act_pickplace_v2").expanduser())
    parser.add_argument("--split", type=Path, default=Path("results/v2/split.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026102501)
    parser.add_argument("--exclude-selection", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = run(args.raw_root, args.split, args.out, args.seed, args.exclude_selection)
    print(json.dumps(result["summary"], indent=2))
