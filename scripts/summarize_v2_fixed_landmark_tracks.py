"""Audit a frozen fixed-landmark run without changing tracks, fits or selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.data.raw import sha256_file


def reconstruct_scan_offsets(points, spec, matrix, rays):
    """Recover the sampled ray number and signed search offset from saved pixels."""
    points, matrix = np.asarray(points, float), np.asarray(matrix, float)
    if matrix.shape != (2, 3) or points.ndim != 2 or points.shape[1] != 2 or rays < 2:
        raise ValueError("2-D points, a 2x3 image transform, and at least two rays required")
    if not np.isfinite(matrix).all() or not np.isfinite(points).all():
        raise ValueError("nonfinite scan geometry")
    ends = np.c_[np.asarray(spec["ends"], float), np.ones(2)] @ matrix.T
    normal = matrix[:, :2] @ np.asarray(spec["normal"], float)
    length = np.linalg.norm(normal)
    if length == 0:
        raise ValueError("degenerate scan normal")
    normal /= length
    basis = np.column_stack([ends[1] - ends[0], normal])
    if abs(np.linalg.det(basis)) < 1e-8:
        raise ValueError("degenerate scan corridor")
    parameters = np.linalg.solve(basis, (points - ends[0]).T).T
    ray_indices = parameters[:, 0] * (rays - 1)
    if (np.any(ray_indices < -1e-6) or np.any(ray_indices > rays - 1 + 1e-6)
            or not np.allclose(ray_indices, np.rint(ray_indices), atol=1e-6, rtol=0)):
        raise ValueError("saved point does not lie on a declared scan ray")
    return np.rint(ray_indices).astype(int), parameters[:, 1]


def distribution(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def audit_edge(edge, spec, matrix, radius, boundary_margin):
    if edge["status"] != "measured":
        return {"status": edge["status"], "range_limited": None}
    indices, offsets = reconstruct_scan_offsets(edge["points"], spec, matrix, edge["rays"])
    if np.any(np.abs(offsets) > radius + 1e-6):
        raise ValueError("retained ray exceeds the frozen search radius")
    near = radius - np.abs(offsets) <= boundary_margin + 1e-6
    return {"status": "measured", "retained_rays": len(offsets),
            "near_search_boundary_ray_indices": indices[near].tolist(),
            "near_search_boundary_offsets_px": np.round(offsets[near], 6).tolist(),
            "offset_min_max_px": np.round([offsets.min(), offsets.max()], 6).tolist(),
            "near_search_boundary_rays": int(near.sum()), "range_limited": bool(near.any())}


def summarize(report_path, plan_path, boundary_margin=2.):
    report, plan = (json.loads(path.read_text()) for path in (report_path, plan_path))
    if report["schema"] != "fixed_landmark_tracks_v2" or plan["schema"] != "fixed_landmark_tracks_plan_v2":
        raise ValueError("the corrected v2 frozen tracker run is required")
    if sha256_file(plan_path) != report["plan_sha256"]:
        raise ValueError("plan hash mismatch")
    snapshot = report_path.parent / "executed_tracker.py"
    tracker_hashes = [value for key, value in plan["source_sha256"].items()
                      if Path(key).name == "diag_v2_fixed_landmark_tracks.py"]
    if len(tracker_hashes) != 1 or sha256_file(snapshot) != tracker_hashes[0]:
        raise ValueError("executed tracker differs from the frozen plan")
    radius = plan["parameters"]["edge_search_radius_px"]
    if not 0 <= boundary_margin < radius:
        raise ValueError("boundary margin must be nonnegative and smaller than the search radius")
    comparisons, edge_rows, model_rows = report["comparisons"], [], []
    declared = {(r["episode"], r["frame"], r["role"]) for r in plan["selections"]}
    observed = {(r["episode"], r["frame"], r["role"]) for r in comparisons}
    if observed != declared or len(observed) != len(comparisons):
        raise ValueError("completed frames do not match the declared selection")
    for row in comparisons:
        if row["static_fit"]["status"] != "fit":
            continue
        matrix = np.asarray(row["static_fit"]["matrix_reference_to_target"])
        edge_audits = {name: audit_edge(row["raw_mat_edges"][name], spec, matrix,
                                      radius, boundary_margin)
                       for name, spec in plan["edge_corridors"].items()}
        edge_rows.append({"episode": row["episode"], "frame": row["frame"], "role": row["role"],
                          "static_support": row["static_support"], "edges": edge_audits,
                          "mat_relative_to_static": row["mat_relative_to_static"]})
        if "model_edge_vs_raw_boundary" in row:
            model_rows.append({"episode": row["episode"], "frame": row["frame"],
                               "static_support": row["static_support"],
                               "edges": {name: {**score, "range_limited": edge_audits[name]["range_limited"]}
                                         for name, score in row["model_edge_vs_raw_boundary"].items()}})
    edge_summary = {}
    for name in plan["edge_corridors"]:
        measured = [r for r in edge_rows if r["edges"][name]["status"] == "measured"]
        relative = [r for r in measured if r["mat_relative_to_static"][name]["status"] == "measured"]
        eligible = [r for r in relative if r["static_support"] == "supported"
                    and not r["edges"][name]["range_limited"]]
        edge_summary[name] = {
            "reference_status": report["reference_edges"][name]["status"],
            "target_measured_frames": len(measured), "relative_measured_frames": len(relative),
            "range_limited_target_frames": sum(r["edges"][name]["range_limited"] for r in measured),
            "supported_uncensored_relative_frames": len(eligible),
            "supported_uncensored_frame_median_signed_normal_px": distribution([
                r["mat_relative_to_static"][name]["median_signed_normal_px"] for r in eligible]),
        }
    support = [r for r in comparisons if r.get("static_support") == "supported"]
    holder = [r for r in support if r.get("heldout_holder_residual", {}).get("n", 0)]
    names = [r["landmark_id"] for r in plan["seeds"] if r["landmark_id"].startswith("candidate_")]
    anchors = {}
    for name in names:
        tracks = [t for r in comparisons for t in r["tracks"] if t["landmark_id"] == name]
        retained = [t for t in tracks if t["accepted"]]
        anchors[name] = {"attempted": len(tracks), "accepted": len(retained),
                         "image_similarity_residual_px": distribution([
                             t["static_similarity_residual_px"] for t in retained])}
    regimes = {}
    for name, group in (("early_episodes_below_7", [r for r in support if r["episode"] < 7]),
                        ("late_episodes_from_7", [r for r in support if r["episode"] >= 7])):
        regimes[name] = {"frames": len(group), "similarity_scale": distribution([
            r["static_fit"]["scale"] for r in group]), "frame_median_static_displacement_px": distribution([
                r["static_fit"]["direct_displacement"]["median_magnitude_px"] for r in group])}
    limited = [{k: r[k] for k in ("episode", "frame", "role")} for r in comparisons
               if r.get("static_support") != "supported"]
    return {"schema": "fixed_landmark_tracking_summary_v1", "physical_calibration_accepted": False,
            "selection_seed": plan["seed"],
            "production_inputs_changed": False, "reserved_episodes_used": [],
            "counts": {"frames": len(comparisons), "fresh_relative_to_declared_inventories": sum(
                r["role"] == "fresh_tracking_check" for r in comparisons),
                "previous_model_frames": len(model_rows), "supported_static_frames": len(support),
                "excluded_previously_selected_frames": len(plan["excluded_previously_selected_frames"])},
            "limited_static_frames": limited, "image_regimes": regimes, "named_anchor_tracks": anchors,
            "withheld_holder_supported_frames": len(holder),
            "withheld_holder_per_frame_p95_px": distribution([
                r["heldout_holder_residual"]["p95_magnitude_px"] for r in holder]),
            "edge_summary": edge_summary, "edge_range_audit": [
                {"episode": row["episode"], "frame": row["frame"],
                 "static_support": row["static_support"],
                 "edges": {name: edge for name, edge in row["edges"].items() if edge["range_limited"]}}
                for row in edge_rows if any(edge["range_limited"] for edge in row["edges"].values())],
            "model_edge_comparisons": model_rows,
            "audit_definition": {"boundary_margin_px": boundary_margin, "search_radius_px": radius,
                "range_limited_rule": "Any retained ray within the stated margin of either frozen search boundary; this flags possible truncation, not proof that the true edge is beyond the range.",
                "no_refitting_or_reselection": True,
                "compact_range_audit": "Only flagged edges are listed; their near-boundary ray indices/offsets and complete retained-offset range are retained. Every offset is reconstructed to compute these flags; original points remain in the frozen report.",
                "uncensored_summary_rule": "Existing measured reference and target edge, two supported static regions, and no retained target ray near a search boundary. This descriptive filter is not a calibration acceptance gate.",
                "distributions": "Unweighted distributions of per-frame statistics; frames within episodes are correlated. No confidence interval is asserted."},
            "limitations": report["limitations"] + [
                "The audit observes retained rays only; rejected rays and stronger unseen edges outside the corridor cannot be reconstructed.",
                "A failed reference bottom line prevents every relative bottom measurement, regardless of target bottom visibility.",
                "The cabinet-handle ROI in the earlier run covered the moving wrist; corrected fitting uses only foot and table clamp.",
                "Similarity support is the original RANSAC mask. Overlay support additionally thresholds the final residual; these need not identify identical tracks."],
            "registration_gate": "Not passed: zero verified multi-height CAD identities and no independent intrinsics/base registration. No joint-zero/gripper correction, FK relabeling or corrected-policy retraining is justified by these pixels.",
            "provenance_sha256": {str(p): sha256_file(p) for p in (report_path, plan_path, snapshot, Path(__file__))}}


def markdown(report):
    counts, edges, regimes = report["counts"], report["edge_summary"], report["image_regimes"]
    holder = report["withheld_holder_per_frame_p95_px"]
    lines = ["# Fixed-base tracking and mat-boundary audit — 25 October 2026", "",
        f"The corrected frozen run tracks **{counts['frames']} source frames**, with two supported fixed regions in **{counts['supported_static_frames']}**. The withheld upper-base region agrees with the fitted image similarity: its maximum per-frame p95 residual on supported frames is **{holder['max']:.2f} px**. This establishes useful local image-coordinate consistency, not physical camera/base calibration.", "",
        "The [summary and per-ray range audit](../results/v2/fixed_landmark_tracking_20261025.json) consume the completed [tracker report](../outputs/v2/fixed_landmark_tracks_20261025/report.json) and its frozen plan. The audit does not refit tracks, change thresholds or select replacement frames.", "",
        "## Corrected reference and selection", "",
        "The 24 October prototype placed a cabinet-handle ROI over the wrist in its arm-clear reference. Those points were moving-arm features; RANSAC rejected them in that completed run. The corrected run removes the ROI and uses only the fixed foot and lower table clamp. The upper fixed-base holder, including tentative E/I image anchors, stays outside the fit. The original run and executed source are retained.", "",
        f"Seed **2026102501** selected one frame from each third of each of the 35 training/validation episodes. All **{counts['excluded_previously_selected_frames']}** frames in the supplied prior tracking, first-frame stability and model-measurement inventories were excluded before decoding. This yields **{counts['fresh_relative_to_declared_inventories']}** newly selected frames plus **{counts['previous_model_frames']}** earlier model-measured frames scored separately. Freshness is relative to those inventories; it is not a claim about every historical inspection. Reserved episodes are excluded. Re-decoded earlier model frames were checked for exact pixel identity.", "",
        "The limited-support frames are " + ", ".join(f"{r['episode']}/{r['frame']}" for r in report["limited_static_frames"]) + ". They remain in the record. The summary below excludes their relative-edge values from the supported, uncensored distribution.", "",
        "## Image motion and withheld base checks", "",
        "| Capture group | Supported frames | Median image scale | Median per-frame static displacement |",
        "|---|---:|---:|---:|"]
    for name, label in (("early_episodes_below_7", "Episodes below 7"), ("late_episodes_from_7", "Episodes from 7")):
        r = regimes[name]
        lines.append(f"| {label} | {r['frames']} | {r['similarity_scale']['median']:.5f} | {r['frame_median_static_displacement_px']['median']:.2f} px |")
    lines += ["", "These are changes in capture coordinates. The similarity can combine camera pose, cropping, stabilization, focus/zoom and parallax. Both fitting regions lie on the left of the image; there is no independently tracked distant background. Agreement at the withheld holder does not quantify extrapolation error across the mat.", "",
              "| Tentative image anchor | Accepted / attempted | Median / maximum similarity residual |", "|---|---:|---:|"]
    for name, row in report["named_anchor_tracks"].items():
        values = row["image_similarity_residual_px"]
        lines.append(f"| {name} | {row['accepted']} / {row['attempted']} | {values['median']:.2f} / {values['max']:.2f} px |")
    lines += ["", "E/I names remain unverified CAD identities. Their successful image tracks do not validate metric points, physical joint zeros or jaw registration.", "",
        "## Raw mat edges and search-range limits", "",
        "Raw signed-contrast scans use the frozen corridors; no segmentation mask enters their extraction. This audit reconstructs each retained ray's offset from the saved scan geometry. It flags a target edge if any retained ray lies within **2 px of the ±32 px search boundary**. A flag indicates possible truncation; it does not prove that the true edge is outside the corridor. Search windows are not widened after inspecting these results.", "",
        "| Edge | Measured target frames | Relative measurements | Range-limited targets | Supported, uncensored relative frames |",
        "|---|---:|---:|---:|---:|"]
    for name, row in edges.items():
        lines.append(f"| {name} | {row['target_measured_frames']} | {row['relative_measured_frames']} | {row['range_limited_target_frames']} | {row['supported_uncensored_relative_frames']} |")
    lines += ["", "The late left-boundary measurements often approach the search limit and must not be treated as complete uncensored displacement estimates. The reference bottom edge fails the predeclared line-consensus threshold, so **no relative bottom measurement is available**, even where a target bottom is visible. All available scan corridors, including the contrast-sampling offsets, remain inside the source images.", "",
        "The following descriptive ranges use only supported static fits and measured reference/target edges without a flagged retained ray. They are per-frame signed normal displacement in reference pixels, not metric mat movement or confidence intervals. Frames from the same episode are correlated.", "",
        "| Edge | Minimum / median / maximum frame-median relative displacement |", "|---|---:|"]
    for name, row in edges.items():
        values = row["supported_uncensored_frame_median_signed_normal_px"]
        value = "Unavailable" if not values["n"] else f"{values['min']:.2f} / {values['median']:.2f} / {values['max']:.2f} px"
        lines.append(f"| {name} | {value} |")
    lines += ["", "## Model boundaries versus direct image boundaries", "",
        "The ten previously model-measured frames are compared without changing their masks, corners or raw-edge rules. On shared measured edges, the model's virtual line and the raw contrast boundary can be compared directly. Each cell below gives median / p95 absolute point-to-line distance; † marks a search-range-limited raw edge.", "",
        "| Episode / frame | Left | Right | Top | Bottom |", "|---|---:|---:|---:|---:|"]
    for row in report["model_edge_comparisons"]:
        values = []
        for name in ("left", "right", "top", "bottom"):
            s = row["edges"].get(name)
            values.append("—" if s is None else f"{s['median_abs_px']:.2f} / {s['p95_abs_px']:.2f}" + ("†" if s["range_limited"] else ""))
        lines.append(f"| {row['episode']} / {row['frame']} | " + " | ".join(values) + " |")
    lines += ["", "Most shared lines agree within a few pixels; episode 26/frame 150 has an approximately 8.08 px median bottom disagreement. This is a boundary disagreement, not a pure segmentation-error measurement: visible contrast, stitching/shadow, occlusions and virtual edge definitions differ. The left measurements marked † retain their range limitation. Earlier shared camera/mat-pose failures therefore cannot be attributed solely to segmentation or solely to physical camera movement.", "",
        "## Physical registration gate", "",
        "The stationary upper opening, side hole and foot offer nominal CAD heights of 69.6, 45.7 and 15.1 mm, but the [landmark review](v2_fixed_base_landmark_review_20261024.md) records zero verified physical identities. Repeated foot apertures, uncertain printed-part revision and unmeasured intrinsics/distortion still prevent an accepted multi-height camera/base fit. No joint-zero/gripper correction or corrected-label retraining follows from this diagnostic.", "",
        "The next useful capture remains a short calibration-board sequence with independently measured board-to-base registration, arm-clear views of named base features, and a measured loaded hold/release. A board on the mat alone does not register the robot base.", "",
        "Reproduce the summary without rerunning tracking:", "", "```sh",
        ".venv/bin/python scripts/summarize_v2_fixed_landmark_tracks.py \\",
        "  --run outputs/v2/fixed_landmark_tracks_20261025 \\",
        "  --out results/v2/fixed_landmark_tracking_20261025.json \\",
        "  --doc docs/v2_fixed_landmark_tracking_20261025.md", "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run / "report.json", args.run / "plan.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.doc.write_text(markdown(result))
    print(json.dumps({"counts": result["counts"], "edge_summary": result["edge_summary"]}, indent=2))
