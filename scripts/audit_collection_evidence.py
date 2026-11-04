"""Inventory local recording evidence without inferring wall time from nominal timestamps.

This is a provenance check, not a replacement for video or trace audits. A candidate
session is matched by its dataset name/FPS only and still requires row-level verification.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from real2sim.artifacts import sha256_file as digest


def inventory_dataset(root, session_paths, policy_hz=10.):
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    fps = float(info["fps"])
    if not np.isfinite([fps, policy_hz]).all() or fps <= 0 or policy_hz <= 0:
        raise ValueError("Dataset FPS and policy rate must be positive and finite")
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise ValueError(f"No data parquet files in {root}")
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    episodes = []
    for episode, rows in frame.groupby("episode_index", sort=True):
        rows = rows.sort_values("frame_index")
        times = rows.timestamp.to_numpy(dtype=float)
        episodes.append({
            "episode_index": int(episode), "rows": len(rows),
            "contiguous_frame_indices": bool(np.array_equal(rows.frame_index, np.arange(len(rows)))),
            "stored_timestamp_span_s": float(times[-1] - times[0]),
            "stored_timestamps_match_nominal_row_clock": bool(np.allclose(
                times, np.arange(len(rows)) / fps, rtol=1e-6, atol=1e-5)),
        })
    sessions = []
    for path in session_paths:
        session = json.loads(path.read_text())
        metadata = session.get("metadata", {})
        if (str(metadata.get("dataset_repo_id", "")).split("/")[-1] != root.name
                or metadata.get("dataset_fps") != fps):
            continue
        attempts = [json.loads(p.read_text()) for p in sorted(path.parent.glob("episode_*.json"))]
        sessions.append({"session": str(path.parent.resolve()), "session_sha256": digest(path),
                         "identity_match": "name_and_fps_only_not_row_verified",
                         "saved_episode_indices": sorted(a["episode_index"] for a in attempts
                                                         if a["status"] == "saved"),
                         "attempt_statuses": [a["status"] for a in attempts],
                         "markers": {name: (path.parent / name).exists()
                                     for name in ("RUNNING.json", "COMPLETE.json", "FAILED.json")}})
    ratio = fps / policy_hz
    stride = round(ratio)
    columns = sorted(set().union(*(f.columns for f in frames)))
    feature_keys = sorted(info["features"])
    return {
        "root": str(root.resolve()), "nominal_fps": fps, "actual_rows": len(frame),
        "actual_episodes": len(episodes), "declared_rows": info.get("total_frames"),
        "declared_episodes": info.get("total_episodes"), "columns": columns,
        "nominal_stride_to_policy": int(stride) if stride >= 1 and abs(ratio - stride) <= 1e-6 else None,
        "policy_hz": policy_hz, "episodes": episodes,
        "candidate_trace_sessions": sessions,
        "physical_clock_status": "unverified_candidate_sidecar" if sessions else "no_trace_sidecar_found",
        "actual_acquisition_cadence_hz": None,
        "requested_sent_separation_in_dataset": all(
            key in columns for key in ("leader_request", "processed_request", "sent_target")),
        "recording_provenance_keys_in_metadata": sorted(k for k in info if any(
            word in k.lower() for word in ("commit", "calibrat", "limiter", "use_degrees", "session"))),
        "clock_or_trace_named_dataset_fields": [k for k in sorted(set(feature_keys + columns))
            if any(word in k.lower() for word in ("timestamp", "clock", "trace", "request", "sent"))],
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in [info_path, *paths]},
    }


def inventory(roots, trace_roots, policy_hz=10.):
    searches = sorted(set([*roots, *trace_roots]))
    session_paths = sorted(set(path for root in searches if root.exists()
                               for path in root.rglob("SESSION.json")))
    return {
        "kind": "LOCAL_COLLECTION_EVIDENCE_INVENTORY", "script_sha256": digest(Path(__file__)),
        "source_sha256": {str(path): digest(path) for path in (
            Path(__file__), Path("src/real2sim/artifacts.py"))},
        "trace_search_roots": [{"path": str(root.resolve()), "exists": root.exists()} for root in searches],
        "trace_session_count": len(session_paths),
        "datasets": [inventory_dataset(root, session_paths, policy_hz) for root in roots],
        "limitations": [
            "Search is limited to the listed local roots; external or differently located traces may exist.",
            "A uniform stored row clock does not establish actual sampling rate, drift or camera freshness.",
            "A same-name/FPS sidecar is only a candidate; use audit_recording_trace.py with --dataset for row parity.",
            "No video re-decode, task-success annotation, hardware test or dataset rewrite was performed.",
            "Old recordings with one action field cannot reconstruct requested-versus-sent clipping.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--trace-root", type=Path, action="append", default=[])
    parser.add_argument("--policy-hz", type=float, default=10.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inventory(args.dataset, args.trace_root, args.policy_hz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output), "datasets": len(report["datasets"]),
                      "trace_sessions": report["trace_session_count"]}))


if __name__ == "__main__":
    main()
