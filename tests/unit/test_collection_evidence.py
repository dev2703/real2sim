"""Stored nominal timestamps and completed sidecars cannot silently certify real timing."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

PATH = Path(__file__).parents[2] / "scripts/audit_collection_evidence.py"
SPEC = importlib.util.spec_from_file_location("collection_evidence", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def dataset(root, fps=25):
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta/info.json").write_text(json.dumps({"fps": fps, "total_frames": 3,
        "total_episodes": 1, "features": {"timestamp": {}, "action": {}, "observation.state": {}}}))
    pd.DataFrame({"episode_index": [0] * 3, "frame_index": [0, 1, 2],
                  "timestamp": [i / fps for i in range(3)], "action": [[10.]] * 3,
                  "observation.state": [[8.]] * 3}).to_parquet(root / "data/rows.parquet")


def test_uniform_nominal_clock_does_not_claim_real_cadence(tmp_path):
    root = tmp_path / "recording"
    dataset(root)
    report = audit.inventory([root], [tmp_path / "absent_traces"])
    result = report["datasets"][0]
    assert result["episodes"][0]["stored_timestamps_match_nominal_row_clock"] is True
    assert result["actual_acquisition_cadence_hz"] is None
    assert result["physical_clock_status"] == "no_trace_sidecar_found"
    assert result["nominal_stride_to_policy"] is None  # 25 -> 10 cannot select every Nth row.
    assert result["requested_sent_separation_in_dataset"] is False
    assert report["trace_session_count"] == 0


def test_same_name_completed_trace_still_requires_row_verification(tmp_path):
    root = tmp_path / "recording"
    dataset(root, fps=20)
    trace = tmp_path / "traces/session_one"
    trace.mkdir(parents=True)
    (trace / "SESSION.json").write_text(json.dumps({"metadata": {
        "dataset_repo_id": "local/recording", "dataset_fps": 20}}))
    (trace / "COMPLETE.json").write_text("{}")
    (trace / "episode_one.json").write_text(json.dumps({"episode_index": 0, "status": "discarded"}))
    report = audit.inventory([root], [trace.parent])
    result = report["datasets"][0]
    assert result["nominal_stride_to_policy"] == 2
    assert result["physical_clock_status"] == "unverified_candidate_sidecar"
    assert result["actual_acquisition_cadence_hz"] is None
    assert result["candidate_trace_sessions"][0]["saved_episode_indices"] == []
