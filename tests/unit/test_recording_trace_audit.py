"""Validate physical-time and label checks with synthetic delayed follower feedback."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).parents[2] / "scripts/audit_recording_trace.py"
SPEC = importlib.util.spec_from_file_location("trace_audit", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def attempt(count=31):
    rows = []
    for i in range(count):
        start = 10**16 + i * 50_000_000
        row = dict(zip(("tick_start_ns", "observe_start_ns", "observe_end_ns", "send_start_ns",
                        "send_end_ns", "record_end_ns"),
                       (start, start + 100, start + 200, start + 300, start + 400, start + 500),
                       strict=True))
        row.update(frame_index=i, leader_request={"joint.pos": 20.},
                   processed_request={"joint.pos": 15.}, sent_target={"joint.pos": 10.},
                   measured_positions={"joint.pos": 8. if i else 0.})
        rows.append(row)
    return {"frames": rows, "status": "saved", "episode_index": 0, "attempt_id": "synthetic"}


def test_actual_stall_is_retained_and_breaks_hold_windows():
    data = attempt()
    for row in data["frames"][15:]:
        for key in row:
            if key.endswith("_ns"):
                row[key] += 200_000_000
    result = audit.audit_attempt(data, 20)
    assert result["tick_interval_ms"]["max"] == pytest.approx(250)
    assert result["intervals_above_2_5_nominal"] == 1
    assert result["measured_tick_hz"] < 20
    joint = result["joints"]["joint.pos"]
    assert joint["leader_minus_sent_absolute"]["mean"] == 10
    assert joint["processed_request_minus_sent_absolute"]["mean"] == 5
    assert joint["latest_prior_sent_minus_measured_absolute"]["max"] == 2
    windows = joint["stable_target_windows"]
    assert [(w["start_frame"], w["end_frame"]) for w in windows] == [(0, 14), (15, 30)]
    assert all(w["signed_residual_median"] == 2 for w in windows)


def test_prior_target_is_used_for_delayed_follower_measurements():
    data = attempt(10)
    for i, row in enumerate(data["frames"]):
        row["sent_target"]["joint.pos"] = i
        row["measured_positions"]["joint.pos"] = max(i - 1, 0)
    result = audit.audit_attempt(data, 20)
    assert result["joints"]["joint.pos"]["latest_prior_sent_minus_measured_absolute"]["max"] == 0


def test_malformed_timestamps_are_rejected():
    data = attempt()
    data["frames"][1]["observe_start_ns"] = data["frames"][0]["observe_start_ns"]
    with pytest.raises(ValueError, match="Non-monotonic"):
        audit.audit_attempt(data, 20)


def test_dataset_validation_rejects_old_requested_action_labels(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"fps": 20, "features": {
        "action": {"names": ["joint.pos"]}}}))
    data = attempt()
    path = tmp_path / "data/episode.parquet"
    frame = pd.DataFrame({"episode_index": [0] * 31, "frame_index": np.arange(31),
                          "action": [[10.]] * 31})
    frame.to_parquet(path)
    session = {"metadata": {"dataset_fps": 20}}
    assert audit.check_dataset(tmp_path, session, [data])["0"]["sent_action_labels"] == "PASS"
    frame["action"] = [[20.]] * 31
    frame.to_parquet(path)
    with pytest.raises(AssertionError):
        audit.check_dataset(tmp_path, session, [data])
