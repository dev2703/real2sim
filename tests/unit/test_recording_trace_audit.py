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
        "action": {"names": ["joint.pos"]},
        "observation.state": {"names": ["joint.pos"]}}}))
    data = attempt()
    path = tmp_path / "data/episode.parquet"
    frame = pd.DataFrame({"episode_index": [0] * 31, "frame_index": np.arange(31),
                          "action": [[10.]] * 31, "observation.state": [[0.]] + [[8.]] * 30,
                          "timestamp": np.arange(31) / 20})
    frame.to_parquet(path)
    session = {"metadata": {"dataset_fps": 20}}
    assert audit.check_dataset(tmp_path, session, [data])["episodes"]["0"]["sent_action_labels"] == "PASS"
    frame["action"] = [[20.]] * 31
    frame.to_parquet(path)
    with pytest.raises(AssertionError):
        audit.check_dataset(tmp_path, session, [data])


def test_drift_and_velocity_scale_expose_nominal_timestamp_error():
    data = attempt()
    # Actual 18.18 Hz for nominal 20 Hz: every stored one-second span is 1.1 physical seconds.
    for i, row in enumerate(data["frames"]):
        for key in row:
            if key.endswith("_ns"):
                row[key] += i * 5_000_000
    result = audit.audit_attempt(data, 20)
    drift = result["clock_drift"]
    assert drift["observation_minus_nominal_end_ms"] == pytest.approx(150)
    assert drift["nominal_velocity_scale_over_physical"]["mean"] == pytest.approx(1.1)
    assert drift["policy_row_stride"]["actual_interval_ms"]["mean"] == pytest.approx(110)
    assert drift["policy_row_stride"]["nominal_velocity_scale_over_physical"]["mean"] == pytest.approx(1.1)
    joint = result["joints"]["joint.pos"]
    assert joint["leader_minus_processed_request_absolute"]["mean"] == 5
    assert joint["processed_requests_changed_by_send"]["changed_fraction"] == 1


def test_stall_residuals_are_separated_and_noninteger_stride_is_not_rounded():
    data = attempt()
    for row in data["frames"][15:]:
        for key in row:
            if key.endswith("_ns"):
                row[key] += 200_000_000
    data["frames"][15]["measured_positions"]["joint.pos"] = 100.
    result = audit.audit_attempt(data, 25)
    assert result["clock_drift"]["policy_row_stride"]["status"] == "noninteger_rate_ratio"
    joint = result["joints"]["joint.pos"]
    assert joint["latest_prior_sent_minus_measured_absolute"]["max"] == 90
    assert joint["prior_sent_minus_measured_excluding_long_gaps_absolute"]["max"] == 2


def test_trace_does_not_certify_untraced_episodes_or_different_observations(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"fps": 20, "features": {
        "action": {"names": ["joint.pos"]}, "observation.state": {"names": ["joint.pos"]}}}))
    data = attempt(4)
    frame = pd.DataFrame({"episode_index": [0] * 4 + [1] * 4, "frame_index": list(range(4)) * 2,
                          "action": [[10.]] * 8, "observation.state": ([[0.]] + [[8.]] * 3) * 2,
                          "timestamp": list(np.arange(4) / 20) * 2})
    path = tmp_path / "data/episode.parquet"
    frame.to_parquet(path)
    session = {"metadata": {"dataset_fps": 20}}
    result = audit.check_dataset(tmp_path, session, [data])
    assert result["coverage"]["untraced_dataset_episodes"] == [1]
    assert result["coverage"]["all_dataset_episodes_traced"] is False
    frame.at[1, "observation.state"] = [99.]
    frame.to_parquet(path)
    with pytest.raises(AssertionError, match="observations differ"):
        audit.check_dataset(tmp_path, session, [data])


@pytest.mark.parametrize("fps,policy_hz", [(0, 10), (20, 0), (float("nan"), 10), (20, float("inf"))])
def test_nonfinite_or_invalid_cadences_are_rejected(fps, policy_hz):
    with pytest.raises(ValueError, match="Rates"):
        audit.audit_attempt(attempt(), fps, policy_hz=policy_hz)
