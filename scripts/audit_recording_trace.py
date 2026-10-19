"""Audit local collector sidecars; never connect to hardware or rewrite recordings."""

import argparse
import json
from pathlib import Path

import numpy as np


def summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)),
            "max": float(values.max())}


def joint_values(rows, key, joint):
    values = [row.get(key, {}).get(joint) for row in rows]
    return np.array([float(value) if value is not None else np.nan for value in values])


def clock_drift(times, fps, policy_hz):
    """Compare actual host observation intervals with nominal row-time assumptions."""
    nominal = np.arange(len(times), dtype=float) / fps
    observed = times[:, 1] - times[0, 1]
    drift = observed - nominal
    result = {"host_observation_start_span_s": float(observed[-1]),
              "nominal_row_span_s": float(nominal[-1]),
              "observation_minus_nominal_end_ms": float(drift[-1] * 1000),
              "observation_minus_nominal_absolute_ms": summary(abs(drift) * 1000),
              "nominal_velocity_scale_over_host_interval": summary(np.diff(observed) * fps)}
    ratio = fps / policy_hz
    stride = int(round(ratio))
    if stride < 1 or not np.isclose(ratio, stride, rtol=0, atol=1e-6):
        result["policy_row_stride"] = {"status": "noninteger_rate_ratio",
                                      "source_fps": fps, "policy_hz": policy_hz}
    else:
        intervals = np.diff(observed[::stride])
        result["policy_row_stride"] = {
            "status": "diagnostic_only", "stride": stride, "policy_hz": policy_hz,
            "actual_interval_ms": summary(intervals * 1000),
            "nominal_velocity_scale_over_host_interval": summary(intervals * policy_hz),
            "intervals_above_1_5_policy_period": int((intervals > 1.5 / policy_hz).sum())}
    return result


def audit_attempt(attempt, fps, hold_tolerance=.5, settle_s=.25, min_hold_s=.5,
                  policy_hz=10.):
    if (not np.isfinite([fps, policy_hz, hold_tolerance, settle_s, min_hold_s]).all()
            or fps <= 0 or policy_hz <= 0 or hold_tolerance < 0 or settle_s < 0 or min_hold_s <= 0):
        raise ValueError("Rates/hold duration must be positive and hold tolerance/settle time nonnegative")
    rows = attempt["frames"]
    if len(rows) < 2:
        raise ValueError("Need at least two recorded rows for a cadence audit")
    if [r["frame_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("Trace row IDs are not contiguous")
    names = ("tick_start_ns", "observe_start_ns", "observe_end_ns", "send_start_ns",
             "send_end_ns", "record_end_ns")
    stamps = np.array([[r[k] for k in names] for r in rows], dtype=np.int64)
    if np.any(np.diff(stamps, axis=1) < 0) or np.any(np.diff(stamps, axis=0) <= 0):
        raise ValueError("Non-monotonic timestamps")
    if np.any(stamps[1:, 0] < stamps[:-1, -1]):
        raise ValueError("Overlapping control ticks")
    # Subtract the integer origin before converting nanoseconds to float seconds.
    times = (stamps - stamps[0, 0]) / 1e9
    intervals = np.diff(times[:, 0])
    period = 1 / fps
    gap_limit = 2.5 * period
    # This tick's observation precedes its command. Use the latest completed send.
    previous = np.searchsorted(stamps[:, 4], stamps[:, 1], side="right") - 1
    eligible = previous >= 0
    command_age = times[eligible, 1] - times[previous[eligible], 4]
    joints = sorted(set().union(*(r["sent_target"] for r in rows)))
    by_joint = {}
    for joint in joints:
        sent = joint_values(rows, "sent_target", joint)
        measured = joint_values(rows, "measured_positions", joint)
        requested = joint_values(rows, "processed_request", joint)
        leader = joint_values(rows, "leader_request", joint)
        prior_errors = sent[previous[eligible]] - measured[eligible]
        comparable_requests = np.isfinite(requested) & np.isfinite(sent)
        changed_requests = abs(requested[comparable_requests] - sent[comparable_requests]) > 1e-6
        holds = []
        start = 0
        while start < len(rows):
            if not np.isfinite(sent[start]):
                start += 1
                continue
            end = start + 1
            low = high = sent[start]
            while end < len(rows) and np.isfinite(sent[end]) and intervals[end - 1] <= gap_limit:
                new_low, new_high = min(low, sent[end]), max(high, sent[end])
                if new_high - new_low > hold_tolerance:
                    break
                low, high = new_low, new_high
                end += 1
            duration = times[end - 1, 1] - times[start, 4]
            if duration >= min_hold_s:
                indices = np.arange(start, end)
                indices = indices[(times[indices, 1] >= times[start, 4] + settle_s)
                                  & (previous[indices] >= start) & np.isfinite(measured[indices])]
                if len(indices):
                    residuals = sent[previous[indices]] - measured[indices]
                    holds.append({"start_frame": start, "end_frame": end - 1,
                        "duration_s": float(duration), "samples_after_settling_window": len(indices),
                        "target_range": [float(low), float(high)],
                        "signed_residual_median": float(np.median(residuals)),
                        "absolute_residual": summary(abs(residuals))})
            start = end
        by_joint[joint] = {
            "missing_or_nonfinite": {key: int((~np.isfinite(value)).sum()) for key, value in
                                     (("leader", leader), ("requested", requested),
                                      ("sent", sent), ("measured", measured))},
            "leader_minus_sent_absolute": summary(abs(leader - sent)),
            "leader_minus_processed_request_absolute": summary(abs(leader - requested)),
            "processed_request_minus_sent_absolute": summary(abs(requested - sent)),
            "processed_requests_changed_by_send": {
                "comparable_rows": int(comparable_requests.sum()),
                "changed_rows": int(changed_requests.sum()),
                "changed_fraction": float(changed_requests.mean()) if len(changed_requests) else None,
                "comparison_tolerance_command_units": 1e-6},
            "latest_prior_sent_minus_measured_absolute": summary(abs(prior_errors)),
            "prior_sent_minus_measured_excluding_long_gaps_absolute": summary(
                abs(prior_errors[command_age <= gap_limit])),
            "stable_target_windows": holds,
        }
    return {"episode_index": attempt["episode_index"], "attempt_id": attempt["attempt_id"],
            "rows": len(rows), "measured_tick_hz": float(1 / intervals.mean()),
            "tick_interval_ms": summary(intervals * 1000),
            "observation_start_interval_ms": summary(np.diff(times[:, 1]) * 1000),
            "send_start_interval_ms": summary(np.diff(times[:, 3]) * 1000),
            "observe_duration_ms": summary((times[:, 2] - times[:, 1]) * 1000),
            "send_duration_ms": summary((times[:, 4] - times[:, 3]) * 1000),
            "tick_to_record_ms": summary((times[:, 5] - times[:, 0]) * 1000),
            "prior_command_age_at_observation_ms": summary(command_age * 1000),
            "clock_drift": clock_drift(times, fps, policy_hz),
            "intervals_above_1_5_nominal": int((intervals > 1.5 * period).sum()),
            "intervals_above_2_5_nominal": int((intervals > gap_limit).sum()),
            "joints": by_joint}


def check_dataset(root, session, attempts):
    import pandas as pd

    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != session["metadata"]["dataset_fps"]:
        raise ValueError("Dataset FPS differs from trace session")
    names = info["features"]["action"]["names"]
    state_names = info["features"]["observation.state"]["names"]
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise ValueError("No data parquet files found")
    frame = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "action",
                                                  "observation.state", "timestamp"])
                       for p in paths], ignore_index=True)
    checks = {}
    for attempt in attempts:
        ep = attempt["episode_index"]
        saved = frame.loc[frame.episode_index == ep].sort_values("frame_index")
        rows = attempt["frames"]
        np.testing.assert_array_equal(saved.frame_index.to_numpy(), np.arange(len(rows)))
        sent = np.array([[row["sent_target"][joint] for joint in names] for row in rows])
        labels = np.stack(saved.action.to_numpy())
        np.testing.assert_allclose(labels, sent, atol=1e-5, rtol=1e-6, equal_nan=False)
        measured = np.array([[row["measured_positions"][joint] for joint in state_names] for row in rows])
        observations = np.stack(saved["observation.state"].to_numpy())
        # This audit supports the collector's default identity position processor.
        # A custom transform needs explicit provenance; it cannot silently validate raw parity.
        np.testing.assert_allclose(observations, measured, atol=1e-5, rtol=1e-6,
                                   equal_nan=False, err_msg="Dataset observations differ from raw trace positions")
        np.testing.assert_allclose(saved.timestamp, np.arange(len(rows)) / info["fps"],
                                   atol=1e-5, rtol=1e-6, equal_nan=False)
        checks[str(ep)] = {"rows": len(rows), "sent_action_labels": "PASS",
                           "raw_observation_positions": "PASS", "nominal_row_timestamps": "PASS",
                           "max_absolute_label_difference": float(abs(labels - sent).max()),
                           "max_absolute_observation_difference": float(abs(observations - measured).max())}
    traced = {a["episode_index"] for a in attempts}
    dataset_episodes = {int(ep) for ep in frame.episode_index.unique()}
    return {"episodes": checks, "coverage": {
        "dataset_episodes": sorted(dataset_episodes), "verified_trace_episodes": sorted(traced),
        "untraced_dataset_episodes": sorted(dataset_episodes - traced),
        "all_dataset_episodes_traced": dataset_episodes == traced,
        "scope": "Only listed traced episodes have timing and command provenance checks."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="The session_UUID directory containing SESSION.json")
    parser.add_argument("--dataset", type=Path, help="Optional matching local LeRobot dataset root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-tolerance", type=float, default=.5, help="Command units; diagnostic only")
    parser.add_argument("--settle-seconds", type=float, default=.25)
    parser.add_argument("--minimum-hold-seconds", type=float, default=.5)
    parser.add_argument("--policy-hz", type=float, default=10., help="Nominal downstream row-stride diagnostic")
    args = parser.parse_args()
    if (not np.isfinite([args.hold_tolerance, args.settle_seconds, args.minimum_hold_seconds,
                        args.policy_hz]).all() or args.hold_tolerance < 0 or args.settle_seconds < 0
            or args.minimum_hold_seconds <= 0 or args.policy_hz <= 0):
        parser.error("Hold tolerance/settle time must be nonnegative; duration/policy rate positive and finite")
    session = json.loads((args.session / "SESSION.json").read_text())
    if session["schema_version"] != 1:
        raise ValueError("Unsupported trace schema")
    fps = float(session["metadata"]["dataset_fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid nominal FPS")
    attempts = [json.loads(p.read_text()) for p in sorted(args.session.glob("episode_*.json"))]
    saved = [a for a in attempts if a["status"] == "saved"]
    if not saved:
        raise ValueError("No saved attempts; interrupted or discarded attempts cannot validate dataset labels")
    if len({a["episode_index"] for a in saved}) != len(saved):
        raise ValueError("Multiple saved attempts claim the same dataset episode")
    result = {"session": str(args.session.resolve()), "provenance": session,
        "trace_session_markers": {name: (args.session / name).exists()
                                  for name in ("RUNNING.json", "COMPLETE.json", "FAILED.json")},
        "nominal_fps": fps, "ignored_attempts": [{k: a[k] for k in ("attempt_id", "episode_index", "status")}
                                               for a in attempts if a["status"] != "saved"],
        "diagnostic_settings": {"hold_tolerance_command_units": args.hold_tolerance,
            "settle_seconds": args.settle_seconds, "minimum_hold_seconds": args.minimum_hold_seconds,
            "hold_break_gap_nominal_intervals": 2.5, "policy_hz": args.policy_hz},
        "episodes": [audit_attempt(a, fps, args.hold_tolerance, args.settle_seconds,
                                   args.minimum_hold_seconds, args.policy_hz) for a in saved],
        "dataset_label_checks": check_dataset(args.dataset, session, saved) if args.dataset else "not_run",
        "limitations": ["Command units follow session calibration; gripper and arm units can differ.",
            "Host read/send intervals do not establish camera exposure timestamps or frame freshness.",
            "Velocity scale is actual host observation interval divided by nominal interval; no label correction is applied.",
            "Changed requests identify send-side command changes, not a motor fault or limiter parameter by themselves.",
            "Dataset observation parity assumes the default identity position observation processor.",
            "Prior-sent residuals include response dynamics; even stable command windows do not prove physical settling.",
            "No measured/sent residual alone diagnoses a motor defect; loaded and unloaded holds need interpretation.",
            "No data acceptance, physical grasp success or Panda gate is granted by this report."],
        "physical_task_success": "not_evaluated", "panda_gate_passed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output), "saved_episodes": len(saved),
                      "label_checks": "PASS" if args.dataset else "not_run"}))


if __name__ == "__main__":
    main()
