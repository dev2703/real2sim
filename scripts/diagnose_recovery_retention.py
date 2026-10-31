"""Audit sustained grasp loss in saved learned-recovery traces, never control a robot.

Simulator held/lifted/ball poses are explicitly scoring-only here. Commanded jaw
intent is kept separate from measured opening; phase predictions are not grasp
labels. This retrospective report can identify slip signatures, not establish
whether seating, friction or acceleration caused a loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.artifacts import sha256_file
from real2sim.policy.phase_recovery_mlp import PHASE_NAMES
from real2sim.serialization import write_json


def audit_trace(rows, *, control_hz=25., minimum_held_s=.12, window_s=.4):
    if (not np.isfinite([control_hz, minimum_held_s, window_s]).all()
            or min(control_hz, minimum_held_s, window_s) <= 0):
        raise ValueError("Positive finite audit timing required")
    minimum = max(1, int(np.ceil(minimum_held_s * control_hz)))
    window = max(2, round(window_s * control_hz))
    events, held_start = [], None
    for index, row in enumerate(rows):
        if row["scoring_only"]["ball_held"]:
            if held_start is None:
                held_start = index
            continue
        if held_start is None:
            continue
        start, held_start = held_start, None
        if index - start < minimum:
            continue
        # All intervals are indexed by saved control ticks, not array positions.
        # The current command drove the just-completed interval in these traces.
        first = max(start, index - window)
        segment = rows[first:index + 1]
        ticks = np.asarray([value["tick"] for value in segment], float)
        if np.any(np.diff(ticks) <= 0):
            raise ValueError("Trace ticks must strictly increase")
        points = np.asarray([value["scoring_only"]["tool_xyz_task"] for value in segment])
        relative = np.asarray([np.asarray(value["scoring_only"]["ball_xyz_task"])
                               - value["scoring_only"]["tool_xyz_task"] for value in segment])
        times = ticks / control_hz
        velocities = np.diff(points, axis=0) / np.diff(times)[:, None]
        midpoints = (times[1:] + times[:-1]) / 2
        acceleration = (np.diff(velocities, axis=0) / np.diff(midpoints)[:, None]
                        if len(velocities) > 1 else np.zeros((1, 3)))
        commands = np.asarray([value["detail"]["requested_common_action"][6] for value in segment])
        gaps = [value["scoring_only"].get("gripper_opening_m") for value in segment]
        phase_id = row["detail"].get("learned_phase_id")
        lifted = any(value["scoring_only"]["lifted"] for value in rows[start:index])
        closed = bool(np.all(commands <= .1))
        event = {
            "tick": row["tick"], "held_start_tick": rows[start]["tick"],
            "held_duration_s": (row["tick"] - rows[start]["tick"]) / control_hz,
            "lifted_during_held_interval": lifted,
            "classification": ("closed_command_loss_after_lift" if lifted and closed
                               else "loss_after_lift" if lifted else "contact_loss_before_lift"),
            "predicted_phase": PHASE_NAMES[phase_id] if phase_id is not None else None,
            "commanded_open_fraction_at_loss": float(commands[-1]),
            "commanded_open_fraction_window_max": float(commands.max()),
            "window_s": float(times[-1] - times[0]),
            "ball_minus_tool_task_xyz_at_window_start_m": relative[0].tolist(),
            "ball_minus_tool_task_xyz_at_loss_m": relative[-1].tolist(),
            "ball_relative_vertical_change_m": float(relative[-1, 2] - relative[0, 2]),
            "measured_jaw_gap_at_window_start_m": gaps[0],
            "measured_jaw_gap_at_loss_m": gaps[-1],
            "peak_measured_tcp_speed_mps": float(np.linalg.norm(velocities, axis=1).max()),
            "peak_measured_tcp_acceleration_mps2": float(np.linalg.norm(acceleration, axis=1).max()),
            "ball_over_bowl_at_loss": bool(row["scoring_only"]["ball_over_bowl"]),
        }
        events.append(event)
    return events


def audit_reports(paths, *, control_hz=25.):
    trials = []
    for path in paths:
        report = json.loads(path.read_text())
        for trial in report["trials"]:
            trace = Path(trial["trace"])
            with trace.open() as stream:
                rows = [json.loads(line) for line in stream]
            trials.append({"robot": report["robot"], "seed": trial["seed"],
                "pick_place_success": trial["pick_place_success"], "abort": trial["abort"],
                "report": str(path), "report_sha256": sha256_file(path),
                "trace": str(trace), "trace_sha256": sha256_file(trace),
                "head_sha256": report["head_sha256"],
                "events": audit_trace(rows, control_hz=control_hz)})
    return {"schema": "recovery_retention_diagnostic_v1", "control_hz": control_hz,
        "scoring_only": True, "causal_claim": False,
        "scope": "Retrospective sustained-contact losses; includes successful releases and failed attempts. No controller changes or physics fitting.",
        "limitations": "Held/contact labels can flicker. Finite-difference acceleration is averaged at the saved control rate. A closed command and jaw resistance do not prove a secure grasp.",
        "trials": trials,
        "summary": {"attempts": len(trials),
            "closed_command_loss_after_lift_events": sum(
                event["classification"] == "closed_command_loss_after_lift"
                for trial in trials for event in trial["events"]),
            "attempts_with_closed_command_loss_after_lift": sum(any(
                event["classification"] == "closed_command_loss_after_lift"
                for event in trial["events"]) for trial in trials)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--control-hz", type=float, default=25.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = audit_reports(args.reports, control_hz=args.control_hz)
    write_json(args.out, report, atomic=True)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
