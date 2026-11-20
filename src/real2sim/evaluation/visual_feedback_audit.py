"""Offline RGB feedback coverage, error and passive-action comparisons.

Runtime feedback at tick t observes the pre-action state. Benchmark scoring at
tick t describes the post-action state, so measurements must be compared with
scoring at t-1. Privileged scoring is used only in this retrospective audit.
Run with ``python -m real2sim.evaluation.visual_feedback_audit REPORT --out OUT``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.artifacts import sha256_file
from real2sim.serialization import write_json


def audit_trace(rows, *, control_hz=25., error_limit_m=.025):
    if not np.isfinite([control_hz, error_limit_m]).all() or min(control_hz, error_limit_m) <= 0:
        raise ValueError("Positive finite audit frequency and error limit required")
    held_ticks = measured_held = lifted_ticks = measured_lifted = 0
    missing_run = longest_missing = support_ticks = unsupported_contact = 0
    errors, held_errors = [], []
    transport_drops, postplacement_losses = [], []
    for index, row in enumerate(rows):
        if row["tick"] != index:
            raise ValueError("Audit requires consecutive benchmark ticks starting at zero")
        if index == 0:
            continue  # The benchmark trace has no scoring state before tick zero.
        state = rows[index - 1]["scoring_only"]
        feedback = row["detail"].get("observable_grasp_feedback", {})
        sphere = feedback.get("fresh_ball_3d")
        held = bool(state["ball_held"])
        lifted = held and bool(state["lifted"])
        held_ticks += held
        lifted_ticks += lifted
        measured_held += held and sphere is not None
        measured_lifted += lifted and sphere is not None
        missing_run = missing_run + 1 if held and sphere is None else 0
        longest_missing = max(longest_missing, missing_run)
        if sphere is not None:
            sphere = np.asarray(sphere, float)
            if sphere.shape != (3,) or not np.isfinite(sphere).all():
                raise ValueError("Audit requires finite three-coordinate RGB measurements")
            error = float(np.linalg.norm(sphere - state["ball_xyz_task"]))
            errors.append(error)
            if held:
                held_errors.append(error)
        supported = bool(feedback.get("visual_hold_supported", False))
        support_ticks += supported
        unsupported_contact += supported and not held
        if feedback.get("loss_suspected", False) and state["ball_over_bowl"]:
            postplacement_losses.append(row["tick"])
        if index >= 2:
            before = rows[index - 2]["scoring_only"]
            # This is a scored contact-loss transition outside delivery, not
            # proof of a sustained drop or a detector true-positive label.
            if (before["ball_held"] and before["lifted"] and not held
                    and not state["ball_over_bowl"]
                    and feedback.get("closed_command", False)):
                transport_drops.append({"tick": row["tick"],
                    "loss_flag_at_transition": bool(feedback.get("loss_suspected", False))})

    def error_summary(values):
        return {"measurements": len(values),
                "median_m": float(np.median(values)) if values else None,
                "p95_m": float(np.quantile(values, .95)) if values else None,
                "max_m": float(max(values)) if values else None,
                "over_limit": sum(value > error_limit_m for value in values)}

    return {"scoring_only": True, "feedback_alignment": "pre-action RGB at t versus scored state t-1",
        "held_ticks": held_ticks, "measured_held_ticks": measured_held,
        "lifted_held_ticks": lifted_ticks, "measured_lifted_held_ticks": measured_lifted,
        "longest_missing_held_s": longest_missing / control_hz,
        "all_measurement_error": error_summary(errors), "held_measurement_error": error_summary(held_errors),
        "error_limit_m": error_limit_m, "visual_support_ticks": support_ticks,
        "support_ticks_without_scored_contact": unsupported_contact,
        "closed_contact_losses_outside_bowl": transport_drops,
        "loss_flag_ticks_over_bowl": postplacement_losses,
        "limitations": "Contact flags can flicker; support is not proof of holding. Bowl-region loss flags are not transport-drop detections. Missing initial scoring state excluded."}


def passive_comparison(baseline, passive):
    """Compare full control/scoring traces after removing disclosed diagnostics."""
    def semantic(row):
        detail = {k: v for k, v in row["detail"].items() if k not in {
            "observable_grasp_feedback", "visual_feedback_enabled", "visual_ball_fit",
            "inference_ms", "head_inference_ms", "controller_latency_ms"}}
        return {k: detail if k == "detail" else v for k, v in row.items() if k != "latency_ms"}

    mismatches = [i for i, (a, b) in enumerate(zip(baseline, passive, strict=False)) if semantic(a) != semantic(b)]
    return {"exact_control_and_scoring_equality": len(baseline) == len(passive) and not mismatches,
        "baseline_ticks": len(baseline), "passive_ticks": len(passive), "mismatched_ticks": mismatches,
        "excluded_fields": "RGB grasp diagnostics, sensing-mode metadata and wall-clock latency only"}


def audit_benchmark(report_path, *, baseline_path=None):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    protocol = json.loads(report_path.with_name("protocol.json").read_text())
    baseline = {} if baseline_path is None else {
        (row["robot"], row["condition"], row["seed"]): row
        for row in json.loads(Path(baseline_path).read_text())["trials"]}
    trials = []
    for trial in report["trials"]:
        trace = Path(trial["trace"])
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        audited = {"robot": trial["robot"], "seed": trial["seed"], "condition": trial["condition"],
            "trace": str(trace), "trace_sha256": sha256_file(trace),
            "pick_place_success": trial["pick_place_success"], "abort": trial["abort"],
            **audit_trace(rows, control_hz=protocol["control_hz"])}
        if baseline_path is not None:
            other = baseline[(trial["robot"], trial["condition"], trial["seed"])]
            baseline_trace = Path(other["trace"])
            comparison = passive_comparison(
                [json.loads(line) for line in baseline_trace.read_text().splitlines()], rows)
            comparison.update(baseline_trace=str(baseline_trace), baseline_trace_sha256=sha256_file(baseline_trace),
                              matched_initial_state=other["initial_state_sha256"] == trial["initial_state_sha256"])
            audited["passive_comparison"] = comparison
        trials.append(audited)
    return {"schema": "visual_feedback_audit_v1", "scoring_only": True,
            "report": str(report_path), "report_sha256": sha256_file(report_path),
            "protocol_sha256": sha256_file(report_path.with_name("protocol.json")),
            "scope": "Retrospective sensing and trajectory audit; run audit_frozen_act_results.py separately for trial integrity",
            "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    write_json(args.out, audit_benchmark(args.report, baseline_path=args.baseline), atomic=True)


if __name__ == "__main__":
    main()
