"""Verify matched frozen-ACT claims against retained per-trial evidence.

A summary is historical evidence until its protocol, trial reports, traces and
matched resets can all be checked locally. This audit does not rerun physics or
establish that the current source tree matches a historical runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real2sim.serialization import write_json
from scripts.benchmark_frozen_act import aggregate, completed_trial, digest_json


def audit_report(path: Path) -> dict:
    path = Path(path)
    report = json.loads(path.read_text())
    protocol = json.loads(path.with_name("protocol.json").read_text())
    errors, missing = [], []
    protocol_body = {k: v for k, v in protocol.items() if k != "protocol_id"}
    protocol_id = protocol.get("protocol_id")
    if protocol_id != digest_json(protocol_body) or report.get("protocol_id") != protocol_id:
        errors.append("Report/protocol identity does not match the recorded protocol digest")
    if report.get("schema") != "matched_frozen_act_results_v1":
        errors.append("Unsupported results schema")
    if report.get("status") != "complete":
        errors.append("Benchmark is incomplete")
    expected = {(robot, condition, seed) for robot in protocol["robots"]
                for condition in protocol["conditions"] for seed in protocol["seeds"]}
    rows = report["trials"]
    keys = [(r["robot"], r["condition"], r["seed"]) for r in rows]
    if set(keys) != expected or len(keys) != len(expected):
        errors.append("Trial identities do not cover the protocol exactly once")
    fingerprints, verified = {}, Counter()
    for row, key in zip(rows, keys, strict=True):
        label = f"{key[0]}/{key[1]}/{key[2]}"
        initial = row.get("initial_state")
        fingerprint = row.get("initial_state_sha256")
        if initial is None or fingerprint != digest_json(initial):
            errors.append(f"{label}: missing or inconsistent reset fingerprint")
        reset_key = (key[0], key[2])
        if reset_key in fingerprints and fingerprints[reset_key] != fingerprint:
            errors.append(f"{label}: conditions used different initial states")
        fingerprints[reset_key] = fingerprint
        if row.get("checkpoint_sha256") != protocol.get("checkpoint_sha256"):
            errors.append(f"{label}: checkpoint differs from protocol")
        if row.get("pick_place_success") and (row.get("abort") or not all(
                row.get(k) for k in ("success", "lifted", "delivered", "released"))):
            errors.append(f"{label}: pick-and-place claim lacks required outcome stages")
        ticks = row["ticks"]
        for name, count in (("act", row["act_ticks"]), ("assistance", row["assistance_ticks"])):
            fraction = count / ticks if ticks else 0.
            if type(count) is not int or not 0 <= count <= ticks or row[f"{name}_fraction"] != fraction:
                errors.append(f"{label}: inconsistent {name} accounting")
        if row["intervention_required"] != bool(row["assistance_ticks"]):
            errors.append(f"{label}: inconsistent intervention accounting")
        if row["condition"] == "raw" and (row["act_ticks"] != ticks or row["assistance_ticks"]):
            errors.append(f"{label}: raw ACT claim includes assistance or missing ACT ticks")
        trace = Path(row["trace"])
        attempt = trace.with_name("report.json")
        if not trace.is_file() or not attempt.is_file():
            missing.append(label)
            continue
        retained = completed_trial(attempt, protocol_id, robot=key[0], condition=key[1], seed=key[2])
        if retained is None or digest_json(retained) != digest_json(row):
            errors.append(f"{label}: retained report/trace fails integrity or differs from summary")
            continue
        try:
            act_ticks = assistance_ticks = trace_ticks = 0
            with trace.open() as stream:
                for tick, line in enumerate(stream):
                    event = json.loads(line)
                    if (event["tick"] != tick or type(event["uses_act"]) is not bool
                            or type(event["assisted"]) is not bool):
                        raise ValueError("tick order or controller flags differ")
                    act_ticks += event["uses_act"]
                    assistance_ticks += event["assisted"]
                    trace_ticks += 1
            if (trace_ticks, act_ticks, assistance_ticks) != (ticks, row["act_ticks"], row["assistance_ticks"]):
                raise ValueError("trace controller counts differ from trial report")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{label}: invalid trace: {exc}")
            continue
        verified[(key[0], key[1])] += 1
    conditions = {}
    for robot in protocol["robots"]:
        for condition in protocol["conditions"]:
            selected = [r for r in rows if r["robot"] == robot and r["condition"] == condition]
            name = f"{robot}/{condition}"
            summary = aggregate(selected)
            if digest_json(summary) != digest_json(report.get("conditions", {}).get(name)):
                errors.append(f"{name}: recorded aggregate differs from trial rows")
            conditions[name] = {
                "claimed_trials": len(selected), "locally_verified_trials": verified[(robot, condition)],
                "pick_place_successes": summary["pick_place_successes"],
                "aborted": summary["aborted"],
                "intervention_rate": summary["intervention_rate"],
                "mean_act_fraction": summary["mean_act_fraction"],
                "mean_assistance_fraction": summary["mean_assistance_fraction"],
            }
    if report.get("matched_initial_states_verified") is not True:
        errors.append("Benchmark did not confirm matched initial states")
    return {
        "schema": "frozen_act_evidence_audit_v1", "report": str(path),
        "status": "invalid" if errors else "incomplete" if missing else "verified",
        "locally_verified": not errors and not missing,
        "claimed_trials": len(rows), "locally_verified_trials": sum(verified.values()),
        "missing_evidence_count": len(missing), "missing_evidence": missing,
        "errors": errors, "conditions": conditions,
        "scope": "Recorded protocol, trial/trace hashes, controller accounting, aggregates and matched resets; no physics rerun or current-runtime equivalence claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit_report(args.report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, result, atomic=True)
    print(json.dumps(result, indent=2))
    return 0 if result["locally_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
