"""A retained aggregate cannot substitute for the individual trial evidence."""

import json

from real2sim.artifacts import sha256_file
from scripts.audit_frozen_act_results import audit_report
from scripts.benchmark_frozen_act import aggregate, digest_json


def benchmark_artifact(tmp_path, conditions=("raw",)):
    protocol = {"robots": ["so101"], "conditions": list(conditions), "seeds": [7],
                "checkpoint_sha256": "frozen ACT"}
    protocol["protocol_id"] = digest_json(protocol)
    rows = []
    for condition in conditions:
        directory = tmp_path / "so101" / condition / "seed_000007" / "attempt_0001"
        directory.mkdir(parents=True)
        trace = directory / "ticks.jsonl"
        assisted = condition != "raw"
        trace.write_text(json.dumps({"tick": 0, "uses_act": not assisted, "assisted": assisted}) + "\n")
        initial = {"robot": "so101", "ball_xyz_task": [0., .1, .02]}
        row = {"robot": "so101", "condition": condition, "seed": 7, "ticks": 1,
               "success": False, "pick_place_success": False, "lifted": False,
               "delivered": False, "released": False, "rolled_out": False,
               "ball_escaped_table": False, "phase_failed": "approach",
               "pick_place_phase_failed": "approach", "ik_rejections": 0,
               "mean_latency_ms": 3., "max_action_saturation": 0., "abort": None,
               "initial_state": initial, "initial_state_sha256": digest_json(initial),
               "checkpoint_sha256": protocol["checkpoint_sha256"],
               "act_ticks": int(not assisted), "assistance_ticks": int(assisted),
               "act_fraction": float(not assisted), "assistance_fraction": float(assisted),
               "intervention_required": assisted, "simulated_time_s": .04, "wall_s": .01,
               "trace": str(trace)}
        attempt = {"status": "complete", "protocol_id": protocol["protocol_id"],
                   "trial": row, "trial_sha256": digest_json(row),
                   "trace_sha256": sha256_file(trace), "runtime_source_changes": [],
                   "lighting_unchanged": True, "checkpoint_unchanged": True}
        (directory / "report.json").write_text(json.dumps(attempt))
        rows.append(row)
    report = {"schema": "matched_frozen_act_results_v1", "status": "complete",
              "protocol_id": protocol["protocol_id"], "trials": rows,
              "matched_initial_states_verified": True,
              "conditions": {f"so101/{condition}": aggregate([rows[i]])
                             for i, condition in enumerate(conditions)}}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return path, report


def save(path, report):
    path.write_text(json.dumps(report))


def test_intact_failed_trials_are_verified_without_becoming_successes(tmp_path):
    path, _ = benchmark_artifact(tmp_path, ("raw", "rgb"))
    result = audit_report(path)
    assert result["status"] == "verified" and result["locally_verified"]
    assert result["locally_verified_trials"] == result["claimed_trials"] == 2
    assert result["conditions"]["so101/raw"]["pick_place_successes"] == 0
    assert result["conditions"]["so101/rgb"]["intervention_rate"] == 1.


def test_pruned_trace_cannot_be_presented_as_locally_verified(tmp_path):
    path, report = benchmark_artifact(tmp_path)
    from pathlib import Path

    Path(report["trials"][0]["trace"]).unlink()
    result = audit_report(path)
    assert result["status"] == "incomplete" and not result["locally_verified"]
    assert result["missing_evidence_count"] == 1 and result["locally_verified_trials"] == 0


def test_edited_summary_cannot_override_retained_trial_or_aggregate(tmp_path):
    path, report = benchmark_artifact(tmp_path)
    report["trials"][0]["pick_place_success"] = True
    save(path, report)
    result = audit_report(path)
    assert result["status"] == "invalid"
    assert any("integrity" in e for e in result["errors"])
    assert any("aggregate" in e for e in result["errors"])
    assert any("required outcome stages" in e for e in result["errors"])


def test_duplicate_trial_is_rejected_even_if_aggregate_is_recomputed(tmp_path):
    path, report = benchmark_artifact(tmp_path)
    report["trials"] *= 2
    report["conditions"]["so101/raw"] = aggregate(report["trials"])
    save(path, report)
    result = audit_report(path)
    assert any("exactly once" in e for e in result["errors"])


def test_different_resets_are_rejected_even_with_consistent_fingerprint_hashes(tmp_path):
    path, report = benchmark_artifact(tmp_path, ("raw", "rgb"))
    row = report["trials"][1]
    row["initial_state"] = {"robot": "so101", "ball_xyz_task": [1., 1., 1.]}
    row["initial_state_sha256"] = digest_json(row["initial_state"])
    save(path, report)
    result = audit_report(path)
    assert any("different initial states" in e for e in result["errors"])


def test_trace_controller_flags_must_agree_with_report_even_after_rehash(tmp_path):
    path, report = benchmark_artifact(tmp_path)
    from pathlib import Path

    trace = Path(report["trials"][0]["trace"])
    trace.write_text('{"tick": 0, "uses_act": false, "assisted": true}\n')
    attempt_path = trace.with_name("report.json")
    attempt = json.loads(attempt_path.read_text())
    attempt["trace_sha256"] = sha256_file(trace)
    save(attempt_path, attempt)
    result = audit_report(path)
    assert any("controller counts differ" in e for e in result["errors"])


def test_protocol_edits_are_rejected_without_valid_identity(tmp_path):
    path, _ = benchmark_artifact(tmp_path)
    protocol_path = path.with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    protocol["checkpoint_sha256"] = "different weights"
    save(protocol_path, protocol)
    result = audit_report(path)
    assert any("protocol identity" in e for e in result["errors"])
