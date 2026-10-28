"""A benchmark must retain failures without resuming partial or stale artifacts."""

import json

from real2sim.artifacts import sha256_file
from scripts.benchmark_frozen_act import aggregate, completed_trial, digest_json


def artifact(tmp_path, *, abort=None):
    trace = tmp_path / "ticks.jsonl"
    trace.write_text("" if abort else '{"tick": 0}\n')
    row = {"robot": "so101", "condition": "raw", "seed": 13000,
           "pick_place_success": False, "ticks": 0 if abort else 1,
           "trace": str(trace), "abort": abort}
    data = {"status": "complete", "protocol_id": "frozen_protocol",
            "runtime_source_changes": [], "lighting_unchanged": True,
            "checkpoint_unchanged": True, "trial": row,
            "trial_sha256": digest_json(row), "trace_sha256": sha256_file(trace)}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data))
    return path, data


def test_complete_failed_trial_is_resumable_and_keeps_its_denominator(tmp_path):
    path, data = artifact(tmp_path, abort="Renderer unavailable")
    assert completed_trial(path, "frozen_protocol", robot="so101", condition="raw", seed=13000) == data["trial"]


def test_partial_edited_or_wrong_seed_attempts_are_not_resumed(tmp_path):
    path, data = artifact(tmp_path)
    assert completed_trial(path, "frozen_protocol") is not None
    assert completed_trial(path, "other_protocol") is None
    assert completed_trial(path, "frozen_protocol", seed=13001) is None
    data["status"] = "running"
    path.write_text(json.dumps(data))
    assert completed_trial(path, "frozen_protocol") is None
    data["status"] = "complete"
    data["trial"]["pick_place_success"] = True
    path.write_text(json.dumps(data))
    assert completed_trial(path, "frozen_protocol") is None
    data["trial_sha256"] = digest_json(data["trial"])
    path.write_text(json.dumps(data))
    (tmp_path / "ticks.jsonl").write_text('{"tick": 0}\n{"tick": 1}\n')
    assert completed_trial(path, "frozen_protocol") is None


def test_aggregate_counts_aborted_and_unsuccessful_trials():
    defaults = {"success": False, "pick_place_success": False, "lifted": False,
                "delivered": False, "released": False, "rolled_out": False,
                "ball_escaped_table": False, "phase_failed": "approach",
                "pick_place_phase_failed": "approach", "ik_rejections": 0,
                "mean_latency_ms": None, "max_action_saturation": 0.,
                "intervention_required": False, "act_fraction": 1.,
                "assistance_fraction": 0., "simulated_time_s": 30., "wall_s": 2.,
                "abort": None}
    rows = [dict(defaults), dict(defaults, abort="setup", phase_failed="aborted",
            pick_place_phase_failed="aborted"), dict(defaults, success=True,
            pick_place_success=True, phase_failed="none", pick_place_phase_failed="none",
            intervention_required=True, act_fraction=.2, assistance_fraction=.8)]
    result = aggregate(rows)
    assert result["n"] == 3 and result["aborted"] == 1
    assert result["pick_place_success_rate"] == 1 / 3
    assert result["intervention_rate"] == 1 / 3
