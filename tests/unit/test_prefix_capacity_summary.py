"""A prefix comparison must reject experimental drift and altered scored inputs."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


def load_summary():
    path = Path(__file__).resolve().parents[2] / "scripts/summarize_v2_prefix_capacity.py"
    spec = importlib.util.spec_from_file_location("prefix_capacity_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plans():
    baseline = {"name": "capacity", "model": {"chunk_size": 25, "n_action_steps": 2,
                "use_vae": False, "kl_weight": 0, "dim_model": 512},
                "train": {"diagnostic_episodes": [11, 8, 23], "seed": 0, "batch_size": 8,
                          "grad_accum": 2, "schedule_steps": 2000}, "experiment": {"kind": "fixed_gate"}}
    prefix = copy.deepcopy(baseline)
    prefix["name"] += "_prefix2"
    prefix["reconstruction_loss"] = {"supervised_prefix_steps": 2}
    prefix["experiment"].update(supervised_prefix_steps=2, comparison="Only temporal support changes")
    return baseline, prefix


def test_only_declared_objective_change_can_enter_the_comparison():
    summary = load_summary()
    baseline, prefix = plans()
    assert summary.verify_plans(baseline, prefix)["matched_effective_batch"] == 16
    prefix["model"]["dim_model"] = 256
    with pytest.raises(ValueError, match="beyond"):
        summary.verify_plans(baseline, prefix)
    baseline, prefix = plans()
    prefix["train"]["seed"] = 1
    with pytest.raises(ValueError, match="beyond"):
        summary.verify_plans(baseline, prefix)
    baseline, prefix = plans()
    prefix["model"]["n_action_steps"] = 3
    with pytest.raises(ValueError, match="two executed"):
        summary.verify_plans(baseline, prefix)


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_payload_and_checkpoint_normalization_tampering_are_rejected(tmp_path):
    summary = load_summary()
    data, checkpoint = tmp_path / "data", tmp_path / "checkpoint"
    data.mkdir()
    checkpoint.mkdir()
    (data / "arrays.npz").write_bytes(b"fixed label payload")
    write_json(data / "stats.json", {"action_std": [1.]})
    files = {name: summary.sha256_file(data / name) for name in ("arrays.npz", "stats.json")}
    manifest = {"files_sha256": files, "action_representation": "cartesian"}
    assert all(summary.verify_payload(data, manifest).values())
    (checkpoint / "model.safetensors").write_bytes(b"fixed weights")
    write_json(checkpoint / "stats.json", {"action_std": [1.]})
    write_json(checkpoint / "real2sim_contract.json", {"derived_files_sha256": files,
                                                        "action_representation": "cartesian"})
    write_json(checkpoint / "config.json", {"chunk_size": 25, "n_action_steps": 2})
    write_json(checkpoint / "run_info.json", {"step": 1000,
        "reconstruction_loss": {"supervised_prefix_steps": 2}})
    stage = {"checkpoint": str(checkpoint), "checkpoint_sha256": summary.sha256_file(checkpoint / "model.safetensors"),
             "examples_seen": 16000}
    _, prefix = plans()
    assert summary.verify_checkpoint(stage, data, manifest, prefix, 2)["normalization_matches"]
    write_json(checkpoint / "stats.json", {"action_std": [2.]})
    with pytest.raises(ValueError, match="normalization"):
        summary.verify_checkpoint(stage, data, manifest, prefix, 2)
    (data / "arrays.npz").write_bytes(b"changed labels")
    with pytest.raises(ValueError, match="payload changed"):
        summary.verify_payload(data, manifest)


def test_changed_scoring_coverage_or_causal_baselines_cannot_be_compared():
    summary = load_summary()
    baseline = {"valid_actions": 983, "executed_actions_per_call": 2,
                "scores": {"hold": {"error": 0.1}, "persistence": {"error": 2.7}}}
    prefix = copy.deepcopy(baseline)
    summary.verify_motion_contract(baseline, prefix)
    prefix["valid_actions"] = 980
    with pytest.raises(ValueError, match="valid_actions"):
        summary.verify_motion_contract(baseline, prefix)
    prefix = copy.deepcopy(baseline)
    prefix["scores"]["hold"]["error"] = .5
    with pytest.raises(ValueError, match="baseline values"):
        summary.verify_motion_contract(baseline, prefix)


def test_frozen_gate_source_must_match_the_recorded_hash(tmp_path):
    summary = load_summary()
    path = tmp_path / "executed_runner.py"
    path.write_text("def gate_from_validation(value):\n    return {'pass': value['pass']}\n")
    ledger = {"source_sha256": {"scripts/run_v2_three_episode_fit.py": summary.sha256_file(path)},
              "gate_version": "unchanged"}
    gate, _, selected = summary.frozen_gate(tmp_path, ledger)
    assert gate({"pass": True}) == {"pass": True} and selected == path
    path.write_text("def gate_from_validation(value):\n    return {'pass': True}\n")
    with pytest.raises(ValueError, match="no archived runner"):
        summary.frozen_gate(tmp_path, ledger)


def test_interrupted_prefix_run_cannot_be_reported_as_final(tmp_path):
    summary = load_summary()
    baseline, prefix = plans()
    import yaml
    dirs = [tmp_path / "baseline", tmp_path / "prefix"]
    for folder, plan in zip(dirs, (baseline, prefix), strict=True):
        folder.mkdir()
        (folder / "plan.yaml").write_text(yaml.safe_dump(plan))
        write_json(folder / "report.json", {"stages": [], "gate_passed": False})
    with pytest.raises(ValueError, match="has not finished"):
        summary.summarize(*dirs)


def test_probe_from_another_checkpoint_cannot_be_attached(tmp_path):
    summary = load_summary()
    path = tmp_path / "object.json"
    write_json(path, {"schema": "canonical_matched_object_probe_v1", "reserved_episodes_used": [],
                      "plan": {"checkpoint": str(tmp_path / "other"), "data": str(tmp_path / "data")}})
    stage = {"checkpoint_integrity": {"checkpoint": str(tmp_path / "selected"), "weights_sha256": "expected"}}
    with pytest.raises(ValueError, match="selected prefix checkpoint"):
        summary.verify_object_probe(path, stage, tmp_path / "data")
