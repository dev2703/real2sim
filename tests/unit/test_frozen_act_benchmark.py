"""A benchmark must retain failures without resuming partial or stale artifacts."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from real2sim.artifacts import sha256_file
from real2sim.policy.adapters import ApplyResult
from scripts import benchmark_frozen_act as benchmark
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


@pytest.mark.parametrize("option", ["head", "head_dir", "checkpoint", "model"])
def test_head_changes_invalidate_protocol_and_recovery_runtime_is_hashed(tmp_path, option):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"frozen ACT")
    head = tmp_path / "head"
    head.mkdir()
    weights = head / "head.pt"
    weights.write_bytes(b"first head")
    # Checkpoint may name an individual file; the other options usually name a directory.
    options = {option: str(weights if option == "checkpoint" else head)}
    first = benchmark.run(tmp_path / "first", ["so101"], ["learned"], [12000],
                          snapshot=snapshot, controller_options=options, plan_only=True)
    hashes = first["runtime_source_sha256"]
    assert hashes[str(weights.resolve())] == sha256_file(weights)
    for name in ("common_recovery.py", "recovery_mlp.py", "phase_recovery_mlp.py",
                 "rgb_reference_tracker.py"):
        dependency = Path("src/real2sim/policy") / name
        assert hashes[str(dependency.resolve())] == sha256_file(dependency)
    weights.write_bytes(b"updated head")
    with pytest.raises(ValueError, match="protocol/inputs changed"):
        benchmark.run(tmp_path / "first", ["so101"], ["learned"], [12000],
                      snapshot=snapshot, controller_options=options, plan_only=True)
    second = benchmark.run(tmp_path / "second", ["so101"], ["learned"], [12000],
                           snapshot=snapshot, controller_options=options, plan_only=True)
    assert second["protocol_id"] != first["protocol_id"]


def test_both_condition_heads_enter_matched_protocol(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"frozen ACT")
    heads = {}
    for condition in ("learned", "no_act"):
        directory = tmp_path / condition
        directory.mkdir()
        (directory / "head.pt").write_bytes(condition.encode())
        heads[condition] = str(directory)
    options = {"heads": heads}
    protocol = benchmark.run(tmp_path / "run", ["so101", "panda"],
        ["learned", "no_act"], [12000], snapshot=snapshot,
        controller_options=options, plan_only=True)
    for directory in heads.values():
        path = Path(directory) / "head.pt"
        assert protocol["runtime_source_sha256"][str(path.resolve())] == sha256_file(path)
    (Path(heads["no_act"]) / "head.pt").write_bytes(b"changed no-ACT head")
    with pytest.raises(ValueError, match="protocol/inputs changed"):
        benchmark.run(tmp_path / "run", ["so101", "panda"], ["learned", "no_act"],
            [12000], snapshot=snapshot, controller_options=options, plan_only=True)


@pytest.mark.parametrize("failure", [None, "reset", "step"])
@pytest.mark.parametrize("condition", ["learned", "no_act"])
def test_custom_controller_releases_hooks_after_completion_or_abort(
        tmp_path, monkeypatch, failure, condition):
    class Controller:
        closes = 0

        def reset(self):
            if failure == "reset":
                raise RuntimeError("reset failed")

        def step(self, obs, env):
            if failure == "step":
                raise RuntimeError("step failed")
            metrics = {"ee_ball_distance_m": .1, "ball_held": False, "lifted": False,
                       "ball_in_bowl": False, "ball_resting": True,
                       "ball_xyz": [0., 0., .02], "ball_xyz_task": [0., 0., .02],
                       "tool_xyz_task": [0., 0., .1]}
            return ApplyResult({}, {"control": {}, "privileged_metrics": metrics}, True, 0.)

        def close(self):
            self.closes += 1

    controller = Controller()
    env = SimpleNamespace(control_hz=25., table_top=0., max_episode_steps=1,
                          task={"episode": {"success_hold_s": .1}, "ball": {"radius": .02}})
    shared = SimpleNamespace(env=env, source=None, robot="so101", reset=lambda *_: ({}, {}))
    monkeypatch.setattr(benchmark, "initial_fingerprint", lambda *_: {"reset": "matched"})
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"frozen reference")
    row = benchmark.run_episode(shared, condition, 12000, tmp_path / "attempt",
                                controller_factory=lambda **_: controller,
                                options={"snapshot": str(tmp_path)})
    assert controller.closes == 1
    assert row["ticks"] == (0 if failure else 1)
    assert row["abort"] == (f"RuntimeError: {failure} failed" if failure else None)
    assert row["checkpoint_sha256"] == sha256_file(checkpoint)


def test_direct_and_module_cli_import_the_same_controller_factory(tmp_path):
    root = Path(__file__).resolve().parents[2]
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"frozen ACT")
    head = tmp_path / "head"
    head.mkdir()
    (head / "head.pt").write_bytes(b"recovery")
    protocols = []
    for name, invocation in (("direct", [str(root / "scripts/benchmark_frozen_act.py")]),
                             ("module", ["-m", "scripts.benchmark_frozen_act"])):
        out = tmp_path / name
        command = [sys.executable, *invocation, "--out", str(out), "--robots", "so101",
                   "--conditions", "learned", "--seeds", "12000", "--snapshot", str(snapshot),
                   "--controller-factory", "scripts.run_recovery_dagger:controller_factory",
                   "--controller-options", json.dumps({"head": str(head)}), "--plan-only"]
        subprocess.run(command, cwd=root, env=dict(os.environ, PYTHONPATH=str(root / "src")),
                       capture_output=True, text=True, check=True)
        protocols.append(json.loads((out / "protocol.json").read_text()))
    assert protocols[0] == protocols[1]
    factory = root / "scripts/run_recovery_dagger.py"
    assert protocols[0]["runtime_source_sha256"][str(factory)] == sha256_file(factory)
