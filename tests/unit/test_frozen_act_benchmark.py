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


@pytest.mark.parametrize("condition", ["rgb", "hf_rgb"])
def test_geometric_rgb_conditions_are_rejected_before_any_run_is_written(tmp_path, condition):
    out = tmp_path / "removed_condition"
    with pytest.raises(ValueError, match="supported conditions"):
        benchmark.run(out, ["so101", "panda"], [condition], [23000],
                      snapshot=tmp_path / "unused", plan_only=True)
    assert not out.exists()
    with pytest.raises(ValueError, match="no longer supported"):
        benchmark.run_episode(None, condition, 23000, out)


def test_benchmark_cli_defaults_to_raw_act_only(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, "argv", ["benchmark_frozen_act.py", "--out", str(tmp_path / "run"), "--plan-only"])
    benchmark.main()
    assert calls[0][0][2] == ["raw"]
    assert calls[0][1]["device"] == "cpu"
    assert calls[0][1]["inference_backend"] == "legacy"


def test_benchmark_cli_forwards_opt_in_device_and_inference_backend(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_frozen_act.py",
            "--out",
            str(tmp_path / "run"),
            "--device",
            "mps",
            "--inference-backend",
            "vectorized",
            "--plan-only",
        ],
    )
    benchmark.main()
    assert calls[0][1]["device"] == "mps"
    assert calls[0][1]["inference_backend"] == "vectorized"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_frozen_act.py",
            "--out",
            str(tmp_path / "run"),
            "--inference-backend",
            "unknown",
            "--plan-only",
        ],
    )
    with pytest.raises(SystemExit):
        benchmark.main()


def test_benchmark_import_does_not_load_geometric_rgb_control():
    program = """
import importlib.abc
import sys
class BlockGeometry(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'scripts.evaluate_hf_recovery', 'real2sim.policy.visual_waypoint',
                        'real2sim.policy.visual_targets', 'real2sim.policy.hf_recovery'}:
            raise AssertionError('benchmark imported geometric RGB control: ' + fullname)
sys.meta_path.insert(0, BlockGeometry())
import scripts.benchmark_frozen_act
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_raw_episode_never_constructs_a_recovery_controller(tmp_path, monkeypatch):
    calls = []

    class NativeDriver:
        def step(self, obs, env):
            calls.append("native ACT")
            metrics = {"ee_ball_distance_m": .1, "ball_held": False, "lifted": False,
                "ball_in_bowl": False, "ball_resting": True, "ball_xyz": [0., 0., .03],
                "ball_xyz_task": [0., 0., .03], "tool_xyz_task": [0., 0., .1]}
            return ApplyResult({}, {"control": {}, "privileged_metrics": metrics}, True, 0.,
                detail={"uses_act": True, "intervention": False})

    env = SimpleNamespace(control_hz=25., table_top=0., max_episode_steps=1,
        task={"episode": {"success_hold_s": .04}, "ball": {"radius": .03}})
    shared = SimpleNamespace(env=env, source=None, robot="so101", reset=lambda *_: ({}, {}))
    monkeypatch.setattr(benchmark, "FrozenDriver", lambda *args: NativeDriver())
    monkeypatch.setattr(benchmark, "initial_fingerprint", lambda *_: {"native_reset": True})
    row = benchmark.run_episode(shared, "raw", 23000, tmp_path / "raw")
    assert calls == ["native ACT"]
    assert row["act_ticks"] == 1 and row["assistance_ticks"] == 0
    assert row["intervention_count"] == 0 and row["perception_log"] == []
    assert row["abort"] is None


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


def test_resume_validates_trace_before_loading_policy_or_scene(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"reference")
    out = tmp_path / "run"
    args = (out, ["so101"], ["raw"], [13000])
    plan = benchmark.run(*args, snapshot=tmp_path, plan_only=True)
    attempt = out / "so101/raw/seed_013000/attempt_0001"
    attempt.mkdir(parents=True)
    path, data = artifact(attempt)
    data["protocol_id"] = plan["protocol_id"]
    data["trial"].update(initial_state_sha256="reset", intervention_required=False)
    data["trial_sha256"] = digest_json(data["trial"])
    path.write_text(json.dumps(data))
    monkeypatch.setattr(benchmark, "aggregate", lambda rows: {"n": len(rows)})
    monkeypatch.setattr(benchmark, "SharedEpisode", lambda *a, **kw: pytest.fail("scene loaded"))
    monkeypatch.setattr(benchmark, "NativeLeRobotACT", lambda *_: pytest.fail("policy loaded"))

    report = benchmark.run(*args, snapshot=tmp_path)
    assert report["trials"] == [data["trial"]]
    (attempt / "ticks.jsonl").write_text("changed")
    with pytest.raises(pytest.fail.Exception, match="policy loaded"):
        benchmark.run(*args, snapshot=tmp_path)


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
    driver_source = Path(benchmark.raw_act.__file__)
    assert hashes[str(driver_source)] == sha256_file(driver_source)
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
@pytest.mark.parametrize("condition", ["learned", "no_act", "hf_learned"])
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


@pytest.mark.parametrize("device", ["", "  ", "cpu ", "unknown", None])
def test_invalid_device_is_rejected_before_writing_output(tmp_path, device):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"frozen ACT")
    out = tmp_path / "invalid_device"
    with pytest.raises(ValueError, match="device"):
        benchmark.run(out, ["so101"], ["raw"], [13000], snapshot=snapshot,
                      device=device, plan_only=True)
    assert not out.exists()


def test_invalid_inference_backend_is_rejected_before_writing_output(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"frozen ACT")
    out = tmp_path / "invalid_backend"
    with pytest.raises(ValueError, match="inference_backend"):
        benchmark.run(out, ["so101"], ["raw"], [13000], snapshot=snapshot,
                      inference_backend="unknown", plan_only=True)
    assert not out.exists()


def test_runtime_protocol_preserves_default_compatibility_and_blocks_incompatible_resume(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"frozen ACT")

    default_out = tmp_path / "default"
    default_plan = benchmark.run(default_out, ["so101"], ["raw"], [13000],
                                 snapshot=snapshot, plan_only=True)
    assert "native_act_runtime" not in default_plan
    assert benchmark.run(default_out, ["so101"], ["raw"], [13000],
                         snapshot=snapshot, device="cpu", inference_backend="legacy",
                         plan_only=True) == default_plan

    for runtime in (("cpu", "vectorized"), ("mps", "legacy"), ("mps", "vectorized")):
        with pytest.raises(ValueError, match="protocol/inputs changed"):
            benchmark.run(default_out, ["so101"], ["raw"], [13000], snapshot=snapshot,
                          device=runtime[0], inference_backend=runtime[1], plan_only=True)

    opt_in_out = tmp_path / "mps_vec"
    opt_in_plan = benchmark.run(opt_in_out, ["so101"], ["raw"], [13000], snapshot=snapshot,
                                device="mps", inference_backend="vectorized", plan_only=True)
    assert opt_in_plan["native_act_runtime"] == {"device": "mps", "inference_backend": "vectorized"}
    assert opt_in_plan["protocol_id"] != default_plan["protocol_id"]
    assert benchmark.run(opt_in_out, ["so101"], ["raw"], [13000], snapshot=snapshot,
                         device="mps", inference_backend="vectorized", plan_only=True) == opt_in_plan

    for incompatible in (("cpu", "legacy"), ("cpu", "vectorized"), ("mps", "legacy")):
        with pytest.raises(ValueError, match="protocol/inputs changed"):
            benchmark.run(opt_in_out, ["so101"], ["raw"], [13000], snapshot=snapshot,
                          device=incompatible[0], inference_backend=incompatible[1], plan_only=True)


def _fake_episode_runtime(monkeypatch, snapshot):
    class Shared:
        def __init__(self, robot, directory, *args, **kwargs):
            self.robot = robot
            self.source = None
            self.env = SimpleNamespace(
                control_hz=25., table_top=0., max_episode_steps=1,
                task={"episode": {"success_hold_s": .04}, "ball": {"radius": .03}},
            )

        def reset(self, seed, scenario=None):
            return {}, {}

        def close(self):
            pass

    class Driver:
        def __init__(self, shared, policy, execute_steps=None):
            self.policy = policy

        def step(self, obs, env):
            metrics = {
                "ee_ball_distance_m": .1, "ball_held": False, "lifted": False,
                "ball_in_bowl": False, "ball_resting": True,
                "ball_xyz": [0., 0., .03], "ball_xyz_task": [0., 0., .03],
                "tool_xyz_task": [0., 0., .1],
            }
            return ApplyResult({}, {"control": {}, "privileged_metrics": metrics}, True, 0.,
                               detail={"uses_act": True, "intervention": False})

    monkeypatch.setattr(benchmark, "SharedEpisode", Shared)
    monkeypatch.setattr(benchmark, "FrozenDriver", Driver)
    monkeypatch.setattr(benchmark, "initial_fingerprint", lambda *_: {"native_reset": True})
    monkeypatch.setattr(benchmark, "lighting", lambda *_: {"ambient": [0., 0., 0.]})
    monkeypatch.setattr(benchmark, "runtime_hashes", lambda *_: {
        str((snapshot / "model.safetensors").resolve()): sha256_file(snapshot / "model.safetensors")
    })


def test_opt_in_runtime_passes_device_and_backend_and_preserves_lazy_resume(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"frozen ACT")
    _fake_episode_runtime(monkeypatch, snapshot)

    loaded = []

    class Policy:
        def __init__(self, path, *, device="cpu", inference_backend="legacy"):
            loaded.append((Path(path), device, inference_backend))
            self.weights_sha256 = sha256_file(weights)
            self.scene_contract = {"camera_keys": ["observation.images.iphone"]}

    monkeypatch.setattr(benchmark, "NativeLeRobotACT", Policy)
    out = tmp_path / "run"
    first = benchmark.run(out, ["so101"], ["raw"], [13000], snapshot=snapshot,
                          device="mps", inference_backend="vectorized")
    assert loaded == [(snapshot, "mps", "vectorized")]
    assert len(first["trials"]) == 1

    monkeypatch.setattr(benchmark, "SharedEpisode", lambda *a, **kw: pytest.fail("scene loaded"))
    monkeypatch.setattr(benchmark, "NativeLeRobotACT", lambda *a, **kw: pytest.fail("policy loaded"))
    resumed = benchmark.run(out, ["so101"], ["raw"], [13000], snapshot=snapshot,
                            device="mps", inference_backend="vectorized")
    assert resumed["trials"] == first["trials"]

    with pytest.raises(ValueError, match="protocol/inputs changed"):
        benchmark.run(out, ["so101"], ["raw"], [13000], snapshot=snapshot)


def test_opt_in_runtime_preserves_no_act_and_forwards_policy_to_learned_factory(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"frozen ACT")
    _fake_episode_runtime(monkeypatch, snapshot)

    seen_policies = []

    class Controller:
        def reset(self):
            pass

        def step(self, obs, env):
            metrics = {
                "ee_ball_distance_m": .1, "ball_held": False, "lifted": False,
                "ball_in_bowl": False, "ball_resting": True,
                "ball_xyz": [0., 0., .03], "ball_xyz_task": [0., 0., .03],
                "tool_xyz_task": [0., 0., .1],
            }
            return ApplyResult({}, {"control": {}, "privileged_metrics": metrics}, True, 0.,
                               detail={"uses_act": False, "intervention": True})

    def fake_factory(*, options, **_kwargs):
        seen_policies.append(options.get("_frozen_act_policy"))
        return Controller()

    monkeypatch.setattr(benchmark, "NativeLeRobotACT", lambda *a, **kw: pytest.fail("no_act loaded policy"))
    monkeypatch.setattr(
        benchmark.importlib,
        "import_module",
        lambda name: SimpleNamespace(factory=fake_factory, __file__=__file__),
    )
    no_act_report = benchmark.run(
        tmp_path / "no_act", ["so101"], ["no_act"], [13000], snapshot=snapshot,
        controller_factory="fake_mod:factory", device="mps", inference_backend="vectorized",
    )
    assert len(no_act_report["trials"]) == 1
    assert seen_policies == [None]

    loaded = []

    class Policy:
        def __init__(self, path, *, device="cpu", inference_backend="legacy"):
            loaded.append((Path(path), device, inference_backend))
            self.weights_sha256 = sha256_file(weights)
            self.scene_contract = None

    monkeypatch.setattr(benchmark, "NativeLeRobotACT", Policy)
    learned_report = benchmark.run(
        tmp_path / "learned", ["so101"], ["learned"], [13000], snapshot=snapshot,
        controller_factory="fake_mod:factory", device="mps", inference_backend="vectorized",
    )
    assert len(learned_report["trials"]) == 1
    assert loaded == [(snapshot, "mps", "vectorized")]
    assert isinstance(seen_policies[1], Policy)
