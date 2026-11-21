"""A raw policy cohort cannot substitute assistance, retries or push-only success."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from real2sim.artifacts import sha256_file
from real2sim.evaluation import raw_act
from real2sim.policy.adapters import ApplyResult
from scripts import benchmark_frozen_act


@pytest.mark.parametrize("runner", [raw_act, benchmark_frozen_act])
@pytest.mark.parametrize("outcome, expected_ticks", [("push_only", 6), ("pick_place", 5), ("truncated", 3)])
def test_runner_stops_on_strict_placement_or_episode_end(tmp_path, monkeypatch, runner, outcome, expected_ticks):
    samples = [(False, True)] * 6
    if outcome == "pick_place":
        samples[2] = (True, False)
    ticks = iter(enumerate(samples, start=1))

    def step(obs, env):
        tick, (held, in_bowl) = next(ticks)
        metrics = {"ee_ball_distance_m": .01, "ball_held": held, "lifted": held,
                   "ball_in_bowl": in_bowl, "ball_over_bowl": in_bowl,
                   "ball_resting": in_bowl and not held, "ball_xyz": [0., 0., .1],
                   "ball_xyz_task": [0., 0., .1], "tool_xyz_task": [0., 0., .1]}
        return ApplyResult({}, {"control": {}, "privileged_metrics": metrics},
                           outcome == "truncated" and tick == 3, 0.,
                           detail={"uses_act": True, "intervention": False})

    env = SimpleNamespace(control_hz=25., table_top=0., max_episode_steps=len(samples),
                          task={"episode": {"success_hold_s": .08}, "ball": {"radius": .03}})
    shared = SimpleNamespace(env=env, robot="so101", reset=lambda *_: ({}, {}))
    policy = SimpleNamespace(weights_sha256="weights")
    monkeypatch.setattr(runner, "FrozenDriver", lambda *_: SimpleNamespace(step=step))
    monkeypatch.setattr(runner, "initial_fingerprint", lambda *_: {"seed": 15000})
    args = (shared,) if runner is raw_act else (shared, "raw")
    row = runner.run_episode(*args, seed=15000, directory=tmp_path / "trial", policy=policy)

    assert row["ticks"] == row["act_ticks"] == expected_ticks
    assert row["success"] is True
    assert row["pick_place_success"] is (outcome == "pick_place")
    assert row["abort"] is None


def test_plan_hashes_live_evaluator_and_checkpoint(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"weights")
    (snapshot / "config.json").write_text('{"chunk_size": 10, "n_action_steps": 5}')

    plan = raw_act.run(tmp_path / "plan", snapshot=snapshot, plan_only=True)
    hashes = plan["runtime_source_sha256"]
    assert hashes[str(weights)] == plan["checkpoint_sha256"] == sha256_file(weights)
    assert hashes[raw_act.__file__] == sha256_file(raw_act.__file__)


def successful_cohort(seeds=tuple(range(15000, 15020))):
    return [{"robot": robot, "seed": seed, "controller_source": "raw", "ticks": 10,
        "act_ticks": 10, "assistance_ticks": 0, "intervention_count": 0,
        "pick_place_success": True, "success": True, "abort": None}
        for robot in raw_act.ROBOTS for seed in seeds]


def test_gate_requires_twenty_complete_unassisted_tasks_on_each_arm():
    seeds = tuple(range(15000, 15020))
    rows = successful_cohort(seeds)
    gate = raw_act.completion_gate(rows, raw_act.ROBOTS, seeds)
    assert gate["objective_met"]
    assert all(row["gate_met"] for row in gate["per_robot"].values())


@pytest.mark.parametrize("change", ["push", "abort", "assistance", "no_act", "missing", "duplicate", "integrity"])
def test_every_failed_missing_or_assisted_attempt_closes_gate(change):
    seeds = tuple(range(15000, 15020))
    rows = successful_cohort(seeds)
    integrity = True
    if change == "push":
        rows[0]["pick_place_success"] = False  # Generic bowl success stays true.
    elif change == "abort":
        rows[0]["abort"] = "physics failed"
    elif change == "assistance":
        rows[0]["assistance_ticks"] = 1
    elif change == "no_act":
        rows[0]["act_ticks"] -= 1
    elif change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0].copy()
    else:
        integrity = False
    assert not raw_act.completion_gate(rows, raw_act.ROBOTS, seeds,
                                       integrity_verified=integrity)["objective_met"]


def test_pilot_or_single_arm_cannot_claim_both_arm_twenty_of_twenty():
    assert not raw_act.completion_gate(successful_cohort((15000,)), raw_act.ROBOTS,
                                       (15000,))["objective_met"]
    seeds = tuple(range(15000, 15020))
    rows = [row for row in successful_cohort(seeds) if row["robot"] == "so101"]
    assert not raw_act.completion_gate(rows, ("so101",), seeds)["objective_met"]


@pytest.mark.parametrize("training,validation", [([15000], []), ([], [15001])])
def test_evaluation_seed_roles_cannot_overlap(training, validation):
    with pytest.raises(ValueError, match="overlap"):
        raw_act.validate_cohort(raw_act.ROBOTS, [15000, 15001], training, validation)


def test_importing_raw_runner_does_not_import_rgb_controllers():
    program = """
import importlib.abc
import sys
class BlockControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'scripts.benchmark_frozen_act', 'scripts.evaluate_hf_recovery',
                        'real2sim.policy.visual_waypoint', 'real2sim.policy.visual_targets'}:
            raise AssertionError('raw runner imported geometric controller: ' + fullname)
sys.meta_path.insert(0, BlockControllers())
import real2sim.evaluation.raw_act
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"chunk_size": 50, "n_action_steps": 10}))
    (snapshot / "model.safetensors").write_bytes(b"frozen native weights")
    (snapshot / "decoder_weight_edit_report.json").write_text(json.dumps({
        "train_episodes": [[0, 10160], [1, 10161]], "validation_episodes": [[0, 11160]]}))

    class Policy:
        def __init__(self, path, **kwargs):
            self.weights_sha256 = sha256_file(snapshot / "model.safetensors")
            self.scene_contract = {"saved": "native"}

    class Shared:
        failure_seed = None

        def __init__(self, robot, directory, *args, **kwargs):
            self.robot = robot
            self.env = SimpleNamespace(control_hz=25., table_top=0., max_episode_steps=3,
                task={"episode": {"success_hold_s": .04}, "ball": {"radius": .03}})
            self.closed = False

        def reset(self, seed, scenario=None):
            self.env.seed, self.env.tick = seed, 0
            return {}, {}

        def close(self):
            self.closed = True

    class Driver:
        def __init__(self, shared, policy, *args):
            self.shared = shared

        def step(self, obs, env):
            if env.seed == Shared.failure_seed:
                raise RuntimeError("actuator failure")
            held = env.tick == 0
            metrics = {"ee_ball_distance_m": .01, "ball_held": held, "lifted": held,
                "ball_in_bowl": not held, "ball_over_bowl": True, "ball_resting": not held,
                "ball_xyz": [0., 0., .1], "ball_xyz_task": [0., 0., .1],
                "tool_xyz_task": [0., 0., .1]}
            env.tick += 1
            return ApplyResult({}, {"control": {}, "privileged_metrics": metrics}, False, 0.,
                               detail={"uses_act": True, "intervention": False})

    monkeypatch.setattr(raw_act, "NativeLeRobotACT", Policy)
    monkeypatch.setattr(raw_act, "SharedEpisode", Shared)
    monkeypatch.setattr(raw_act, "FrozenDriver", Driver)
    monkeypatch.setattr(raw_act, "initial_fingerprint", lambda shared, info: {"seed": shared.env.seed})
    monkeypatch.setattr(raw_act, "runtime_hashes", lambda *args:
                        {str(snapshot / "model.safetensors"): sha256_file(snapshot / "model.safetensors")})
    return snapshot, Shared


def test_full_cohort_and_recomputed_trace_audit_open_gate(fake_runtime, tmp_path):
    snapshot, _ = fake_runtime
    report = raw_act.run(tmp_path / "run", snapshot=snapshot)
    assert report["objective_met"] and report["evidence_verified"]
    assert len(report["trials"]) == 40
    assert all(row["ticks"] == row["act_ticks"] == 2 for row in report["trials"])
    assert raw_act.audit_report(tmp_path / "run/report.json")["locally_verified"]


def test_aborted_attempts_stay_in_denominator_and_cannot_be_overwritten(fake_runtime, tmp_path):
    snapshot, shared = fake_runtime
    shared.failure_seed = 15000
    out = tmp_path / "run"
    report = raw_act.run(out, snapshot=snapshot)
    assert not report["objective_met"] and report["evidence_verified"]
    assert len(report["trials"]) == 40
    assert all(value["attempted"] == 20 and value["pick_place_successes"] == 19
               for value in report["gate"]["per_robot"].values())
    assert len(list(out.glob("*/seed_*/report.json"))) == 40
    with pytest.raises(FileExistsError, match="failures cannot be overwritten"):
        raw_act.run(out, snapshot=snapshot)


def test_known_export_fit_roles_reject_evaluation_leak_before_writing(fake_runtime, tmp_path):
    snapshot, _ = fake_runtime
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="overlap"):
        raw_act.run(out, snapshot=snapshot, seeds=(10160,))
    assert not out.exists()


def test_decoder_adaptation_report_seed_lists_and_episode_roles_are_all_excluded(tmp_path):
    (tmp_path / "decoder_adaptation_report.json").write_text(json.dumps({
        "train_episodes": [[0, 20010]], "validation_episodes": [[1, 21010]],
        "training_seeds": [20011, 20012], "validation_seeds": [21011]}))
    train, validation, declarations = raw_act.known_fitting_seeds(tmp_path)
    assert train == {20010, 20011, 20012}
    assert validation == {21010, 21011}
    assert len(declarations) == 1


@pytest.mark.parametrize("tamper", ["trace", "leaf", "score"])
def test_audit_rejects_changed_evidence_and_false_task_claims(fake_runtime, tmp_path, tamper):
    snapshot, _ = fake_runtime
    out = tmp_path / "run"
    raw_act.run(out, snapshot=snapshot, seeds=(15000,))
    report_path = out / "report.json"
    report = json.loads(report_path.read_text())
    row = report["trials"][0]
    trace = Path(row["trace"])
    if tamper == "trace":
        trace.write_text("")
    elif tamper == "leaf":
        (trace.parent / "report.json").write_text("{}")
    else:
        ticks = [json.loads(line) for line in trace.read_text().splitlines()]
        for tick in ticks:
            tick["scoring_only"].update(ball_held=False, lifted=False)
        trace.write_text("".join(json.dumps(tick) + "\n" for tick in ticks))
        row["trace_sha256"] = sha256_file(trace)
        leaf_path = trace.parent / "report.json"
        leaf = json.loads(leaf_path.read_text())
        leaf.update(trial=row, trial_sha256=raw_act.digest_json(row))
        leaf_path.write_text(json.dumps(leaf))
        report_path.write_text(json.dumps(report))
    audit = raw_act.audit_report(report_path)
    assert not audit["locally_verified"] and not audit["gate"]["objective_met"]
