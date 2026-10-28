"""DAgger's expert clock follows executed jaw intent, not hypothetical labels."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.artifacts import sha256_file
from scripts import run_recovery_dagger as dagger_module
from scripts.evaluate_hf_recovery import StaticBowlRecovery
from scripts.run_recovery_dagger import (
    CartesianConvergenceTeacher,
    snapshot_sources,
    teacher_common_action,
)


@pytest.mark.parametrize("phase,selected,paused", [
    ("CLOSE", 1., True), ("CLOSE", 0., False),
    ("RELEASE", 0., True), ("RELEASE", 1., False),
])
def test_expert_dwell_does_not_count_opposing_executed_command(monkeypatch, phase, selected, paused):
    teacher = object.__new__(CartesianConvergenceTeacher)
    teacher.phase, teacher.phase_tick = phase, 12
    teacher.target_q = None
    teacher.executed_semantic_gripper = selected

    def step(self, obs, env):
        self.phase_tick += 1
        return self.phase_tick

    monkeypatch.setattr(StaticBowlRecovery, "step", step)
    assert teacher.step({}, object()) == (12 if paused else 13)


def test_panda_equivalent_tcp_allows_phase_progress_without_original_joint_branch(monkeypatch):
    teacher = object.__new__(CartesianConvergenceTeacher)
    teacher.phase, teacher.phase_tick = "PRE_GRASP", 12
    teacher.target_q = np.arange(7, dtype=float)
    teacher.n = 7
    measured = np.zeros(7)
    teacher._native = lambda: measured
    teacher._grip = lambda opened: .08 if opened else 0.
    teacher._forward = lambda *_: np.eye(4)
    monkeypatch.setattr(StaticBowlRecovery, "step", lambda *_: None)
    teacher.step({}, SimpleNamespace(ee_pose=lambda: np.eye(4)))
    np.testing.assert_array_equal(teacher.target_q, measured)


def test_panda_expert_cartesian_endpoint_is_not_native_nullspace_motion():
    target = np.eye(4)
    target[:3, 3] = [.1, .2, .3]
    teacher = SimpleNamespace(robot="panda", phase="PRE_GRASP", target_q=np.zeros(7),
        _grip=lambda _: .08, _forward=lambda *_: target)
    executor = SimpleNamespace(env=SimpleNamespace(ee_pose=lambda: np.eye(4)),
        native_to_common=lambda _: np.r_[np.zeros(6), 1.])
    action = teacher_common_action(teacher, SimpleNamespace(detail={
        "requested_native": np.zeros(8), "phase": "PRE_GRASP"}), executor)
    assert np.linalg.norm(action[:3]) == pytest.approx(.03)
    np.testing.assert_allclose(action[:3] / np.linalg.norm(action[:3]),
                               target[:3, 3] / np.linalg.norm(target[:3, 3]))
    assert action[6] == 1.


@pytest.mark.parametrize("phase,next_phase,phase_tick,stable", [
    ("CLOSE", "LIFT", 17, 0), ("LOWER", "RELEASE", 2, 1),
])
def test_transition_labels_belong_to_issued_command_and_never_borrow_next_endpoint(
        phase, next_phase, phase_tick, stable):
    teacher = object.__new__(CartesianConvergenceTeacher)
    teacher.robot, teacher.n = "panda", 7
    teacher.phase, teacher.phase_tick, teacher.tick, teacher._stable = phase, phase_tick, 0, stable
    teacher.phase_log, teacher.target_q, teacher.failure = [], np.zeros(7), None
    teacher._native = lambda: np.zeros(7)
    teacher._grip = lambda opened: .08 if opened else 0.
    teacher._forward = lambda *_: np.eye(4)
    teacher._safe = lambda *_: True
    teacher.max_step_m = .03
    env = SimpleNamespace(control_hz=25., max_velocity=np.ones(7), ee_pose=lambda: np.eye(4),
        step=lambda _: ({}, 0., False, False, {"control": {"limit_clipped": [False]}}))
    teacher.env = env
    result = teacher.step({}, env)
    assert teacher.phase == next_phase
    assert result.detail["phase"] == phase
    assert result.detail["requested_native"][-1] == 0.

    # Even if a future implementation prepares the next phase endpoint early,
    # it cannot change the Cartesian label for the command just issued.
    teacher.target_q = np.ones(7)
    teacher._forward = lambda *_: pytest.fail("Used next phase endpoint for previous command")
    expected = np.r_[.001, np.zeros(6)]
    executor = SimpleNamespace(native_to_common=lambda _: expected.copy())
    np.testing.assert_array_equal(teacher_common_action(teacher, result, executor), expected)


def test_executed_source_snapshot_survives_later_edits_and_references_binary_artifacts(tmp_path):
    source, weights = tmp_path / "controller.py", tmp_path / "head.pt"
    source.write_text("action = 1\n")
    weights.write_bytes(b"binary artifact")
    out = tmp_path / "run"
    out.mkdir()
    hashes = {str(path): sha256_file(path) for path in (source, weights)}
    ledger = json.loads(Path(snapshot_sources(out, hashes)).read_text())
    rows = {row["source"]: row for row in ledger["entries"]}
    saved = out / rows[str(source)]["snapshot"]
    assert sha256_file(saved) == hashes[str(source)]
    source.write_text("action = 2\n")
    assert saved.read_text() == "action = 1\n"
    assert rows[str(weights)]["snapshot"] is None
    assert rows[str(weights)]["sha256"] == hashes[str(weights)]


def test_executed_source_snapshot_refuses_source_drift(tmp_path):
    source = tmp_path / "controller.py"
    source.write_text("action = 1\n")
    hashes = {str(source): sha256_file(source)}
    source.write_text("action = 2\n")
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(RuntimeError, match="changed while taking"):
        snapshot_sources(out, hashes)


def test_learned_trace_records_exact_causal_inputs_and_reference_ages(monkeypatch):
    class References:
        def __init__(self, _env):
            self.ball, self.bowl = np.array([.1, .2, .03]), np.array([.3, .1, .03])
            self.log = [{"tick": 0}]

        def update(self, tick, *, force=False):
            return None

        def observation_metadata(self, tick):
            return {"visibility": (False, True), "last_seen_ticks": (0, 5)}

    class Executor:
        def __init__(self, _episode):
            pass

        def step(self, _action):
            return SimpleNamespace(detail={})

    captured = {}

    def predict(observations, base, features):
        captured.update(observations=observations.copy(), base_actions=base.copy(), features=features.copy())
        return np.r_[np.zeros(6), 1.]

    head = SimpleNamespace(metadata={"use_act": False, "observation_names": ["fresh_rgb_ball_relative_x"]},
        head_sha256="head", predict=predict)
    env = SimpleNamespace(control_hz=25., ee_pose=lambda: np.eye(4), gripper_semantic=lambda: .5)
    episode = SimpleNamespace(env=env, robot="panda", source=None)
    monkeypatch.setattr(dagger_module, "CommonCartesianExecutor", Executor)
    monkeypatch.setattr(dagger_module, "RGBReferenceTracker", References)
    controller = dagger_module.LearnedRecoveryController(episode, head)
    controller.tick = 5
    detail = controller.step({}, env).detail
    assert detail["rgb_reference_visibility"] == [False, True]
    assert detail["rgb_reference_age_s"] == [.2, 0.]
    for name, values in captured.items():
        np.testing.assert_array_equal(detail["recovery_inputs"][name], values)
