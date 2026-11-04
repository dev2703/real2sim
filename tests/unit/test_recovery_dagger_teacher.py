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
def test_expert_dwell_resets_after_opposing_issued_command(monkeypatch, phase, selected, paused):
    teacher = object.__new__(CartesianConvergenceTeacher)
    teacher.phase, teacher.phase_tick = phase, 12
    teacher.target_q = None
    teacher.executed_semantic_gripper = selected

    def step(self, obs, env):
        self.phase_tick += 1
        return self.phase_tick

    monkeypatch.setattr(StaticBowlRecovery, "step", step)
    assert teacher.step({}, object()) == (0 if paused else 13)


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


@pytest.mark.parametrize("feature_dim", [256, 512])
def test_learned_trace_records_exact_causal_inputs_and_reference_ages(monkeypatch, feature_dim):
    class References:
        def __init__(self, _env):
            self.ball, self.bowl = np.array([.1, .2, .03]), np.array([.3, .1, .03])
            self.log = [{"tick": 0}]

        def update(self, tick, *, force=False):
            return None

        def observation_metadata(self, tick):
            return {"visibility": (False, True), "last_seen_ticks": (0, 5)}

    class Executor:
        def __init__(self, _episode, *, require_source_mapping):
            assert require_source_mapping is False
            pass

        def step(self, _action):
            return SimpleNamespace(detail={})

    captured = {}

    def predict(observations, base, features):
        captured.update(observations=observations.copy(), base_actions=base.copy(), features=features.copy())
        return np.r_[np.zeros(6), 1.]

    head = SimpleNamespace(metadata={"use_act": False, "observation_names": ["fresh_rgb_ball_relative_x"],
        "act_feature_dim": feature_dim},
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
    assert captured["features"].shape == (feature_dim,)


def test_empty_pack_keeps_checkpoint_feature_width(tmp_path):
    dagger_module.save_pack(tmp_path / "pack.npz", [], feature_dim=512)
    with np.load(tmp_path / "pack.npz", allow_pickle=False) as pack:
        assert pack["features"].shape == (0, 512)


@pytest.mark.parametrize("robot", ["so101", "panda"])
def test_grasp_monitor_and_rgb_close_anchor_follow_issued_jaw_without_changing_head_history(
        monkeypatch, robot):
    class References:
        def __init__(self, _env):
            self.ball, self.bowl = np.array([.01, 0., .02]), np.array([.3, .1, .03])
            self.log = [{"tick": 0}]
            self.tick = 0

        def update(self, tick, *, force=False):
            self.tick = tick

        def observation_metadata(self, tick):
            return {"visibility": (True, True), "last_seen_ticks": (tick, tick)}

    # SO101 may reject the entire native command, including jaw closure. A
    # Panda arm rejection currently preserves selected jaw intent; the second
    # parameter case also covers actual native-to-semantic width conversion.
    issued = iter([1., 0., 0.])

    class Executor:
        def __init__(self, _episode, *, require_source_mapping):
            assert require_source_mapping is False

        def step(self, _action):
            semantic = next(issued)
            native = np.r_[np.zeros(5), semantic] if robot == "so101" else np.r_[
                np.zeros(7), semantic * .08]
            return SimpleNamespace(detail={"requested_native": native.tolist()})

    observed_history = []

    def predict(observations, _base, _features):
        observed_history.append(observations.copy())
        return np.zeros(7)  # The head requests closure every tick.

    head = SimpleNamespace(metadata={"use_act": False,
        "observation_names": ["fresh_rgb_ball_relative_x"]}, head_sha256="head", predict=predict)
    env = SimpleNamespace(control_hz=25., ee_pose=lambda: np.eye(4), max_opening=.08,
        gripper_semantic=lambda: .7, sim_state=lambda: np.r_[np.zeros(5), .7],
        mapping=SimpleNamespace(gripper_sim_to_real_percent=lambda jaw: jaw * 90.))
    episode = SimpleNamespace(env=env, robot=robot, source=None)
    monkeypatch.setattr(dagger_module, "CommonCartesianExecutor", Executor)
    monkeypatch.setattr(dagger_module, "RGBReferenceTracker", References)
    controller = dagger_module.LearnedRecoveryController(episode, head)
    controller.step({}, env)
    assert controller.grasp_anchor is None
    detail = controller.step({}, env).detail
    assert detail["observable_grasp_feedback"]["commanded_opening"] == 1.
    assert not detail["observable_grasp_feedback"]["closed_command"]
    np.testing.assert_allclose(controller.grasp_anchor, [.01, 0., .02])
    detail = controller.step({}, env).detail
    assert detail["observable_grasp_feedback"]["commanded_opening"] == 0.
    assert detail["observable_grasp_feedback"]["closed_command"]
    # The checkpoint was trained on selected actions; retain that exact causal
    # contract while supplying actual issued command separately to diagnosis.
    assert observed_history[1][-33 + 29] == 0.


@pytest.mark.parametrize("condition,use_act", [("learned", True), ("no_act", False), ("hf_learned", False)])
def test_matched_factory_selects_condition_head_and_reuses_frozen_backbone(
        tmp_path, monkeypatch, condition, use_act):
    (tmp_path / "model.safetensors").write_bytes(b"frozen")
    digest = sha256_file(tmp_path / "model.safetensors")
    backbone = SimpleNamespace(weights_sha256=digest)
    heads = {"learned": str(tmp_path / "act"), "no_act": str(tmp_path / "no_act"),
             "hf_learned": str(tmp_path / "prefix")}
    loaded = []

    def load(directory, base_hash):
        loaded.append((directory, base_hash))
        return SimpleNamespace(metadata={"use_act": use_act})

    monkeypatch.setattr(dagger_module, "load_recovery_head", load)
    monkeypatch.setattr(dagger_module, "NativeLeRobotACT",
                        lambda *_: pytest.fail("Reloaded the shared frozen ACT"))
    monkeypatch.setattr(dagger_module, "LearnedRecoveryController",
                        lambda shared, head, policy, **kwargs: (shared, policy))
    monkeypatch.setattr(dagger_module, "frozen_prefix_controller",
                        lambda shared, recovery, policy: (shared, policy))
    shared, actual_policy = dagger_module.controller_factory(
        env="env", source="source", robot="panda", snapshot=tmp_path,
        condition=condition, options={"heads": heads, "_frozen_act_policy": backbone})
    assert loaded == [(Path(heads[condition]), digest)]
    assert shared.env == "env" and shared.robot == "panda"
    assert actual_policy is (backbone if use_act or condition == "hf_learned" else None)


def test_matched_factory_rejects_shared_backbone_with_wrong_weights(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"frozen")
    monkeypatch.setattr(dagger_module, "load_recovery_head",
                        lambda *_: SimpleNamespace(metadata={"use_act": True}))
    with pytest.raises(ValueError, match="different frozen checkpoint"):
        dagger_module.controller_factory(env=None, source=None, robot="panda", snapshot=tmp_path,
            condition="learned", options={"head": "head", "_frozen_act_policy": SimpleNamespace(
                weights_sha256="other")})


def test_source_frame_routes_observations_actions_and_history_consistently(monkeypatch):
    from real2sim.policy import source_recovery_frame

    proxy, executed, histories = object(), [], []
    source_action = np.r_[.001, np.zeros(5), .4]

    class Frame:
        def __init__(self, env, source, *, source_offset):
            assert source == "source" and len(source_offset) == 3

        def reset(self):
            pass

        def update(self):
            return proxy

        def map_action(self, action):
            return action + np.r_[.002, np.zeros(6)]

        def diagnostics(self):
            return {"schema": "source"}

    class Encoder:
        def __init__(self, robot, hz):
            assert robot == "so101"

        def reset(self):
            pass

        def encode(self, env, *args, **kwargs):
            assert env is proxy
            return np.zeros(99)

        def selected_action(self, action, tick):
            histories.append(action.copy())

    class Executor:
        def __init__(self, episode, *, require_source_mapping):
            assert require_source_mapping is True

        def step(self, action):
            executed.append(action.copy())
            return SimpleNamespace(detail={"requested_native": np.r_[np.zeros(7), .032]})

    monkeypatch.setattr(source_recovery_frame, "SourceRecoveryFrame", Frame)
    monkeypatch.setattr(dagger_module, "RecoveryObservation", Encoder)
    monkeypatch.setattr(dagger_module, "CommonCartesianExecutor", Executor)
    monkeypatch.setattr(dagger_module, "initial_references",
                        lambda *_: (np.zeros(3), np.ones(3), {}))
    head = SimpleNamespace(metadata={"use_act": False, "training_robot_ids": [0]},
        predict=lambda *_: source_action.copy(), head_sha256="head")
    env = SimpleNamespace(control_hz=25., ee_pose=lambda: np.eye(4), max_opening=.08,
                          gripper_semantic=lambda: .4)
    controller = dagger_module.LearnedRecoveryController(
        SimpleNamespace(env=env, source="source", robot="panda"), head, recovery_frame="source")
    detail = controller.step({}, env).detail
    np.testing.assert_allclose(histories[0], source_action)
    np.testing.assert_allclose(executed[0], source_action + np.r_[.002, np.zeros(6)])
    assert detail["recovery_frame"] == "source"
    assert detail["source_recovery_frame"] == {"schema": "source"}


def test_source_frame_rejects_heads_trained_in_another_schema():
    episode = SimpleNamespace(env=None, source=None, robot="panda")
    head = SimpleNamespace(metadata={"use_act": False, "training_robot_ids": [0, 1]})
    with pytest.raises(ValueError, match="SO-101-only"):
        dagger_module.LearnedRecoveryController(episode, head, recovery_frame="source")


@pytest.mark.parametrize("geometry", ["tcp", "grasp"])
@pytest.mark.parametrize("clock", ["all_switches", "task_events"])
def test_achieved_history_controller_remembers_motion_and_issued_jaw_after_a_rejected_request(monkeypatch, geometry, clock):
    from real2sim.policy.common_recovery import (
        TRACKED_ACHIEVED_OBSERVATION_NAMES,
        TRACKED_TASK_EVENT_OBSERVATION_NAMES,
    )
    from real2sim.policy.progress_geometry import observation_geometry_names

    pose = np.eye(4)
    observations = []

    class References:
        def __init__(self, _env):
            self.ball, self.bowl = np.ones(3), np.ones(3) * 2
            self.log = [{}]

        def update(self, *_args, **_kwargs):
            pass

        def observation_metadata(self, tick):
            return {"visibility": (True, True), "last_seen_ticks": (tick, tick)}

    class Executor:
        def __init__(self, *_args, **_kwargs):
            pass

        def step(self, _action):
            pose[0, 3] += .001  # Residual measured motion despite a held command.
            return SimpleNamespace(detail={
                "requested_native": np.r_[np.zeros(7), .08],
                "achieved_common_action": [.001, 0., 0., 0., 0., 0., 1.]})

    def predict(observed, *_args):
        observations.append(observed.copy())
        return np.array([.03, 0., 0., 0., .2, 0., 0.])

    monkeypatch.setattr(dagger_module, "RGBReferenceTracker", References)
    monkeypatch.setattr(dagger_module, "CommonCartesianExecutor", Executor)
    env = SimpleNamespace(control_hz=25., ee_pose=lambda: pose.copy(), max_opening=.08,
                          gripper_semantic=lambda: .8)
    names = TRACKED_TASK_EVENT_OBSERVATION_NAMES if clock == "task_events" else TRACKED_ACHIEVED_OBSERVATION_NAMES
    head = SimpleNamespace(metadata={"use_act": False,
        "observation_names": observation_geometry_names(names, geometry)},
        predict=predict, head_sha256="head")
    controller = dagger_module.LearnedRecoveryController(
        SimpleNamespace(env=env, robot="panda", source=None), head)
    first = controller.step({}, env).detail
    controller.step({}, env)
    assert first["history_mode"] == "achieved"
    assert first["observation_geometry"] == geometry
    assert first["gripper_clock"] == clock
    assert observations[0][-33 + 13] == pytest.approx(1.008 if geometry == "grasp" else 1.)
    np.testing.assert_allclose(observations[1][-33 + 23:-33 + 30], [.001, 0., 0., 0., 0., 0., 1.])
    assert observations[1][-1] == pytest.approx(0. if clock == "task_events" else .04)


@pytest.mark.parametrize("tracked", [True, False])
def test_rgb_handover_preserves_observation_ages_and_cached_static_bowl(tracked):
    controller = object.__new__(dagger_module.LearnedRecoveryController)
    controller.env = SimpleNamespace(control_hz=25.)
    controller.references = SimpleNamespace(log=[]) if tracked else None
    prefix = SimpleNamespace(ball=np.array([.1, .2, .03]), bowl=np.array([.3, .1, .02]),
        last_seen={"ball": 45, "bowl": 10}, visible={"ball": True, "bowl": False})
    controller.inherit_rgb_references(prefix, 50)
    assert controller.rgb_reference_handover["target_ages_s"] == {"ball": .2, "bowl": 1.6}
    assert controller.rgb_reference_handover["configured_object_position_fallback"] is False
    if tracked:
        assert controller.references.last_seen == {"ball": -5, "bowl": -40}
        assert controller.references.visible == {"ball": True, "bowl": False}
        assert len(controller.references.log) == 1
        prefix.bowl[:] = 99
        np.testing.assert_allclose(controller.references.bowl, [.3, .1, .02])
    else:
        assert controller.reference_seed_metadata["last_seen_ticks"] == (-5, -40)
        np.testing.assert_allclose(controller.bowl, [.3, .1, .02])


def test_rgb_handover_rejects_future_or_unobserved_reference_times():
    controller = object.__new__(dagger_module.LearnedRecoveryController)
    controller.env = SimpleNamespace(control_hz=25.)
    controller.references = None
    prefix = SimpleNamespace(ball=np.zeros(3), bowl=np.ones(3),
        last_seen={"ball": 51, "bowl": 10}, visible={"ball": True, "bowl": False})
    with pytest.raises(ValueError, match="previously observed"):
        controller.inherit_rgb_references(prefix, 50)
