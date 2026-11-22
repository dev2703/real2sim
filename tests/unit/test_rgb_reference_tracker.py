"""References update only from visible RGB measurements and retain honest ages."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from real2sim.policy import rgb_reference_tracker as tracker_module
from real2sim.policy.common_recovery import FrozenACTFeatures
from real2sim.policy.rgb_reference_tracker import RGBReferenceTracker
from real2sim.policy.visual_targets import VisualTarget
from scripts import run_recovery_dagger as dagger_module


def target(x, y):
    return VisualTarget(np.array([x, y, .03]), np.zeros(2), 100., .0001, .9)


def setup(monkeypatch, frames):
    iterator = iter(frames)
    monkeypatch.setattr(tracker_module, "locate_objects", lambda *_a, **_kw: next(iterator))
    monkeypatch.setattr(tracker_module, "camera_intrinsics_px", lambda *_: np.eye(3))
    env = SimpleNamespace(render=lambda _: np.zeros((64, 64, 3), np.uint8),
        model=SimpleNamespace(camera=lambda _: SimpleNamespace(id=0)),
        data=SimpleNamespace(cam_xpos=np.array([[0., 0., 1.]]), cam_xmat=np.eye(3).reshape(1, 9)),
        offset=np.zeros(3), table_top=0., task={"ball": {"radius": .03}},
        bowl={"height": .03, "rim_radius": .055, "wall": .002},
        config={"ik": {"workspace_min": [-1., -1., 0.], "workspace_max": [1., 1., 1.]}},
        ee_pose=lambda: np.eye(4))
    return RGBReferenceTracker(env)


def test_displaced_ball_uses_fresh_rgb_and_occluded_bowl_keeps_its_age(monkeypatch):
    tracker = setup(monkeypatch, [
        {"ball": target(.1, .2), "bowl": target(.3, .1)},
        {"ball": target(.13, .22), "bowl": None},
    ])
    tracker.update(0)
    assert tracker.update(4) is None
    update = tracker.update(5)
    np.testing.assert_allclose(tracker.ball, [.13, .22, .03])
    np.testing.assert_allclose(tracker.bowl, [.3, .1, .03])
    assert update["ball_reference_shift_m"] > .03
    assert tracker.observation_metadata(5) == {
        "visibility": (True, False), "last_seen_ticks": (5, 0)}


def test_missing_measurements_never_create_configured_goal_fallback(monkeypatch):
    tracker = setup(monkeypatch, [{"ball": None, "bowl": target(.3, .1)}])
    tracker.update(0)
    assert tracker.ball is None
    with pytest.raises(RuntimeError, match="observed reference"):
        tracker.observation_metadata(0)


def test_expert_replans_displaced_free_ball_but_leaves_observed_bowl_contents(monkeypatch):
    tracker = setup(monkeypatch, [{"ball": target(.1, .2), "bowl": target(.3, .1)}])
    tracker.update(0)
    teacher = SimpleNamespace(phase="DESCEND", ball_in_tool=np.zeros(3))
    update = {"ball_reference_shift_m": .02}
    assert tracker.expert_needs_replan(teacher, update)
    tracker.ball = tracker.bowl.copy()
    assert not tracker.expert_needs_replan(teacher, update)


def test_small_refresh_shifts_replan_once_cumulative_drift_leaves_planned_goal(monkeypatch):
    tracker = setup(monkeypatch, [
        {"ball": target(.1 + shift, .2), "bowl": target(.3, .1)}
        for shift in (0., .002, .004, .007)
    ])
    tracker.update(0)
    teacher = SimpleNamespace(phase="DESCEND", ball=tracker.ball.copy(), ball_in_tool=np.zeros(3))
    assert not tracker.expert_needs_replan(teacher, tracker.update(5))
    assert not tracker.expert_needs_replan(teacher, tracker.update(10))
    update = tracker.update(15)
    assert update["ball_reference_shift_m"] < .006
    assert tracker.expert_needs_replan(teacher, update)
    # Replanning to this observation resets the cumulative displacement.
    teacher.ball = tracker.ball.copy()
    assert not tracker.expert_needs_replan(teacher, update)
    assert not tracker.expert_needs_replan(teacher, None)


def test_retained_reference_does_not_trigger_replan_without_fresh_ball(monkeypatch):
    tracker = setup(monkeypatch, [
        {"ball": target(.1, .2), "bowl": target(.3, .1)},
        {"ball": None, "bowl": target(.3, .1)},
    ])
    tracker.update(0)
    teacher = SimpleNamespace(phase="DESCEND", ball=np.zeros(3), ball_in_tool=np.zeros(3))
    assert not tracker.expert_needs_replan(teacher, tracker.update(5))


@pytest.mark.parametrize("phase", ["CLOSE", "LIFT", "TRANSPORT", "LOWER"])
def test_unmoved_visible_ball_exposes_empty_pickup_in_active_carry_phases(monkeypatch, phase):
    tracker = setup(monkeypatch, [
        {"ball": target(.1, .2), "bowl": target(.3, .1)},
        {"ball": target(.1, .2), "bowl": target(.3, .1)},
    ])
    tracker.update(0)
    teacher = SimpleNamespace(phase=phase, ball=tracker.ball.copy(), ball_in_tool=np.zeros(3))
    pose = np.eye(4)
    pose[:3, 3] = tracker.ball + [0., 0., .09]
    tracker.env.ee_pose = lambda: pose
    update = tracker.update(5)
    assert update["ball_reference_shift_m"] == 0.
    assert tracker.expert_needs_replan(teacher, update)
    # A matching measured tool/ball geometry does not invent a pickup failure.
    pose[:3, 3] = tracker.ball
    assert not tracker.expert_needs_replan(teacher, update)


@pytest.mark.parametrize("phase", ["ACQUIRE", "DONE", "FAILED", "RELEASE", "RETREAT"])
def test_terminal_or_post_release_phases_never_start_another_pickup(monkeypatch, phase):
    tracker = setup(monkeypatch, [{"ball": target(.1, .2), "bowl": target(.3, .1)}])
    update = tracker.update(0)
    teacher = SimpleNamespace(phase=phase, ball=np.zeros(3), ball_in_tool=np.zeros(3))
    assert not tracker.expert_needs_replan(teacher, update)


@pytest.mark.parametrize("phase", ["PRE_GRASP", "DESCEND"])
def test_early_approach_distance_alone_does_not_signal_a_missed_pickup(monkeypatch, phase):
    tracker = setup(monkeypatch, [{"ball": target(.1, .2), "bowl": target(.3, .1)}])
    update = tracker.update(0)
    teacher = SimpleNamespace(phase=phase, ball=tracker.ball.copy(), ball_in_tool=np.zeros(3))
    assert not tracker.expert_needs_replan(teacher, update)


def test_single_decision_render_cache_deduplicates_only_overlapping_ticks(monkeypatch):
    render_calls = []
    tracker_frames = []
    act_frames = []
    sim_state = {"step": 0}

    def render(camera):
        render_calls.append((sim_state["step"], camera))
        base = sim_state["step"] + (100 if camera == "top" else 0)
        return np.full((64, 64, 3), base, np.uint8)

    def fake_locate_objects(rgb, *_args, **_kwargs):
        tracker_frames.append((sim_state["step"], int(rgb[0, 0, 0])))
        return {"ball": target(.1, .2), "bowl": target(.3, .1)}

    monkeypatch.setattr(tracker_module, "locate_objects", fake_locate_objects)
    monkeypatch.setattr(tracker_module, "camera_intrinsics_px", lambda *_: np.eye(3))

    class FakePolicy:
        def __init__(self):
            self.config = SimpleNamespace(n_action_steps=3, chunk_size=10, dim_model=256)
            self.camera_keys = ["observation.images.top", "observation.images.iphone"]
            self.action_head = nn.Identity()
            self.eval_called = False
            self.policy = self
            self.model = SimpleNamespace(action_head=self.action_head)

        def eval(self):
            return self

        def requires_grad_(self, _flag):
            return self

        def reset(self):
            pass

        def predict_action_chunk(self, _state, images):
            act_frames.append((
                sim_state["step"],
                int(images["observation.images.iphone"][0, 0, 0]),
                int(images["observation.images.top"][0, 0, 0]),
            ))
            features = torch.zeros((1, 10, 256))
            self.action_head(features)
            return np.zeros((10, 6))

    class FakeExecutor:
        def __init__(self, _episode, *, require_source_mapping):
            assert require_source_mapping is True

        def hf_to_common(self, _raw):
            return np.zeros(7)

        def step(self, _action):
            sim_state["step"] += 1
            return SimpleNamespace(detail={"requested_native": np.zeros(6).tolist()})

    monkeypatch.setattr(dagger_module, "CommonCartesianExecutor", FakeExecutor)
    env = SimpleNamespace(
        control_hz=25.,
        camera_bindings={"observation.images.top": "top", "observation.images.iphone": "iphone"},
        render=render,
        model=SimpleNamespace(camera=lambda _: SimpleNamespace(id=0)),
        data=SimpleNamespace(cam_xpos=np.array([[0., 0., 1.]]), cam_xmat=np.eye(3).reshape(1, 9)),
        offset=np.zeros(3),
        table_top=0.,
        task={"ball": {"radius": .03}},
        bowl={"height": .03, "rim_radius": .055, "wall": .002},
        config={"ik": {"workspace_min": [-1., -1., 0.], "workspace_max": [1., 1., 1.]}},
        ee_pose=lambda: np.eye(4),
        sim_state=lambda: np.r_[np.zeros(5), 36.],
        real_state=lambda: np.zeros(6, np.float32),
        mapping=SimpleNamespace(gripper_sim_to_real_percent=lambda value: value),
    )
    head = SimpleNamespace(
        metadata={"use_act": True, "observation_names": ["fresh_rgb_ball_relative_x"], "act_feature_dim": 256},
        head_sha256="head",
        predict=lambda *_args: np.r_[np.zeros(6), 1.],
    )
    controller = dagger_module.LearnedRecoveryController(
        SimpleNamespace(env=env, robot="so101", source=None), head, FakePolicy()
    )
    try:
        per_tick_iphone_renders = {}
        post_step_frames = {}
        for tick in range(16):
            before = len(render_calls)
            controller.step({}, env)
            tick_calls = render_calls[before:]
            per_tick_iphone_renders[tick] = sum(1 for _step, cam in tick_calls if cam == "iphone")
            # Post-step video/inspection render must still observe post-step simulator state.
            post_step_frames[tick] = int(env.render("iphone")[0, 0, 0])

        # Overlapping ticks (0 and 15): tracker refresh + ACT query share one iPhone render.
        assert per_tick_iphone_renders[0] == 1
        assert per_tick_iphone_renders[15] == 1
        # Non-overlapping tracker-only ticks (5, 10) and ACT-only ticks (3, 6, 9, 12): 1 iPhone render each.
        for tick in (3, 5, 6, 9, 10, 12):
            assert per_tick_iphone_renders[tick] == 1
        # Idle ticks (neither refreshes): 0 iPhone renders.
        for tick in (1, 2, 4, 7, 8, 11, 13, 14):
            assert per_tick_iphone_renders[tick] == 0

        # Tracker saw ticks 0, 5, 10, 15; ACT saw ticks 0, 3, 6, 9, 12, 15 — never stale tick-5 frame at tick 6.
        assert tracker_frames == [(0, 0), (5, 5), (10, 10), (15, 15)]
        assert act_frames == [(0, 0, 100), (3, 3, 103), (6, 6, 106), (9, 9, 109), (12, 12, 112), (15, 15, 115)]
        # Post-step renders always see the stepped simulator state (tick + 1).
        assert all(post_step_frames[tick] == tick + 1 for tick in range(16))

        # Invalidation before a tracker refresh tick (tick 20) still shares the single iPhone render.
        for _ in range(4):
            controller.step({}, env)  # ticks 16, 17, 18 (ACT at 18), 19
        controller.act.invalidate("test_replan")
        before = len(render_calls)
        controller.step({}, env)  # tick 20: tracker refresh + invalidated ACT query coincide
        tick20_calls = render_calls[before:]
        assert tick20_calls == [(20, "iphone"), (20, "top")]
    finally:
        controller.close()


def test_default_update_and_frozen_act_next_preserve_direct_render_behavior(monkeypatch):
    calls = []
    monkeypatch.setattr(tracker_module, "locate_objects", lambda *_a, **_kw: {
        "ball": target(.1, .2), "bowl": target(.3, .1)})
    monkeypatch.setattr(tracker_module, "camera_intrinsics_px", lambda *_: np.eye(3))
    env = SimpleNamespace(
        camera_bindings={"observation.images.top": "top", "observation.images.iphone": "iphone"},
        render=lambda camera: (calls.append(camera), np.zeros((64, 64, 3), np.uint8))[1],
        model=SimpleNamespace(camera=lambda _: SimpleNamespace(id=0)),
        data=SimpleNamespace(cam_xpos=np.array([[0., 0., 1.]]), cam_xmat=np.eye(3).reshape(1, 9)),
        offset=np.zeros(3),
        table_top=0.,
        task={"ball": {"radius": .03}},
        bowl={"height": .03, "rim_radius": .055, "wall": .002},
        config={"ik": {"workspace_min": [-1., -1., 0.], "workspace_max": [1., 1., 1.]}},
        real_state=lambda: np.zeros(6, np.float32),
    )
    tracker = RGBReferenceTracker(env)
    tracker.update(0)
    assert calls == ["iphone"]

    action_head = nn.Identity()

    def predict(_state, _images):
        action_head(torch.zeros((1, 4, 256)))
        return np.zeros((4, 6))

    policy = SimpleNamespace(
        config=SimpleNamespace(n_action_steps=2, chunk_size=4, dim_model=256),
        camera_keys=["observation.images.top", "observation.images.iphone"],
        policy=SimpleNamespace(model=SimpleNamespace(action_head=action_head)),
        predict_action_chunk=predict,
        reset=lambda: None,
    )
    stream = FrozenACTFeatures(SimpleNamespace(env=env, source=None, robot="so101"), policy)
    try:
        stream.next()
        assert calls == ["iphone", "top", "iphone"]
    finally:
        stream.close()
