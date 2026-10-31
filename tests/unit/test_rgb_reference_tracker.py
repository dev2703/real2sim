"""References update only from visible RGB measurements and retain honest ages."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.policy import rgb_reference_tracker as tracker_module
from real2sim.policy.rgb_reference_tracker import RGBReferenceTracker
from real2sim.policy.visual_targets import VisualTarget


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
