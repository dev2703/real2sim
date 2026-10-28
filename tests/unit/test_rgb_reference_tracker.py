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
