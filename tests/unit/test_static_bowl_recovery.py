"""Recovery may remember an observed static bowl, never an unseen movable ball."""

from types import SimpleNamespace

import numpy as np

from real2sim.policy.visual_waypoint import VisualWaypointController
from scripts.evaluate_hf_recovery import StaticBowlRecovery


def controller(monkeypatch, frames, *, static=True):
    recovery = object.__new__(StaticBowlRecovery)
    recovery.env = SimpleNamespace(task={"bowl": {"static": static}})
    recovery.perception_log = []
    recovery.ball = recovery.bowl = None
    iterator = iter(frames)

    def acquire(self):
        targets = next(iterator)
        self.perception_log.append({"targets": targets})
        if all(target is not None for target in targets.values()):
            self.ball = np.asarray(targets["ball"]["center_task"], float)
            self.bowl = np.asarray(targets["bowl"]["center_task"], float)
            return True
        return False

    monkeypatch.setattr(VisualWaypointController, "_acquire", acquire)
    return recovery


def target(x, y, z):
    return {"center_task": [x, y, z]}


def test_static_bowl_memory_requires_fresh_ball_and_tracks_its_new_position(monkeypatch):
    recovery = controller(monkeypatch, [
        {"ball": target(.1, .2, .03), "bowl": target(.3, .1, .03)},
        {"ball": target(.14, .23, .03), "bowl": None},
    ])
    assert recovery._acquire()
    assert recovery._acquire()
    np.testing.assert_allclose(recovery.ball, [.14, .23, .03])
    np.testing.assert_allclose(recovery.bowl, [.3, .1, .03])
    assert "static_bowl_memory_from_initial_rgb" in recovery.perception_log[-1]
    recovery.bowl[0] = 99
    np.testing.assert_allclose(recovery.bowl_memory, [.3, .1, .03])


def test_remembered_bowl_does_not_manufacture_an_occluded_ball(monkeypatch):
    recovery = controller(monkeypatch, [
        {"ball": target(.1, .2, .03), "bowl": target(.3, .1, .03)},
        {"ball": None, "bowl": None},
    ])
    assert recovery._acquire()
    assert not recovery._acquire()


def test_missing_bowl_without_prior_observation_has_no_configured_fallback(monkeypatch):
    recovery = controller(monkeypatch, [{"ball": target(.1, .2, .03), "bowl": None}])
    assert not recovery._acquire()
    assert getattr(recovery, "bowl_memory", None) is None


def test_observed_movable_bowl_cannot_be_used_as_static_memory(monkeypatch):
    recovery = controller(monkeypatch, [
        {"ball": target(.1, .2, .03), "bowl": target(.3, .1, .03)},
        {"ball": target(.1, .2, .03), "bowl": None},
    ], static=False)
    assert recovery._acquire()
    assert not recovery._acquire()
