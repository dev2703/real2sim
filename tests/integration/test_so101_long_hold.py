"""Retention scoring distinguishes proximity, load, and ball motion."""
import copy
import importlib
from pathlib import Path

import numpy as np
import pytest

from real2sim.sim.env import GRIPPER_BODIES

pytestmark = pytest.mark.integration


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    return importlib.import_module("diag_so101_long_hold")


def held():
    return {"jaws": {name: {"normal_force_n": .1} for name in GRIPPER_BODIES},
        "height_m": .04, "ball_tool_m": [0., 0., 0.], "tool_target_error_m": .001,
        "robot_table_penetration_m": 0., "robot_support_penetration_m": 0.}


def test_legacy_held_or_contact_count_does_not_replace_positive_bilateral_force(diagnostic):
    row = held()
    assert diagnostic.force_bilateral(row)
    assert not diagnostic.ready([row] * 4)
    assert diagnostic.ready([row] * 5)
    row["jaws"][GRIPPER_BODIES[1]]["normal_force_n"] = 0.
    row["legacy_ball_held"] = True
    row["bilateral"] = True
    assert not diagnostic.force_bilateral(row)
    assert not diagnostic.ready([row] * 5)


def test_one_lost_sample_or_relative_slip_fails_entire_hold(diagnostic):
    anchor = held()
    rows = [copy.deepcopy(anchor) for _ in range(20)]
    assert diagnostic.summarize_hold(anchor, rows, 10)["retained_entire_hold"]
    rows[5]["jaws"][GRIPPER_BODIES[0]]["normal_force_n"] = .001
    summary = diagnostic.summarize_hold(anchor, rows, 10)
    assert not summary["retained_entire_hold"]
    assert summary["end_force_bilateral"]
    assert summary["first_force_bilateral_loss_s"] == .6
    rows[5] = copy.deepcopy(anchor)
    rows[7]["ball_tool_m"] = [.011, 0., 0.]
    summary = diagnostic.summarize_hold(anchor, rows, 10)
    assert not summary["retained_entire_hold"]
    assert summary["first_force_bilateral_loss_s"] is None
    assert summary["first_strict_retention_failure_s"] == .8


def test_support_height_is_not_counted_as_loaded_lift(diagnostic):
    row = held()
    row["height_above_rest_m"] = .02
    assert not diagnostic.ready([row] * 5)
    assert not diagnostic.summarize_hold(row, [row] * 20, 10)["retained_entire_hold"]


def test_encoder_latch_uses_requested_closure_and_measured_angle_only(diagnostic):
    encoder = diagnostic.EncoderWidth(.0003)
    target = np.zeros(6)
    for tick in range(3):
        np.testing.assert_array_equal(encoder.command(target, .2, "CLOSE", tick), target)
    result = encoder.command(target, .2, "CLOSE", 3)
    assert result[5] == pytest.approx(.1997)
    assert encoder.event == {"tick": 3, "position_rad": .2, "phase": "CLOSE"}
    # Subsequent arm commands remain intact; releasing bypasses the latch.
    np.testing.assert_array_equal(encoder.command(np.ones(6), .3, "LIFT", 4)[:5], 1.)
    np.testing.assert_array_equal(encoder.command(target, .3, "RELEASE", 5), target)


@pytest.mark.parametrize("impratio,noslip", [(None, None), (100., 3)])
def test_contact_options_apply_before_any_settling_or_acquisition(
        diagnostic, monkeypatch, impratio, noslip):
    class ReachedSettling(Exception):
        pass

    def inspect_options_before_reset(env, seed):
        assert env.steps == 0 and not env._ready
        assert env.model.opt.impratio == (10. if impratio is None else impratio)
        assert env.model.opt.noslip_iterations == (0 if noslip is None else noslip)
        raise ReachedSettling

    monkeypatch.setattr(diagnostic, "settle_start_pose", inspect_options_before_reset)
    with pytest.raises(ReachedSettling):
        diagnostic.trial(Path("configs/sim_so101.yaml"), 0,
                         impratio=impratio, noslip_iterations=noslip)


@pytest.mark.parametrize("options", [{"impratio": 0}, {"impratio": float("nan")},
                                     {"noslip_iterations": -1}, {"noslip_iterations": 1.5}])
def test_invalid_contact_options_fail_before_loading_a_scene(diagnostic, options):
    with pytest.raises(ValueError):
        diagnostic.trial(Path("nonexistent-scene.yaml"), 0, **options)
