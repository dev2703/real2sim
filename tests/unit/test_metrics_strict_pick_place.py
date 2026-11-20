"""Strict placement requires a lift and bowl release in the same carry."""

import pytest

from real2sim.evaluation.metrics import TrialTracker


def sample(*, held=False, lifted=False, over_bowl=False, in_bowl=False, resting=False):
    return {"ee_ball_distance_m": .02, "ball_held": held, "lifted": lifted,
            "ball_over_bowl": over_bowl or in_bowl, "ball_in_bowl": in_bowl, "ball_resting": resting,
            "ball_xyz": [0., 0., .1], "ball_xyz_task": [0., 0., .1], "tool_xyz_task": [0., 0., .1]}


@pytest.fixture
def tracker():
    return TrialTracker(success_hold_ticks=2, approach_distance_m=.03, table_top_z=0., ball_radius_m=.03)


@pytest.mark.parametrize("shallow_regrasp", [False, True])
def test_drop_then_push_in_does_not_reuse_an_earlier_lift(tracker, shallow_regrasp):
    tracker.update(sample(held=True, lifted=True))
    tracker.update(sample(resting=True))
    if shallow_regrasp:
        tracker.update(sample(held=True, in_bowl=True))
    for _ in range(2):
        tracker.update(sample(in_bowl=True, resting=True))
    assert tracker.lifted and tracker.released and tracker.delivered and tracker.success
    assert not tracker.pick_place_success


@pytest.mark.parametrize("after_drop", [False, True])
def test_valid_placement_requires_fresh_held_lift_and_full_dwell(tracker, after_drop):
    if after_drop:
        tracker.update(sample(held=True, lifted=True))
        tracker.update(sample(resting=True))
    tracker.update(sample(held=True))
    tracker.update(sample(held=True, lifted=True))
    tracker.update(sample(held=True, over_bowl=True))
    tracker.update(sample(over_bowl=True))
    tracker.update(sample(in_bowl=True, resting=True))
    assert not tracker.pick_place_success
    tracker.update(sample(in_bowl=True, resting=True))
    assert tracker.pick_place_success


def test_ball_that_leaves_bowl_after_release_needs_a_new_carry(tracker):
    tracker.update(sample(held=True, lifted=True, over_bowl=True))
    tracker.update(sample(in_bowl=True))
    tracker.update(sample(resting=True))
    for _ in range(2):
        tracker.update(sample(in_bowl=True, resting=True))
    assert tracker.lifted and tracker.rolled_out and tracker.success
    assert not tracker.pick_place_success
