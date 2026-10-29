"""Scoring semantics of TrialTracker (docs/PROJECT_SUMMARY.md), edge cases."""

import pytest

from real2sim.evaluation.metrics import PHASES, TRIAL_COLUMNS, TrialTracker, summarize


def pm(*, dist=0.2, held=False, lifted=False, in_bowl=False, resting=False, z=0.05):
    return {"ee_ball_distance_m": dist, "ball_held": held, "lifted": lifted, "ball_in_bowl": in_bowl,
            "ball_resting": resting, "ball_xyz": [0.3, 0.0, z], "ball_xyz_task": [0.3, 0.0, z],
            "tool_xyz_task": [0.3, 0.0, 0.1]}


def tracker(hold=3):
    return TrialTracker(success_hold_ticks=hold, approach_distance_m=0.03, table_top_z=0.0, ball_radius_m=0.0335)


def run(seq, hold=3):
    t = tracker(hold)
    for kw in seq:
        t.update(pm(**kw))
    return t


# ------------------------------------------------------------------ phase attribution
def test_never_close_is_approach():
    assert run([{"dist": 0.2}] * 5).phase_failed() == "approach"


def test_bumped_into_air_without_grasp_is_still_approach():
    t = run([{"dist": 0.2, "lifted": True, "z": 0.1}] * 5)
    assert not t.lifted and t.phase_failed() == "approach"


def test_reached_but_never_held_is_grasp():
    assert run([{"dist": 0.2}, {"dist": 0.02}, {"dist": 0.1}]).phase_failed() == "grasp"


def test_held_but_never_lifted_is_grasp():
    t = run([{"dist": 0.2}, {"dist": 0.2, "held": True}, {"dist": 0.2, "held": True}])
    assert t.held_ever and not t.lifted and t.phase_failed() == "grasp"


def test_lifted_while_held_but_never_in_bowl_is_transport():
    t = run([{"dist": 0.01, "held": True}, {"dist": 0.01, "held": True, "lifted": True}, {"dist": 0.01}])
    assert t.lifted and t.phase_failed() == "transport"


def test_in_bowl_but_not_settled_is_release():
    t = run([{"dist": 0.01, "held": True, "lifted": True}, {"dist": 0.01, "held": True, "in_bowl": True}])
    assert t.phase_failed() == "release"


def test_threshold_is_inclusive_for_reaching():
    """min distance exactly equal to the threshold counts as reached (grasp), not approach."""
    assert run([{"dist": 0.03}]).phase_failed() == "grasp"


# ------------------------------------------------------------------ success
def test_success_needs_consecutive_settled_unheld_ticks():
    settled = {"dist": 0.05, "in_bowl": True, "resting": True}
    t = run([settled, settled, {"dist": 0.05, "in_bowl": True, "resting": True, "held": True}, settled, settled])
    assert not t.success  # the held flicker reset the counter: 2 + 2 < 3
    t.update(pm(**settled))
    assert t.success and t.phase_failed() == "none"


def test_success_is_latched():
    settled = {"dist": 0.05, "in_bowl": True, "resting": True}
    t = run([settled] * 3 + [{"dist": 0.05}] * 3)
    assert t.success


def test_held_ball_resting_at_bowl_mouth_is_not_success():
    t = run([{"dist": 0.0, "in_bowl": True, "resting": True, "held": True}] * 10)
    assert not t.success and t.phase_failed() == "release"


@pytest.mark.parametrize("hold", [0, -1])
def test_zero_hold_window_is_rejected(hold):
    with pytest.raises(ValueError, match="success_hold_ticks"):
        tracker(hold)


def test_non_positive_approach_distance_is_rejected():
    with pytest.raises(ValueError):
        TrialTracker(success_hold_ticks=3, approach_distance_m=0.0, table_top_z=0.0, ball_radius_m=0.03)


# ------------------------------------------------------------------ release / roll-out
def test_release_detected_on_held_to_unheld_transition_only():
    assert not run([{"dist": 0.1}] * 3).released
    assert run([{"held": True}, {"held": False}]).released


def test_bounce_out_of_band_then_success_is_not_rolled_out():
    settled = {"dist": 0.05, "in_bowl": True, "resting": True}
    t = run([{"held": True, "in_bowl": True}, {"in_bowl": True}, {"in_bowl": False, "z": 0.12},
             settled, settled, settled])
    assert t.success
    assert t.row(**_static())["rolled_out"] is False


def test_release_then_leave_bowl_without_success_is_rolled_out():
    t = run([{"held": True, "in_bowl": True}, {"in_bowl": True}, {"in_bowl": False}, {"in_bowl": False}])
    assert t.row(**_static())["rolled_out"] is True and t.phase_failed() == "release"


def test_ball_below_table_is_escaped():
    assert run([{"z": -0.1}]).ball_escaped_table


# ------------------------------------------------------------------ rows and summary
def _static(**kw):
    base = {"experiment_id": "x", "controller_source": "LEARNED_ACT", "robot": "so101", "seed": 0,
            "checkpoint_sha256": None, "checkpoint_step": None, "mean_latency_ms": None, "video_path": ""}
    return base | kw


def test_row_has_exactly_the_published_columns_in_order():
    row = run([{"dist": 0.2}]).row(**_static())
    assert list(row) == TRIAL_COLUMNS and row["abort"] is None


def test_aborted_trial_is_its_own_phase_and_counted():
    row = run([{"dist": 0.2}]).row(abort="IK seed outside limits", **_static())
    assert row["phase_failed"] == "aborted" and "aborted" in PHASES
    s = summarize([row, run([{"dist": 0.2}]).row(**_static())])
    assert s["aborted"] == 1 and s["phase_failed_counts"]["aborted"] == 1 and s["phase_failed_counts"]["approach"] == 1


def test_row_missing_static_column_raises():
    with pytest.raises(KeyError):
        run([{"dist": 0.2}]).row(experiment_id="x")


def test_summary_of_no_rows():
    s = summarize([])
    assert s["n"] == 0 and s["success_rate"] is None and s["mean_latency_ms"] is None
