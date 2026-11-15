"""Coverage and passive-control claims must use the causal scoring alignment."""

import copy

import pytest

from real2sim.evaluation.visual_feedback_audit import audit_trace, passive_comparison


def row(tick, *, held=False, ball=(.2, 0., .03), sphere=None, bowl=False, loss=False):
    return {"tick": tick, "latency_ms": 5., "control": [0.],
            "detail": {"observable_grasp_feedback": {"fresh_ball_3d": sphere,
                "closed_command": True, "loss_suspected": loss}, "requested_common_action": [0.] * 7},
            "scoring_only": {"ball_held": held, "lifted": held, "ball_over_bowl": bowl,
                             "ball_xyz_task": list(ball)}}


def test_rgb_is_compared_with_pre_action_scoring_and_missing_runs_are_honest():
    rows = [row(0, held=True), row(1, held=True),
            row(2, held=False, ball=(.5, 0., .03), sphere=[.2, 0., .03]),
            row(3, sphere=[.5, 0., .03])]
    result = audit_trace(rows)
    assert result["held_ticks"] == 2 and result["measured_held_ticks"] == 1
    assert result["longest_missing_held_s"] == .04
    assert result["all_measurement_error"]["max_m"] == 0.
    assert result["closed_contact_losses_outside_bowl"] == [{"tick": 3, "loss_flag_at_transition": False}]


def test_bowl_loss_flags_do_not_count_as_transport_drop_detections():
    rows = [row(0, held=True), row(1, bowl=True), row(2, loss=True)]
    result = audit_trace(rows)
    assert result["closed_contact_losses_outside_bowl"] == []
    assert result["loss_flag_ticks_over_bowl"] == [2]
    assert result["held_measurement_error"]["max_m"] is None


def test_passive_comparison_checks_inputs_commands_controls_and_whole_scored_state():
    baseline = [row(0)]
    passive = copy.deepcopy(baseline)
    passive[0]["detail"]["visual_ball_fit"] = "arcs"
    passive[0]["detail"]["observable_grasp_feedback"]["fresh_ball_3d"] = [.2, 0., .03]
    passive[0]["latency_ms"] = 50.
    assert passive_comparison(baseline, passive)["exact_control_and_scoring_equality"]
    for part, key, value in [("detail", "recovery_inputs", {"observations": [1.]}),
                             ("detail", "requested_common_action", [1.] * 7),
                             ("scoring_only", "ball_xyz_task", [0., 0., 0.])]:
        changed = copy.deepcopy(passive)
        changed[0][part][key] = value
        assert not passive_comparison(baseline, changed)["exact_control_and_scoring_equality"]
    assert not passive_comparison(baseline, [*passive, row(1)])["exact_control_and_scoring_equality"]


def test_invalid_timing_and_measurements_are_rejected():
    with pytest.raises(ValueError, match="consecutive"):
        audit_trace([row(1)])
    with pytest.raises(ValueError, match="finite"):
        audit_trace([row(0), row(1, sphere=[float("nan"), 0., 0.])])
