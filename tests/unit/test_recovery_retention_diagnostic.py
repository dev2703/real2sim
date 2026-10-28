"""Saved command intent and sustained scoring evidence remain distinct."""

import numpy as np
import pytest

from scripts.diagnose_recovery_retention import audit_trace


def trace(*, command=0., lifted=True, held_ticks=5):
    rows = []
    for tick in range(held_ticks + 1):
        tool = np.array([.1, .2, .03 + .001 * tick])
        rows.append({"tick": tick, "detail": {"requested_common_action": [0.] * 6 + [command],
            "learned_phase_id": 4}, "scoring_only": {"ball_held": tick < held_ticks,
            "lifted": lifted, "tool_xyz_task": tool.tolist(),
            "ball_xyz_task": (tool + [0., 0., -.002 * tick]).tolist(),
            "ball_over_bowl": False, "gripper_opening_m": .064 - .001 * tick}})
    return rows


def test_closed_command_loss_records_slip_separately_from_phase_and_jaw_readback():
    event, = audit_trace(trace())
    assert event["classification"] == "closed_command_loss_after_lift"
    assert event["predicted_phase"] == "LIFT"
    assert event["ball_relative_vertical_change_m"] == pytest.approx(-.01)
    assert event["peak_measured_tcp_speed_mps"] == pytest.approx(.025)
    assert event["peak_measured_tcp_acceleration_mps2"] == pytest.approx(0., abs=1e-10)
    assert event["measured_jaw_gap_at_loss_m"] == pytest.approx(.059)


def test_current_command_controls_loss_classification_even_with_closed_previous_command():
    rows = trace()
    rows[-1]["detail"]["requested_common_action"][6] = 1.
    event, = audit_trace(rows)
    assert event["classification"] == "loss_after_lift"


def test_one_tick_contact_chatter_and_empty_pickup_do_not_become_lifted_retention_loss():
    assert audit_trace(trace(held_ticks=1)) == []
    event, = audit_trace(trace(lifted=False))
    assert event["classification"] == "contact_loss_before_lift"


def test_irregular_saved_ticks_use_elapsed_time_for_motion():
    rows = trace()
    for row in rows:
        row["tick"] *= 2
    event, = audit_trace(rows)
    assert event["peak_measured_tcp_speed_mps"] == pytest.approx(.0125)
