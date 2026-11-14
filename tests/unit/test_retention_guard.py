"""Causal retention, delivery, loss and native rejection streams."""

import numpy as np
import pytest

from real2sim.policy.retention_guard import RetentionGuard


def step(guard, *, jaw=0., position=(0, 0, .1), predicted=(.2, 0, .12),
         bowl=(.2, 0, .1), ball_age=0., bowl_age=0., **feedback):
    action, detail = guard.filter(np.array([.01, 0, .002, 0, 0, 0, jaw]), feedback,
        position=position, predicted_ball=predicted, bowl=bowl,
        ball_age_s=ball_age, bowl_age_s=bowl_age)
    guard.observe_execution(rejected=False, achieved_action=np.r_[action[:6], action[6]])
    return action, detail


def carry(guard):
    step(guard)
    step(guard, grasp_candidate=True, candidate_ever=True)
    assert guard.stage == "TRANSPORT"


def test_empty_close_can_lift_but_cannot_carry_or_release_and_eventually_retries():
    guard = RetentionGuard(25, close_wait_s=.12)
    step(guard)
    action, _ = step(guard, jaw=1.)
    assert action[6] == 0 and action[0] == 0 and action[2] > 0
    for _ in range(4):
        action, _ = step(guard, jaw=1.)
    assert guard.stage == "REACQUIRE"
    assert action[6] == 1 and not action[:6].any()


def test_premature_open_is_latched_and_stale_ball_memory_does_not_block_valid_carry():
    guard = RetentionGuard(25)
    carry(guard)
    for _ in range(10):
        action, _ = step(guard, jaw=1., predicted=(0, 0, .2), ball_age=20., candidate_ever=True)
        assert action[6] == 0
    assert guard.stage == "TRANSPORT"


@pytest.mark.parametrize("kwargs", [dict(predicted=None), dict(bowl_age=5.),
    dict(predicted=(.2, 0, .09)), dict(predicted=(.2, 0, .20)),
    dict(retention_warning=True), dict(candidate_ever=False)])
def test_release_requires_fresh_bowl_supported_height_and_no_retention_warning(kwargs):
    guard = RetentionGuard(25)
    carry(guard)
    evidence = dict(candidate_ever=True)
    evidence.update(kwargs)
    for _ in range(10):
        action, _ = step(guard, jaw=1., **evidence)
        assert action[6] == 0


def test_delivery_dwell_resets_on_motion_and_release_waits_for_measured_open_before_retreat():
    guard = RetentionGuard(25, release_dwell_s=.12)
    carry(guard)
    step(guard, jaw=1., candidate_ever=True)
    step(guard, jaw=1., position=(.01, 0, .1), candidate_ever=True)
    assert guard.delivery_ticks == 0
    for _ in range(2):
        action, _ = step(guard, jaw=1., position=(.01, 0, .1), candidate_ever=True)
        assert action[6] == 0
    action, _ = step(guard, jaw=1., position=(.01, 0, .1), candidate_ever=True)
    assert guard.stage == "RELEASE" and action[6] == 1 and not action[:6].any()
    step(guard, measured_opening=.2)
    assert guard.stage == "RELEASE"
    action, _ = step(guard, measured_opening=.8)
    assert guard.stage == "RETREAT" and action[6] == 1


def test_loss_discards_old_goal_and_requires_a_detection_after_loss_before_new_attempt():
    guard = RetentionGuard(25, retry_pause_s=.08)
    carry(guard)
    action, detail = step(guard, loss_suspected=True, ball_age=.1)
    assert detail["stage_changed"] and guard.stage == "REACQUIRE"
    for _ in range(4):
        action, _ = step(guard, ball_age=.3)
        assert guard.stage == "REACQUIRE" and not action[:6].any()
    action, _ = step(guard, ball_age=0.)
    assert guard.stage == "ACQUIRE" and not action[:6].any()
    step(guard)
    assert guard.stage == "CLOSE"


def test_stale_reacquisition_times_out_and_stop_preserves_closed_intent():
    guard = RetentionGuard(25, reacquire_wait_s=.12)
    carry(guard)
    step(guard, loss_suspected=True)
    for _ in range(5):
        action, _ = step(guard, ball_age=10., commanded_opening=0.)
    assert guard.stage == "STOP" and not action.any()
    step(guard, ball_age=0.)
    assert guard.stage == "STOP"
    guard.reset()
    assert guard.stage == "ACQUIRE" and guard.retries == 0


def test_repeated_rejected_proposals_stop_with_closed_jaw_and_achieved_velocity_feedback():
    guard = RetentionGuard(25, rejection_ticks=2)
    carry(guard)
    for _ in range(2):
        guard.observe_execution(rejected=True, achieved_action=np.zeros(7))
    action, detail = step(guard, jaw=1., commanded_opening=0.)
    assert not action.any()
    assert detail["stop_reason"] == "repeated_native_rejection"


def test_acquisition_blocks_stale_targets_and_retry_count_is_bounded():
    guard = RetentionGuard(25, max_retries=1, retry_pause_s=.04)
    action, _ = step(guard, ball_age=1.)
    assert not action[:6].any()
    carry(guard)
    step(guard, loss_suspected=True)
    step(guard)
    carry(guard)
    action, _ = step(guard, loss_suspected=True)
    assert guard.stage == "STOP" and not action[:6].any()


@pytest.mark.parametrize("kwargs", [dict(hz=0), dict(hz=25, rejection_ticks=1.5),
    dict(hz=25, fresh_s=float("nan"))])
def test_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        RetentionGuard(**kwargs)
