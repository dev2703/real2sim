"""Observable feedback distinguishes a closed command from supported lifting."""

import numpy as np
import pytest

from real2sim.policy.grasp_feedback import ObservableGraspMonitor, RecoveryPhaseGuard


def sample(monitor, tick, *, z=0., measured=.7, command=0., **kwargs):
    return monitor.update(tick, [0., 0., z], measured, command, **kwargs)


def test_closed_command_and_tcp_lift_do_not_establish_a_grasp_with_empty_jaws():
    monitor = ObservableGraspMonitor(25.)
    for tick in range(30):
        feedback = sample(monitor, tick, z=tick * .002, measured=.01)
        assert feedback["closed_command"]
        assert not feedback["grasp_candidate"]
        assert not feedback["held_verified"]
        assert not feedback["loss_suspected"]
    assert feedback["status"] == "empty_closed"


def test_resisted_jaw_requires_dwell_and_lift_before_candidate_and_detects_empty_loss():
    monitor = ObservableGraspMonitor(25.)
    for tick in range(5):
        assert not sample(monitor, tick)["grasp_candidate"]
    feedback = sample(monitor, 5, z=.025)
    assert feedback["grasp_candidate"]
    assert feedback["candidate_ever"]
    assert not feedback["held_verified"]
    assert not sample(monitor, 6, z=.03, measured=.01)["loss_suspected"]
    feedback = sample(monitor, 7, z=.032, measured=.01)
    assert feedback["loss_suspected"]
    assert feedback["loss_reason"] == "previously_resisted_jaw_closed_to_empty_range"


def test_gradual_jaw_narrowing_warns_before_empty_closure_without_claiming_verified_loss():
    monitor = ObservableGraspMonitor(25.)
    for tick in range(6):
        feedback = sample(monitor, tick, z=.005 * tick, measured=.77)
    assert feedback["grasp_candidate"]
    feedback = sample(monitor, 6, z=.03, measured=.61)
    assert feedback["status"] == "retention_warning"
    assert feedback["jaw_narrowing_since_candidate"] == pytest.approx(.16)
    assert not feedback["loss_suspected"]
    assert not feedback["held_verified"]


def test_jaw_open_intentionally_resets_candidate_and_never_reports_loss():
    monitor = ObservableGraspMonitor(25.)
    for tick in range(6):
        sample(monitor, tick, z=tick * .005)
    feedback = sample(monitor, 6, z=.03, measured=.01, command=1.)
    assert feedback["status"] == "open"
    assert not feedback["loss_suspected"]
    assert not feedback["candidate_ever"]
    assert feedback["jaw_narrowing_since_candidate"] == 0.


def test_fresh_rgb_ball_far_from_predicted_grasp_detects_missed_pickup_without_jaw_proof():
    monitor = ObservableGraspMonitor(25.)
    feedback = sample(monitor, 0, measured=.01,
                      fresh_ball=[.1, .2, .03], predicted_ball=[.1, .2, .11])
    assert feedback["loss_suspected"]
    assert feedback["loss_reason"] == "fresh_on_table_rgb_ball_separated_from_predicted_grasp"
    assert feedback["fresh_rgb_separation_m"] == pytest.approx(.08)
    assert not feedback["candidate_ever"]
    # No fresh measurement is represented as missing, not as a cached goal.
    feedback = sample(monitor, 1, measured=.01)
    assert feedback["fresh_rgb_separation_m"] is None
    assert feedback["loss_suspected"]


def test_loss_ignores_unpaired_or_nearby_rgb_measurements():
    monitor = ObservableGraspMonitor(25.)
    assert not sample(monitor, 0, fresh_ball=[0, 0, 0])["loss_suspected"]
    assert not sample(monitor, 1, fresh_ball=[0, 0, 0], predicted_ball=[0, 0, .03])["loss_suspected"]


@pytest.mark.parametrize("kwargs", [{"hz": 0}, {"hz": float("nan")},
    {"hz": 25, "empty_opening": .2, "resistance_opening": .1},
    {"hz": 25, "narrowing_warning": 1.}])
def test_invalid_monitor_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ObservableGraspMonitor(**kwargs)


def test_ticks_openings_and_positions_are_validated_and_reset_restarts_history():
    monitor = ObservableGraspMonitor(25.)
    sample(monitor, 7)
    with pytest.raises(ValueError, match="consecutive"):
        sample(monitor, 9)
    with pytest.raises(ValueError, match="fractions"):
        sample(monitor, 8, measured=1.1)
    with pytest.raises(ValueError, match="finite coordinates"):
        monitor.update(8, [0, 1], .5, 0.)
    sample(monitor, 8)
    monitor.reset()
    sample(monitor, 0)


def test_guard_hysteresis_and_jaw_latch_prevent_carry_regression_and_early_release():
    guard = RecoveryPhaseGuard(25., phase_confirm_s=.08)
    close = np.r_[np.zeros(6), 0.]
    opened = np.r_[np.zeros(6), 1.]
    feedback = {"candidate_ever": True}
    guard.filter(close, "CLOSE", feedback)
    _, detail = guard.filter(close, "CLOSE", feedback)
    assert detail["execution_phase"] == "CLOSE"
    guard.filter(close, "LIFT", feedback)
    guard.filter(close, "LIFT", feedback)
    for phase in ("PRE_GRASP", "RELEASE", "CLOSE"):
        guard.filter(opened, phase, feedback)
        action, detail = guard.filter(opened, phase, feedback)
        assert detail["execution_phase"] == "LIFT"
        assert action[6] == 0.
    guard.filter(close, "LOWER", feedback)
    guard.filter(close, "LOWER", feedback)
    action, detail = guard.filter(opened, "RELEASE", feedback)
    assert action[6] == 0.
    action, detail = guard.filter(opened, "RELEASE", feedback)
    assert action[6] == 1.
    assert detail["execution_phase"] == "RELEASE"


@pytest.mark.parametrize("candidate", [False, True])
def test_skipping_lift_blocks_actual_transport_motion_even_with_jaw_candidate(candidate):
    guard = RecoveryPhaseGuard(25., phase_confirm_s=.04)
    proposal = np.r_[.01, .002, .003, np.zeros(3), 0.]
    filtered, detail = guard.filter(proposal, "TRANSPORT", {"candidate_ever": candidate})
    np.testing.assert_array_equal(filtered[:6], np.zeros(6))
    assert detail["execution_phase"] == "ACQUIRE"
    assert "carry_requires_lift_and_observable_grasp_candidate" in detail["intervention_reasons"]


@pytest.mark.parametrize("phase", ["TRANSPORT", "LOWER", "RELEASE"])
def test_unsafe_phase_request_is_blocked_before_hysteresis_can_confirm_it(phase):
    guard = RecoveryPhaseGuard(25., phase_confirm_s=.12)
    proposal = np.r_[.01, .002, .003, np.zeros(3), 1.]
    filtered, detail = guard.filter(proposal, phase, {})
    np.testing.assert_array_equal(filtered, np.zeros(7))
    assert detail["execution_phase"] == "ACQUIRE"
    assert "carry_requires_lift_and_observable_grasp_candidate" in detail["intervention_reasons"]


@pytest.mark.parametrize("phase", ["PRE_GRASP", "CLOSE", "RELEASE", "RETREAT"])
def test_first_carry_regression_or_early_release_request_immediately_stops_arm(phase):
    guard = RecoveryPhaseGuard(25., phase_confirm_s=.12)
    proposal = np.r_[0., 0., .002, np.zeros(3), 0.]
    feedback = {"candidate_ever": True}
    for _ in range(3):
        guard.filter(proposal, "LIFT", feedback)
    filtered, detail = guard.filter(np.r_[.01, np.zeros(5), 1.], phase, feedback)
    np.testing.assert_array_equal(filtered, np.zeros(7))
    assert detail["execution_phase"] == "LIFT"


def test_lift_can_establish_candidate_but_transport_wait_is_bounded_and_can_retry():
    guard = RecoveryPhaseGuard(25., phase_confirm_s=.04, candidate_wait_s=.12,
                               retry_pause_s=.08)
    lift = np.r_[0., 0., .002, np.zeros(3), 0.]
    filtered, detail = guard.filter(lift, "LIFT", {})
    assert filtered[2] > 0
    assert detail["execution_phase"] == "LIFT"
    filtered, detail = guard.filter(lift, "TRANSPORT", {})
    assert filtered[2] == 0.
    filtered, detail = guard.filter(lift, "TRANSPORT", {})
    assert "observable_candidate_timeout_retry" in detail["intervention_reasons"]
    assert filtered[6] == 1.
    guard.filter(lift, "TRANSPORT", {})
    # A new learned approach/close/lift sequence can establish a new attempt.
    guard.filter(lift, "PRE_GRASP", {})
    guard.filter(lift, "CLOSE", {})
    guard.filter(lift, "LIFT", {})
    filtered, detail = guard.filter(lift, "TRANSPORT", {"candidate_ever": True})
    assert detail["execution_phase"] == "TRANSPORT"
    assert filtered[2] > 0


def test_stable_phase_regression_stops_its_arm_command_as_well_as_latching_jaws():
    guard = RecoveryPhaseGuard(25., phase_confirm_s=.04)
    closed = np.r_[0., 0., .002, np.zeros(3), 0.]
    guard.filter(closed, "LIFT", {"candidate_ever": True})
    filtered, detail = guard.filter(np.r_[.01, np.zeros(5), 1.], "PRE_GRASP",
                                  {"candidate_ever": True})
    np.testing.assert_array_equal(filtered, np.zeros(7))
    assert detail["execution_phase"] == "LIFT"


def test_guard_observable_loss_pause_is_bounded_and_allows_new_pickup_attempt():
    guard = RecoveryPhaseGuard(25., phase_confirm_s=.04, retry_pause_s=.08)
    action = np.r_[.01, np.zeros(5), 0.]
    guard.filter(action, "LIFT", {})
    loss = {"loss_suspected": True}
    for _ in range(2):
        filtered, detail = guard.filter(action, "LIFT", loss)
        np.testing.assert_array_equal(filtered, np.r_[np.zeros(6), 1.])
        assert detail["execution_phase"] == "ACQUIRE"
    # Persistent evidence does not restart the pause forever.
    filtered, detail = guard.filter(action, "PRE_GRASP", loss)
    assert detail["execution_phase"] == "PRE_GRASP"
    assert filtered[0] > 0
    assert detail["retry_pause_ticks_remaining"] == 0


def test_motion_limits_bound_speed_and_acceleration_without_inventing_a_goal():
    hz, acceleration, speed = 25., .8, .12
    guard = RecoveryPhaseGuard(hz, max_speed_mps=speed, max_acceleration_mps2=acceleration)
    previous_velocity = np.zeros(3)
    for direction in (1., -1., 0.):
        for _ in range(12):
            original = np.r_[.03 * direction, np.zeros(5), 1.]
            action, detail = guard.filter(original, "PRE_GRASP", {})
            velocity = action[:3] * hz
            assert np.linalg.norm(velocity) <= speed + 1e-12
            assert np.linalg.norm(velocity - previous_velocity) * hz <= acceleration + 1e-12
            assert action[1] == action[2] == 0.
            previous_velocity = velocity
    assert detail["assistance_enabled"]


def test_monitor_has_no_environment_and_guard_rejects_unknown_phase():
    guard = RecoveryPhaseGuard(25.)
    with pytest.raises(ValueError, match="nonterminal learned"):
        guard.filter(np.zeros(7), "privileged_held", {})
    with pytest.raises(ValueError, match="seven-value"):
        guard.filter(np.zeros(8), "LIFT", {})
    guard.reset()
    np.testing.assert_array_equal(guard.velocity, np.zeros(3))
