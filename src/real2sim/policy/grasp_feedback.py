"""Observable grasp evidence and optional stateful execution assistance.

Jaw command, jaw readback and measured TCP motion cannot prove that a ball is
held. They can establish a conservative grasp candidate, and closure after a
previously resisted jaw can flag a suspected loss. A fresh on-table RGB ball
far from the predicted grasp supplies independent missed-pickup/loss evidence.
No contacts, object truth, teacher phase or task-success flags enter these APIs.
"""

from __future__ import annotations

import math

import numpy as np


def _position(value, name):
    result = np.asarray(value, float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite coordinates")
    return result


class ObservableGraspMonitor:
    """Track causal evidence, keeping a closed command distinct from a grasp.

    Openings are semantic fractions, with zero fully closed and one fully open.
    Thresholds are explicit diagnostic assumptions, not learned contact labels.
    In particular, a jaw obstruction can satisfy the candidate test without a
    ball. Absence of visible RGB loss evidence never establishes a held object.
    """

    def __init__(self, hz, *, empty_opening=.08, resistance_opening=.18,
                 settle_s=.2, minimum_lift_m=.02, loss_dwell_s=.08,
                 free_ball_separation_m=.06, narrowing_warning=.12):
        values = [hz, empty_opening, resistance_opening, settle_s, minimum_lift_m,
                  loss_dwell_s, free_ball_separation_m, narrowing_warning]
        if (not np.isfinite(values).all() or min(values) <= 0
                or not empty_opening < resistance_opening < 1 or narrowing_warning >= 1):
            raise ValueError("Positive finite grasp thresholds and ordered opening fractions required")
        self.hz = float(hz)
        self.empty_opening = float(empty_opening)
        self.resistance_opening = float(resistance_opening)
        self.settle_ticks = max(1, math.ceil(settle_s * self.hz))
        self.minimum_lift_m = float(minimum_lift_m)
        self.loss_ticks = max(1, math.ceil(loss_dwell_s * self.hz))
        self.free_ball_separation_m = float(free_ball_separation_m)
        self.narrowing_warning = float(narrowing_warning)
        self.reset()

    @property
    def config(self):
        return {"hz": self.hz, "empty_opening": self.empty_opening,
                "resistance_opening": self.resistance_opening,
                "settle_s": self.settle_ticks / self.hz,
                "minimum_lift_m": self.minimum_lift_m,
                "loss_dwell_s": self.loss_ticks / self.hz,
                "free_ball_separation_m": self.free_ball_separation_m,
                "narrowing_warning": self.narrowing_warning}

    def reset(self):
        self.previous_tick = None
        self.close_tick = None
        self.close_position = None
        self.resistance_ticks = 0
        self.empty_ticks = 0
        self.candidate_ever = False
        self.candidate_peak_opening = None
        self.loss_reason = None

    def update(self, tick, tcp_position, measured_opening, commanded_opening, *,
               fresh_ball=None, predicted_ball=None):
        position = _position(tcp_position, "TCP position")
        if (not isinstance(tick, (int, np.integer)) or tick < 0
                or (self.previous_tick is not None and tick != self.previous_tick + 1)):
            raise ValueError("Grasp feedback requires consecutive nonnegative integer ticks")
        openings = np.asarray([measured_opening, commanded_opening], float)
        if not np.isfinite(openings).all() or np.any(openings < 0) or np.any(openings > 1):
            raise ValueError("Measured and commanded openings must be fractions in 0..1")
        ball = None if fresh_ball is None else _position(fresh_ball, "Fresh RGB ball")
        expected = None if predicted_ball is None else _position(predicted_ball, "Predicted ball")
        self.previous_tick = int(tick)
        closed = bool(commanded_opening < .5)
        if not closed:
            self.close_tick = self.close_position = None
            self.resistance_ticks = self.empty_ticks = 0
            self.candidate_ever = False
            self.candidate_peak_opening = None
            self.loss_reason = None
        elif self.close_tick is None:
            self.close_tick, self.close_position = int(tick), position.copy()
        dwell = 0. if self.close_tick is None else (tick - self.close_tick + 1) / self.hz
        lift = 0. if self.close_position is None else float(position[2] - self.close_position[2])
        resisted = bool(closed and measured_opening >= self.resistance_opening)
        empty = bool(closed and measured_opening <= self.empty_opening)
        self.resistance_ticks = self.resistance_ticks + 1 if resisted else 0
        self.empty_ticks = self.empty_ticks + 1 if empty else 0
        candidate = bool(closed and self.resistance_ticks >= self.settle_ticks
                         and lift >= self.minimum_lift_m)
        self.candidate_ever |= candidate
        if candidate:
            self.candidate_peak_opening = max(measured_opening,
                self.candidate_peak_opening if self.candidate_peak_opening is not None else 0.)
        narrowing = (0. if self.candidate_peak_opening is None
                     else max(0., float(self.candidate_peak_opening - measured_opening)))
        warning = bool(closed and narrowing >= self.narrowing_warning)
        separation = (None if ball is None or expected is None
                      else float(np.linalg.norm(ball - expected)))
        if closed:
            if separation is not None and separation > self.free_ball_separation_m:
                self.loss_reason = "fresh_on_table_rgb_ball_separated_from_predicted_grasp"
            elif self.candidate_ever and self.empty_ticks >= self.loss_ticks:
                self.loss_reason = "previously_resisted_jaw_closed_to_empty_range"
        suspected = self.loss_reason is not None
        status = ("loss_suspected" if suspected else "retention_warning" if warning
                  else "grasp_candidate" if candidate
                  else "empty_closed" if empty else "closing" if closed else "open")
        return {"status": status, "closed_command": closed,
                "measured_opening": float(measured_opening),
                "commanded_opening": float(commanded_opening),
                "closure_dwell_s": dwell, "tcp_lift_since_close_m": lift,
                "jaw_resisted": resisted, "grasp_candidate": candidate,
                "candidate_ever": self.candidate_ever, "loss_suspected": suspected,
                "loss_reason": self.loss_reason, "fresh_rgb_separation_m": separation,
                "jaw_narrowing_since_candidate": narrowing, "retention_warning": warning,
                "held_verified": False}


class RecoveryPhaseGuard:
    """Optional disclosed assistance around a learned phase/action prediction.

    This guard adds temporal hysteresis, preserves closed intent during carry,
    requires a LIFT prediction and observable grasp candidate before transport,
    and permits release after a stable LOWER prediction. A candidate timeout
    resets the attempt; a candidate is still not proof of holding an object.
    It never invents a
    goal or queries a geometric teacher. Suspected loss briefly stops the arm
    and opens the jaws, then resets phase memory so the learner can try again.
    Translation speed and acceleration limits isolate abrupt motion as a
    retention variable; native workspace/collision gates remain downstream.
    """

    phases = ("ACQUIRE", "PRE_GRASP", "DESCEND", "CLOSE", "LIFT", "TRANSPORT",
              "LOWER", "RELEASE", "RETREAT")
    carry = ("CLOSE", "LIFT", "TRANSPORT", "LOWER")

    def __init__(self, hz, *, phase_confirm_s=.12, retry_pause_s=.2,
                 max_speed_mps=.15, max_acceleration_mps2=1., candidate_wait_s=1.2):
        values = [hz, phase_confirm_s, retry_pause_s, max_speed_mps,
                  max_acceleration_mps2, candidate_wait_s]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Positive finite phase-guard timing and motion limits required")
        self.hz = float(hz)
        self.confirm_ticks = max(1, math.ceil(phase_confirm_s * self.hz))
        self.pause_ticks = max(1, math.ceil(retry_pause_s * self.hz))
        self.candidate_wait_ticks = max(1, math.ceil(candidate_wait_s * self.hz))
        self.max_speed_mps = float(max_speed_mps)
        self.max_acceleration_mps2 = float(max_acceleration_mps2)
        self.reset()

    @property
    def config(self):
        return {"hz": self.hz, "phase_confirm_s": self.confirm_ticks / self.hz,
                "retry_pause_s": self.pause_ticks / self.hz,
                "max_speed_mps": self.max_speed_mps,
                "max_acceleration_mps2": self.max_acceleration_mps2,
                "candidate_wait_s": self.candidate_wait_ticks / self.hz}

    def reset(self):
        self.phase = "ACQUIRE"
        self.pending_phase, self.pending_ticks = None, 0
        self.closed_latch = False
        self.retry_remaining = 0
        self.loss_handled = False
        self.lift_seen = False
        self.waiting_ticks = 0
        self.velocity = np.zeros(3)

    def _start_retry(self):
        self.retry_remaining = self.pause_ticks
        self.phase, self.pending_phase, self.pending_ticks = "ACQUIRE", None, 0
        self.closed_latch, self.lift_seen, self.waiting_ticks = False, False, 0

    def filter(self, action, phase_name, feedback):
        action = np.asarray(action, float)
        if action.shape != (7,) or not np.isfinite(action).all() or not 0 <= action[6] <= 1:
            raise ValueError("Phase guard requires a finite seven-value common action")
        if phase_name not in self.phases:
            raise ValueError("Phase guard requires a nonterminal learned phase name")
        result = action.copy()
        reasons = []
        loss = bool(feedback.get("loss_suspected", False))
        if loss and not self.loss_handled:
            self._start_retry()
            self.loss_handled = True
        elif not loss:
            self.loss_handled = False
        if self.retry_remaining:
            self.retry_remaining -= 1
            self.velocity[:] = 0.
            result[:6], result[6] = 0., 1.
            reasons.append("observable_loss_retry_pause")
        else:
            if phase_name == self.pending_phase:
                self.pending_ticks += 1
            else:
                self.pending_phase, self.pending_ticks = phase_name, 1
            candidate = bool(feedback.get("candidate_ever", False))
            allowed = True
            if self.closed_latch:
                if phase_name in self.carry:
                    allowed = self.carry.index(phase_name) >= self.carry.index(self.phase)
                elif phase_name == "RELEASE":
                    allowed = self.phase == "LOWER"
                else:
                    allowed = False
            if phase_name in ("TRANSPORT", "LOWER", "RELEASE") and (
                    not self.lift_seen or not candidate):
                allowed = False
                reasons.append("carry_requires_lift_and_observable_grasp_candidate")
            blocked_motion = not allowed
            if blocked_motion:
                reasons.append("phase_regression_or_premature_release")
            # Hysteresis controls phase advancement, while preconditions and
            # carry regression checks constrain the very first unsafe request.
            if allowed and self.pending_ticks >= self.confirm_ticks:
                self.phase = phase_name
                if self.phase == "LIFT":
                    self.lift_seen = True
            waiting = not candidate and (self.phase == "LIFT" or (
                phase_name in ("TRANSPORT", "LOWER", "RELEASE")))
            self.waiting_ticks = self.waiting_ticks + 1 if waiting else 0
            if self.waiting_ticks >= self.candidate_wait_ticks:
                self._start_retry()
                self.retry_remaining -= 1
                self.velocity[:] = 0.
                result[:6], result[6] = 0., 1.
                reasons.append("observable_candidate_timeout_retry")
                return result, self._detail(action, result, phase_name, reasons)
            if self.phase in self.carry:
                self.closed_latch = True
            elif self.phase in ("RELEASE", "RETREAT"):
                self.closed_latch = False
            if self.closed_latch and result[6] != 0.:
                result[6] = 0.
                reasons.append("carry_closed_jaw_latch")
            if blocked_motion:
                # A rejected phase must also reject its arm motion, rather
                # than only changing the execution-phase label in the log.
                self.velocity[:] = 0.
                result[:6] = 0.
                if phase_name in ("TRANSPORT", "LOWER", "RELEASE"):
                    result[6] = 0.
                return result, self._detail(action, result, phase_name, reasons)
            requested_velocity = result[:3] * self.hz
            speed = float(np.linalg.norm(requested_velocity))
            if speed > self.max_speed_mps:
                requested_velocity *= self.max_speed_mps / speed
            change = requested_velocity - self.velocity
            change_length = float(np.linalg.norm(change))
            maximum_change = self.max_acceleration_mps2 / self.hz
            if change_length > maximum_change:
                change *= maximum_change / change_length
            self.velocity += change
            result[:3] = self.velocity / self.hz
            if not np.allclose(result[:3], action[:3], atol=1e-12, rtol=0.):
                reasons.append("translation_speed_or_acceleration_limit")
        return result, self._detail(action, result, phase_name, reasons)

    def _detail(self, action, result, phase_name, reasons):
        return {"assistance_enabled": True, "requested_phase": phase_name,
                "execution_phase": self.phase, "closed_jaw_latched": self.closed_latch,
                "retry_pause_ticks_remaining": self.retry_remaining,
                "observable_candidate_wait_ticks": self.waiting_ticks,
                "intervention_reasons": reasons,
                "action_changed": bool(not np.array_equal(result, action)),
                "grasp_feedback_is_proof_of_hold": False}
