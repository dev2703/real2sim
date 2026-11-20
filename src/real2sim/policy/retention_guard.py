"""Observable execution sequencing for heads without learned phase outputs.

This is explicit, optional assistance, not a learned stage prediction or proof
of holding an object. It filters proposals; it never generates goal directions.
"""

from __future__ import annotations

import math

import numpy as np


class RetentionGuard:
    def __init__(self, hz, *, mode="full", fresh_s=.4, close_wait_s=1.2, release_dwell_s=.2,
                 retry_pause_s=.2, reacquire_wait_s=2., rejection_ticks=8,
                 max_retries=3, delivery_xy_m=.025, delivery_height_m=.05,
                 max_speed_mps=.10, max_acceleration_mps2=.5,
                 visual_wait_s=2., release_wait_s=2.):
        values = locals().copy()
        values.pop("self")
        values.pop("mode")
        if mode not in ("full", "jaw", "visual"):
            raise ValueError("Retention mode must be full, jaw or visual")
        if not np.isfinite(list(values.values())).all() or min(values.values()) <= 0:
            raise ValueError("Retention guard requires positive finite limits")
        for name in ("rejection_ticks", "max_retries"):
            if isinstance(values[name], bool) or int(values[name]) != values[name]:
                raise ValueError("Retention guard counts must be integers")
        self.config = {**values, "mode": mode}
        self.mode = mode
        self.hz = float(hz)
        self.reset()

    def _ticks(self, key):
        return max(1, math.ceil(self.config[key] * self.hz))

    def reset(self):
        self.stage = "ACQUIRE"
        self.ticks = self.stage_ticks = self.retries = self.rejections = 0
        self.delivery_ticks = 0
        self.visual_wait_ticks = self.release_ticks = 0
        self.visual_hold_seen = False
        self.release_validated = False
        self.previous_release_ball = None
        self.previous_position = None
        self.velocity = np.zeros(3)
        self.transition_tick = 0
        self.stop_reason = None

    def _stage(self, stage):
        if stage != self.stage:
            self.stage, self.stage_ticks = stage, 0
            self.transition_tick = self.ticks
            self.delivery_ticks = 0

    def _retry(self, reason):
        self.retries += 1
        self.rejections = 0
        self.visual_hold_seen = False
        self.visual_wait_ticks = 0
        if self.retries > self.config["max_retries"]:
            self.stop_reason = reason
            self._stage("STOP")
        else:
            self._stage("REACQUIRE")

    @property
    def needs_rgb(self):
        return (self.stage in ("REACQUIRE", "RELEASE") if self.mode == "visual"
                else self.stage in ("ACQUIRE", "REACQUIRE"))

    def observe_execution(self, *, rejected, achieved_action):
        """Use actual motion for the next rate limit, never rejected intent."""
        achieved = np.asarray(achieved_action, float)
        if achieved.shape != (7,) or not np.isfinite(achieved).all():
            raise ValueError("Retention feedback requires seven finite achieved values")
        self.rejections = self.rejections + 1 if rejected else 0
        self.velocity = achieved[:3] * self.hz

    def filter(self, action, feedback, *, position, predicted_ball, bowl,
               ball_age_s, bowl_age_s):
        action = np.asarray(action, float)
        ball_age_s = float("inf") if ball_age_s is None else ball_age_s
        bowl_age_s = float("inf") if bowl_age_s is None else bowl_age_s
        position, bowl = np.asarray(position, float), np.asarray(bowl, float)
        predicted = None if predicted_ball is None else np.asarray(predicted_ball, float)
        if (action.shape != (7,) or position.shape != (3,) or bowl.shape != (3,)
                or not np.isfinite(np.r_[action, position, bowl]).all()
                or np.isnan([ball_age_s, bowl_age_s]).any()
                or min(ball_age_s, bowl_age_s) < 0 or not 0 <= action[6] <= 1
                or (predicted is not None and (predicted.shape != (3,) or not np.isfinite(predicted).all()))):
            raise ValueError("Retention guard requires finite actions, positions and nonnegative reference ages")
        before = self.stage
        result, reasons = action.copy(), []
        speed = (0. if self.previous_position is None else
                 float(np.linalg.norm(position - self.previous_position) * self.hz))
        self.previous_position = position.copy()
        fresh = ball_age_s <= self.config["fresh_s"]
        carry = self.stage in ("CLOSE", "LIFT", "TRANSPORT", "LOWER")
        if carry and feedback.get("visual_hold_supported", False):
            self.visual_hold_seen = True
        if self.stage not in ("STOP", "REACQUIRE", "RELEASE", "RETREAT"):
            if carry and feedback.get("loss_suspected", False):
                self._retry("observable_loss")
                reasons.append("observable_loss")
            elif self.rejections >= self.config["rejection_ticks"]:
                # Holding closed avoids dropping a possible payload when IK
                # fails; do not open merely because a motion was unreachable.
                if (self.mode == "visual" and not feedback.get("candidate_ever", False)
                        and not self.visual_hold_seen):
                    self._retry("unreachable_acquisition")
                    reasons.append("unreachable_acquisition")
                else:
                    self.stop_reason = "repeated_native_rejection"
                    self._stage("STOP")
                    reasons.append(self.stop_reason)
        if self.stage == "ACQUIRE":
            if not fresh:
                result[:6] = 0.
                result[6] = float(feedback.get("commanded_opening", 1.))
                reasons.append("acquisition_requires_fresh_rgb")
            elif action[6] < .5:
                self._stage("CLOSE")
        elif self.stage == "REACQUIRE":
            result[:6] = 0.
            if (self.stage_ticks >= self._ticks("retry_pause_s") and fresh
                    and self.ticks - ball_age_s * self.hz > self.transition_tick):
                self._stage("ACQUIRE")
            elif self.stage_ticks >= self._ticks("reacquire_wait_s"):
                self.stop_reason = "reacquisition_timeout"
                self._stage("STOP")
            reasons.append("loss_pause_and_fresh_rgb_reacquisition")
        elif self.stage in ("CLOSE", "LIFT"):
            # Let the model establish lift, while suppressing lateral carry
            # and rotation until resisted jaw + achieved lift give a candidate.
            if feedback.get("grasp_candidate", False):
                self._stage("TRANSPORT")
            elif self.stage_ticks >= self._ticks("close_wait_s"):
                if self.mode == "visual" and self.visual_hold_seen:
                    self.stop_reason = "jaw_candidate_timeout_with_possible_payload"
                    self._stage("STOP")
                    reasons.append(self.stop_reason)
                else:
                    self._retry("grasp_candidate_timeout")
                    reasons.append("grasp_candidate_timeout")
            else:
                if action[2] > 0:
                    self._stage("LIFT")
                if self.mode == "full":
                    result[[0, 1, 3, 4, 5]] = 0.
                    reasons.append("lift_before_lateral_transport")
        elif self.stage in ("TRANSPORT", "LOWER"):
            visual_hold = self.mode != "visual" or feedback.get("visual_hold_supported", False)
            self.visual_hold_seen |= bool(feedback.get("visual_hold_supported", False))
            self.visual_wait_ticks = 0 if visual_hold else self.visual_wait_ticks + 1
            if not visual_hold:
                # Initial jaw resistance can precede an actual pickup. Keep
                # the learned pickup motion during the bounded observation
                # window; once visual carrying was established, its expiry
                # immediately holds the arm with closed intent.
                if self.visual_hold_seen:
                    result[:6] = 0.
                reasons.append("awaiting_visual_hold")
                if self.visual_wait_ticks >= self._ticks("visual_wait_s"):
                    self.stop_reason = "visual_hold_timeout"
                    self._stage("STOP")
            delivery_ball = predicted
            if self.mode == "visual":
                measured_ball = feedback.get("fresh_ball_3d")
                delivery_ball = None if measured_ball is None else np.asarray(measured_ball, float)
                if delivery_ball is not None and (delivery_ball.shape != (3,)
                        or not np.isfinite(delivery_ball).all()):
                    raise ValueError("Delivery feedback requires a finite 3D RGB ball")
            aligned = (delivery_ball is not None and bowl_age_s <= self.config["fresh_s"]
                       and np.linalg.norm(delivery_ball[:2] - bowl[:2]) <= self.config["delivery_xy_m"])
            in_height = (aligned and 0 <= delivery_ball[2] - bowl[2] <= self.config["delivery_height_m"])
            # LOWER is established by achieved placement, not an open proposal.
            if aligned and self.stage == "TRANSPORT":
                self._stage("LOWER")
            stable = (in_height and visual_hold and speed <= .03 and feedback.get("candidate_ever", False)
                      and not feedback.get("retention_warning", False) and not self.rejections)
            self.delivery_ticks = self.delivery_ticks + 1 if stable else 0
            if (self.stage != "STOP" and action[6] >= .5
                    and self.delivery_ticks >= self._ticks("release_dwell_s")):
                self._stage("RELEASE")
                result[:6] = 0.
                reasons.append("stable_observable_delivery_release")
        elif self.stage == "RELEASE":
            result[:6] = 0.
            opened = feedback.get("measured_opening", 0.) >= .5
            if self.mode == "visual":
                sphere = feedback.get("fresh_ball_3d")
                sphere = None if sphere is None else np.asarray(sphere, float)
                if sphere is not None and (sphere.shape != (3,) or not np.isfinite(sphere).all()):
                    raise ValueError("Release feedback requires a finite 3D RGB ball")
                inside = (opened and sphere is not None and bowl_age_s <= self.config["fresh_s"]
                    and np.linalg.norm(sphere[:2] - bowl[:2]) <= self.config["delivery_xy_m"]
                    and -.05 <= sphere[2] - bowl[2] <= .01)
                settled = (inside and self.previous_release_ball is not None
                    and np.linalg.norm(sphere - self.previous_release_ball) * self.hz <= .03)
                self.release_ticks = self.release_ticks + 1 if settled else 0
                self.previous_release_ball = None if sphere is None else sphere.copy()
                self.release_validated = self.release_ticks >= self._ticks("release_dwell_s")
                if self.release_validated:
                    self._stage("RETREAT")
                elif self.stage_ticks >= self._ticks("release_wait_s"):
                    self.stop_reason = "release_unconfirmed"
                    self._stage("STOP")
                reasons.append("release_requires_observed_bowl_ball_and_open_jaw")
            elif opened:
                self._stage("RETREAT")

        if self.stage in ("CLOSE", "LIFT", "TRANSPORT", "LOWER"):
            result[6] = 0.
            if action[6] != 0:
                reasons.append("retain_closed_until_delivery")
        elif self.stage in ("RELEASE", "RETREAT", "REACQUIRE"):
            result[6] = 1.
        elif self.stage == "STOP":
            result[:6] = 0.
            result[6] = float(feedback.get("commanded_opening", 0.))
            reasons.append(self.stop_reason)

        # Freeze arm on acquisition/retry/stop rather than carrying residual
        # commanded velocity into a stale-target or loss interval.
        hold = (self.stage in ("REACQUIRE", "STOP", "RELEASE")
                or (before == "REACQUIRE") or (self.stage == "ACQUIRE" and not fresh))
        if hold:
            result[:6] = 0.
        elif self.mode == "full":
            velocity = result[:3] * self.hz
            norm = np.linalg.norm(velocity)
            velocity *= min(1., self.config["max_speed_mps"] / max(norm, 1e-12))
            change = velocity - self.velocity
            maximum = self.config["max_acceleration_mps2"] / self.hz
            change *= min(1., maximum / max(np.linalg.norm(change), 1e-12))
            result[:3] = (self.velocity + change) / self.hz
            result[:3] *= min(1., self.config["max_speed_mps"] /
                             max(np.linalg.norm(result[:3]) * self.hz, 1e-12))
            if self.stage in ("CLOSE", "LIFT"):
                # A phase constraint takes priority over velocity smoothing:
                # residual lateral velocity must not bypass the lift gate.
                result[[0, 1, 3, 4, 5]] = 0.
            if not np.allclose(result[:3], action[:3], atol=1e-12, rtol=0):
                reasons.append("achieved_motion_speed_or_acceleration_limit")
        detail = {"assistance_enabled": True, "execution_stage": self.stage,
                  "mode": self.mode,
                  "stage_changed": before != self.stage, "retry_count": self.retries,
                  "native_rejection_streak": self.rejections,
                  "delivery_dwell_ticks": self.delivery_ticks,
                  "visual_wait_ticks": self.visual_wait_ticks,
                  "visual_hold_seen": self.visual_hold_seen,
                  "release_validation_ticks": self.release_ticks,
                  "release_validated": self.release_validated,
                  "intervention_reasons": reasons, "stop_reason": self.stop_reason,
                  "action_changed": not np.array_equal(result, action),
                  "held_verified": False}
        self.ticks += 1
        self.stage_ticks += 1
        return result, detail
