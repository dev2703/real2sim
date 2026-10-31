"""Retain a closed, accepted command after a waypoint carry-planning failure.

This opt-in variant uses phase and accepted-command history only. It does not
infer that a ball is held, consume contacts/success flags, or plan a new route.
The failed trial remains a failed trial through the configured timeout.
"""

from __future__ import annotations

import numpy as np

from real2sim.policy.adapters import ApplyResult
from real2sim.policy.visual_waypoint import VisualWaypointController


class FailureHoldWaypointController(VisualWaypointController):
    """Freeze the last safe closed-gripper command on LIFT/TRANSPORT/LOWER failure."""

    def reset(self):
        super().reset()
        self._last_safe_command = None
        self.failure_hold_command = None
        self.failure_hold_source_phase = None

    def _transition(self, phase):
        source = self.phase
        retained = (self._last_safe_command.copy()
                    if phase == "FAILED" and source in {"LIFT", "TRANSPORT", "LOWER"}
                    and self._last_safe_command is not None else None)
        super()._transition(phase)
        if retained is not None:
            self.failure_hold_command = retained
            self.failure_hold_source_phase = source

    def step(self, obs, env):
        if env is not self.env:
            raise ValueError("controller is bound to its construction environment")
        if self.failure_hold_command is None:
            result = super().step(obs, env)
            if self.phase != "FAILED" and not result.ik_rejected:
                key = "applied_radians" if self.robot == "so101" else "applied"
                self._last_safe_command = np.asarray(result.info["control"][key], float).copy()
            result.detail["failure_hold_armed"] = self.failure_hold_command is not None
            return result

        command = self.failure_hold_command.copy()
        accepted = self._safe(command[:self.n], command[-1])
        if not accepted:
            # Retain closure while holding measured arm joints if the static
            # map no longer accepts the previously checked command.
            command[:self.n] = self._native()[:self.n]
        observed, _, _, truncated, info = env.step(command)
        self.tick += 1
        self.phase_tick += 1
        return ApplyResult(observed, info, truncated,
            float(np.mean(info["control"]["limit_clipped"])), ik_rejected=not accepted,
            detail={"phase": "FAILED", "ik_reason": "failure_hold" if accepted else "failure_hold_FK_rejected",
                    "requested_native": command.tolist(), "failure": self.failure,
                    "failure_hold_armed": True,
                    "failure_hold_source_phase": self.failure_hold_source_phase,
                    "recovery_scope": "retain accepted arm and closed-gripper command; no new goal or placement claim",
                    "object_source": "phase_and_accepted_command_history_only"})
