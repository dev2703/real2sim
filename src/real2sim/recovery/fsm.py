"""Recovery gate finite-state machine (plan §4 M6 step 3, architecture §4.5).

``ACT -> DEVIATION_CONFIRMED -> SPECIALIST -> HANDOFF_CHECK -> ACT`` plus the absorbing ``STOP``.

* takeover (``ACT -> DEVIATION_CONFIRMED``) emits ``flush_chunk=True``: queued ACT actions are
  discarded and the specialist's action is used from this tick on;
* ``HANDOFF_CHECK`` needs ``H`` consecutive ticks of: EE clearance above the table, ball visible,
  slow EE, EE inside the pre-grasp corridor and no active deviation; then ``ACT`` is re-entered
  with ``force_fresh_observation=True`` (new observation, new ACT chunk);
* maximum recovery duration or retries, or a detector STOP recommendation -> ``STOP``, which is
  absorbing until :meth:`RecoveryGate.reset`.

The gate is pure and deterministic: it only reads the detector :class:`Decision` and the
observable features, and keeps a transcript ``(tick, state, reason)`` for logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from real2sim.recovery.detector import Decision, ObservableFeatures, inside_box


class GateState(StrEnum):
    ACT = "ACT"
    DEVIATION_CONFIRMED = "DEVIATION_CONFIRMED"
    SPECIALIST = "SPECIALIST"
    HANDOFF_CHECK = "HANDOFF_CHECK"
    STOP = "STOP"


@dataclass(frozen=True)
class GateOutput:
    tick: int
    state: GateState
    use_specialist: bool  # execute the specialist's action instead of ACT's
    flush_chunk: bool  # discard queued ACT actions (takeover tick only)
    force_fresh_observation: bool  # new observation + new ACT chunk (handoff tick only)
    stop: bool
    reason: str


class RecoveryGate:
    def __init__(self, config: dict[str, Any]) -> None:
        fsm = config.get("fsm", config)
        self.max_recovery_ticks = int(fsm["max_recovery_ticks"])
        self.max_retries = int(fsm["max_retries"])
        h = fsm["handoff"]
        self.handoff_ticks = int(h["ticks"])
        self.z_min = float(h["z_min_m"])
        self.handoff_ball_conf_min = float(h["ball_conf_min"])
        self.ee_speed_max = float(h["ee_speed_max"])
        corridor = h.get("corridor", "pre_grasp_corridor")
        if isinstance(corridor, str):
            corridor = config["detector"][corridor]
        self.corridor = corridor
        self.reset()

    def reset(self) -> None:
        self.state = GateState.ACT
        self.retries = 0
        self.recovery_start_tick: int | None = None
        self.handoff_count = 0
        self.transcript: list[tuple[int, str, str]] = []
        self.stop_reason = ""

    # ------------------------------------------------------------------ helpers
    def handoff_ok(self, f: ObservableFeatures, decision: Decision) -> bool:
        """Every handoff condition on this tick (architecture §4.5)."""
        if decision.deviation:
            return False
        # A missing, malformed or non-finite tracker/proprioception reading must never
        # satisfy handoff. In particular, NaN bypasses ordinary threshold comparisons.
        ee_pos = np.asarray(f.ee_pos, dtype=np.float64)
        ee_vel = np.asarray(f.ee_vel, dtype=np.float64)
        ball_rel = None if f.ball_rel is None else np.asarray(f.ball_rel, dtype=np.float64)
        ball_conf = float(f.ball_conf)
        if (
            ee_pos.shape != (3,)
            or ee_vel.shape != (3,)
            or ball_rel is None
            or ball_rel.shape != (3,)
            or not np.isfinite(ee_pos).all()
            or not np.isfinite(ee_vel).all()
            or not np.isfinite(ball_rel).all()
            or not np.isfinite(ball_conf)
            or not 0.0 <= ball_conf <= 1.0
            or ball_conf < self.handoff_ball_conf_min
        ):
            return False
        if ee_pos[2] <= self.z_min:
            return False
        if float(np.linalg.norm(ee_vel)) >= self.ee_speed_max:
            return False
        return inside_box(ee_pos, self.corridor)

    def _enter(self, tick: int, state: GateState, reason: str) -> None:
        self.state = state
        self.transcript.append((tick, state.value, reason))

    def _stop(self, tick: int, reason: str) -> GateOutput:
        self.stop_reason = reason
        self._enter(tick, GateState.STOP, reason)
        return GateOutput(tick, GateState.STOP, False, False, False, True, reason)

    def _recovery_timed_out(self, tick: int) -> bool:
        return self.recovery_start_tick is not None and tick - self.recovery_start_tick >= self.max_recovery_ticks

    # --------------------------------------------------------------------- step
    def step(self, tick: int, decision: Decision, f: ObservableFeatures) -> GateOutput:
        s = self.state
        if s is GateState.STOP:
            return GateOutput(tick, s, False, False, False, True, self.stop_reason)

        if decision.stop_recommended:
            return self._stop(tick, decision.reason)

        if s is GateState.ACT:
            if not decision.deviation:
                return GateOutput(tick, s, False, False, False, False, "none")
            if self.retries >= self.max_retries:
                return self._stop(tick, f"max_retries:{decision.reason}")
            self.retries += 1
            self.recovery_start_tick = tick
            self.handoff_count = 0
            self._enter(tick, GateState.DEVIATION_CONFIRMED, decision.reason)
            return GateOutput(tick, self.state, True, True, False, False, decision.reason)

        if self._recovery_timed_out(tick):
            return self._stop(tick, "max_recovery_ticks")

        if s is GateState.DEVIATION_CONFIRMED:
            self._enter(tick, GateState.SPECIALIST, "takeover")
            return GateOutput(tick, self.state, True, False, False, False, "takeover")

        if s is GateState.SPECIALIST:
            if self.handoff_ok(f, decision):
                self.handoff_count = 1
                self._enter(tick, GateState.HANDOFF_CHECK, "handoff_candidate")
                return GateOutput(tick, self.state, True, False, False, False, "handoff_candidate")
            return GateOutput(tick, s, True, False, False, False, "recovering")

        if s is GateState.HANDOFF_CHECK:
            if not self.handoff_ok(f, decision):
                self.handoff_count = 0
                self._enter(tick, GateState.SPECIALIST, "handoff_failed")
                return GateOutput(tick, self.state, True, False, False, False, "handoff_failed")
            self.handoff_count += 1
            if self.handoff_count >= self.handoff_ticks:
                self.recovery_start_tick = None
                self.handoff_count = 0
                self._enter(tick, GateState.ACT, "handoff")
                return GateOutput(tick, self.state, False, False, True, False, "handoff")
            return GateOutput(tick, s, True, False, False, False, "handoff_check")

        raise AssertionError(f"unreachable state {s}")
