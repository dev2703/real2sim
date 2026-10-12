"""Detector + gate composed on one observable feature stream (plan §4 M6 step 3).

:class:`RecoveryLoop` is the piece the closed-loop evaluation should drive: one call per policy
tick with the observable features, returning the detector :class:`Decision` and the gate
:class:`GateOutput`. It owns the one interaction the two parts cannot express alone: on every
takeover tick (``GateOutput.flush_chunk``) the detector is re-armed, so the ``ball_rolled_away``
reference is re-latched at the ball's post-roll position and the specialist can hand off once
it has reached it. Without this the stale reference keeps the deviation active for the whole
recovery and every roll-away ends in ``STOP`` (``max_recovery_ticks``).

Nothing here reads simulator ground truth; both inputs are :class:`ObservableFeatures`.
"""

from __future__ import annotations

from typing import Any

from real2sim.recovery.detector import Decision, DeviationDetector, ObservableFeatures
from real2sim.recovery.fsm import GateOutput, GateState, RecoveryGate


class RecoveryLoop:
    def __init__(self, config: dict[str, Any]) -> None:
        self.detector = DeviationDetector(config)
        self.gate = RecoveryGate(config)
        self.rearm_ticks: list[int] = []

    def reset(self) -> None:
        self.detector.reset()
        self.gate.reset()
        self.rearm_ticks = []

    @property
    def state(self) -> GateState:
        return self.gate.state

    def step(self, tick: int, f: ObservableFeatures) -> tuple[Decision, GateOutput]:
        decision = self.detector.update(f)
        out = self.gate.step(tick, decision, f)
        if out.flush_chunk:
            # Takeover: the specialist is now in charge and will approach the ball where it is
            # *now*; measure further roll-aways from that position, not from the pre-roll one.
            self.detector.rearm(f)
            self.rearm_ticks.append(tick)
        return decision, out
