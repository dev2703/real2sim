"""Recovery package (plan §4 M6): observable-only detector, gate FSM, specialist MLP, dataset.

Nothing here reads simulator ground truth; the specialist input is the 29-float observable
vector of decision D-005 (``detector.FEATURE_NAMES``).
"""

from real2sim.recovery.detector import (
    FEATURE_NAMES,
    Decision,
    DeviationDetector,
    ObservableFeatures,
)
from real2sim.recovery.fsm import GateOutput, GateState, RecoveryGate
from real2sim.recovery.loop import RecoveryLoop

__all__ = [
    "FEATURE_NAMES",
    "Decision",
    "DeviationDetector",
    "GateOutput",
    "GateState",
    "ObservableFeatures",
    "RecoveryGate",
    "RecoveryLoop",
]
