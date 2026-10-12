"""Transparent, rule-based deviation detector (plan §4 M6 steps 1-2).

Inputs are *observable* only: the canonical proprioception (D-008), the colour-tracker estimates
of the ball and bowl relative to the end-effector, their confidences, the last ACT-proposed
action and a gripper-contact flag derived from gripper state. There is no simulator ground
truth anywhere in :class:`ObservableFeatures`; ``tests/unit/test_recovery_fsm.py`` asserts it.

Every rule is a threshold plus a persistence counter (``configs/recovery.yaml``, provisional
until measured on replays), with hysteresis where a quantity can hover around the threshold.

The ``ball_rolled_away`` rule measures displacement from a latched reference ball position.
That reference is valid only for the controller that was in charge when it was latched: after
a takeover the specialist is *supposed* to go to the ball's new position, so the reference must
be re-latched or the rule reports the same roll forever and the gate can never hand off.
:meth:`DeviationDetector.rearm` does that; :class:`real2sim.recovery.loop.RecoveryLoop` calls it
on every takeover tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from real2sim.data.labels import ACTION_NAMES, PROPRIO_NAMES

REASONS = ("none", "ball_rolled_away", "approach_miss", "ee_drift", "ball_lost")

FEATURE_NAMES: list[str] = (
    list(PROPRIO_NAMES)
    + ["ball_rel_x", "ball_rel_y", "ball_rel_z"]
    + ["bowl_rel_x", "bowl_rel_y", "bowl_rel_z"]
    + ["ball_conf", "bowl_conf"]
    + [f"base_proposed_{a}" for a in ACTION_NAMES]
    + ["gripper_contact"]
)
assert len(FEATURE_NAMES) == 29, "D-005 fixes the specialist input at 29 floats"


@dataclass
class ObservableFeatures:
    """One policy tick of observable state. All geometry in the task frame, metres/radians."""

    ee_pos: np.ndarray  # (3,)
    ee_rot6d: np.ndarray  # (6,)
    ee_vel: np.ndarray  # (3,) m/s, finite difference at the policy clock
    gripper_obs: float  # semantic gripper opening in [0, 1]
    ball_rel: np.ndarray | None  # (3,) estimated ball centre minus EE, None if not visible
    bowl_rel: np.ndarray  # (3,) estimated bowl centre minus EE
    ball_conf: float  # tracker confidence in [0, 1]
    bowl_conf: float
    gripper_contact: bool  # derived from gripper state (commanded closed but blocked)
    last_action: np.ndarray | None = None  # (7,) last ACT-proposed action, None before the first
    phase_hint: str | None = None  # optional: approach | grasp | transport | release

    def to_vector(self) -> np.ndarray:
        """The 29-float specialist input (D-005), ordered as :data:`FEATURE_NAMES`."""
        ball = np.zeros(3) if self.ball_rel is None else np.asarray(self.ball_rel, np.float64)
        ball_conf = 0.0 if self.ball_rel is None else float(self.ball_conf)
        act = np.zeros(7) if self.last_action is None else np.asarray(self.last_action, np.float64)
        if act.shape != (7,):
            raise ValueError(f"last_action must have shape (7,), got {act.shape}")
        v = np.concatenate(
            [
                np.asarray(self.ee_pos, np.float64),
                np.asarray(self.ee_rot6d, np.float64),
                np.asarray(self.ee_vel, np.float64),
                [float(self.gripper_obs)],
                ball,
                np.asarray(self.bowl_rel, np.float64),
                [ball_conf, float(self.bowl_conf)],
                act,
                [1.0 if self.gripper_contact else 0.0],
            ]
        )
        if v.shape != (29,):
            raise ValueError(f"feature vector has shape {v.shape}, expected (29,)")
        return v


@dataclass(frozen=True)
class Decision:
    deviation: bool
    reason: str  # one of REASONS
    confidence: float  # in [0, 1]: persistence progress scaled by tracker confidence
    stop_recommended: bool = False  # True only for ball_lost

    def __post_init__(self) -> None:
        if self.reason not in REASONS:
            raise ValueError(f"reason {self.reason!r} not in {REASONS}")


@dataclass
class _Counters:
    ball_lost: int = 0
    rolled_away: int = 0
    approach_miss: int = 0
    ee_drift: int = 0
    ball_ref_task: np.ndarray | None = None  # pre-grasp reference ball position, task frame
    history: list[str] = field(default_factory=list)


def ball_task_position(f: ObservableFeatures) -> np.ndarray:
    """Estimated ball centre in the task frame (EE position plus tracker-relative estimate)."""
    if f.ball_rel is None:
        raise ValueError("ball is not visible; no task-frame ball position")
    return np.asarray(f.ee_pos, np.float64) + np.asarray(f.ball_rel, np.float64)


def inside_box(p: np.ndarray, box: dict[str, Any]) -> bool:
    lo = np.asarray(box["min_xyz"], np.float64)
    hi = np.asarray(box["max_xyz"], np.float64)
    p = np.asarray(p, np.float64)
    return bool(np.all(p >= lo) and np.all(p <= hi))


class DeviationDetector:
    """Per-tick deviation rules with persistence counters (no simulator truth)."""

    def __init__(self, config: dict[str, Any]) -> None:
        det = config["detector"] if "detector" in config else config
        self.cfg = det
        self.ball_conf_min = float(det["ball_conf_min"])
        self.ball_lost_ticks = int(det["ball_lost_ticks"])
        self.roll = det["rolled_away"]
        self.miss = det["approach_miss"]
        self.drift = det["ee_drift"]
        self.corridor = det["pre_grasp_corridor"]
        self.pre_grasp_phases = set(det.get("pre_grasp_phases", ["approach", "grasp"]))
        if float(self.roll["release_m"]) > float(self.roll["displacement_m"]):
            raise ValueError("rolled_away.release_m must be <= displacement_m (hysteresis)")
        self.reset()

    def reset(self) -> None:
        self._c = _Counters()

    @property
    def counters(self) -> dict[str, int]:
        return {
            "ball_lost": self._c.ball_lost,
            "rolled_away": self._c.rolled_away,
            "approach_miss": self._c.approach_miss,
            "ee_drift": self._c.ee_drift,
        }

    def set_ball_reference(self, ball_task: np.ndarray) -> None:
        """Pin the pre-grasp ball reference (default: first confident observation)."""
        self._c.ball_ref_task = np.asarray(ball_task, np.float64).copy()

    @property
    def ball_reference(self) -> np.ndarray | None:
        """Latched reference ball position in the task frame (``None`` until first latched)."""
        return None if self._c.ball_ref_task is None else self._c.ball_ref_task.copy()

    def rearm(self, f: ObservableFeatures | None = None) -> None:
        """Re-arm the rules for a new controller (call on every takeover).

        Clears all persistence counters and the ``ball_rolled_away`` reference. If ``f`` is given
        and the ball is confidently visible in it, the reference is re-latched at the ball's
        *current* position immediately; otherwise the next confident observation latches it.
        Transcript history is kept so the deviation that caused the takeover remains logged.
        """
        history = self._c.history
        self._c = _Counters(history=history)
        if f is not None and self.ball_visible(f):
            self._c.ball_ref_task = ball_task_position(f)

    def ball_visible(self, f: ObservableFeatures) -> bool:
        return f.ball_rel is not None and float(f.ball_conf) >= self.ball_conf_min

    def _pre_grasp(self, f: ObservableFeatures) -> bool:
        return f.phase_hint is None or f.phase_hint in self.pre_grasp_phases

    def update(self, f: ObservableFeatures) -> Decision:
        c = self._c
        held = bool(f.gripper_contact)
        visible = self.ball_visible(f)
        pre_grasp = self._pre_grasp(f)

        # ball_lost: K consecutive ticks without a confident estimate while not holding it.
        if not visible and not held:
            c.ball_lost += 1
        else:
            c.ball_lost = 0

        # ball_rolled_away: xy displacement from the pre-grasp reference while not held.
        if visible and not held and pre_grasp:
            ball_task = ball_task_position(f)
            if c.ball_ref_task is None:
                c.ball_ref_task = ball_task.copy()
            disp = float(np.linalg.norm((ball_task - c.ball_ref_task)[:2]))
            if disp >= float(self.roll["displacement_m"]):
                c.rolled_away += 1
            elif disp < float(self.roll["release_m"]):
                c.rolled_away = 0
            # between release_m and displacement_m: hold the counter (hysteresis)
        else:
            c.rolled_away = 0

        # approach_miss: close to the ball, gripper commanded closed, but no contact.
        closing = f.last_action is not None and float(f.last_action[6]) < float(
            self.miss["gripper_closed_cmd"]
        )
        if (
            visible
            and not held
            and pre_grasp
            and closing
            and float(np.linalg.norm(np.asarray(f.ball_rel, np.float64)))
            <= float(self.miss["reach_radius_m"])
        ):
            c.approach_miss += 1
        else:
            c.approach_miss = 0

        # ee_drift: outside the familiar pre-grasp corridor while not holding the ball.
        if not held and pre_grasp and not inside_box(f.ee_pos, self.corridor):
            c.ee_drift += 1
        else:
            c.ee_drift = 0

        # Priority: stop condition first, then the geometric rules.
        if c.ball_lost >= self.ball_lost_ticks:
            conf = min(1.0, c.ball_lost / self.ball_lost_ticks)
            return self._emit(Decision(True, "ball_lost", conf, stop_recommended=True))
        n = int(self.roll["ticks"])
        if c.rolled_away >= n:
            return self._emit(Decision(True, "ball_rolled_away", min(1.0, c.rolled_away / n) * float(f.ball_conf)))
        m = int(self.miss["ticks"])
        if c.approach_miss >= m:
            return self._emit(Decision(True, "approach_miss", min(1.0, c.approach_miss / m) * float(f.ball_conf)))
        k = int(self.drift["ticks"])
        if c.ee_drift >= k:
            return self._emit(Decision(True, "ee_drift", min(1.0, c.ee_drift / k)))
        return self._emit(Decision(False, "none", 0.0))

    def _emit(self, d: Decision) -> Decision:
        self._c.history.append(d.reason)
        return d
