"""Per-trial outcome accounting from the env's privileged metrics (scoring only, plan section 7).

The tracker consumes ``info['privileged_metrics']`` after every tick. It is the only consumer of
simulator ground truth in the evaluation loop; the policy batch is built elsewhere and never sees
these values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRIAL_COLUMNS = [
    "experiment_id", "controller_source", "robot", "seed", "checkpoint_sha256", "checkpoint_step",
    "success", "pick_place_success", "lifted", "delivered", "released", "rolled_out", "ball_escaped_table",
    "phase_failed", "min_ee_ball_distance_m", "ticks", "ik_rejections", "mean_latency_ms",
    "max_action_saturation", "video_path", "abort",
]
# ``aborted``: the trial stopped on an exception (IK/physics/action error); kept as a row and never
# merged into ``approach`` so a systematic harness bug cannot masquerade as a policy failure.
PHASES = ("approach", "grasp", "transport", "release", "none", "aborted")


@dataclass
class TrialTracker:
    """Accumulates success/failure evidence over one trial.

    ``success`` latches once ``ball_in_bowl and ball_resting and not ball_held`` has held for
    ``success_hold_ticks`` consecutive ticks. ``approach_distance_m`` is the tool-site-to-ball-centre
    distance below which a failed trial counts as having reached the ball (``grasp`` rather than
    ``approach``); it must match docs/experiment_protocol.md section 3.

    ``lifted`` means the ball was above the lift height *while held* (plan M7 "lift success"); a
    ball knocked into the air without a grasp is not a lift. ``rolled_out`` means a released ball
    left the bowl and the trial did not end in success (a bounce that settles back is not a roll).
    """

    success_hold_ticks: int
    approach_distance_m: float
    table_top_z: float
    ball_radius_m: float
    ticks: int = 0
    success: bool = False
    pick_place_success: bool = False
    lifted: bool = False
    delivered: bool = False
    held_ever: bool = False
    released: bool = False
    rolled_out: bool = False
    ball_escaped_table: bool = False
    min_ee_ball_distance_m: float = float("inf")
    max_action_saturation: float = 0.0
    ik_rejections: int = 0
    _settled: int = 0
    _in_bowl_prev: bool = False
    _held_prev: bool = False
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if int(self.success_hold_ticks) < 1:
            raise ValueError("success_hold_ticks must be >= 1 (0 would declare success on tick 1)")
        if not self.approach_distance_m > 0:
            raise ValueError("approach_distance_m must be positive")

    def update(self, privileged: dict, *, saturation: float = 0.0, ik_rejected: bool = False) -> bool:
        """Record one tick; returns True once the success condition has held long enough."""
        self.ticks += 1
        self.max_action_saturation = max(self.max_action_saturation, float(saturation))
        self.ik_rejections += int(ik_rejected)
        self.min_ee_ball_distance_m = min(self.min_ee_ball_distance_m,
                                          float(privileged["ee_ball_distance_m"]))
        held = bool(privileged["ball_held"])
        self.lifted |= bool(privileged["lifted"]) and held
        self.held_ever |= held
        if self._held_prev and not held:
            self.released = True
        in_bowl = bool(privileged["ball_in_bowl"])
        self.delivered |= in_bowl
        # Only a *released* ball leaving the bowl counts as rolling out: the ball must be unheld
        # on both the previous and the current tick. A held ball lowered through the rim+radius
        # band toggles in_bowl without any roll, and a release that coincides with leaving the
        # band is a release, not a roll-out.
        if self._in_bowl_prev and not in_bowl and not held and not self._held_prev:
            self.rolled_out = True
        self._held_prev = held
        self._in_bowl_prev = in_bowl
        z = float(privileged["ball_xyz"][2])
        if z < self.table_top_z - self.ball_radius_m:
            self.ball_escaped_table = True
        # Success needs the ball resting inside the bowl *and out of the gripper*: a ball clamped
        # motionless at the bowl mouth satisfies the geometric/velocity tests but was never placed.
        settled = in_bowl and bool(privileged["ball_resting"]) and not held
        self._settled = self._settled + 1 if settled else 0
        if self._settled >= self.success_hold_ticks:
            self.success = True
            self.pick_place_success |= self.lifted and self.released
        self.history.append({
            "tick": self.ticks, "in_bowl": in_bowl, "held": held, "resting": bool(privileged["ball_resting"]),
            "ee_ball_distance_m": float(privileged["ee_ball_distance_m"]),
            "ball_xyz_task": [float(v) for v in privileged["ball_xyz_task"]],
            "tool_xyz_task": [float(v) for v in privileged["tool_xyz_task"]],
        })
        return self.success

    def phase_failed(self) -> str:
        """Furthest phase reached (docs/experiment_protocol.md section 3).

        approach: never held and never came within ``approach_distance_m``; grasp: reached or
        touched the ball but never lifted it while held; transport: lifted but never in the bowl;
        release: in the bowl but not a stable, released success.
        """
        # Checked from the furthest phase back, so the label is the furthest phase reached.
        if self.success:
            return "none"
        if self.delivered:
            return "release"
        if self.lifted:
            return "transport"
        if self.held_ever or self.min_ee_ball_distance_m <= self.approach_distance_m:
            return "grasp"
        return "approach"

    def row(self, *, abort: str | None = None, **static) -> dict:
        row = dict(static)
        row.update({
            "success": self.success, "pick_place_success": self.pick_place_success and not bool(abort),
            "lifted": self.lifted, "delivered": self.delivered,
            "released": self.released, "rolled_out": self.rolled_out and not self.success,
            "ball_escaped_table": self.ball_escaped_table,
            "phase_failed": "aborted" if abort else self.phase_failed(),
            "abort": abort or None,
            "min_ee_ball_distance_m": (None if not np.isfinite(self.min_ee_ball_distance_m)
                                       else round(self.min_ee_ball_distance_m, 5)),
            "ticks": self.ticks, "ik_rejections": self.ik_rejections,
            "max_action_saturation": round(self.max_action_saturation, 4),
        })
        missing = [c for c in TRIAL_COLUMNS if c not in row]
        if missing:
            raise KeyError(f"trial row lacks columns {missing}")
        return {c: row[c] for c in TRIAL_COLUMNS}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    count = lambda key: int(sum(bool(r[key]) for r in rows))  # noqa: E731
    lat = [r["mean_latency_ms"] for r in rows if r["mean_latency_ms"] is not None]
    return {
        "n": n,
        "aborted": int(sum(bool(r.get("abort")) for r in rows)),
        "successes": count("success"),
        "success_rate": (count("success") / n) if n else None,
        "pick_place_successes": sum(bool(r.get("pick_place_success", False)) for r in rows),
        "pick_place_success_rate": (sum(bool(r.get("pick_place_success", False)) for r in rows) / n) if n else None,
        "lifted": count("lifted"), "delivered": count("delivered"), "released": count("released"),
        "rolled_out": count("rolled_out"), "ball_escaped_table": count("ball_escaped_table"),
        "phase_failed_counts": {p: sum(r["phase_failed"] == p for r in rows) for p in PHASES},
        "ik_rejections_total": int(sum(r["ik_rejections"] for r in rows)),
        "mean_latency_ms": (float(np.mean(lat)) if lat else None),
        "max_action_saturation": (max(r["max_action_saturation"] for r in rows) if rows else None),
    }
