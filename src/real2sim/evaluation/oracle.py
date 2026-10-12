"""E4b oracle geometry diagnostic (plan M2 step 3, section 7 row E4b; gate G2 grasp physics).

A scripted pick-and-place controller for :class:`real2sim.sim.env.So101PickPlaceEnv` that reads
the simulator ground truth (``info['privileged_metrics']`` ``ball_xyz`` / ``bowl_xyz``) and drives
the arm through the diagnostic IK (:class:`real2sim.sim.kinematics.NativeKinematics`). It has no
policy, no observation adapter and no image input; it exists only to answer "can the simulated
gripper physically grasp, carry and place the simulated ball?" (plan 8 fallback) and is never an
upper bound for any learned result.

Labels are mandatory: ``controller_source = ORACLE_GEOMETRY_DIAGNOSTIC``, caption
``ORACLE (privileged GT, scripted)``. Never ``LEARNED_ACT``.

Grasp orientation (documented choice). FK (``So101FK.fk_real``) of the 12 held-ball states in
``results/audit/camera_fit/report.json`` (the ``docs/calibration.md`` hold episodes) puts the
gripper-frame x axis (finger direction, tips at x = 0) 45-76 deg away from straight down, pointing
radially away from the base, with the jaw-opening axis (site z) pointing up: the fixed jaw under
the ball, the moving jaw closing from above. In the simulated model that orientation cannot grasp a
67 mm ball resting on the table: the fixed jaw is 2.7 cm thick below its finger surface, so for
tilts >= 45 deg the IK rejects the grasp pose for table penetration ('collision'); the dataset
poses are consistent with the 1.2 cm base lift pending in D-014. A pure top-down pose (tilt 0) is
grasped but the 8 cm lifted poses are unreachable (IK 'residual_rejected'). The oracle therefore
uses a near-vertical approach, ``approach_tilt_deg`` = 15 from vertical in the vertical plane
through the ball, pinching the ball near the fingertips (``ball_in_site``).

Start pose: the clamped dataset rest pose penetrates the table (D-014 pending). Each trial starts
from ``scenario['joint_positions']`` = that rest pose after 30 ticks of physics settling under the
held initial target (``settle_start_pose``), which has zero robot-table contacts.

    uv run python -m real2sim.evaluation.oracle --config configs/sim_so101.yaml \
        --n-trials 10 --seed-start 0 --out results/metrics/oracle_so101 --video results/media/oracle
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from real2sim.config import load_yaml
from real2sim.evaluation.metrics import TRIAL_COLUMNS, TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo, policy_view_to_bgr
from real2sim.policy.adapters import IPHONE_KEY, ApplyResult
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics
from real2sim.sim.model import project_path

CONTROLLER_SOURCE = "ORACLE_GEOMETRY_DIAGNOSTIC"
EXPERIMENT_ID = "E4b_oracle_geometry_diagnostic"
CAPTION = "ORACLE (privileged GT, scripted)"
SETTLE_TICKS = 30
PHASES = ("PRE_GRASP", "DESCEND", "CLOSE", "LIFT", "TRANSPORT", "LOWER", "RELEASE", "RETREAT", "DONE")


@dataclass
class OracleParams:
    """Geometry knobs of the scripted grasp (metres, degrees, gripper percent)."""

    approach_tilt_deg: float = 15.0      # finger axis from straight down, in the ball's radial plane
    # Ball centre expressed in the gripper-frame site at the grasp pose: x along the fingers
    # (tips at x = 0, negative = towards the wrist), z along the jaw opening (positive = towards
    # the moving jaw; the fixed finger surface is at z = -0.020). Measured sweep (seed 0): held
    # and lifted for tilt 15 deg with bx in [-0.01, 0], bz in [0.02, 0.04]; deeper pinches
    # (bx <= -0.02) or tilts >= 45 deg put the fixed jaw into the table (IK 'collision').
    ball_in_site: tuple[float, float, float] = (-0.010, 0.0, 0.030)
    pre_grasp_height: float = 0.08
    lift_height: float = 0.08
    transport_clearance: float = 0.03    # ball bottom above the bowl rim during transport
    lower_depth: float = 0.015           # descend before opening
    retreat_height: float = 0.05
    gripper_open_pct: float = 90.0
    gripper_closed_pct: float = 0.0
    reach_tolerance_m: float = 0.012
    phase_tick_limits: dict = field(default_factory=lambda: {
        "PRE_GRASP": 40, "DESCEND": 30, "CLOSE": 20, "LIFT": 25, "TRANSPORT": 50,
        "LOWER": 15, "RELEASE": 10, "RETREAT": 20})
    close_hold_ticks: int = 3            # consecutive ball_held ticks before lifting
    # sim_so101.yaml ik.workspace_min z (0.015) keeps policy targets off the table; a grasp of a
    # 67 mm ball on the table needs the fingertips lower than that. Overridden in the oracle's own
    # IK copy only (the IK collision check still rejects table penetration > 2 mm).
    ik_workspace_min_z: float = 0.0
    # Per-tick Cartesian waypoint clamp towards the phase target (architecture section 4.2 uses
    # the same 3 cm for the learned controller). Without it the IK can jump to a far joint
    # configuration whose rate-limited execution drags the ball into the table (measured).
    max_step_m: float = 0.03
    # Orientation tolerance for the carry phases (the 5-DOF arm cannot hold the 15 deg tilt
    # everywhere above the table; position is what matters once the ball is pinched). The
    # grasp phases keep sim_so101.yaml ik.orientation_tolerance_rad.
    carry_orientation_tolerance_rad: float = 0.7
    carry_orientation_weight_m_per_rad: float = 0.01
    # A phase whose IK is rejected this many consecutive ticks has reached the arm's limit for
    # that target; move on instead of waiting for the tick limit (the ball slips slowly while
    # the gripper stalls).
    stall_ticks: int = 4


def settle_start_pose(env: So101PickPlaceEnv, seed: int, ticks: int = SETTLE_TICKS) -> tuple[np.ndarray, dict]:
    """Rest pose after ``ticks`` control ticks holding the initial target; clipped to env.limits.

    Returns the native pose and the start-pose penetration report before/after settling.
    """
    _, info = env.reset(seed=seed)
    before = dict(info["privileged_metrics"]["start_pose"])
    for _ in range(ticks):
        _, _, _, _, info = env.step(env.initial_qpos)
    pm = info["privileged_metrics"]
    after = {k: pm[k] for k in ("robot_table_contacts", "robot_table_penetration_m",
                                "tool_height_above_table_m")}
    q = np.clip(env.sim_state(), env.limits[:, 0], env.limits[:, 1])
    return q, {"settle_ticks": ticks, "before": before, "after": after, "qpos": q.tolist()}


def grasp_orientation(ball_task_xy: np.ndarray, tilt_deg: float) -> np.ndarray:
    """``R_base_site`` for the tilted approach: site x = finger axis, site z = jaw opening.

    x is rotated ``tilt_deg`` away from straight down towards the radial direction of the ball
    (away from the base), z = up-ish component perpendicular to x inside the same vertical plane,
    y = z cross x (horizontal, the moving-jaw hinge axis).
    """
    radial = np.array([ball_task_xy[0], ball_task_xy[1], 0.0])
    n = np.linalg.norm(radial)
    radial = radial / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    t = math.radians(tilt_deg)
    x = math.sin(t) * radial + math.cos(t) * np.array([0.0, 0.0, -1.0])
    z = math.cos(t) * radial + math.sin(t) * np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


class OracleGeometryController:
    """Phase FSM over privileged ball/bowl positions. ORACLE_GEOMETRY_DIAGNOSTIC only."""

    controller_source = CONTROLLER_SOURCE

    def __init__(self, env: So101PickPlaceEnv, params: OracleParams | None = None):
        self.env = env
        self.p = params or OracleParams()
        cfg = copy.deepcopy(env.config)
        cfg["ik"]["workspace_min"] = [*cfg["ik"]["workspace_min"][:2], self.p.ik_workspace_min_z]
        self.ik_grasp = NativeKinematics(env.model, cfg)
        carry = copy.deepcopy(cfg)
        carry["ik"]["orientation_tolerance_rad"] = self.p.carry_orientation_tolerance_rad
        carry["ik"]["orientation_weight_m_per_rad"] = self.p.carry_orientation_weight_m_per_rad
        self.ik_carry = NativeKinematics(env.model, carry)
        self.latencies_ms: list[float] = []
        self.n_policy_calls = 0  # there is no policy; stays 0 by construction
        self.reset()

    def reset(self) -> None:
        """Clear all per-trial state, including IK latencies (per-row ``mean_latency_ms`` is per trial)."""
        self.phase = "PRE_GRASP"
        self.phase_tick = 0
        self.phase_log: list[dict] = []
        self.ik_rejections = 0
        self.latencies_ms.clear()
        self._held_streak = 0
        self._reject_streak = 0
        self._grasp_T: np.ndarray | None = None
        self._first_held_tick: int | None = None
        self.tick = 0

    # ------------------------------------------------------------------ geometry
    def _grasp_pose(self, ball_task: np.ndarray) -> np.ndarray:
        R = grasp_orientation(ball_task[:2], self.p.approach_tilt_deg)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = ball_task - R @ np.asarray(self.p.ball_in_site, float)
        return T

    @staticmethod
    def _lifted(T: np.ndarray, dz: float) -> np.ndarray:
        out = T.copy()
        out[2, 3] += dz
        return out

    def _target(self, pm: dict) -> tuple[np.ndarray, float]:
        """(T_base_site target, gripper percent) for the current phase."""
        p = self.p
        ball = np.asarray(pm["ball_xyz_task"], float)
        bowl = np.asarray(pm["bowl_xyz_task"], float)
        rim_z = float(self.env.bowl["height"])
        radius = float(self.env.task["ball"]["radius"])
        if self.phase in ("PRE_GRASP", "DESCEND", "CLOSE"):
            self._grasp_T = self._grasp_pose(ball)
        grasp = self._grasp_T if self._grasp_T is not None else self._grasp_pose(ball)
        if self.phase == "PRE_GRASP":
            return self._lifted(grasp, p.pre_grasp_height), p.gripper_open_pct
        if self.phase == "DESCEND":
            return grasp, p.gripper_open_pct
        if self.phase == "CLOSE":
            return grasp, p.gripper_closed_pct
        if self.phase == "LIFT":
            return self._lifted(grasp, p.lift_height), p.gripper_closed_pct
        # Carry pose over the bowl: same orientation family, ball bottom clear of the rim.
        place_ball = np.array([bowl[0], bowl[1], rim_z + radius + p.transport_clearance])
        place = self._grasp_pose(place_ball)
        if self.phase == "TRANSPORT":
            return place, p.gripper_closed_pct
        if self.phase == "LOWER":
            return self._lifted(place, -p.lower_depth), p.gripper_closed_pct
        if self.phase == "RELEASE":
            return self._lifted(place, -p.lower_depth), p.gripper_open_pct
        return self._lifted(place, p.retreat_height), p.gripper_open_pct  # RETREAT / DONE

    # ------------------------------------------------------------------ FSM
    def _advance(self, pm: dict, pos_err: float, ik_rejected: bool) -> None:
        p = self.p
        limit = p.phase_tick_limits.get(self.phase, 10**9)
        self._reject_streak = self._reject_streak + 1 if ik_rejected else 0
        stalled = self._reject_streak >= p.stall_ticks
        reached = pos_err <= p.reach_tolerance_m
        timed_out = self.phase_tick >= limit or (stalled and self.phase != "CLOSE")
        nxt = None
        if self.phase == "PRE_GRASP" and (reached or timed_out):
            nxt = "DESCEND"
        elif self.phase == "DESCEND" and (reached or timed_out):
            nxt = "CLOSE"
        elif self.phase == "CLOSE":
            self._held_streak = self._held_streak + 1 if pm["ball_held"] else 0
            if pm["ball_held"] and self._first_held_tick is None:
                self._first_held_tick = self.tick
            if self._held_streak >= p.close_hold_ticks or timed_out:
                nxt = "LIFT"
        elif self.phase == "LIFT" and (reached or timed_out):
            nxt = "TRANSPORT"
        elif self.phase == "TRANSPORT" and (reached or timed_out):
            nxt = "LOWER"
        elif self.phase == "LOWER" and (reached or timed_out):
            nxt = "RELEASE"
        elif self.phase == "RELEASE" and timed_out:
            nxt = "RETREAT"
        elif self.phase == "RETREAT" and (reached or timed_out):
            nxt = "DONE"
        if nxt is not None:
            self.phase_log.append({"phase": self.phase, "end_tick": self.tick, "ticks": self.phase_tick,
                                   "reached": bool(reached), "timed_out": bool(timed_out),
                                   "ik_stalled": bool(stalled),
                                   "pos_err_m": round(float(pos_err), 4),
                                   "ball_held": bool(pm["ball_held"]), "lifted": bool(pm["lifted"]),
                                   "ball_in_bowl": bool(pm["ball_in_bowl"])})
            self.phase, self.phase_tick, self._held_streak, self._reject_streak = nxt, 0, 0, 0

    def step(self, obs: dict, env: So101PickPlaceEnv) -> ApplyResult:
        """One control tick. ``obs`` is ignored: the oracle reads env.info() ground truth."""
        del obs  # no policy input; privileged metrics only (labelled diagnostic)
        pm = env.info()["privileged_metrics"]
        target, grip_pct = self._target(pm)
        lo, hi = env.limits[:, 0], env.limits[:, 1]
        seed = np.clip(env.sim_state(), lo, hi)
        grip = float(np.clip(env.mapping.gripper_real_to_sim_rad(grip_pct), lo[5], hi[5]))
        # Waypoint: at most max_step_m towards the phase target, with the target orientation.
        cur = env.ee_pose()[:3, 3]
        waypoint = target.copy()
        d = target[:3, 3] - cur
        if (n := float(np.linalg.norm(d))) > self.p.max_step_m:
            waypoint[:3, 3] = cur + d * (self.p.max_step_m / n)
        if self.phase == "DONE":
            # Task complete: hold the current pose while the tracker's success hold runs out.
            # No IK solve, so no rejection is attributed to the finished motion.
            q = seed.copy()
            q[5] = grip
            ik_rejected, ik_reason, ik_pos_err, ik_rot_err = False, "done_hold", None, None
        else:
            ik = self.ik_grasp if self.phase in ("DESCEND", "CLOSE") else self.ik_carry
            t0 = time.perf_counter()
            result = ik.solve(waypoint, seed, gripper=grip)
            self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            ik_rejected, ik_reason = not result.accepted, result.reason
            ik_pos_err, ik_rot_err = result.position_error_m, result.orientation_error_rad
            if result.accepted:
                q = np.asarray(result.qpos, float)
            else:
                self.ik_rejections += 1
                q = seed.copy()       # hold the arm, still honour the gripper command
                q[5] = grip
        # NativeKinematics clips to model jnt_range; env.limits also intersects the actuator
        # ctrlrange (wrist_flex differs by 3e-6 rad). Clip here so a joint resting on its IK limit
        # is not reported as actuator saturation by env.step's limit_clipped.
        q = np.clip(q, lo, hi)
        obs, _, _, truncated, info = env.step(q)
        pm = info["privileged_metrics"]
        self.tick += 1
        self.phase_tick += 1
        pos_err = float(np.linalg.norm(env.ee_pose()[:3, 3] - target[:3, 3]))
        phase_before = self.phase
        self._advance(pm, pos_err, ik_rejected)
        saturation = float(np.mean(info["control"]["limit_clipped"]))
        return ApplyResult(obs, info, truncated, saturation, ik_rejected=ik_rejected,
                           detail={"phase": phase_before, "ik_reason": ik_reason,
                                   "position_error_m": ik_pos_err,
                                   "orientation_error_rad": ik_rot_err,
                                   "tracking_error_m": pos_err, "gripper_pct": grip_pct,
                                   "ball_held": bool(pm["ball_held"])})


# ---------------------------------------------------------------------- trials
ROLLED_OUT_NOTE = (
    "rolled_out = an in_bowl True->False transition at a tick where the ball was not held "
    "(computed from the tracker history). TrialTracker.rolled_out latches any such transition, "
    "including the held ball crossing the rim+radius z band during LOWER/RELEASE; that raw value "
    "is kept per trial as tracker_rolled_out_raw.")


def rolled_out_unheld(history: list[dict]) -> bool:
    """True if the ball left the bowl (in_bowl True -> False) while not held by the gripper.

    ``history`` is ``TrialTracker.history``: one dict per tick with ``in_bowl`` and ``held``. A
    transition is counted only when the ball is unheld on both the tick before and the tick after,
    so a grasped ball carried through the bowl's in/out boundary during placement is ignored.
    """
    for prev, cur in zip(history, history[1:], strict=False):
        if prev["in_bowl"] and not cur["in_bowl"] and not prev["held"] and not cur["held"]:
            return True
    return False


def run_oracle_trial(env: So101PickPlaceEnv, ctrl: OracleGeometryController, *, seed: int,
                     start_qpos: np.ndarray, max_ticks: int, tracker: TrialTracker,
                     video: CaptionedVideo | None = None, upscale: int = 2) -> list[dict]:
    obs, info = env.reset(seed=seed, scenario={"joint_positions": start_qpos})
    ctrl.reset()
    if video is not None:
        video.write(policy_view_to_bgr(obs[IPHONE_KEY], upscale), f"{CAPTION} seed={seed} tick=0")
    ticks = []
    for tick in range(1, max_ticks + 1):
        result = ctrl.step(obs, env)
        obs = result.obs
        done = tracker.update(result.info["privileged_metrics"], saturation=result.saturation,
                              ik_rejected=result.ik_rejected)
        ticks.append({"tick": tick, **result.detail})
        if video is not None:
            video.write(policy_view_to_bgr(obs[IPHONE_KEY], upscale),
                        f"{CAPTION} seed={seed} tick={tick} {result.detail['phase']}")
        if done or result.truncated:
            break
    return ticks


def run_trials(*, config: Path, eval_config: Path, n_trials: int, seed_start: int, out: Path,
               video_dir: Path | None, params: OracleParams | None = None,
               max_ticks: int | None = None, contacts: dict | None = None) -> dict:
    eval_cfg = load_yaml(eval_config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = So101PickPlaceEnv(config, render_images=video_dir is not None, contacts=contacts)
    task = env.task
    hold_ticks = int(math.ceil(task["episode"]["success_hold_s"] * env.control_hz))
    ticks_cap = env.max_episode_steps if max_ticks is None else min(int(max_ticks), env.max_episode_steps)
    params = params or OracleParams()
    ctrl = OracleGeometryController(env, params)
    vcfg = eval_cfg["video"]
    rows, trials = [], []
    t_start = time.perf_counter()
    for i in range(n_trials):
        seed = seed_start + i
        start_q, settle = settle_start_pose(env, seed)
        tracker = TrialTracker(success_hold_ticks=hold_ticks,
                               approach_distance_m=float(eval_cfg["phase"]["approach_distance_m"]),
                               table_top_z=env.table_top, ball_radius_m=float(task["ball"]["radius"]))
        writer, video_path = None, ""
        if video_dir is not None:
            h, w = env.data_config["image"]["height"], env.data_config["image"]["width"]
            video_path = str(Path(video_dir) / f"trial_seed{seed:04d}_{CONTROLLER_SOURCE}.mp4")
            writer = CaptionedVideo(video_path, vcfg["fps"], (h * vcfg["upscale"], w * vcfg["upscale"]),
                                    vcfg["codec"])
        t0 = time.perf_counter()
        tick_log: list[dict] = []
        try:
            tick_log = run_oracle_trial(env, ctrl, seed=seed, start_qpos=start_q, max_ticks=ticks_cap,
                                        tracker=tracker, video=writer, upscale=int(vcfg["upscale"]))
            abort = None
        except (RuntimeError, ValueError) as exc:  # physics/IK abort = failed trial, never dropped
            abort = str(exc)
        finally:
            if writer is not None:
                writer.close()
        lat = ctrl.latencies_ms  # cleared by ctrl.reset(): this trial's IK solves only
        row = tracker.row(experiment_id=EXPERIMENT_ID, controller_source=CONTROLLER_SOURCE,
                          robot="so101", seed=seed, checkpoint_sha256=None, checkpoint_step=None,
                          mean_latency_ms=(round(float(np.mean(lat)), 2) if lat else None),
                          video_path=video_path)
        tracker_rolled_out_raw = bool(row["rolled_out"])
        row["rolled_out"] = rolled_out_unheld(tracker.history)
        rows.append(row)
        trials.append({**row, "abort": abort, "wall_s": round(time.perf_counter() - t0, 2),
                       "tracker_rolled_out_raw": tracker_rolled_out_raw,
                       "policy_calls": ctrl.n_policy_calls, "final_phase": ctrl.phase,
                       "first_held_tick": ctrl._first_held_tick, "phase_log": ctrl.phase_log,
                       "start_pose": settle, "history": tracker.history, "ticks_detail": tick_log})
        print(f"[trial seed={seed}] source={CONTROLLER_SOURCE} success={row['success']} "
              f"phase_failed={row['phase_failed']} fsm={ctrl.phase} held_at={ctrl._first_held_tick} "
              f"lifted={row['lifted']} delivered={row['delivered']} ticks={row['ticks']} "
              f"ik_rej={row['ik_rejections']} wall_s={trials[-1]['wall_s']}"
              + (f" ABORT={abort}" if abort else ""), flush=True)
    env.close()
    with (out / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    summary = {
        "experiment_id": EXPERIMENT_ID, "controller_source": CONTROLLER_SOURCE, "robot": "so101",
        "caption": CAPTION,
        "note": ("Scripted controller on simulator ground truth (privileged ball/bowl positions). "
                 "Not a policy; never an upper bound for a learned result."),
        "rolled_out_note": ROLLED_OUT_NOTE,
        "ik_rejections_note": ("ik_rejections count IK solves rejected during PRE_GRASP..RETREAT "
                               "only; the DONE phase holds the current pose without solving IK."),
        "summary": summarize(rows),
        "final_phase_counts": {p: sum(t["final_phase"] == p for t in trials) for p in PHASES},
        "trials": trials,
        "checkpoint": {"checkpoint": None, "checkpoint_sha256": None, "checkpoint_step": None,
                       "kind": "oracle"},
        "contract": None,
        "oracle_params": {**params.__dict__, "ball_in_site": list(params.ball_in_site)},
        "contacts_override": contacts,
        "resolved": {"sim_config": env.config, "task_config": task, "eval_config": eval_cfg,
                     "data_config": env.data_config, "max_ticks": ticks_cap,
                     "success_hold_ticks": hold_ticks},
        "platform": {"python": platform.python_version(), "machine": platform.machine(),
                     "system": platform.system()},
        "wall_s_total": round(time.perf_counter() - t_start, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    print(f"successes {summary['summary']['successes']}/{n_trials} -> {out}", flush=True)
    return summary


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.ndarray, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=project_path("configs/sim_so101.yaml"))
    p.add_argument("--eval-config", type=Path, default=project_path("configs/eval_so101.yaml"))
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video", type=Path, default=None, help="directory for captioned mp4 per trial")
    p.add_argument("--max-ticks", type=int, default=None)
    p.add_argument("--tilt-deg", type=float, default=None, help="approach tilt from vertical")
    p.add_argument("--friction", type=float, default=None,
                   help="sliding friction override (sim_so101.yaml contacts.sweep_sliding_friction)")
    args = p.parse_args(argv)
    params = OracleParams()
    if args.tilt_deg is not None:
        params.approach_tilt_deg = args.tilt_deg
    contacts = None
    if args.friction is not None:
        base = load_yaml(args.config)["contacts"]["friction"]
        contacts = {"friction": [args.friction, base[1], base[2]]}
    run_trials(config=args.config, eval_config=args.eval_config, n_trials=args.n_trials,
               seed_start=args.seed_start, out=args.out, video_dir=args.video, params=params,
               max_ticks=args.max_ticks, contacts=contacts)


if __name__ == "__main__":
    main()
