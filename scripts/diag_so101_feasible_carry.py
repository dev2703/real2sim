"""Headless privileged controller diagnostic; never a learned-policy benchmark.

Compare the oracle FSM with unchanged native IK acceptance against a carry controller
that plans ball-position goals in the arm's five-joint configuration space. Every
projected native target passes the original IK workspace/collision/limit checks.
No physics, success rule, camera calibration or learned checkpoint is changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from real2sim.config import load_yaml
from real2sim.evaluation.metrics import TrialTracker
from real2sim.evaluation.oracle import (
    OracleGeometryController,
    OracleParams,
    grasp_orientation,
    settle_start_pose,
)
from real2sim.policy.adapters import ApplyResult
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics

CARRY_PHASES = ("LIFT", "TRANSPORT", "LOWER", "RELEASE", "RETREAT")


def strict_params(env):
    cfg = env.config["ik"]
    return OracleParams(carry_position_tolerance_m=cfg["position_tolerance_m"],
        carry_orientation_tolerance_rad=cfg["orientation_tolerance_rad"],
        carry_orientation_weight_m_per_rad=cfg["orientation_weight_m_per_rad"])


class FeasibleCarry(OracleGeometryController):
    """Privileged task-space planning with unconstrained carry orientation.

    Only carry motion differs. Approach and grasp use the identical strict IK/FSM.
    Orientation is a soft planning preference, then the exact FK target is checked
    with the unchanged native IK tolerances. This is an explicit task retargeting
    diagnostic, not an alteration to what an existing policy requested.
    """

    def __init__(self, env, params=None):
        params = params or strict_params(env)
        expected = strict_params(env)
        for name in ("carry_position_tolerance_m", "carry_orientation_tolerance_rad",
                     "carry_orientation_weight_m_per_rad"):
            if not math.isclose(getattr(params, name), getattr(expected, name)):
                raise ValueError("This diagnostic requires unchanged native IK acceptance")
        super().__init__(env, params)
        for key in ("workspace_min", "workspace_max"):
            if not np.allclose(self.ik_grasp.config[key], env.config["ik"][key], atol=1e-12, rtol=0):
                raise ValueError("This diagnostic requires unchanged native IK workspace")

    def reset(self):
        super().reset()
        self.carry_anchor = None
        self.ball_in_tool = None
        self.carry_rotation = None

    def _ball_goal(self, pm):
        if self.carry_anchor is None:
            pose = self.env.ee_pose()
            self.carry_anchor = np.asarray(pm["ball_xyz_task"], float)
            self.ball_in_tool = pose[:3, :3].T @ (self.carry_anchor - pose[:3, 3])
            self.carry_rotation = pose[:3, :3].copy()
        if self.phase == "LIFT":
            return self.carry_anchor + [0, 0, self.p.lift_height]
        bowl = np.asarray(pm["bowl_xyz_task"], float)
        goal = bowl + [0, 0, self.env.bowl["height"]
                       + self.env.task["ball"]["radius"] + self.p.transport_clearance]
        if self.phase in ("LOWER", "RELEASE"):
            goal[2] -= self.p.lower_depth
        elif self.phase == "RETREAT":
            goal[2] += self.p.retreat_height
        return goal

    def step(self, obs, env):
        if self.phase not in CARRY_PHASES:
            return super().step(obs, env)
        pm = env.info()["privileged_metrics"]
        goal = self._ball_goal(pm)
        lo, hi = env.limits.T
        current = np.clip(env.sim_state(), lo, hi)
        grip_pct = self.p.gripper_open_pct if self.phase in ("RELEASE", "RETREAT") else self.p.gripper_closed_pct
        grip = float(np.clip(env.mapping.gripper_real_to_sim_rad(grip_pct), lo[5], hi[5]))
        ik = self.ik_grasp  # original residual and collision thresholds in all phases

        def residual(arm):
            pose = ik.forward(np.r_[arm, grip])
            ball = pose[:3, 3] + pose[:3, :3] @ self.ball_in_tool
            rotation = Rotation.from_matrix(self.carry_rotation @ pose[:3, :3].T).as_rotvec()
            return np.r_[ball - goal, .002 * rotation, .0001 * (arm - current[:5])]

        plan = least_squares(residual, current[:5], bounds=(lo[:5], hi[:5]),
                             max_nfev=40, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        cap = env.max_velocity[:5] / env.control_hz
        step = np.clip(plan.x - current[:5], -cap, cap)
        target_q = np.r_[current[:5] + step, grip]
        initial = ik.forward(current)
        target = ik.forward(target_q)
        distance = np.linalg.norm(target[:3, 3] - initial[:3, 3])
        if distance > self.p.max_step_m:
            target_q[:5] = current[:5] + step * self.p.max_step_m / distance
            target = ik.forward(target_q)
        # Exact FK target and in-range seed: collision/workspace checks remain authoritative.
        checked = ik.solve(target, target_q, gripper=grip)
        rejected = not checked.accepted
        if rejected:
            target_q = current.copy()
            target_q[5] = grip
            self.ik_rejections += 1
        obs, _, _, truncated, info = env.step(target_q)
        self.tick += 1
        self.phase_tick += 1
        phase = self.phase
        pm = info["privileged_metrics"]
        pose = env.ee_pose()
        predicted_ball = pose[:3, 3] + pose[:3, :3] @ self.ball_in_tool
        position_error = float(np.linalg.norm(predicted_ball - goal))
        self._advance(pm, position_error, rejected)
        return ApplyResult(obs, info, truncated, float(np.mean(info["control"]["limit_clipped"])),
            ik_rejected=rejected, detail={"phase": phase, "ik_reason": checked.reason,
            "tracking_error_m": position_error, "planner_cost": float(plan.cost)})


class MatchedCarryPose(FeasibleCarry):
    """Same carry ball goals/offset as FeasibleCarry, full-pose strict native IK."""

    def _target(self, pm):
        if self.phase not in CARRY_PHASES:
            return OracleGeometryController._target(self, pm)
        goal = self._ball_goal(pm)
        pose = np.eye(4)
        pose[:3, :3] = (self.carry_rotation if self.phase == "LIFT"
                        else grasp_orientation(goal[:2], self.p.approach_tilt_deg))
        pose[:3, 3] = goal - pose[:3, :3] @ self.ball_in_tool
        grip = self.p.gripper_open_pct if self.phase in ("RELEASE", "RETREAT") else self.p.gripper_closed_pct
        return pose, grip

    def step(self, obs, env):
        return OracleGeometryController.step(self, obs, env)


class ReachableGrasp(FeasibleCarry):
    """Preselect a grasp tilt/IK branch by feasibility and joint travel, never outcomes."""

    def reset(self):
        super().reset()
        self.grasp_search = None
        self.grasp_plan = None
        self.p.approach_tilt_deg = 15.

    def _choose_grasp(self, env):
        cfg = copy.deepcopy(env.config)
        cfg["ik"].update(position_convergence_m=1e-5, orientation_convergence_rad=1e-4)
        ik = NativeKinematics(env.model, cfg)
        lo, hi = env.limits.T
        current = np.clip(env.sim_state(), lo, hi)
        ball = np.asarray(env.info()["privileged_metrics"]["ball_xyz_task"], float)
        pan = float(np.arctan2(ball[1], ball[0]))
        seeds = [current] + [np.clip(np.r_[pan, a, b, c, 0., current[5]], lo, hi)
            for a, b, c in ((-1.3, 1.3, 1.), (-.8, 1.3, .6), (-.3, .7, .6),
                            (.5, -1.3, 1.), (-.5, .6, -1.))]
        grip = float(np.clip(env.mapping.gripper_real_to_sim_rad(self.p.gripper_open_pct), lo[5], hi[5]))
        reasons, candidates = Counter(), []

        def valid_path(start, end):
            count = max(1, math.ceil(float(np.max(abs(end - start)
                / (env.max_velocity / env.control_hz)))))
            for alpha in np.linspace(0, 1, count + 1):
                q = start + alpha * (end - start)
                target = ik.forward(q)
                result = ik.solve(target, q, gripper=float(q[5]))
                if not result.accepted:
                    return result.reason
            return "accepted"

        for tilt in (10., 15., 20., 25.):
            target = np.eye(4)
            target[:3, :3] = grasp_orientation(ball[:2], tilt)
            target[:3, 3] = ball - target[:3, :3] @ self.p.ball_in_site
            pre = target.copy()
            pre[2, 3] += self.p.pre_grasp_height
            for index, seed in enumerate(seeds):
                result = ik.solve(pre, seed, gripper=grip)
                reasons["pre_" + result.reason] += 1
                if not result.accepted:
                    continue
                grasp = ik.solve(target, result.qpos, gripper=grip)
                reasons["grasp_" + grasp.reason] += 1
                if not grasp.accepted:
                    continue
                pre_q, grasp_q = np.clip(result.qpos, lo, hi), np.clip(grasp.qpos, lo, hi)
                path = valid_path(current, pre_q)
                if path == "accepted":
                    path = valid_path(pre_q, grasp_q)
                reasons["path_" + path] += 1
                if path != "accepted":
                    continue
                score = float(np.linalg.norm(pre_q[:5] - current[:5])
                    + np.linalg.norm(grasp_q[:5] - pre_q[:5]))
                candidates.append({"tilt_deg": tilt, "seed_index": index, "score": score,
                    "pre_qpos": pre_q.tolist(), "grasp_qpos": grasp_q.tolist()})
        self.grasp_plan = min(candidates, key=lambda c: (c["score"], c["tilt_deg"], c["seed_index"])) if candidates else None
        self.grasp_search = {"candidate_count": len(candidates), "reasons": dict(reasons),
            "selection": self.grasp_plan, "used_task_outcomes": False,
            "fallback": None if candidates else "original_15_degree_approach"}
        if self.grasp_plan:
            self.p.approach_tilt_deg = self.grasp_plan["tilt_deg"]

    def step(self, obs, env):
        if self.grasp_search is None:
            self._choose_grasp(env)  # before the first control tick, without trial outcomes
        if self.grasp_plan is None or self.phase not in ("PRE_GRASP", "DESCEND"):
            return super().step(obs, env)  # original CLOSE FSM and feasible carry unchanged
        lo, hi = env.limits.T
        current = np.clip(env.sim_state(), lo, hi)
        goal = np.asarray(self.grasp_plan[
            "pre_qpos" if self.phase == "PRE_GRASP" else "grasp_qpos"], float)
        step = goal - current
        # Follow the same joint-space line checked during preflight. Independent
        # component clipping follows a different path and can introduce collisions.
        ratio = float(np.max(abs(step) / (env.max_velocity / env.control_hz)))
        step *= 1 / max(1., ratio)
        initial = self.ik_grasp.forward(current)
        for _ in range(5):
            native = current + step
            target = self.ik_grasp.forward(native)
            distance = float(np.linalg.norm(target[:3, 3] - initial[:3, 3]))
            if distance <= self.p.max_step_m:
                break
            step *= self.p.max_step_m / distance * .999
        checked = self.ik_grasp.solve(target, native, gripper=float(native[5]))
        rejected = not checked.accepted
        if rejected:
            native = current
            self.ik_rejections += 1
        obs, _, _, truncated, info = env.step(native)
        self.tick += 1
        self.phase_tick += 1
        phase = self.phase
        actual, desired = env.ee_pose(), self.ik_grasp.forward(goal)
        position_error = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
        rotation_error = float(np.linalg.norm(Rotation.from_matrix(
            desired[:3, :3] @ actual[:3, :3].T).as_rotvec()))
        # A joint-space approach must reach the grasp orientation before descending.
        self._advance(info["privileged_metrics"], max(position_error,
            rotation_error * self.p.reach_tolerance_m / env.config["ik"]["orientation_tolerance_rad"]), rejected)
        return ApplyResult(obs, info, truncated, float(np.mean(info["control"]["limit_clipped"])),
            ik_rejected=rejected, detail={"phase": phase, "ik_reason": checked.reason,
                "tracking_error_m": position_error, "orientation_error_rad": rotation_error})


def run(config, mode, seeds, *, tilt_deg=15.):
    rows = []
    with So101PickPlaceEnv(config, render_images=False) as env:
        params = strict_params(env)
        params.approach_tilt_deg = float(tilt_deg)
        classes = {"strict_pose": OracleGeometryController, "matched_pose": MatchedCarryPose,
                   "feasible_carry": FeasibleCarry, "reachable_grasp": ReachableGrasp}
        ctrl = classes[mode](env, params)
        for seed in seeds:
            start, settling = settle_start_pose(env, seed)
            obs, _ = env.reset(seed=seed, scenario={"joint_positions": start})
            ctrl.reset()
            tracker = TrialTracker(math.ceil(env.task["episode"]["success_hold_s"] * env.control_hz),
                .03, env.table_top, env.task["ball"]["radius"])
            reasons, phases = Counter(), Counter()
            max_penetration = 0.
            held_ticks = 0
            for _ in range(env.max_episode_steps):
                result = ctrl.step(obs, env)
                obs = result.obs
                pm = result.info["privileged_metrics"]
                reasons[result.detail["ik_reason"]] += 1
                phases[result.detail["phase"]] += 1
                max_penetration = max(max_penetration, pm["robot_table_penetration_m"])
                held_ticks += int(pm["ball_held"])
                if tracker.update(pm, saturation=result.saturation, ik_rejected=result.ik_rejected) or result.truncated:
                    break
            row = {"mode": mode, "seed": seed, "success": tracker.success,
                "pick_place_success": tracker.pick_place_success, "held": tracker.held_ever,
                "held_ticks": held_ticks, "lifted": tracker.lifted, "delivered": tracker.delivered,
                "phase_failed": tracker.phase_failed(), "final_fsm_phase": ctrl.phase,
                "min_tool_ball_distance_m": tracker.min_ee_ball_distance_m,
                "ik_rejections": tracker.ik_rejections, "ik_reasons": dict(reasons),
                "phase_ticks": dict(phases), "phase_log": ctrl.phase_log,
                "max_table_penetration_m": max_penetration,
                "start_settling": settling}
            if isinstance(ctrl, ReachableGrasp):
                row["grasp_search"] = ctrl.grasp_search
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("mode", "seed", "success", "pick_place_success",
                "held", "lifted", "phase_failed", "ik_rejections")}), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/sim_so101.yaml"))
    p.add_argument("--modes", nargs="+", choices=("strict_pose", "matched_pose", "feasible_carry", "reachable_grasp"),
                   default=["strict_pose", "matched_pose", "feasible_carry"])
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--tilt-deg", type=float, default=15.)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    (args.out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    cfg = load_yaml(args.config)
    sources = [args.config, Path(cfg["task_config"]), Path(cfg["data_config"]),
        Path(cfg["model"]), Path(cfg["manifest"]), Path(__file__),
        Path("src/real2sim/config.py"), Path("src/real2sim/evaluation/oracle.py"),
        Path("src/real2sim/evaluation/metrics.py"), Path("src/real2sim/sim/env.py"),
        Path("src/real2sim/sim/kinematics.py"), Path("src/real2sim/sim/model.py"),
        Path("src/real2sim/sim/cameras.py"), Path("src/real2sim/geometry/so101_fk.py"),
        Path("src/real2sim/geometry/so3.py"), Path("src/real2sim/data/transforms.py"),
        Path("src/real2sim/policy/adapters.py")]
    report = {"controller_source": "PRIVILEGED_FEASIBLE_CARRY_DIAGNOSTIC", "learned_policy": False,
        "changes_to_physics_scoring_limits_collisions": False,
        "scene": str(args.config), "geometry_independently_calibrated": False,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "ik_acceptance": cfg["ik"], "approach_tilt_deg": args.tilt_deg,
        "executed_script": "executed_script.py", "trials": []}
    for mode in args.modes:
        seeds = range(args.seed_start, args.seed_start + args.n_trials)
        report["trials"].extend(run(args.config, mode, seeds, tilt_deg=args.tilt_deg))
        report["summary"] = {}
        for completed_mode in dict.fromkeys(t["mode"] for t in report["trials"]):
            trials = [t for t in report["trials"] if t["mode"] == completed_mode]
            report["summary"][completed_mode] = {"n": len(trials),
                "successes": sum(t["success"] for t in trials),
                "pick_place_successes": sum(t["pick_place_success"] for t in trials),
                "lifts": sum(t["lifted"] for t in trials),
                "ik_rejections": sum(t["ik_rejections"] for t in trials)}
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
