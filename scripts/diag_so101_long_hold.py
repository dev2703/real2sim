"""Long loaded holds with strict contact/relative-pose scoring; privileged diagnostic.

The reachable-grasp controller still reads simulator object positions. The optional
encoder-width closure modification uses only gripper positions and requested targets.
Neither this runner nor that modification changes gains, friction, collision gates,
joint limits, the environment's task score, or any learned policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from diag_so101_contact_retention import snapshot
from diag_so101_feasible_carry import ReachableGrasp, strict_params

from real2sim.config import load_yaml
from real2sim.evaluation.oracle import settle_start_pose
from real2sim.sim.env import GRIPPER_BODIES, So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics


def force_bilateral(row, threshold_n=.01):
    """Positive force on both actual jaws, rather than proximity/one-jaw contact."""
    return all(row["jaws"][name]["normal_force_n"] > threshold_n for name in GRIPPER_BODIES)


def ready(rows, count=5):
    return len(rows) >= count and all(force_bilateral(r)
                                      and r.get("height_above_rest_m", r["height_m"]) > .03
                                      for r in rows[-count:])


def summarize_hold(anchor, rows, hz, max_displacement_m=.01):
    """Require sustained bilateral load and stable ball-to-tool pose for the full hold."""
    displacement = [float(np.linalg.norm(np.asarray(r["ball_tool_m"])
                                         - anchor["ball_tool_m"])) for r in rows]
    good = [force_bilateral(r) and r.get("height_above_rest_m", r["height_m"]) > .03
            and d <= max_displacement_m
            for r, d in zip(rows, displacement, strict=True)]
    first_loss = next(((i + 1) / hz for i, r in enumerate(rows) if not force_bilateral(r)), None)
    return {"duration_s": len(rows) / hz, "retained_entire_hold": bool(rows) and all(good),
        "first_force_bilateral_loss_s": first_loss,
        "first_strict_retention_failure_s": next(((i + 1) / hz for i, ok in enumerate(good)
                                                  if not ok), None),
        "max_ball_displacement_tool_m": max(displacement, default=None),
        "final_ball_displacement_tool_m": displacement[-1] if displacement else None,
        "end_force_bilateral": force_bilateral(rows[-1]) if rows else False,
        "max_tool_target_error_m": max((r["tool_target_error_m"] for r in rows), default=None),
        "max_robot_table_penetration_m": max((r["robot_table_penetration_m"] for r in rows), default=0.),
        "max_robot_support_penetration_m": max((r["robot_support_penetration_m"] for r in rows), default=0.)}


class EncoderWidth:
    """Latch a measured jaw angle after closing stalls, using encoder values only.

    This deliberately simple rejected/accepted candidate is not force control. The
    detector is fixed before trials: three readings span <0.015 rad after at least
    four closing samples, with requested closure lag >0.03 rad.
    """

    def __init__(self, preload_rad):
        self.preload_rad = float(preload_rad)
        self.positions = []
        self.target = None
        self.event = None

    def command(self, target, position, phase, tick):
        target = np.asarray(target, float).copy()
        if phase in ("CLOSE", "LIFT", "TRANSPORT", "LOWER"):
            self.positions.append(float(position))
            if (self.target is None and len(self.positions) >= 4
                    and np.ptp(self.positions[-3:]) < .015 and target[5] < position - .03):
                self.target = float(position - self.preload_rad)
                self.event = {"tick": tick, "position_rad": float(position), "phase": phase}
            if self.target is not None:
                target[5] = self.target
        return target


def trial(config, seed, *, mode="fixed_close", seconds=30., grasp_x=-.01, grasp_z=.03,
          preload_rad=.0003, max_acquisition_ticks=250, grasp_tilts_deg=(10., 15., 20., 25.),
          impratio=None, noslip_iterations=None):
    if impratio is not None and (not np.isfinite(impratio) or impratio <= 0):
        raise ValueError("impratio must be finite and positive")
    if noslip_iterations is not None and (not isinstance(noslip_iterations, int)
                                         or noslip_iterations < 0):
        raise ValueError("noslip_iterations must be a nonnegative integer")
    with So101PickPlaceEnv(config, render_images=False) as env:
        # Explicit numerical/contact-formulation probes, applied before settling or
        # reset. Default None preserves the upstream options and all task physics.
        if impratio is not None:
            env.model.opt.impratio = impratio
        if noslip_iterations is not None:
            env.model.opt.noslip_iterations = noslip_iterations
        solver_options = {"impratio": float(env.model.opt.impratio),
            "noslip_iterations": int(env.model.opt.noslip_iterations),
            "noslip_tolerance": float(env.model.opt.noslip_tolerance),
            "solver_iterations": int(env.model.opt.iterations)}
        hold_ticks = round(seconds * env.control_hz)
        env.max_episode_steps = max_acquisition_ticks + hold_ticks + 40
        start, settling = settle_start_pose(env, seed)
        obs, _ = env.reset(seed=seed, scenario={"joint_positions": start})
        params = strict_params(env)
        params.ball_in_site = (grasp_x, 0., grasp_z)
        ctrl = ReachableGrasp(env, params, grasp_tilts_deg=grasp_tilts_deg)
        encoder = EncoderWidth(preload_rad)
        raw_step = env.step
        if mode == "encoder_width":
            def step(target):
                return raw_step(encoder.command(target, env.sim_state()[5], ctrl.phase, env.steps))
            env.step = step
        acquisition = []
        target = None
        for _ in range(max_acquisition_ticks):
            result = ctrl.step(obs, env)
            obs = result.obs
            row = snapshot(env)
            pm = result.info["privileged_metrics"]
            row.update(height_above_rest_m=float(env.data.xpos[env.ball_body, 2]
                                                - env._ball_rest_center_z),
                robot_table_penetration_m=pm["robot_table_penetration_m"],
                robot_support_penetration_m=pm.get("robot_support_penetration_m", 0.))
            acquisition.append(row)
            if ready(acquisition):
                target = np.asarray(result.info["control"]["applied_radians"]).copy()
                break
            if ctrl.phase in ("RELEASE", "RETREAT", "DONE") or result.truncated:
                break
        common = {"seed": seed, "mode": mode, "grasp_ball_in_site_m": params.ball_in_site,
            "mujoco_version": mujoco.__version__,
            "actual_solver_options": solver_options,
            "numerical_warning_counts": env.data.warning.number.tolist(),
            "grasp_tilts_deg": list(grasp_tilts_deg),
            "acquisition_ticks": len(acquisition), "hold_acquired": target is not None,
            "encoder_event": encoder.event, "grasp_search": ctrl.grasp_search,
            "start_settling": settling, "acquisition_final": acquisition[-1],
            "acquisition_max_robot_table_penetration_m": max(
                r["robot_table_penetration_m"] for r in acquisition),
            "acquisition_max_robot_support_penetration_m": max(
                r["robot_support_penetration_m"] for r in acquisition)}
        if target is None:
            return common
        anchor = acquisition[-1]
        if mode == "contact_width":
            target[5] = float(env.sim_state()[5] - preload_rad)
        # Freeze exactly one native target. The encoder candidate has already latched;
        # it receives no privileged contacts, positions or force during this hold.
        target_pose = NativeKinematics(env.model, env.config).forward(target)
        rows = []
        for _ in range(hold_ticks):
            _, _, _, _, info = raw_step(target)
            row = snapshot(env)
            pm = info["privileged_metrics"]
            row.update(height_above_rest_m=float(env.data.xpos[env.ball_body, 2]
                                                - env._ball_rest_center_z),
                tool_target_error_m=float(np.linalg.norm(
                env.ee_pose()[:3, 3] - target_pose[:3, 3])),
                robot_table_penetration_m=pm["robot_table_penetration_m"],
                robot_support_penetration_m=pm.get("robot_support_penetration_m", 0.),
                gripper_native_rad=float(env.sim_state()[5]),
                gripper_actuator_force_nm=float(env.data.actuator_force[env.actuator_ids[5]]),
                legacy_ball_held=pm["ball_held"], ball_touching_table=pm["ball_touching_table"],
                ball_touching_support=pm.get("ball_touching_support", False))
            rows.append(row)
        common.update(hold_target_native=target.tolist(), anchor=anchor,
            hold=summarize_hold(anchor, rows, env.control_hz),
            numerical_warning_counts=env.data.warning.number.tolist())
        # One-Hz snapshots plus the loss neighborhood retain the mechanism evidence
        # without producing megabytes of redundant 10-Hz contact rows.
        loss = next((i for i, r in enumerate(rows) if not force_bilateral(r)), len(rows) - 1)
        indices = sorted(set(range(0, len(rows), round(env.control_hz)))
                         | {len(rows) - 1} | set(range(max(0, loss - 5), min(len(rows), loss + 3))))
        common["hold_samples"] = [{"elapsed_s": (i + 1) / env.control_hz, **rows[i]} for i in indices]
        return common


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 4, 20])
    p.add_argument("--modes", nargs="+", choices=["fixed_close", "contact_width", "encoder_width"],
                   default=["fixed_close"])
    p.add_argument("--seconds", type=float, default=30.)
    p.add_argument("--grasp-x", type=float, default=-.01)
    p.add_argument("--grasp-z", type=float, default=.03)
    p.add_argument("--grasp-tilts-deg", type=float, nargs="+", default=[10, 15, 20, 25])
    p.add_argument("--preload-rad", type=float, default=.0003)
    p.add_argument("--impratio", type=float, default=None)
    p.add_argument("--noslip-iterations", type=int, default=None)
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 20:
        p.error("--seconds must be finite and at least 20")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    cfg = load_yaml(args.config)
    sources = [Path(__file__), Path("scripts/diag_so101_feasible_carry.py"),
        Path("scripts/diag_so101_contact_retention.py"), args.config, Path(cfg["task_config"]),
        Path(cfg["model"]), Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/model.py"),
        Path("src/real2sim/sim/ball_support.py"),
        Path("src/real2sim/sim/kinematics.py"), Path("src/real2sim/evaluation/oracle.py")]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for path in sources:
        dest = args.out / "source" / path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
    report = {"kind": "PRIVILEGED_LONG_HOLD_DIAGNOSTIC_NOT_LEARNED_POLICY",
        "mujoco_version": mujoco.__version__,
        "physical_parameters_changed": False,
        "numerical_contact_options_requested": args.impratio is not None
            or args.noslip_iterations is not None,
        "geometry_independently_calibrated": False, "source_sha256_at_start": before,
        "protocol": {"seconds": args.seconds, "bilateral_force_threshold_n": .01,
            "max_ball_tool_displacement_m": .01,
            "min_ball_lift_above_initial_rest_m": .03,
            "acquisition": "Five positive-force bilateral samples lifted30mm above initial support rest height",
            "changes_to_gains_friction_collisions_limits_task_scoring": False,
            "encoder_width_uses": "Only requested target and measured gripper position; acquisition remains GT oracle",
            "grasp_x": args.grasp_x, "grasp_z": args.grasp_z, "preload_rad": args.preload_rad,
            "grasp_tilts_deg": args.grasp_tilts_deg,
            "impratio": args.impratio, "noslip_iterations": args.noslip_iterations,
            "numerical_intervention_timing": "Before all settling/reset/acquisition; None preserves defaults"},
        "trials": []}
    for mode in args.modes:
        for seed in args.seeds:
            row = trial(args.config, seed, mode=mode, seconds=args.seconds,
                grasp_x=args.grasp_x, grasp_z=args.grasp_z, preload_rad=args.preload_rad,
                grasp_tilts_deg=args.grasp_tilts_deg,
                impratio=args.impratio, noslip_iterations=args.noslip_iterations)
            report["trials"].append(row)
            (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: row[k] for k in ("seed", "mode", "hold_acquired")}
                             | {"hold": row.get("hold")}), flush=True)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report["source_changed_during_run"] = [path for path in before if before[path] != after[path]]
    report["source_sha256_at_end"] = after
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
