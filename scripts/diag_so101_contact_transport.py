"""Matched privileged transport/release checks for diagnostic NoSlip settings.

This is a numerical contact sensitivity experiment, not a learned policy result
or a physical calibration. No production defaults or model parameters are edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from diag_so101_feasible_carry import ReachableGrasp, strict_params

from real2sim.config import load_yaml
from real2sim.evaluation.metrics import TrialTracker
from real2sim.evaluation.oracle import settle_start_pose
from real2sim.sim.env import So101PickPlaceEnv

PHYSICAL_ARRAYS = (
    "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_solref",
    "geom_solimp", "geom_priority", "geom_contype", "geom_conaffinity",
    "body_mass", "body_inertia", "actuator_gainprm", "actuator_biasprm",
    "actuator_forcerange", "actuator_ctrlrange", "jnt_range",
)


def physical_hashes(model):
    return {name: hashlib.sha256(getattr(model, name).tobytes()).hexdigest()
            for name in PHYSICAL_ARRAYS}


def trial(config, seed, noslip_iterations):
    with So101PickPlaceEnv(config, render_images=False) as env:
        physical_before = physical_hashes(env.model)
        env.model.opt.noslip_iterations = noslip_iterations
        physical_after_intervention = physical_hashes(env.model)
        assert physical_before == physical_after_intervention
        options = {"noslip_iterations": int(env.model.opt.noslip_iterations),
                   "noslip_tolerance": float(env.model.opt.noslip_tolerance),
                   "impratio": float(env.model.opt.impratio),
                   "solver_iterations": int(env.model.opt.iterations),
                   "physics_hz": env.config["physics_hz"], "control_hz": env.control_hz}
        start, settling = settle_start_pose(env, seed)
        obs, initial = env.reset(seed=seed, scenario={"joint_positions": start})
        ctrl = ReachableGrasp(env, strict_params(env))
        tracker = TrialTracker(math.ceil(env.task["episode"]["success_hold_s"] * env.control_hz),
                               .03, env.table_top, env.task["ball"]["radius"])
        phases, reasons = Counter(), Counter()
        max_penetration = {"table": 0., "support": 0.}
        release_samples = []
        rows = []
        abort = None
        try:
            for _ in range(env.max_episode_steps):
                result = ctrl.step(obs, env)
                obs = result.obs
                pm = result.info["privileged_metrics"]
                phase = result.detail["phase"]
                phases[phase] += 1
                reasons[result.detail["ik_reason"]] += 1
                max_penetration["table"] = max(max_penetration["table"], pm["robot_table_penetration_m"])
                max_penetration["support"] = max(max_penetration["support"], pm.get("robot_support_penetration_m", 0.))
                row = {"tick": tracker.ticks + 1, "phase": phase,
                       "ball_xyz_task": pm["ball_xyz_task"], "held": bool(pm["ball_held"]),
                       "lifted": bool(pm["lifted"]), "in_bowl": bool(pm["ball_in_bowl"]),
                       "resting": bool(pm["ball_resting"]),
                       "ball_speed_mps": float(np.linalg.norm(env.data.qvel[env._ball_v:env._ball_v + 3])),
                       "gripper_rad": float(env.sim_state()[5]),
                       "gripper_command_rad": float(result.info["control"]["applied_radians"][5])}
                if phase in ("RELEASE", "RETREAT", "DONE"):
                    release_samples.append(row)
                if tracker.ticks % 10 == 0:
                    rows.append(row)
                if tracker.update(pm, saturation=result.saturation, ik_rejected=result.ik_rejected) or result.truncated:
                    break
        except (RuntimeError, ValueError) as exc:
            abort = f"{type(exc).__name__}: {exc}"
        return {"config": str(config), "seed": seed, "noslip_iterations": noslip_iterations,
                "success": tracker.success, "pick_place_success": tracker.pick_place_success,
                "lifted": tracker.lifted, "delivered": tracker.delivered,
                "released": tracker.released, "rolled_out": tracker.rolled_out and not tracker.success,
                "ball_escaped_table": tracker.ball_escaped_table, "ticks": tracker.ticks,
                "phase_failed": tracker.phase_failed(), "final_fsm_phase": ctrl.phase,
                "phase_ticks": dict(phases), "phase_log": ctrl.phase_log,
                "ik_rejections": tracker.ik_rejections, "ik_reasons": dict(reasons),
                "max_robot_penetration_m": max_penetration,
                "max_action_saturation": tracker.max_action_saturation,
                "start_settling": settling, "initial_ball_xyz_task": initial["privileged_metrics"]["ball_xyz_task"],
                "grasp_search": ctrl.grasp_search, "actual_solver_options": options,
                "physical_arrays_unchanged_by_intervention": physical_before == physical_after_intervention,
                "model_arrays_changed_by_resets": [name for name, digest in physical_hashes(env.model).items()
                                                  if digest != physical_after_intervention[name]],
                "physical_array_sha256": physical_before,
                "warning_counts": env.data.warning.number.tolist(), "abort": abort,
                "release_samples": release_samples, "one_hz_samples": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    scenes = [("original_no_spacer", Path("outputs/v2/feasible_carry_20261023_no_spacer_config/sim.yaml"), (0, 4, 20)),
              ("image10mm_fixed", Path("outputs/v2/image_support_scenes_20261023/image_support_10mm_sim.yaml"), (0,))]
    paths = {Path(__file__), Path("scripts/diag_so101_feasible_carry.py"),
             Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/base_env.py"), Path("src/real2sim/sim/model.py"), Path("src/real2sim/artifacts.py"),
             Path("src/real2sim/sim/ball_support.py"), Path("src/real2sim/sim/kinematics.py"),
             Path("src/real2sim/evaluation/oracle.py"), Path("src/real2sim/serialization.py"), Path("src/real2sim/evaluation/metrics.py")}
    for _, config, _ in scenes:
        cfg = load_yaml(config)
        paths.update([config, Path(cfg["task_config"]), Path(cfg["model"]), Path(cfg["manifest"]), Path(cfg["data_config"])])

    def source_hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}

    before = source_hashes()
    report = {"kind": "PRIVILEGED_MATCHED_CONTACT_TRANSPORT_RELEASE_SENSITIVITY_NOT_LEARNED_POLICY",
              "mujoco_version": mujoco.__version__, "physical_calibration_accepted": False,
              "production_defaults_changed": False, "source_sha256_at_start": before,
              "protocol": {"settings_applied": "Immediately after model compilation, before first reset or settling",
                           "controller": "ReachableGrasp + feasible carry, unchanged original script",
                           "options": [0, 3], "episode_timeout_s": 30,
                           "success_rule": "Original TrialTracker: held lift, transport, released in bowl and resting for 1s",
                           "unchanged": "Physical parameters, geometry, native limits/collision/workspace gates, controller, scoring, resets and scene seeds",
                           "fixed_image_case_not_independent_seed_trials": True,
                           "contact_postprocessing_changes_simulated_dynamics": True}, "trials": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for scene, config, seeds in scenes:
        for seed in seeds:
            for option in (0, 3):
                row = trial(config, seed, option)
                row["scene"] = scene
                report["trials"].append(row)
                args.out.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: row[k] for k in ("scene", "seed", "noslip_iterations", "pick_place_success", "phase_failed", "ticks", "abort", "max_robot_penetration_m")}), flush=True)
    after = source_hashes()
    report["source_sha256_at_end"] = after
    report["source_changed_during_run"] = [path for path in before if before[path] != after[path]]
    report["summary"] = {str(option): {
        "n": len(rows := [r for r in report["trials"] if r["noslip_iterations"] == option]),
        "pick_place_successes": sum(r["pick_place_success"] for r in rows),
        "released": sum(r["released"] for r in rows),
        "aborts": sum(r["abort"] is not None for r in rows),
        "max_table_penetration_m": max(r["max_robot_penetration_m"]["table"] for r in rows),
        "max_support_penetration_m": max(r["max_robot_penetration_m"]["support"] for r in rows),
    } for option in (0, 3)}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
