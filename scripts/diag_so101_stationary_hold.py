"""Isolate zero-action drift from vision, labels, reference initialization and ball contact."""

import argparse
import copy
import json
import traceback
from pathlib import Path

import numpy as np
from run_so101_experiments import atomic_json, hash_file, now, queue_lock

from real2sim.geometry.so3 import delta_action
from real2sim.policy.adapters import CartesianActionAdapter
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics

OUT = Path("outputs/so101_root_cause_20261017/stationary_hold")
CONFIG = Path("results/audit/geometry_control_20261016/g2_fitted_xy_local/sim_f0.6_d1.0_b12.0.yaml")
MODES = ("fixed_joint_target", "zero_delta_current_pose", "fixed_initial_pose_feedback",
         "zero_delta_accepted_target")


def probe(config, state, mode, ticks=200):
    if mode not in MODES:
        raise ValueError(mode)
    with So101PickPlaceEnv(config, render_images=False) as env:
        q = sim_from_real(env.mapping, state)
        if np.any(q < env.limits[:, 0]) or np.any(q > env.limits[:, 1]):
            raise ValueError("Stationary hold requires an in-limit reference; do not clip it")
        env.max_episode_steps = ticks
        env.reset(seed=0, scenario={"joint_positions": q, "ball_xy": [.5, .3]})
        initial = env.info()["privileged_metrics"]
        if initial["robot_table_penetration_m"] > env.config["ik"]["penetration_tolerance_m"]:
            raise ValueError("Invalid stationary-hold start")
        initial_pose = env.ee_pose().copy()
        cfg = copy.deepcopy(env.config)
        cfg["ik"].update(position_convergence_m=1e-5, orientation_convergence_rad=1e-4)
        adapter = CartesianActionAdapter(env, NativeKinematics(env.model, cfg), target_reference=(
            "accepted_target" if mode == "zero_delta_accepted_target" else "current_pose"))
        rows = []
        for tick in range(ticks):
            before = env.ee_pose()
            if mode == "fixed_joint_target":
                _, _, _, _, info = env.step(q)
                rejected = False
            else:
                requested = (delta_action(before, initial_pose)
                             if mode == "fixed_initial_pose_feedback" else np.zeros(6))
                result = adapter.apply(np.r_[requested, state[5] / 100])
                info, rejected = result.info, result.ik_rejected
            error = delta_action(initial_pose, env.ee_pose())
            metrics = info["privileged_metrics"]
            rows.append({"tick": tick + 1, "translation_drift_m": float(np.linalg.norm(error[:3])),
                "rotation_drift_rad": float(np.linalg.norm(error[3:])), "ik_rejected": rejected,
                "native_target_drift_rad": float(np.max(abs(np.array(info["control"]["applied_radians"]) - q))),
                "joint_limit_clipped": info["control"]["limit_clipped"],
                "joint_rate_limited": info["control"]["rate_limited"],
                "robot_table_penetration_m": metrics["robot_table_penetration_m"],
                "ball_held": metrics["ball_held"],
                "ball_non_table_contact": bool(env._ball_contacts() - {env.table_geom})})
        return {"mode": mode, "ticks": ticks, "seconds": ticks / env.control_hz,
            "initial_reference": "Measured initial simulated pose, shared by every route; no future recorded targets",
            "end_drift_mm": rows[-1]["translation_drift_m"] * 1000,
            "max_drift_mm": max(r["translation_drift_m"] for r in rows) * 1000,
            "ik_rejections": sum(r["ik_rejected"] for r in rows),
            "ball_non_table_contact_ticks": sum(r["ball_non_table_contact"] for r in rows),
            "max_robot_table_penetration_m": max(r["robot_table_penetration_m"] for r in rows),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    out, config = args.out, args.config
    with queue_lock(Path("outputs/experiments_20261016_stage1")), queue_lock(out):
        if (out / "COMPLETE.json").exists():
            print("Stationary probe already complete; no duplicate work.")
            return
        if (out / "report.json").exists() or (out / "FAILED.json").exists():
            raise FileExistsError("Existing partial probe retained; choose a new --out path")
        atomic_json(out / "RUNNING.json", {"at": now()})
        try:
            joint = dict(np.load(".cache/derived_joint/arrays.npz"))
            source_files = [Path(__file__), config, Path(".cache/derived_joint/arrays.npz"),
                Path("src/real2sim/sim/env.py"),
                Path(yaml_task := __import__("yaml").safe_load(config.read_text())["task_config"]),
                Path("src/real2sim/policy/adapters.py"), Path("src/real2sim/sim/kinematics.py")]
            hashes = {str(p): hash_file(p) for p in source_files}
            report = {"kind": "NO_POLICY_STATIONARY_HOLD_CONTROL_DIAGNOSTIC",
                "source_sha256": hashes, "safety_thresholds_changed": False,
                "panda_allowed": False, "runs": []}
            (out / "executed_script.py").write_text(Path(__file__).read_text())
            for ep, timestamp in ((1, 18.), (2, 18.), (3, 20.)):
                selected = np.flatnonzero(joint["episode_index"] == ep)
                index = selected[np.argmin(abs(joint["timestamp"][selected] - timestamp))]
                for mode in args.modes:
                    run = probe(config, joint["state"][index], mode)
                    run.update(episode=ep, reference_timestamp=float(joint["timestamp"][index]))
                    report["runs"].append(run)
                    atomic_json(out / "report.json", report)
                    print(json.dumps({k: v for k, v in run.items() if k != "rows"}), flush=True)
            for path, expected in hashes.items():
                assert hash_file(Path(path)) == expected
            atomic_json(out / "COMPLETE.json", {"at": now(), "status": "diagnostic_complete",
                "runs": len(report["runs"]), "grasp_success": "not_evaluated", "panda_allowed": False})
        except BaseException as error:
            atomic_json(out / "FAILED.json", {"at": now(), "error": repr(error),
                                               "traceback": traceback.format_exc()})
            raise
        finally:
            (out / "RUNNING.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
