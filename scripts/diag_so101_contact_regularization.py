"""Exact-loaded-state friction regularization sensitivity, never physical calibration.

Acquire once under the unchanged nominal simulator, then restore the full integration
state for each predeclared option. Friction, mass, gains, geometry, limits, commands
and collision gates stay fixed. Only MuJoCo's friction regularization/postprocessor
changes after acquisition; no production model/config is edited.
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
from diag_so101_long_hold import ready, summarize_hold

from real2sim.config import load_yaml
from real2sim.evaluation.oracle import settle_start_pose
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics


def acquire(config, seed, grasp_z):
    with So101PickPlaceEnv(config, render_images=False) as env:
        env.max_episode_steps = 1000
        start, settling = settle_start_pose(env, seed)
        obs, _ = env.reset(seed=seed, scenario={"joint_positions": start})
        params = strict_params(env)
        params.ball_in_site = (-.01, 0., grasp_z)
        controller = ReachableGrasp(env, params)
        rows = []
        for _ in range(250):
            result = controller.step(obs, env)
            obs = result.obs
            row = snapshot(env)
            row["height_above_rest_m"] = float(env.data.xpos[env.ball_body, 2]
                                               - env._ball_rest_center_z)
            rows.append(row)
            if ready(rows):
                state_type = mujoco.mjtState.mjSTATE_INTEGRATION
                state = np.zeros(mujoco.mj_stateSize(env.model, state_type))
                mujoco.mj_getState(env.model, env.data, state, state_type)
                jaw_contacts = []
                for contact in env.data.contact[:env.data.ncon]:
                    geoms = (int(contact.geom1), int(contact.geom2))
                    if env.ball_geom not in geoms or contact.efc_address < 0:
                        continue
                    other = geoms[1] if geoms[0] == env.ball_geom else geoms[0]
                    if not any(other in members for members in env.gripper_geoms.values()):
                        continue
                    jaw_contacts.append({"geom": env.model.geom(other).name or str(other),
                        "dimension": int(contact.dim), "friction": contact.friction.tolist(),
                        "solref": contact.solref.tolist(), "solimp": contact.solimp.tolist(),
                        "jaw_priority": int(env.model.geom_priority[other]),
                        "ball_priority": int(env.model.geom_priority[env.ball_geom])})
                return {"state": state.tolist(), "state_type": int(state_type),
                    "target": result.info["control"]["applied_radians"],
                    "ball_rest_center_z": env._ball_rest_center_z, "anchor": row,
                    "acquisition_ticks": len(rows), "grasp_search": controller.grasp_search,
                    "settling": settling, "active_jaw_contact_parameters": jaw_contacts,
                    "source_model_options": {"impratio": env.model.opt.impratio,
                        "noslip_iterations": env.model.opt.noslip_iterations,
                        "cone": int(env.model.opt.cone), "solver": int(env.model.opt.solver)}}
            if controller.phase in ("RELEASE", "RETREAT", "DONE") or result.truncated:
                break
        raise RuntimeError("Predeclared baseline did not acquire a loaded bilateral hold")


def compare(config, loaded, seconds, impratio, noslip):
    with So101PickPlaceEnv(config, render_images=False) as env:
        env.reset(seed=0)
        env.max_episode_steps = round(seconds * env.control_hz) + 10
        state_type = mujoco.mjtState(loaded["state_type"])
        state = np.asarray(loaded["state"])
        mujoco.mj_setState(env.model, env.data, state, state_type)
        check = np.zeros_like(state)
        mujoco.mj_getState(env.model, env.data, check, state_type)
        np.testing.assert_array_equal(check, state)
        env._ball_rest_center_z = loaded["ball_rest_center_z"]
        env.model.opt.impratio = impratio
        env.model.opt.noslip_iterations = noslip
        mujoco.mj_forward(env.model, env.data)
        target = np.asarray(loaded["target"])
        target_pose = NativeKinematics(env.model, env.config).forward(target)
        rows = []
        for _ in range(round(seconds * env.control_hz)):
            _, _, _, _, info = env.step(target)
            row = snapshot(env)
            pm = info["privileged_metrics"]
            row.update(height_above_rest_m=float(env.data.xpos[env.ball_body, 2]
                                                - env._ball_rest_center_z),
                tool_target_error_m=float(np.linalg.norm(env.ee_pose()[:3, 3]
                                                        - target_pose[:3, 3])),
                robot_table_penetration_m=pm["robot_table_penetration_m"],
                robot_support_penetration_m=pm.get("robot_support_penetration_m", 0.))
            rows.append(row)
        early = rows[:round(2 * env.control_hz)]
        contacts = [c for r in early for c in r["contacts"] if c["jaw"]]
        positive = [c for c in contacts if c["sliding_friction_utilization"] is not None]
        total_normal = sum(c["normal_force_n"] for c in positive)
        summary = {"impratio": impratio, "noslip_iterations": noslip,
            "integration_state_restored_bitwise": True,
            "hold": summarize_hold(loaded["anchor"], rows, env.control_hz),
            "first2s_mean_surface_slip_mps": float(np.mean(
                [c["tangential_surface_slip_mps"] for c in contacts])),
            "first2s_mean_friction_utilization": float(np.mean(
                [c["sliding_friction_utilization"] for c in contacts
                 if c["sliding_friction_utilization"] is not None])),
            "first2s_max_friction_utilization": max(
                c["sliding_friction_utilization"] for c in contacts
                if c["sliding_friction_utilization"] is not None),
            "first2s_force_weighted_friction_utilization": sum(
                c["normal_force_n"] * c["sliding_friction_utilization"] for c in positive)
                / total_normal,
            "first2s_normal_force_fraction_on_nearly_saturated_contacts": sum(
                c["normal_force_n"] for c in positive
                if c["sliding_friction_utilization"] >= .95) / total_normal,
            "first2s_mean_sum_jaw_normal_n": float(np.mean(
                [sum(j["normal_force_n"] for j in r["jaws"].values()) for r in early])),
            "first2s_ball_tool_displacement_m": float(np.linalg.norm(
                np.asarray(early[-1]["ball_tool_m"]) - loaded["anchor"]["ball_tool_m"])),
            "samples": [{"elapsed_s": (i + 1) / env.control_hz, **r}
                        for i, r in enumerate(rows) if i % round(env.control_hz) == 0
                        or i == len(rows) - 1]}
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--grasp-z", type=float, default=.03)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not np.isfinite(args.seconds) or args.seconds < 30:
        parser.error("requires at least 30 seconds")
    config = load_yaml(args.config)
    paths = [Path(__file__), Path("scripts/diag_so101_long_hold.py"),
        Path("scripts/diag_so101_contact_retention.py"),
        Path("scripts/diag_so101_feasible_carry.py"),
        Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/model.py"),
        Path("src/real2sim/sim/ball_support.py"), Path("src/real2sim/sim/kinematics.py"),
        Path("src/real2sim/evaluation/oracle.py"), args.config,
        Path(config["task_config"]), Path(config["model"]), Path(config["manifest"])]
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    args.out.mkdir(parents=True)
    for i, path in enumerate(paths):
        destination = args.out / "sources" / f"{i:02d}_{path.name}"
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(path.read_bytes())
    loaded = acquire(args.config, args.seed, args.grasp_z)
    report = {"kind": "EXACT_STATE_CONTACT_MODEL_SENSITIVITY_NOT_CALIBRATION_OR_POLICY",
        "mujoco_version": mujoco.__version__, "source_sha256_at_start": before,
        "protocol": {"seconds": args.seconds, "seed": args.seed,
            "grasp_ball_in_site_m": [-.01, 0., args.grasp_z],
            "physical_contact_parameters_verified": False,
            "production_model_or_config_changed": False,
            "unchanged": "geometry, friction, mass, gains, commands, limits and collision gates",
            "changed_after_identical_acquisition": "friction regularization ratio or NoSlip postprocessor only",
            "strict_retention": "bilateral>0.01N, lift>30mm above original rest and drift<=10mm for30s",
            "official_reference": "https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip",
            "limitations": "One privileged loaded state, not independent scenes. NoSlip can introduce instability and is not calibrated contact physics."},
        "loaded_state": loaded, "trials": []}
    for impratio, noslip in ((10., 0), (100., 0), (10., 3)):
        row = compare(args.config, loaded, args.seconds, impratio, noslip)
        report["trials"].append(row)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
    after = hashes()
    report["source_sha256_at_end"] = after
    report["source_changed_during_run"] = [p for p in before if before[p] != after[p]]
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
