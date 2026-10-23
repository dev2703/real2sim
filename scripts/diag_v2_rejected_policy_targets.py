"""Replay learned targets at logged measured joints, retaining every acceptance gate.

This is an offline controller diagnostic, not a learned task-success measurement.
Alternative orientation weights affect numerical optimization only. Their candidates
still have to satisfy the original position/rotation/workspace/contact constraints.
The existing Jacobian projection changes the requested action and is labelled as such.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from contextlib import closing
from pathlib import Path

import mujoco
import numpy as np

from real2sim.geometry.so3 import delta_action
from real2sim.policy.kinematic_actions import bounded_joint_step
from real2sim.policy.paired import file_sha256
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics


def contacts(ik):
    rows = []
    tolerated = {"ball", "gripper", "moving_jaw_so101_v1"}
    for contact in ik.data.contact:
        bodies = {ik.model.body(ik.model.geom_bodyid[g]).name
                  for g in (contact.geom1, contact.geom2)}
        if "ball" in bodies or bodies <= tolerated:
            continue
        if contact.dist < -ik.config["penetration_tolerance_m"]:
            rows.append({"bodies": sorted(bodies), "penetration_m": -float(contact.dist),
                         "geoms": [ik.model.geom(g).name for g in (contact.geom1, contact.geom2)]})
    return rows


def result_row(result, ik, target):
    pose = ik.forward(ik.data.qpos[ik.qadr].copy())
    residual = delta_action(pose, target)
    return {"accepted": result.accepted, "reason": result.reason,
            "position_error_m": result.position_error_m,
            "orientation_error_rad": result.orientation_error_rad,
            "translation_residual_m": residual[:3].tolist(),
            "rotation_residual_rad": residual[3:].tolist(), "contacts": contacts(ik)}


def project(ik, env, target, seed, grip):
    current = ik.forward(seed)
    jp, jr = np.zeros((3, ik.model.nv)), np.zeros((3, ik.model.nv))
    mujoco.mj_jacSite(ik.model, ik.data, jp, jr, ik.site)
    jacobian = np.vstack([jp, jr])[:, ik.vadr[:5]]
    step = bounded_joint_step(
        jacobian, delta_action(current, target), seed[:5], ik.limits[:5],
        env.max_velocity[:5] / env.control_hz,
        rotation_weight=env.config["ik"]["orientation_weight_m_per_rad"], damping=.001)
    candidate = seed.copy()
    candidate[:5] += step
    candidate[5] = grip
    projected = ik.forward(candidate)
    changed = delta_action(projected, target)
    result = ik.solve(projected, seed, gripper=grip)
    return {**result_row(result, ik, projected),
            "requested_position_change_m": float(np.linalg.norm(changed[:3])),
            "requested_orientation_change_rad": float(np.linalg.norm(changed[3:]))}


def summarize(rows, mode):
    selected = [r[mode] for r in rows]
    return {"n": len(selected), "accepted": sum(r["accepted"] for r in selected),
            "reasons": dict(Counter(r["reason"] for r in selected)),
            "position_error_m_median": float(np.median([r["position_error_m"] for r in selected])),
            "orientation_error_rad_max": max(r["orientation_error_rad"] for r in selected)}


def run(args):
    original = json.loads((args.rollouts / "report.json").read_text())
    rows = []
    with closing(So101PickPlaceEnv(original["config"], render_images=False)) as env:
        modes = {"original": NativeKinematics(env.model, env.config)}
        for weight in (.008, .002, .0005):
            cfg = copy.deepcopy(env.config)
            cfg["ik"]["orientation_weight_m_per_rad"] = weight
            modes[f"optimization_weight_{weight:g}"] = NativeKinematics(env.model, cfg)
        for directory in sorted(args.rollouts.glob("seed_*")):
            previous = env.initial_qpos.copy()
            records = [json.loads(line) for line in (directory / "ticks.jsonl").read_text().splitlines()]
            for record in records:
                seed = np.clip(previous, env.limits[:, 0], env.limits[:, 1])
                previous = np.array(record["q_native_rad"])
                if not record["ik_rejected"]:
                    continue
                detail = record["detail"]
                target = np.array(detail["bounded_target_pose"])
                grip = detail["bounded_gripper_rad"]
                row = {"seed": int(directory.name[5:]), "tick": record["tick"],
                       "logged_reason": detail["ik_reason"], "q_native_rad": seed.tolist(),
                       "target_xyz": target[:3, 3].tolist(),
                       "ee_ball_distance_m_scoring_only": record["ee_ball_distance_m"]}
                for name, ik in modes.items():
                    result = ik.solve(target, seed, gripper=grip)
                    row[name] = result_row(result, ik, target)
                row["projected_target"] = project(modes["original"], env, target, seed, grip)
                rows.append(row)
    summary = {mode: summarize(rows, mode) for mode in (*modes, "projected_target")}
    per_seed = {str(seed): {mode: summarize([r for r in rows if r["seed"] == seed], mode)
                           for mode in (*modes, "projected_target")}
                for seed in sorted({r["seed"] for r in rows})}
    report = {"kind": "OFFLINE_REJECTED_LEARNED_TARGET_DIAGNOSTIC",
              "physical_calibration_accepted": False, "learned_task_success": None,
              "limitations": "Only previously rejected actions, at their exact previous measured "
              "native joints. No physics or policy rollout. Original gates remain unchanged. "
              "Projection intentionally changes targets; reduced optimizer weights do not. "
              "Ball positions appear only as copied scoring context and never enter any solver.",
              "original_acceptance": {key: env.config["ik"][key] for key in (
                  "position_tolerance_m", "orientation_tolerance_rad", "penetration_tolerance_m",
                  "workspace_min", "workspace_max")},
              "summary": summary, "per_seed": per_seed,
              "reproduced_logged_reasons": sum(r["logged_reason"] == r["original"]["reason"] for r in rows),
              "input_sha256": {str(p): file_sha256(p) for p in (
                  args.rollouts / "report.json", *(args.rollouts.glob("seed_*/ticks.jsonl")),
                  Path(original["config"]), Path("src/real2sim/sim/kinematics.py"),
                  Path("src/real2sim/policy/kinematic_actions.py"), Path(__file__))},
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": summary, "per_seed": per_seed,
                      "reproduced_logged_reasons": report["reproduced_logged_reasons"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, default=Path("outputs/v2/full_policy_rollouts_20261023"))
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    if options.out.exists():
        raise FileExistsError(options.out)
    run(options)
