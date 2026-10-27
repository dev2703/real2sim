"""Check the declared positive-layout SO101 transport goal without moving it.

Position-only multistart fitting removes the controller's soft orientation cost.
A separate triangle-inequality certificate bounds the pinned model's reach; it
depends on that model, task datum and the declared fixed ball/tool offset.
Neither calculation establishes physical robot geometry or a measured bowl pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from real2sim.artifacts import sha256_file as sha256
from real2sim.serialization import json_default as as_json
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics

SCENE = Path("configs/v2/positive_setup_20261026/sim_so101.yaml")
TRIALS = Path("outputs/v2/rgb_waypoint_so101_20261026_positive/report.json")
BALL_IN_TOOL = np.array([-.010, 0., .030])


def reach_certificate(env, goal):
    model = env.model
    ids = env.joint_ids[:5]
    if not np.allclose(model.jnt_pos[ids], 0):
        raise ValueError("certificate assumes the pinned arm joint origins coincide with body origins")
    pan_body = int(model.jnt_bodyid[ids[0]])
    lift_body = int(model.jnt_bodyid[ids[1]])
    pan_origin = model.body_pos[pan_body].copy()
    pan_rotation = np.empty(9)
    mujoco.mju_quat2Mat(pan_rotation, model.body_quat[pan_body])
    if not np.allclose(np.abs(pan_rotation.reshape(3, 3) @ model.jnt_axis[ids[0]]), [0, 0, 1]):
        raise ValueError("certificate requires a vertical shoulder-pan axis")
    lift_offset = pan_rotation.reshape(3, 3) @ model.body_pos[lift_body]
    circle_center = pan_origin + [0, 0, lift_offset[2]]
    circle_radius = float(np.linalg.norm(lift_offset[:2]))
    site_rotation = np.empty(9)
    mujoco.mju_quat2Mat(site_rotation, model.site_quat[env.site_id])
    ball_in_gripper_body = model.site_pos[env.site_id] + site_rotation.reshape(3, 3) @ BALL_IN_TOOL
    lengths = {model.body(int(model.jnt_bodyid[j])).name:
               float(np.linalg.norm(model.body_pos[int(model.jnt_bodyid[j])]))
               for j in ids[2:5]}
    lengths["gripper_body_origin_to_assumed_ball"] = float(np.linalg.norm(ball_in_gripper_body))
    bound = sum(lengths.values())
    relative = np.asarray(goal) - circle_center
    nearest = float(np.hypot(abs(np.linalg.norm(relative[:2]) - circle_radius), relative[2]))
    return {"method": "Triangle inequality from shoulder-lift pivot; unrestricted pan circle and arbitrary downstream rotations make this an intentionally generous outer bound.",
            "shoulder_pan_origin_task_m": pan_origin,
            "shoulder_lift_pivot_circle_center_task_m": circle_center,
            "shoulder_lift_pivot_circle_radius_m": circle_radius,
            "downstream_link_lengths_m": lengths,
            "gripper_body_to_assumed_ball_vector_m": ball_in_gripper_body,
            "maximum_distance_from_lift_pivot_outer_bound_m": bound,
            "minimum_goal_distance_to_unrestricted_pivot_circle_m": nearest,
            "certified_position_residual_lower_bound_m": max(0., nearest - bound),
            "scope": "Pinned model, identity task/base axes, fixed ball-in-tool offset and declared goal only; not hardware reach calibration."}


def run(out, starts=64):
    if out.exists():
        raise FileExistsError(out)
    previous = json.loads(TRIALS.read_text())
    inputs = [Path(__file__), Path("src/real2sim/artifacts.py"), Path("src/real2sim/serialization.py"),
              SCENE, TRIALS, Path("configs/v2/positive_setup_20261026/task.yaml"),
              Path("assets/menagerie/robotstudio_so101/so101.xml"),
              Path("src/real2sim/sim/kinematics.py")]
    before = {str(p): sha256(p) for p in inputs}
    report = {"schema": "positive_layout_so101_transport_reach_20261026_v1",
              "physical_calibration_accepted": False, "physics_actions_executed": 0,
              "native_limits_and_gates_changed": False, "position_only_objective": True,
              "position_only_optimizer_has_orientation_or_posture_penalty": False,
              "ball_in_tool_m": BALL_IN_TOOL,
              "source_sha256_before": before, "goals": [],
              "assumption_warning": "Bowl +x bearing, operator/task axes identity and base datum are assigned conventions, not independently measured registration. Results must not be described as physical hardware reach failure."}
    with So101PickPlaceEnv(SCENE, render_images=False) as env:
        env.reset(seed=0)
        ik = NativeKinematics(env.model, env.config)
        lo, hi = env.limits[:5].T
        grip = float(np.clip(env.mapping.gripper_real_to_sim_rad(0.), *env.limits[5]))
        trial = previous["trials"][0]
        bowl = np.asarray(trial["perception_log"][0]["targets"]["bowl"]["center_task"])
        nominal = np.r_[env.task["bowl"]["center_task_xy"], env.table_top - env.offset[2] + env.bowl["height"]]
        seeds = [np.asarray(trial["plan_log"][-1]["target_qpos"]), np.zeros(5)]
        seeds.extend(np.random.default_rng(20261026).uniform(lo, hi, (starts, 5)))
        for name, rim in (("actual_rgb_transport_goal", bowl), ("nominal_declared_transport_goal", nominal)):
            goal = rim + [0, 0, env.task["ball"]["radius"] + .03]

            def residual(q, goal=goal):
                pose = ik.forward(np.r_[q, grip])
                return pose[:3, 3] + pose[:3, :3] @ BALL_IN_TOOL - goal

            candidates = []
            for i, seed in enumerate(seeds):
                fitted = least_squares(residual, np.clip(seed, lo, hi), bounds=(lo, hi),
                    max_nfev=300, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                pose = ik.forward(np.r_[fitted.x, grip])
                error = float(np.linalg.norm(residual(fitted.x)))
                # Request the actual desired ball point with the candidate's own
                # orientation. This is not a tautological FK(candidate) target.
                desired_pose = pose.copy()
                desired_pose[:3, 3] = goal - pose[:3, :3] @ BALL_IN_TOOL
                checked = ik.solve(desired_pose, np.r_[fitted.x, grip], gripper=grip)
                ik.forward(np.r_[fitted.x, grip])
                candidates.append({"start": i, "optimizer_success": bool(fitted.success),
                    "nfev": fitted.nfev, "qpos_arm_rad": fitted.x,
                    "position_residual_m": error,
                    "within_original_3mm_position_gate": error <= ik.config["position_tolerance_m"],
                    "candidate_fk_collision": ik._collision(),
                    "original_desired_pose_gate_accepted": checked.accepted,
                    "original_desired_pose_gate_reason": checked.reason})
            candidates.sort(key=lambda c: c["position_residual_m"])
            certificate = reach_certificate(env, goal)
            if candidates[0]["position_residual_m"] + 1e-8 < certificate["certified_position_residual_lower_bound_m"]:
                raise AssertionError("numerical fit contradicts the conservative geometric bound")
            report["goals"].append({"name": name, "goal_ball_task_m": goal,
                "goal_horizontal_radius_from_task_base_m": float(np.linalg.norm(goal[:2])),
                "multistart_count": len(seeds), "multistart_is_global_proof": False,
                "best_position_residual_m": candidates[0]["position_residual_m"],
                "candidates_within_original_position_tolerance": sum(c["within_original_3mm_position_gate"] for c in candidates),
                "original_desired_pose_gate_acceptances": sum(c["original_desired_pose_gate_accepted"] for c in candidates),
                "reach_certificate": certificate, "candidates": candidates})
    report["source_sha256_after"] = {str(p): sha256(p) for p in inputs}
    report["runtime_source_changes"] = [p for p in before if before[p] != report["source_sha256_after"][p]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=as_json, allow_nan=False) + "\n")
    print(json.dumps([{k: g[k] for k in ("name", "multistart_count", "best_position_residual_m", "candidates_within_original_position_tolerance", "original_desired_pose_gate_acceptances")} for g in report["goals"]], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/v2/positive_transport_reach_20261026.json"))
    parser.add_argument("--random-starts", type=int, default=64)
    args = parser.parse_args()
    run(args.out, args.random_starts)
