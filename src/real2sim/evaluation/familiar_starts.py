"""Closed-loop diagnostic at estimated training-episode ball positions.

Uses existing camera-fit estimates restricted to training IDs. These are approximate starts,
not physically calibrated matches. Recorded poses are used only at reset, never as controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.evaluation.run import run_trials, sha256_file
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real


def make_scenarios(data: Path, backprojection: Path, env) -> tuple[list, list, list]:
    manifest = json.loads((data / "manifest.json").read_text())
    estimate = json.loads(backprojection.read_text())["estimates"]["start"]
    if len(estimate["points_xy"]) != len(estimate["episodes_frames"]):
        raise ValueError("start positions and provenance have different lengths")
    arrays = np.load(data / "arrays.npz")
    joints = arrays["joint_state"] if "joint_state" in arrays else arrays["state"]
    ball_only, ball_and_pose, sources = [], [], []
    for xy, (episode, frame) in zip(estimate["points_xy"], estimate["episodes_frames"], strict=True):
        if episode not in manifest["split"]["train"]:
            continue
        ticks = np.flatnonzero(arrays["episode_index"] == episode)
        first = int(ticks[np.argmin(arrays["frame_index"][ticks])])
        raw = joints[first]
        native = sim_from_real(env.mapping, raw)
        clamped = np.clip(native, env.limits[:, 0], env.limits[:, 1])
        ball_only.append({"ball_xy": xy})
        ball_and_pose.append({"ball_xy": xy, "joint_positions": clamped.tolist()})
        sources.append({"training_episode": episode, "ball_detection_frame": frame,
                        "reset_frame": int(arrays["frame_index"][first]),
                        "ball_xy_estimated_m": xy, "recorded_joint_state": raw.tolist(),
                        "reset_clamp_radians": (clamped - native).tolist()})
    if not sources:
        raise ValueError("no estimated starts belong to training episodes")
    return ball_only, ball_and_pose, sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backprojection", type=Path, default=Path("results/audit/camera_fit/backprojection.json"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    sim, ev = Path("configs/sim_so101.yaml"), Path("configs/eval_so101.yaml")
    with So101PickPlaceEnv(sim, render_images=False) as env:
        balls, poses, sources = make_scenarios(args.data, args.backprojection, env)
        reset_diagnostics = []
        for scenario in poses:
            _, info = env.reset(seed=0, scenario=scenario)
            before = env.sim_state()
            _, _, _, _, after = env.step(before)
            reset_diagnostics.append({"at_reset": info["privileged_metrics"]["start_pose"],
                "after_one_hold_tick": {key: after["privileged_metrics"][key] for key in (
                    "robot_table_contacts", "robot_table_penetration_m", "tool_height_above_table_m")},
                "joint_drift_one_hold_tick_radians": (env.sim_state() - before).tolist(),
                "tool_motion_one_hold_tick_m": float(np.linalg.norm(
                    np.array(after["privileged_metrics"]["tool_xyz"])
                    - info["privileged_metrics"]["tool_xyz"]))})
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"sources": sources, "reset_diagnostics": reset_diagnostics,
              "backprojection_sha256": sha256_file(args.backprojection),
              "note": "Approximate training positions from existing camera fit; uncertain extrinsics and joint mapping remain. No calibration changes. Checkpoint predicts all actions after reset. These diagnostic trials do not join the primary benchmark.",
              "results": {}}
    for condition, scenarios in (("random_ball", None), ("familiar_ball", balls), ("familiar_ball_and_pose", poses)):
        result = run_trials(robot="so101", checkpoint=args.checkpoint, config=sim, eval_config=ev,
                            n_trials=len(sources), seed_start=0, out=args.out / condition,
                            video=False, controller="learned", device=args.device,
                            experiment_id=f"DIAGNOSTIC_{condition}", reset_scenarios=scenarios)
        report["results"][condition] = result["summary"]
        (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
