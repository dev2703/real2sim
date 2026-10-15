"""Trace frozen pilot seed zero without changing commands or benchmark settings."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from real2sim.config import load_yaml
from real2sim.evaluation.run import build_learned_controller, make_env


def main():
    root = Path("results/metrics/dinov2_act_pilot_20261015")
    out = Path("results/audit/dinov2_ik_deadband_20261015")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    checkpoint = Path("outputs/act_cartesian_dinov2_pilot_20261015/checkpoints/001250")
    reports = {}
    for robot in ("so101", "panda"):
        env = make_env(robot, root / "configs" / f"sim_{robot}.yaml")
        controller, _, meta = build_learned_controller(
            env, checkpoint, load_yaml(root / "configs" / f"eval_{robot}.yaml"),
            "cpu", robot=robot)
        adapter = controller.action_adapter
        ik = adapter.ik if robot == "so101" else adapter.panda.ik
        original = ik.solve
        rows = []

        def traced(target, seed, _solve=original, _robot=robot, _rows=rows, **kwargs):
            result = _solve(target, seed, **kwargs)
            n = 5 if _robot == "so101" else 7
            _rows.append({
                "accepted": bool(result.accepted), "reason": result.reason,
                "iterations": int(result.iterations),
                "arm_command_delta_rad": float(np.max(np.abs(result.qpos[:n] - seed[:n]))),
                "position_residual_m": float(result.position_error_m),
                "orientation_residual_rad": float(result.orientation_error_rad),
            })
            return result

        ik.solve = traced
        obs, _ = env.reset(seed=0)
        controller.reset()
        baseline = json.loads((root / f"pilot_{robot}" / "summary.json").read_text())
        history = baseline["trials"][0]["history"]
        max_difference = 0.0
        for tick in range(300):
            result = controller.step(obs, env)
            obs = result.obs
            action = np.asarray(result.detail["action"])
            rows[-1].update(tick=tick + 1, action=action.tolist(),
                            translation_m=float(np.linalg.norm(action[:3])),
                            rotation_rad=float(np.linalg.norm(action[3:6])))
            metrics = result.info["privileged_metrics"]
            # Ground truth is used only to verify the unchanged trace, never as policy input.
            for key in ("tool_xyz_task", "ball_xyz_task"):
                max_difference = max(max_difference, float(np.max(np.abs(
                    np.asarray(metrics[key]) - np.asarray(history[tick][key])))))
            if result.truncated:
                break
        report = {
            "checkpoint": meta, "seed": 0, "ticks": len(rows),
            "accepted_unchanged_arm": sum(r["accepted"] and r["arm_command_delta_rad"] < 1e-12
                                          for r in rows),
            "rejections": sum(not r["accepted"] for r in rows),
            "zero_iteration_solves": sum(r["iterations"] == 0 for r in rows),
            "median_translation_m": float(np.median([r["translation_m"] for r in rows])),
            "median_rotation_rad": float(np.median([r["rotation_rad"] for r in rows])),
            "max_baseline_trajectory_difference_m": max_difference,
            "rows": rows,
        }
        (out / f"{robot}.json").write_text(json.dumps(report, indent=2) + "\n")
        reports[robot] = {k: v for k, v in report.items() if k != "rows"}
        env.close()
        print(json.dumps(reports[robot]), flush=True)
    (out / "review.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
