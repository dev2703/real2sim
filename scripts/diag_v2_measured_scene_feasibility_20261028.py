"""Oracle-initialized geometric feasibility, never learned ACT task success.

The ordinary RGB frontend is scored at the unchanged configured start. One
initial simulator object-pose read then supplies geometric teacher goals; every
native IK/path gate, actuator, contact setting and success rule stays unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from real2sim.artifacts import sha256_file
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.policy.visual_waypoint import VisualWaypointController
from real2sim.serialization import json_default, write_json


class OracleInitialWaypoint(VisualWaypointController):
    controller_source = "ORACLE_INITIAL_OBJECT_GEOMETRIC_TEACHER"

    def reset(self):
        super().reset()
        self.rgb_acquisition = None
        self.oracle_initial_goals = None

    def _acquire(self):
        # Score the real frontend first. The oracle goals below are disclosed
        # and never counted as RGB acquisition or learned-policy success.
        acquired = super()._acquire()
        detected = self.perception_log[-1]["targets"]
        self.rgb_acquisition = {
            "both": acquired,
            "ball": detected["ball"] is not None,
            "bowl": detected["bowl"] is not None,
            "targets": detected,
        }
        metrics = self.env.info()["privileged_metrics"]
        self.ball = np.asarray(metrics["ball_xyz_task"], float)
        self.bowl = np.asarray(metrics["bowl_xyz_task"], float) + [
            0.0,
            0.0,
            self.env.bowl["height"],
        ]
        self.oracle_initial_goals = {
            "ball_xyz_task": self.ball.copy(),
            "bowl_rim_xyz_task": self.bowl.copy(),
            "supplied_once_at_initial_acquisition": True,
        }
        return True


def run(robot, config, out, seeds):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    if robot == "so101":
        from real2sim.sim.env import So101PickPlaceEnv as Environment
    else:
        from real2sim.panda.env import PandaPickPlaceEnv as Environment
    source_paths = [
        Path(__file__),
        config,
        Path("src/real2sim/policy/visual_waypoint.py"),
        Path("src/real2sim/evaluation/metrics.py"),
        Path("physical_measurements.md"),
    ]
    hashes = {str(path): sha256_file(path) for path in source_paths}
    report = {
        "schema": "measured_scene_oracle_feasibility_20261028_v1",
        "status": "running",
        "robot": robot,
        "config": str(config),
        "seeds": seeds,
        "learned_ACT_evaluated": False,
        "oracle_initial_object_pose_used": True,
        "lighting_changed": False,
        "physical_calibration_accepted": False,
        "source_sha256_before": hashes,
        "trials": [],
        "purpose": "Separate initial RGB failure from native geometric reach, contact retention and released resting placement in the measured-scalar scene.",
    }
    rows = []
    with Environment(config, render_images=False) as env:
        task_path = Path(env.config["task_config"])
        hashes[str(task_path)] = sha256_file(task_path)
        report["task"] = env.task
        report["sim"] = env.config
        report["resolved_lighting"] = {
            "headlight_ambient": env.model.vis.headlight.ambient.copy(),
            "headlight_diffuse": env.model.vis.headlight.diffuse.copy(),
            "headlight_specular": env.model.vis.headlight.specular.copy(),
            "light_ambient": env.model.light_ambient.copy(),
            "light_diffuse": env.model.light_diffuse.copy(),
            "light_specular": env.model.light_specular.copy(),
        }
        write_json(out / "report.json", report)
        controller = OracleInitialWaypoint(env, robot=robot)
        for seed in seeds:
            trial = out / f"seed_{seed:03d}"
            trial.mkdir()
            controller.reset()
            tracker = TrialTracker(
                success_hold_ticks=math.ceil(
                    env.task["episode"]["success_hold_s"] * env.control_hz
                ),
                approach_distance_m=0.03,
                table_top_z=env.table_top,
                ball_radius_m=env.task["ball"]["radius"],
            )
            initial = None
            abort = None
            native_states, commands, distances = [], [], []
            started = time.perf_counter()
            try:
                observation, info = env.reset(seed=seed)
                initial = info["privileged_metrics"]
                if not cv2.imwrite(str(trial / "initial.png"), env.render("iphone")[:, :, ::-1]):
                    raise OSError("Could not save initial RGB")
                with (trial / "ticks.jsonl").open("w") as stream:
                    for tick in range(env.max_episode_steps):
                        state = env.sim_state().copy()
                        result = controller.step(observation, env)
                        observation = result.obs
                        metrics = result.info["privileged_metrics"]
                        done = tracker.update(
                            metrics, saturation=result.saturation, ik_rejected=result.ik_rejected
                        )
                        native_states.append(state)
                        commands.append(result.detail["requested_native"])
                        distances.append(metrics["ee_ball_distance_m"])
                        stream.write(
                            json.dumps(
                                {
                                    "tick": tick,
                                    "phase": controller.phase,
                                    "source": controller.controller_source,
                                    "readback_before": state,
                                    "control": result.info["control"],
                                    "detail": result.detail,
                                    "oracle_initial_goals": controller.oracle_initial_goals,
                                    "scoring_only": metrics,
                                },
                                default=json_default,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        if done or result.truncated:
                            break
            except (RuntimeError, ValueError, OSError) as exc:
                abort = f"{type(exc).__name__}: {exc}"
            row = tracker.row(
                experiment_id="MEASURED_SCENE_ORACLE_FEASIBILITY_20261028",
                controller_source=controller.controller_source,
                robot=robot,
                seed=seed,
                checkpoint_sha256=None,
                checkpoint_step=None,
                mean_latency_ms=None,
                video_path="",
                abort=abort,
            )
            rows.append(row)
            for plan in controller.plan_log:
                plan["uses_object_ground_truth"] = True
                plan["object_goal_source"] = (
                    "Initial oracle object centres; no later object feedback"
                )
            np.savez(
                trial / "native_trajectory.npz",
                readback_before=np.asarray(native_states),
                requested_native=np.asarray(commands),
                tool_ball_distance_m=np.asarray(distances),
            )
            report["trials"].append(
                {
                    **row,
                    "held_ever": tracker.held_ever,
                    "initial_scoring_only": initial,
                    "rgb_acquisition": controller.rgb_acquisition,
                    "oracle_initial_goals": controller.oracle_initial_goals,
                    "final_phase": controller.phase,
                    "controller_failure": controller.failure,
                    "plan_log": controller.plan_log,
                    "phase_log": controller.phase_log,
                    "ik_calls": controller.ik_calls,
                    "safety_checks": controller.safety_checks,
                    "wall_s": time.perf_counter() - started,
                    "trace": str(trial / "ticks.jsonl"),
                }
            )
            report["summary"] = summarize(rows)
            write_json(out / "report.json", report)
            print(
                json.dumps(
                    {
                        "robot": robot,
                        "seed": seed,
                        "rgb": controller.rgb_acquisition,
                        "success": row["pick_place_success"],
                        "lifted": row["lifted"],
                        "phase": row["pick_place_phase_failed"],
                        "failure": controller.failure,
                        "abort": abort,
                    },
                    default=json_default,
                ),
                flush=True,
            )
    report["status"] = "complete"
    report["source_sha256_after"] = {path: sha256_file(path) for path in hashes}
    report["runtime_source_changes"] = [
        path for path in hashes if hashes[path] != report["source_sha256_after"][path]
    ]
    report["rgb_frontend_summary"] = {
        key: sum(
            bool(row["rgb_acquisition"] and row["rgb_acquisition"][key]) for row in report["trials"]
        )
        for key in ("ball", "bowl", "both")
    }
    write_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("so101", "panda"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = args.config or Path(
        f"configs/v2/measured_registration_20261028/sim_{args.robot}_positive_all_scalars.yaml"
    )
    run(args.robot, config, args.out, args.seeds)
