"""Evaluate the RGB waypoint controller with the existing pick/place scoring.

This is an image-guided geometric controller, not a learned ACT checkpoint or a
claim of frozen-policy transfer. Simulator object state is read only for scoring.
The configured initial pose and full episode horizon are retained by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from real2sim.artifacts import sha256_file as digest
from real2sim.config import REPO_ROOT, load_yaml
from real2sim.evaluation.metrics import TRIAL_COLUMNS, TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo, policy_view_to_bgr
from real2sim.serialization import json_default
from real2sim.serialization import write_json as _write_json
from real2sim.sim.contact_solver import contact_solver_settings

SOURCE = "RGB_WAYPOINT_CONTROLLER"
DEFAULT_CONFIGS = {
    "so101": Path("configs/v2/sim_so101.waypoint_20261026.yaml"),
    "panda": Path("configs/v2/sim_panda.waypoint_20261026.yaml"),
}


def write_json(path, value):
    _write_json(path, value, atomic=True)


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def loaded_project_sources():
    """Hash loaded package modules, rather than unrelated experiments/checkpoints."""
    paths = {Path(__file__).resolve()}
    for name, module in list(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if name.startswith("real2sim.") and source:
            path = Path(source).resolve()
            if path.suffix == ".py" and path.is_relative_to(REPO_ROOT):
                paths.add(path)
    return paths


def save_picture(path, obs, key):
    bgr = policy_view_to_bgr(obs[key], 1)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"could not write image: {path}")


def table_penetration(metrics):
    """Use the same measured robot/table quantity across the two env schemas."""
    if "robot_table_penetration_m" in metrics:
        return float(metrics["robot_table_penetration_m"])
    from real2sim.panda.retarget import classify_contacts

    return float(classify_contacts(metrics["contacts"])["robot_table_penetration_m"])


def run(*, robot, config, seeds, out, video=False, settle_start=False):
    # Lazy import leaves --help usable before opening a renderer or loading a model.
    from real2sim.policy.visual_waypoint import VisualWaypointController

    if settle_start and robot != "so101":
        raise ValueError("--settle-start is supported only for SO-101")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("supply at least one unique trial seed")
    config = project_path(config)
    cfg = load_yaml(config)
    if cfg.get("robot", "so101") != robot:
        raise ValueError(f"--config robot differs from --robot {robot}")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    if robot == "so101":
        from real2sim.sim.env import So101PickPlaceEnv

        env_class = So101PickPlaceEnv
    else:
        from real2sim.panda.env import PandaPickPlaceEnv

        env_class = PandaPickPlaceEnv
    if settle_start:
        from real2sim.evaluation.oracle import settle_start_pose

    configuration_paths = {"sim": config, "task": project_path(cfg["task_config"]),
                           "data": project_path(cfg["data_config"])}
    sources = loaded_project_sources() | set(configuration_paths.values())
    sources |= {project_path(cfg[key]) for key in ("model", "manifest") if key in cfg}
    hashes = {str(path): digest(path) for path in sorted(sources)}
    for name, path in configuration_paths.items():
        (out / f"{name}.yaml").write_bytes(path.read_bytes())
    (out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    report = {
        "schema": "rgb_waypoint_evaluation_v1", "status": "running",
        "controller_source": SOURCE, "learned_policy": False, "robot": robot,
        "physical_calibration_accepted": False, "frozen_act_transfer_demonstrated": False,
        "config": str(config), "seeds": seeds, "settle_start_requested": settle_start,
        "reset_protocol": ("explicit SO-101 settled start, then reset with settled joints"
                           if settle_start else "configured initial pose; no replacement or clipping"),
        "scoring": "Unchanged TrialTracker held lift, release over bowl and resting placement; full configured horizon.",
        "limitations": "RGB geometric goals in the configured simulator. Camera projection, object dimensions and static collision geometry (including bowl pose) are supplied scene parameters, not new physical calibration. Carry orientation is flexible for the five-DOF SO-101; ball-waypoint error must satisfy the native position tolerance. This run does not train or evaluate ACT weights.",
        "source_sha256_before": hashes, "trials": [],
    }
    write_json(out / "report.json", report)
    rows = []
    try:
        with closing(env_class(config)) as env:
            ctrl = VisualWaypointController(env, robot=robot)
            report["resolved"] = {
                "sim": env.config, "task": env.task, "data": env.data_config,
                "control_hz": env.control_hz, "max_ticks": env.max_episode_steps,
                "contact_solver": contact_solver_settings(env.model),
                "native_joint_limits": env.limits,
                "approach_distance_m": .03,
            }
            # Include modules imported during controller/environment construction.
            for path in sorted(loaded_project_sources()):
                hashes.setdefault(str(path), digest(path))
            write_json(out / "report.json", report)
            for seed in seeds:
                directory = out / f"seed_{seed:03d}"
                directory.mkdir()
                tracker = TrialTracker(
                    success_hold_ticks=math.ceil(env.task["episode"]["success_hold_s"] * env.control_hz),
                    approach_distance_m=.03, table_top_z=env.table_top,
                    ball_radius_m=float(env.task["ball"]["radius"]))
                writer, obs, abort, initial, settling = None, None, None, None, None
                max_table, max_support = 0., 0.
                first_event = {"held": None, "held_lift": None, "release": None}
                durations = []
                started = time.perf_counter()
                ctrl.reset()
                try:
                    if settle_start:
                        start, settling = settle_start_pose(env, seed)
                        obs, info = env.reset(seed=seed, scenario={"joint_positions": start})
                    else:
                        obs, info = env.reset(seed=seed)
                    initial = info["privileged_metrics"]
                    initial_readback = env.sim_state().copy()
                    key = ("observation.images.iphone" if "observation.images.iphone" in obs
                           else env.camera_keys[0])
                    save_picture(directory / "initial.png", obs, key)
                    if video:
                        height, width = obs[key].shape[-2:]
                        writer = CaptionedVideo(str(directory / "rollout.mp4"), env.control_hz,
                                                (height * 2, width * 2), "mp4v")
                        writer.write(policy_view_to_bgr(obs[key], 2),
                                     f"RGB waypoint ({robot}; geometric) seed={seed} tick=0")
                    with (directory / "ticks.jsonl").open("w") as trace:
                        for tick in range(1, env.max_episode_steps + 1):
                            before = env.sim_state().copy()
                            t0 = time.perf_counter()
                            result = ctrl.step(obs, env)
                            durations.append((time.perf_counter() - t0) * 1000)
                            obs = result.obs
                            pm = result.info["privileged_metrics"]
                            released_before = tracker.released
                            done = tracker.update(pm, saturation=result.saturation,
                                                  ik_rejected=result.ik_rejected)
                            for name, event in (("held", bool(pm["ball_held"])),
                                                ("held_lift", bool(pm["ball_held"] and pm["lifted"])),
                                                ("release", tracker.released and not released_before)):
                                if event and first_event[name] is None:
                                    first_event[name] = {"tick": tick, "time_s": tick / env.control_hz}
                            penetration = table_penetration(pm)
                            max_table = max(max_table, penetration)
                            max_support = max(max_support, float(pm.get("robot_support_penetration_m", 0)))
                            record = {
                                "tick": tick, "time_s": tick / env.control_hz,
                                "phase": ctrl.phase, "detail": result.detail,
                                "control": result.info["control"],
                                "readback_before": before, "readback_after": env.sim_state(),
                                "readback_units": ("six native joint radians" if robot == "so101"
                                                   else "nine readbacks: seven native arm radians plus two individual finger slide positions in metres"),
                                "ik_rejected": result.ik_rejected, "saturation": result.saturation,
                                "control_and_render_ms": durations[-1],
                                "scoring_only": {name: pm[name] for name in (
                                    "ball_xyz_task", "tool_xyz_task", "ee_ball_distance_m",
                                    "ball_held", "lifted", "ball_in_bowl", "ball_resting")},
                                "robot_table_penetration_m": penetration,
                                "robot_support_penetration_m": pm.get("robot_support_penetration_m", 0),
                            }
                            trace.write(json.dumps(record, default=json_default, allow_nan=False) + "\n")
                            if writer:
                                writer.write(policy_view_to_bgr(obs[key], 2),
                                             f"RGB waypoint ({robot}; geometric) seed={seed} tick={tick} {ctrl.phase}")
                            if tick % 125 == 0:
                                print(f"seed={seed} tick={tick}/{env.max_episode_steps} phase={ctrl.phase} "
                                      f"held={tracker.held_ever} lifted={tracker.lifted}", flush=True)
                            if done or result.truncated:
                                break
                except (RuntimeError, ValueError, OSError) as exc:
                    abort = f"{type(exc).__name__}: {exc}"
                finally:
                    if writer:
                        writer.close()
                if obs is not None:
                    save_picture(directory / "final.png", obs, key)
                row = tracker.row(
                    experiment_id=f"RGB_WAYPOINT_{robot.upper()}", controller_source=SOURCE,
                    robot=robot, seed=seed, checkpoint_sha256=None, checkpoint_step=None,
                    mean_latency_ms=float(np.mean(durations)) if durations else None,
                    video_path=str(directory / "rollout.mp4") if video else "", abort=abort)
                rows.append(row)
                report["trials"].append({
                    **row, "held_ever": tracker.held_ever,
                    "first_scored_events": first_event, "initial_scoring_only": initial,
                    "initial_native_readback": initial_readback if initial is not None else None,
                    "final_phase": ctrl.phase, "controller_failure": ctrl.failure,
                    "phase_log": ctrl.phase_log, "plan_log": ctrl.plan_log,
                    "perception_log": ctrl.perception_log, "ik_calls": ctrl.ik_calls,
                    "safety_checks": ctrl.safety_checks, "start_settling": settling,
                    "maximum_robot_table_penetration_m": max_table,
                    "maximum_robot_support_penetration_m": max_support,
                    "wall_s": time.perf_counter() - started,
                    "latency_includes_rendering_and_planning": True,
                    "trace": str(directory / "ticks.jsonl"),
                })
                report["summary"] = summarize(rows)
                write_json(out / "report.json", report)
                print(json.dumps({name: row[name] for name in (
                    "seed", "pick_place_success", "lifted", "phase_failed", "abort")}), flush=True)
            report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["setup_or_runner_failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["source_sha256_after"] = {path: digest(path) for path in hashes}
        report["runtime_source_changes"] = [path for path, before in hashes.items()
                                              if report["source_sha256_after"][path] != before]
        late = loaded_project_sources() - {Path(path) for path in hashes}
        report["late_import_source_sha256"] = {str(path): digest(path) for path in sorted(late)}
        with (out / "trials.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRIAL_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=tuple(DEFAULT_CONFIGS), default="so101")
    parser.add_argument("--config", type=Path,
                        help="scene YAML; defaults to existing valid SO-101 recorded reset or Panda scene")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--settle-start", action="store_true",
                        help="explicitly use the SO-101 oracle's settled-start procedure")
    args = parser.parse_args()
    run(robot=args.robot, config=args.config or DEFAULT_CONFIGS[args.robot],
        seeds=args.seeds, out=args.out, video=args.video, settle_start=args.settle_start)


if __name__ == "__main__":
    main()
