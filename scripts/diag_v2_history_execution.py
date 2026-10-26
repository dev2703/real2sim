"""Compare queue2, queue1 and aligned temporal ensembling on one frozen ACT.

Development diagnostic only: no reserved examples, retraining, reset replacement,
physics tuning or task-success gate changes. Uses the checkpoint's original
provisional geometry, separately from the measured-radius waypoint baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from real2sim.config import REPO_ROOT, load_yaml, project_path
from real2sim.evaluation.run import run_trials
from real2sim.policy.adapters import ChunkExecutor


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=json_default, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(checkpoint, scene, out, *, video=True, torch_threads=2):
    checkpoint, scene, out = map(Path, (checkpoint, scene, out))
    if out.exists():
        raise FileExistsError(out)
    required = [checkpoint / name for name in ("model.safetensors", "config.json", "stats.json",
                                               "real2sim_contract.json", "run_info.json")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(torch_threads)
    out.mkdir(parents=True)
    cfg = load_yaml(scene)
    task = project_path(cfg["task_config"])
    data = project_path(cfg["data_config"])
    sources = [Path(__file__), scene, task, data, *required,
               project_path(cfg["model"]), project_path(cfg["manifest"]),
               REPO_ROOT / "src/real2sim/config.py",
               REPO_ROOT / "src/real2sim/evaluation/run.py",
               REPO_ROOT / "src/real2sim/evaluation/metrics.py",
               REPO_ROOT / "src/real2sim/policy/act.py",
               REPO_ROOT / "src/real2sim/policy/adapters.py",
               REPO_ROOT / "src/real2sim/policy/temporal_ensemble.py",
               REPO_ROOT / "src/real2sim/policy/observation_variants.py",
               REPO_ROOT / "src/real2sim/sim/env.py",
               REPO_ROOT / "src/real2sim/sim/model.py",
               REPO_ROOT / "src/real2sim/sim/cameras.py",
               REPO_ROOT / "src/real2sim/sim/contact_solver.py",
               REPO_ROOT / "src/real2sim/data/transforms.py",
               REPO_ROOT / "src/real2sim/geometry/so101_fk.py",
               REPO_ROOT / "src/real2sim/geometry/so3.py",
               REPO_ROOT / "src/real2sim/sim/kinematics.py"]
    hashes = {str(path.resolve()): digest(path) for path in sources}
    for name, path in (("sim.yaml", scene), ("task.yaml", task), ("data.yaml", data),
                       ("executed_script.py", Path(__file__))):
        (out / name).write_bytes(path.read_bytes())
    info = json.loads((checkpoint / "run_info.json").read_text())
    report = {
        "schema": "history4_execution_20261026_v1", "status": "running",
        "controller_source": "LEARNED_ACT", "scope": "One common development seed0, full30s; not reserved evaluation",
        "checkpoint": str(checkpoint), "checkpoint_step": info["step"],
        "scene": str(scene), "task_sha256": digest(task), "physical_calibration_accepted": False,
        "new_measured_radius_scene": False,
        "reset_protocol": "Original configured native joints, common seeded objects, no settling/replacement",
        "changes_to_physics_scoring_limits": False, "torch_threads": torch_threads,
        "source_sha256_before": hashes, "modes": {},
    }
    write_json(out / "report.json", report)
    original_step = ChunkExecutor.step
    modes = {"queue2": {"n_action_steps": 2}, "queue1": {"n_action_steps": 1},
             "ensemble_0p01": {"n_action_steps": 1, "temporal_ensemble_coeff": .01}}
    for mode, inference in modes.items():
        directory = out / mode
        directory.mkdir()
        evaluation = load_yaml("configs/v2/eval_so101.yaml")
        evaluation["task_config"] = str(task)
        evaluation["inference"].update(inference)
        eval_path = directory / "eval.yaml"
        eval_path.write_text(yaml.safe_dump(evaluation, sort_keys=False))
        trace_path = directory / "ticks.jsonl"
        initial = {}
        started = time.perf_counter()
        tick = 0
        max_penetration = 0.
        with trace_path.open("w") as handle:
            def traced_step(controller, obs, env, initial=initial, started=started):
                nonlocal tick, max_penetration
                before = env.sim_state()
                if tick == 0:
                    initial.update(native_readback=before.tolist(), ee_pose=env.ee_pose().tolist(),
                        image_sha256={key: hashlib.sha256(obs[key].tobytes()).hexdigest()
                                      for key in env.camera_keys})
                step_started = time.perf_counter()
                result = original_step(controller, obs, env)
                elapsed = (time.perf_counter() - step_started) * 1000
                tick += 1
                pm = result.info["privileged_metrics"]
                penetration = float(pm["robot_table_penetration_m"])
                max_penetration = max(max_penetration, penetration)
                record = {
                    "tick": tick, "sim_time_s": tick / env.control_hz,
                    "host_elapsed_s": time.perf_counter() - started,
                    "full_controller_step_ms": elapsed,
                    "detail": result.detail, "control": result.info["control"],
                    "readback_before_native_radians": before,
                    "readback_after_native_radians": env.sim_state(),
                    "achieved_tool_pose": env.ee_pose(), "ik_rejected": result.ik_rejected,
                    "scoring_only": {key: pm[key] for key in (
                        "ball_xyz_task", "bowl_xyz_task", "ee_ball_distance_m", "ball_held",
                        "lifted", "ball_in_bowl", "ball_resting", "robot_table_penetration_m")},
                }
                handle.write(json.dumps(record, default=json_default, allow_nan=False) + "\n")
                return result

            with patch.object(ChunkExecutor, "step", traced_step):
                summary = run_trials(robot="so101", checkpoint=checkpoint, config=scene,
                    eval_config=eval_path, n_trials=1, seed_start=0, out=directory,
                    video=video, controller="learned", device="cpu",
                    experiment_id=f"DEVELOPMENT_HISTORY4_{mode.upper()}")
        report["modes"][mode] = {
            "summary": summary["summary"], "trial": summary["trials"][0],
            "execution_interface": summary["execution_interface"], "initial": initial,
            "trace": str(trace_path), "trace_sha256": digest(trace_path),
            "summary_path": str(directory / "summary.json"),
            "max_robot_table_penetration_m": max_penetration,
        }
        write_json(out / "report.json", report)
    first = report["modes"]["queue2"]["initial"]
    report["identical_initial_readbacks_poses_and_images"] = all(
        entry["initial"] == first for entry in report["modes"].values())
    report["source_sha256_after"] = {path: digest(path) for path in hashes}
    report["runtime_source_changes"] = [path for path in hashes
        if hashes[path] != report["source_sha256_after"][path]]
    report["status"] = "complete"
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    run(args.checkpoint, args.scene, args.out, video=not args.no_video,
        torch_threads=args.torch_threads)


if __name__ == "__main__":
    main()
