"""Test native geometric-teacher -> common Cartesian action -> live native IK.

Teacher decisions use RGB, measured joints and the supplied static collision
map. Dynamic simulator object metrics only score the live result. The shadow
teacher's MjData is synchronized from the live environment before each label;
its phase history persists, so this tests teacher labels on the executed state
distribution rather than replaying a stored successful trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from real2sim.artifacts import sha256_file
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.policy.common_recovery import CommonCartesianExecutor
from real2sim.serialization import json_default, write_json

if __package__:
    from scripts.benchmark_frozen_act import SharedEpisode
    from scripts.evaluate_hf_recovery import lighting
    from scripts.run_recovery_dagger import CartesianConvergenceTeacher, teacher_common_action
else:
    from benchmark_frozen_act import SharedEpisode
    from evaluate_hf_recovery import lighting
    from run_recovery_dagger import CartesianConvergenceTeacher, teacher_common_action


def run(out, robot, seeds):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    live = SharedEpisode(robot, out / "live_setup")
    shadow = SharedEpisode(robot, out / "shadow_setup")
    watched = [Path(__file__), Path("src/real2sim/policy/common_recovery.py"),
        Path("scripts/run_recovery_dagger.py"),
        Path("scripts/benchmark_frozen_act.py"), Path("src/real2sim/policy/visual_waypoint.py"),
        Path("src/real2sim/policy/visual_targets.py"), Path("scripts/evaluate_hf_recovery.py")]
    hashes = {str(p): sha256_file(p) for p in watched}
    report = {"schema": "rgb_teacher_common_cartesian_roundtrip_v1", "status": "running",
        "robot": robot, "seeds": seeds, "no_hf_inference": True,
        "common_action_representation": "TCP translation delta, world/task rotation-vector delta, semantic gripper opening",
        "privileged_object_state_drives_control": False,
        "shadow_teacher_state": "mj_copyData live into independently constructed identical shadow before each label; teacher phase memory persists",
        "lighting_before": lighting(live.env), "source_sha256_before": hashes, "trials": []}
    write_json(out / "report.json", report, atomic=True)
    try:
        executor = CommonCartesianExecutor(live)
        teacher = CartesianConvergenceTeacher(shadow.env, robot=robot)
        for seed in seeds:
            directory = out / f"seed_{seed}"
            directory.mkdir()
            live.reset(seed)
            teacher_obs, _ = shadow.reset(seed)
            teacher.reset()
            teacher.bowl_memory = None
            tracker = TrialTracker(success_hold_ticks=math.ceil(
                live.env.task["episode"]["success_hold_s"] * live.env.control_hz),
                approach_distance_m=.03, table_top_z=live.env.table_top,
                ball_radius_m=live.env.task["ball"]["radius"])
            reasons, errors, abort, first_divergence = {}, [], None, None
            with (directory / "ticks.jsonl").open("w") as stream:
                try:
                    for tick in range(live.env.max_episode_steps):
                        mujoco.mj_copyData(shadow.env.data, shadow.env.model, live.env.data)
                        shadow.env.steps = live.env.steps
                        shadow.env._ready = live.env._ready
                        expert = teacher.step(teacher_obs, shadow.env)
                        native = np.asarray(expert.detail["requested_native"], float)
                        common = teacher_common_action(teacher, expert, executor)
                        result = executor.step(common)
                        achieved = np.asarray(result.detail["requested_native"], float)
                        error = float(np.max(abs(achieved[:teacher.n] - native[:teacher.n])))
                        errors.append(error)
                        reason = result.detail["ik_reason"]
                        reasons[reason] = reasons.get(reason, 0) + 1
                        if first_divergence is None and (result.ik_rejected or error > .03):
                            first_divergence = {"tick": tick, "phase": teacher.phase,
                                "ik_reason": reason, "max_native_joint_error_rad": error}
                        done = tracker.update(result.info["privileged_metrics"],
                            saturation=result.saturation, ik_rejected=result.ik_rejected)
                        stream.write(json.dumps({"tick": tick, "teacher_phase": teacher.phase,
                            "teacher_detail": expert.detail, "common_action": common,
                            "executor_detail": result.detail,
                            "max_native_joint_error_rad": error,
                            "scoring_only": result.info["privileged_metrics"]},
                            default=json_default, allow_nan=False) + "\n")
                        teacher_obs = expert.obs
                        if done or result.truncated:
                            break
                except (RuntimeError, ValueError) as exc:
                    abort = f"{type(exc).__name__}: {exc}"
            row = tracker.row(experiment_id="COMMON_ACTION_TEACHER_ROUNDTRIP",
                controller_source="RGB_TEACHER_COMMON_ACTION_NATIVE_IK", robot=robot, seed=seed,
                checkpoint_sha256=None, checkpoint_step=None, mean_latency_ms=None,
                video_path="", abort=abort)
            row.update(ik_reasons=reasons, teacher_phase=teacher.phase,
                teacher_failure=teacher.failure, first_divergence=first_divergence,
                mean_native_joint_reconstruction_error_rad=float(np.mean(errors)) if errors else None,
                maximum_native_joint_reconstruction_error_rad=max(errors, default=None),
                teacher_perception_log=teacher.perception_log,
                trace=str(directory / "ticks.jsonl"))
            report["trials"].append(row)
            report["summary"] = summarize(report["trials"])
            write_json(out / "report.json", report, atomic=True)
            print(json.dumps({"robot": robot, "seed": seed, "placed": row["pick_place_success"],
                "ik_reasons": reasons, "phase": teacher.phase, "failure": teacher.failure,
                "first_divergence": first_divergence, "abort": abort}), flush=True)
        report.update(status="complete", lighting_after=lighting(live.env),
            source_sha256_after={p: sha256_file(Path(p)) for p in hashes})
        report["runtime_source_changes"] = [p for p in hashes
            if hashes[p] != report["source_sha256_after"][p]]
        write_json(out / "report.json", report, atomic=True)
        return report
    finally:
        shadow.close()
        live.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--robot", choices=("so101", "panda"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[12000, 12001])
    args = parser.parse_args()
    run(args.out, args.robot, args.seeds)
