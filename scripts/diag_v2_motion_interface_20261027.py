"""Reconstruct successful demonstrations through an explicitly lossy SO action interface.

This is a teacher-command replay diagnostic, not ACT inference or learned-policy
success. The source commands, timing, reset, physics and existing guards are fixed.
Source joint targets are used only to derive the declared XYZ/orientation/gripper
reference. Cartesian IK is seeded only from the live robot state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from contextlib import closing
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from real2sim.config import load_yaml, project_path
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo
from real2sim.policy.follower_joint import check_joint_request
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics

if __package__:
    from scripts import run_v2_semantic_act_20261027 as semantic
else:
    import run_v2_semantic_act_20261027 as semantic

SOURCE = Path("outputs/v2/matched_scene_act_20261026/data")
SCENE = semantic.SO_SCENE
MODES = ("native", "xyz_binary", "orientation_binary", "xyz_continuous", "orientation_continuous")
SEEDS = (100, 101, 102, 103, 104)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=serial, allow_nan=False) + "\n")


def command_reference(env, fk, native, raw_native):
    """Return explicit semantic information; native arm targets never seed IK."""
    pose = fk.forward(native)
    return {"point": semantic.point(pose, semantic.SO_OFFSET), "rotation": pose[:3, :3].copy(),
            "gripper_native": float(native[5]),
            "opened": bool(env.mapping.gripper_sim_to_real_percent(raw_native[5]) >= 45.)}


class ContinuousPointAdapter(semantic.PointAdapter):
    """Current PointAdapter numerical solve with explicitly retained hinge command."""

    def apply_continuous(self, desired_point, gripper_native):
        env = self.env
        q = env.sim_state()
        current = semantic.point(env.ee_pose(), self.offset)
        delta = desired_point - current
        requested = current + delta / max(1., np.linalg.norm(delta) / .024)
        grip = float(gripper_native)
        if not np.isfinite(grip) or not env.limits[5, 0] <= grip <= env.limits[5, 1]:
            raise ValueError("source continuous gripper command exceeds existing limits")
        reference = (semantic.grasp_orientation(self.objects[:2], 15.)
                     if self.orientation_reference is None else self.orientation_reference)

        def residual(arm):
            pose = self.ik.forward(np.r_[arm, grip])
            rotation = Rotation.from_matrix(reference @ pose[:3, :3].T).as_rotvec()
            return np.r_[semantic.point(pose, self.offset) - requested,
                         .002 * rotation, .0001 * (arm - q[:5])]

        lower = env.limits[:5, 0] + self.joint_margin
        upper = env.limits[:5, 1] - self.joint_margin
        fitted = least_squares(residual, np.clip(q[:5], lower, upper), bounds=(lower, upper),
                               max_nfev=35, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        target = np.r_[fitted.x, grip]
        error = np.linalg.norm(semantic.point(self.ik.forward(target), self.offset) - requested)
        self.ik.data.qpos[:] = env.data.qpos
        accepted, reason, _ = check_joint_request(self.ik, q, env.data.ctrl[env.actuator_ids],
                                                 target, env.max_velocity / env.control_hz, samples=17)
        if error > .003:
            accepted, reason = False, "point_residual"
        obs, _, _, truncated, info = env.step(target if accepted else q)
        return obs, truncated, info, accepted, reason, requested


def distribution(values):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())} if len(values) else None


def run_trial(env, arrays, seed, mode, directory, *, video):
    ids = np.flatnonzero(arrays["episode_index"] == seed)
    if not len(ids):
        raise ValueError(f"source episode {seed} not present")
    env.reset(seed=seed)
    # Camera object estimates only select the existing fixed SO jaw orientation.
    objects = semantic.acquire(env)
    if objects is None:
        raise RuntimeError("initial RGB ball/bowl detection unavailable")
    source_fk = NativeKinematics(env.model, env.config)
    guard = NativeKinematics(env.model, env.config)
    cls = ContinuousPointAdapter if mode.endswith("continuous") else semantic.PointAdapter
    adapter = cls(env, "so101", objects, joint_margin=.01)
    tracker = TrialTracker(success_hold_ticks=math.ceil(env.control_hz), approach_distance_m=.03,
                           table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
    writer = CaptionedVideo(directory / "rollout.mp4", env.control_hz, (360, 640)) if video else None
    reasons, errors = {}, {k: [] for k in ("grasp_tracking_m", "orientation_tracking_rad",
                                          "gripper_tracking_rad", "source_readback_rad")}
    abort, max_penetration = None, 0.
    with (directory / "ticks.jsonl").open("w") as handle:
        try:
            for tick in range(env.max_episode_steps):
                # Keep failed replays in the full original horizon, holding the final
                # source request after the successful source recording terminates.
                index = ids[min(tick, len(ids) - 1)]
                native = sim_from_real(env.mapping, arrays["action"][index])
                reference = command_reference(env, source_fk, native,
                                              arrays["requested_native_command_rad"][index])
                before = env.sim_state()
                if mode == "native":
                    guard.data.qpos[:] = env.data.qpos
                    accepted, reason, _ = check_joint_request(guard, before,
                        env.data.ctrl[env.actuator_ids], native,
                        env.max_velocity / env.control_hz, samples=17)
                    _, _, _, truncated, info = env.step(native if accepted else before)
                    requested_point = reference["point"]
                else:
                    if mode.startswith("orientation"):
                        adapter.orientation_reference = reference["rotation"]
                    if mode.endswith("continuous"):
                        _, truncated, info, accepted, reason, requested_point = adapter.apply_continuous(
                            reference["point"], reference["gripper_native"])
                    else:
                        _, truncated, info, accepted, reason, requested_point = adapter.apply(
                            reference["point"], reference["opened"])
                achieved = env.ee_pose()
                error = {"grasp_tracking_m": float(np.linalg.norm(
                    semantic.point(achieved, semantic.SO_OFFSET) - reference["point"])),
                    "orientation_tracking_rad": float(Rotation.from_matrix(
                        reference["rotation"] @ achieved[:3, :3].T).magnitude()),
                    "gripper_tracking_rad": float(abs(env.sim_state()[5] - native[5])),
                    "source_readback_rad": float(np.linalg.norm(env.sim_state() -
                        arrays["achieved_successor_native_rad"][index]))}
                for key, value in error.items():
                    errors[key].append(value)
                reasons[reason] = reasons.get(reason, 0) + 1
                pm = info["privileged_metrics"]
                max_penetration = max(max_penetration, pm["robot_table_penetration_m"])
                done = tracker.update(pm, ik_rejected=not accepted,
                    saturation=float(np.mean(info["control"]["limit_clipped"])))
                handle.write(json.dumps({"tick": tick, "source_index": int(index),
                    "source_recording_held_at_end": tick >= len(ids),
                    "source_accepted_native": native, "semantic_reference": reference,
                    "requested_grasp_point": requested_point, "accepted": accepted, "reason": reason,
                    "readback_before_native": before, "readback_after_native": env.sim_state(),
                    "tracking_error": error, "control": info["control"], "scoring_only": pm},
                    default=serial, allow_nan=False) + "\n")
                if writer:
                    writer.write(env.render("iphone")[:, :, ::-1].copy(),
                                 f"TEACHER REPLAY DIAGNOSTIC: {mode}; no ACT inference")
                if done or truncated:
                    break
        except (RuntimeError, ValueError) as exc:
            abort = f"{type(exc).__name__}: {exc}"
        finally:
            if writer:
                writer.close()
    row = tracker.row(experiment_id="MOTION_INTERFACE_20261027", controller_source="TEACHER_REPLAY",
                      robot="so101", seed=seed, checkpoint_sha256=None, checkpoint_step=None,
                      mean_latency_ms=None, video_path=str(directory / "rollout.mp4") if video else "",
                      abort=abort)
    return {**row, "mode": mode, "source_ticks": len(ids), "guard_reasons": reasons,
            "max_robot_table_penetration_m": max_penetration,
            "tracking_errors": {k: distribution(v) for k, v in errors.items()},
            "trace": str(directory / "ticks.jsonl")}


def run(out, seeds=SEEDS, modes=MODES, video=True):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    if manifest["policy_hz"] != 12.5 or manifest["physical_calibration_accepted"]:
        raise ValueError("unexpected source timing/provenance")
    arrays = dict(np.load(SOURCE / "arrays.npz", allow_pickle=False))
    collection = json.loads((SOURCE / "teacher_trials.json").read_text())
    for seed in seeds:
        rows = [r for r in collection["trials"] if r["seed"] == seed]
        if len(rows) != 1 or not rows[0]["pick_place_success"]:
            raise ValueError(f"source episode {seed} is not a proven successful whole demonstration")
    config = load_yaml(SCENE)
    task = project_path(config["task_config"])
    if digest(task) != manifest["calibration_sha256"]:
        raise ValueError("source scene task differs from recorded teacher configuration")
    dependencies = [Path(__file__), Path(semantic.__file__), SCENE, task,
        Path("src/real2sim/artifacts.py"), Path("src/real2sim/serialization.py"),
        project_path(config["data_config"]), SOURCE / "arrays.npz", SOURCE / "manifest.json",
        SOURCE / "teacher_trials.json", Path("src/real2sim/policy/follower_joint.py"),
        Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/base_env.py"), Path("assets/menagerie/robotstudio_so101/so101.xml")]
    before = {str(p): digest(p) for p in dependencies}
    # Check weights unchanged even though no policy is loaded by this diagnostic.
    frozen_weights = [Path("outputs/v2/semantic_act_transfer_20261027_v2/checkpoint/model.safetensors"),
                      Path("outputs/v2/matched_scene_act_20261026/training/checkpoints/002000/model.safetensors")]
    before_weights = {str(p): digest(p) for p in frozen_weights if p.exists()}
    shutil.copyfile(__file__, out / "executed_runner.py")
    report = {"schema": "motion_interface_20261027_v1", "status": "running", "seeds": list(seeds),
        "modes": list(modes), "source": str(SOURCE), "source_selection": "first five successful train seeds in manifest, declared before execution",
        "ACT_inference": False, "ACT_training": False, "physical_calibration_accepted": False,
        "joint_margin_rad": .01, "solver_seed": "live measured arm only; no source joint seed",
        "position_step_bound_m": .024, "same_weak_orientation_weight_m_per_rad": .002,
        "continuous_gripper": "accepted source servo setpoint, not achieved successor readback",
        "native_replay": "recorded accepted/rate-bounded commands rechecked through unchanged guard",
        "binary_gripper": "raw source intent >=45 percent; map to 90/0 percent endpoints",
        "physics_hz": 500, "control_hz": 12.5, "timeout_s": 30.,
        "unchanged_guard": "native bounds, workspace, 17-sample joint paths, original 2mm penetration gate",
        "source_sha256_before": before, "frozen_weights_sha256_before": before_weights,
        "trials": [], "summaries": {}}
    save(out / "report.json", report)
    with closing(So101PickPlaceEnv(SCENE, render_images=False)) as env:
        for mode in modes:
            directory = out / mode
            directory.mkdir()
            for seed in seeds:
                trial = directory / f"seed_{seed}"
                trial.mkdir()
                row = run_trial(env, arrays, seed, mode, trial, video=video)
                report["trials"].append(row)
                report["summaries"][mode] = summarize([r for r in report["trials"] if r["mode"] == mode])
                save(out / "report.json", report)
                print(json.dumps({"mode": mode, "seed": seed, "placed": row["pick_place_success"],
                    "lifted": row["lifted"], "ticks": row["ticks"], "guard": row["guard_reasons"],
                    "abort": row["abort"]}), flush=True)
    after = {name: digest(name) for name in before}
    after_weights = {name: digest(name) for name in before_weights}
    report.update(status="complete", source_sha256_after=after,
                  runtime_source_changes=[name for name in before if before[name] != after[name]],
                  frozen_weights_sha256_after=after_weights,
                  frozen_weights_unchanged=before_weights == after_weights)
    save(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run(args.out, args.seeds, args.modes, video=not args.no_video)
