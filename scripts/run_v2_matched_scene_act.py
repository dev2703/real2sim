"""One bounded ACT experiment on commands/images from a matched simulator teacher.

This generates new simulation data; it never relabels real demonstrations or
claims physical calibration. Pre-action images/readbacks supervise the native
servo setpoint accepted during that control interval. Final learned feedback
rollouts have no teacher goals, corrective IK, phase input or object-state input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
import yaml

from real2sim.config import REPO_ROOT, load_yaml, project_path
from real2sim.data.prepare import compute_stats
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo, policy_view_to_bgr
from real2sim.policy.act import load_checkpoint
from real2sim.policy.adapters import (
    JOINT_STATE_NAMES,
    ApplyResult,
    ChunkExecutor,
    JointActionAdapter,
    ObservationAdapter,
)
from real2sim.policy.follower_joint import check_joint_request
from real2sim.policy.train import train
from real2sim.policy.visual_waypoint import VisualWaypointController
from real2sim.sim.env import So101PickPlaceEnv, real_from_sim, sim_from_real
from real2sim.sim.kinematics import NativeKinematics

TRAIN_SEEDS = tuple(range(100, 120))
VALIDATION_SEEDS = tuple(range(200, 206))
DEVELOPMENT_SEEDS = tuple(range(300, 305))
VERSION = "matched_scene_native_commands_20261026_v1"
SOURCE = "SIMULATED_RGB_WAYPOINT_TEACHER"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=serializable, allow_nan=False) + "\n")
    temporary.replace(path)


def tracker_for(env):
    return TrialTracker(
        success_hold_ticks=math.ceil(env.task["episode"]["success_hold_s"] * env.control_hz),
        approach_distance_m=.03, table_top_z=env.table_top,
        ball_radius_m=float(env.task["ball"]["radius"]))


def outcome(tracker, *, seed, source, checkpoint=None, abort=None):
    return tracker.row(experiment_id=VERSION, controller_source=source, robot="so101",
        seed=seed, checkpoint_sha256=sha256(checkpoint / "model.safetensors") if checkpoint else None,
        checkpoint_step=2000 if checkpoint else None, mean_latency_ms=None,
        video_path="", abort=abort)


def collect(scene: Path, out: Path) -> dict:
    """Collect whole successful teacher trials without changing predefined partitions."""
    data = out / "data"
    data.mkdir()
    report = {"schema": VERSION, "teacher_source": SOURCE,
        "train_seeds_declared_before_collection": list(TRAIN_SEEDS),
        "validation_seeds_declared_before_collection": list(VALIDATION_SEEDS),
        "development_seeds_not_collected": list(DEVELOPMENT_SEEDS),
        "selection": "Whole successful teacher trials only; all failed attempts retained in report.",
        "trials": [], "physical_calibration_accepted": False, "cases": {}}
    episodes, all_images = [], []
    with closing(So101PickPlaceEnv(scene)) as env:
        key = env.camera_keys[0]
        if env.camera_keys != ["observation.images.iphone"]:
            raise ValueError("the declared pilot requires the one-camera iPhone scene")
        for partition, seeds in (("train", TRAIN_SEEDS), ("val", VALIDATION_SEEDS)):
            for seed in seeds:
                states, commands, images, raw_commands, readbacks = [], [], [], [], []
                obs, _ = env.reset(seed=seed)
                teacher = VisualWaypointController(env, robot="so101")
                tracker, abort = tracker_for(env), None
                started = time.perf_counter()
                try:
                    for _tick in range(env.max_episode_steps):
                        # All training inputs exist BEFORE the servo request at tick t.
                        state = env.real_state().copy()
                        image = np.rint(obs[key].transpose(1, 2, 0) * 255).astype(np.uint8)
                        result = teacher.step(obs, env)
                        native = np.asarray(result.info["control"]["applied_radians"])
                        requested = np.asarray(result.info["control"]["requested_radians"])
                        states.append(state)
                        images.append(image)
                        commands.append(real_from_sim(env.mapping, native).astype(np.float32))
                        raw_commands.append(requested)
                        readbacks.append(env.sim_state())
                        done = tracker.update(result.info["privileged_metrics"],
                            saturation=result.saturation, ik_rejected=result.ik_rejected)
                        obs = result.obs
                        if done or result.truncated:
                            break
                except (RuntimeError, ValueError) as exc:
                    abort = f"{type(exc).__name__}: {exc}"
                row = outcome(tracker, seed=seed, source=SOURCE, abort=abort)
                included = bool(row["pick_place_success"])
                trial = {**row, "partition": partition, "included_in_supervision": included,
                    "teacher_failure": teacher.failure, "ik_calls": teacher.ik_calls,
                    "ticks_recorded": len(states), "wall_s": time.perf_counter() - started}
                report["trials"].append(trial)
                if included:
                    start = sum(len(ep["state"]) for ep in episodes)
                    length = len(states)
                    episodes.append({"state": np.asarray(states, np.float32),
                        "action": np.asarray(commands, np.float32),
                        "joint_state": np.asarray(states, np.float32),
                        "joint_action": np.asarray(commands, np.float32),
                        "episode_index": np.full(length, seed, np.int64),
                        "frame_index": np.arange(length, dtype=np.int64),
                        "timestamp": np.arange(length, dtype=np.float64) / env.control_hz,
                        "ep_start": np.full(length, start, np.int64),
                        "ep_len": np.full(length, length, np.int64),
                        "label_is_extrapolated": np.zeros(length, bool),
                        "requested_native_command_rad": np.asarray(raw_commands, np.float32),
                        "achieved_successor_native_rad": np.asarray(readbacks, np.float32)})
                    all_images.append(np.asarray(images, np.uint8))
                write_json(out / "collection.json", report)
                print(json.dumps({"stage": "collect", "seed": seed,
                    "success": included, "failure": teacher.failure, "ticks": len(states)}), flush=True)
        split = {"train": [r["seed"] for r in report["trials"]
                           if r["partition"] == "train" and r["included_in_supervision"]],
                 "val": [r["seed"] for r in report["trials"]
                         if r["partition"] == "val" and r["included_in_supervision"]],
                 "reserved": [], "development_rollout_seeds": list(DEVELOPMENT_SEEDS)}
        if not split["train"] or not split["val"]:
            raise RuntimeError("the fixed trial set supplied no successful teacher train/val episodes")
        arrays = {key: np.concatenate([ep[key] for ep in episodes]) for key in episodes[0]}
        np.savez(data / "arrays.npz", **arrays)
        np.save(data / "images_iphone.npy", np.concatenate(all_images))
        training = np.isin(arrays["episode_index"], split["train"])
        stats = {"observation.state": compute_stats(arrays["state"][training]),
            "action": compute_stats(arrays["action"][training]),
            "image_normalization": {"mode": "imagenet", "mean": [.485, .456, .406],
                                    "std": [.229, .224, .225]},
            "computed_on": "Successful simulated teacher training episodes only; no val/dev statistics",
            "train_episodes": split["train"]}
        write_json(data / "stats.json", stats)
        write_json(data / "teacher_trials.json", report)
        task_path = project_path(env.config["task_config"])
        manifest = {"prepare_version": VERSION, "contract_version": VERSION,
            "mode": "joint", "action_representation": "canonical_joint",
            "source": {"repo_id": "sim_teacher/so101_rgb_waypoint_20261026",
                "revision": sha256(data / "teacher_trials.json"), "root": str(out.resolve()),
                "codebase_version": "real2sim_simulation_teacher_v1", "fps": env.control_hz},
            "policy_hz": env.control_hz, "stride": 1,
            "camera_keys": env.camera_keys,
            "image_files": {"observation.images.iphone": "images_iphone.npy"},
            "image_shape_hwc": [192, 256, 3],
            "letterbox": {"src_w": 640, "src_h": 360, "dst_w": 256, "dst_h": 192,
                          "scale": .4, "resized_w": 256, "resized_h": 144,
                          "pad_left": 0, "pad_top": 24},
            "state_names": JOINT_STATE_NAMES, "action_names": JOINT_STATE_NAMES,
            "label_semantics": "Pre-action measured joints/RGB at simulator tick t -> accepted absolute native servo setpoint during [t,t+1), expressed in degree/percent mapping. This is command supervision, not achieved-next-state supervision.",
            "controller_semantics": "current_pose", "calibration_sha256": sha256(task_path),
            "physical_calibration_accepted": False,
            "timing_basis": "Exact synchronous MuJoCo control clock at12.5Hz; host inference does not advance simulator time",
            "split": split, "n_ticks": len(training), "n_ticks_train": int(training.sum()),
            "n_ticks_val": int((~training).sum()),
            "teacher_provenance": {"source": SOURCE, "successful_trials_only": True,
                                   "failed_trials_retained": str(out / "collection.json")}}
        manifest["files_sha256"] = {name: sha256(data / name)
            for name in ("arrays.npz", "stats.json", "images_iphone.npy", "teacher_trials.json")}
        write_json(data / "manifest.json", manifest)
        report["split"] = split
        report["supervised_ticks"] = len(training)
        report["data_manifest_sha256"] = sha256(data / "manifest.json")
        report["cases"] = {name: summarize([r for r in report["trials"] if r["partition"] == name])
                           for name in ("train", "val")}
        write_json(out / "collection.json", report)
        return report


class CommandPathGuard(JointActionAdapter):
    """Reject unsafe learned commands; never generate object-directed corrections."""

    def __init__(self, env):
        super().__init__(env)
        self.ik = NativeKinematics(env.model, env.config)

    def apply(self, action):
        desired = sim_from_real(self.env.mapping, np.asarray(action, float))
        current = self.env.sim_state()
        start = self.env.data.ctrl[self.env.actuator_ids].copy()
        self.ik.data.qpos[:] = self.env.data.qpos
        accepted, reason, applied = check_joint_request(self.ik, current, start, desired,
            self.env.max_velocity / self.env.control_hz, samples=17)
        obs, _, _, truncated, info = self.env.step(desired if accepted else current)
        return ApplyResult(obs, info, truncated, float(np.mean(info["control"]["limit_clipped"])),
            ik_rejected=not accepted,
            detail={"requested_native_rad": desired.tolist(),
                    "checked_applied_native_rad": applied.tolist(), "guard_reason": reason,
                    "task_corrections": 0, "corrective_ik_calls": 0})


def evaluate(checkpoint, scene, out, seeds):
    """Run the frozen ACT on live simulated feedback, without the teacher."""
    policy, normalizer, contract = load_checkpoint(checkpoint, "cpu")
    if contract["source"]["repo_id"] != "sim_teacher/so101_rgb_waypoint_20261026":
        raise ValueError("this diagnostic requires its explicit simulation-teacher checkpoint")
    before = sha256(checkpoint / "model.safetensors")
    rows, trials = [], []
    out.mkdir()
    with closing(So101PickPlaceEnv(scene)) as env:
        if sha256(project_path(env.config["task_config"])) != contract["calibration_sha256"]:
            raise ValueError("learned execution requires the exact teacher task configuration")
        adapter = ObservationAdapter(contract, normalizer)
        controller = ChunkExecutor(policy, normalizer, adapter, CommandPathGuard(env),
            supervised_prefix_steps=8)
        for seed in seeds:
            directory = out / f"seed_{seed:03d}"
            directory.mkdir()
            tracker, abort, max_penetration = tracker_for(env), None, 0.
            obs, _ = env.reset(seed=seed)
            controller.reset()
            writer = CaptionedVideo(directory / "rollout.mp4", env.control_hz, (384, 512))
            reasons = {}
            with (directory / "ticks.jsonl").open("w") as handle:
                try:
                    for tick in range(env.max_episode_steps):
                        before_q = env.sim_state()
                        result = controller.step(obs, env)
                        obs = result.obs
                        pm = result.info["privileged_metrics"]
                        reason = result.detail["guard_reason"]
                        reasons[reason] = reasons.get(reason, 0) + 1
                        max_penetration = max(max_penetration, pm["robot_table_penetration_m"])
                        record = {"tick": tick, "detail": result.detail,
                            "control": result.info["control"], "readback_before_rad": before_q,
                            "readback_after_rad": env.sim_state(), "scoring_only": pm}
                        handle.write(json.dumps(record, default=serializable, allow_nan=False) + "\n")
                        writer.write(policy_view_to_bgr(obs["observation.images.iphone"], 2),
                            "LEARNED ACT from simulation commands; no teacher/IK corrections")
                        done = tracker.update(pm, saturation=result.saturation,
                                              ik_rejected=result.ik_rejected)
                        if done or result.truncated:
                            break
                except (RuntimeError, ValueError) as exc:
                    abort = f"{type(exc).__name__}: {exc}"
                finally:
                    writer.close()
            row = outcome(tracker, seed=seed, source="LEARNED_ACT_WITH_STATIC_COMMAND_GUARD",
                          checkpoint=checkpoint, abort=abort)
            row["video_path"] = str(directory / "rollout.mp4")
            rows.append(row)
            trials.append({**row, "guard_reasons": reasons,
                "maximum_robot_table_penetration_m": max_penetration,
                "policy_calls": controller.n_policy_calls,
                "mean_policy_call_ms": float(np.mean(controller.latencies_ms)),
                "mean_control_step_ms": float(np.mean(controller.control_step_latencies_ms)),
                "trace": str(directory / "ticks.jsonl"),
                "teacher_goals_or_ik_corrections": False})
            write_json(out / "report.json", {"summary": summarize(rows), "trials": trials})
            print(json.dumps({"stage": "learned_rollout", "seed": seed,
                "success": row["pick_place_success"], "phase_failed": row["phase_failed"],
                "closest_m": row["min_ee_ball_distance_m"], "guard": reasons}), flush=True)
    result = {"schema": VERSION, "checkpoint": str(checkpoint), "checkpoint_sha256": before,
        "checkpoint_unchanged": before == sha256(checkpoint / "model.safetensors"),
        "physical_calibration_accepted": False, "teacher_or_ik_corrections": False,
        "static_collision_map_supplied": True, "summary": summarize(rows), "trials": trials}
    write_json(out / "report.json", result)
    return result


def run(scene, out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    sources = [Path(__file__), scene, project_path(load_yaml(scene)["task_config"]),
        *[REPO_ROOT / "src/real2sim" / name for name in (
            "policy/train.py", "policy/act.py", "policy/dataset.py", "policy/adapters.py",
            "policy/visual_waypoint.py", "policy/visual_targets.py", "sim/env.py", "sim/kinematics.py")]]
    hashes = {str(p.resolve()): sha256(p) for p in sources}
    report = {"schema": VERSION, "status": "collecting", "scene": str(scene),
        "training_jobs": 1, "maximum_training_examples": 32000,
        "physical_calibration_accepted": False, "real_demonstrations_used": False,
        "source_sha256_before": hashes}
    write_json(out / "report.json", report)
    report["collection"] = collect(scene, out)
    cfg = {
        "name": VERSION, "data": str((out / "data").resolve()),
        "model": {"chunk_size": 8, "n_action_steps": 1,
            "vision_backbone": "resnet18", "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
            "freeze_backbone": True, "attention_backend": "sdpa", "dim_model": 128,
            "n_heads": 4, "dim_feedforward": 512, "n_encoder_layers": 2, "n_decoder_layers": 1,
            "use_vae": False, "latent_dim": 16, "n_vae_encoder_layers": 1,
            "dropout": 0., "kl_weight": 0., "optimizer_lr": 3e-4,
            "optimizer_lr_backbone": 1e-5, "optimizer_weight_decay": 0.},
        "scheduler": {"warmup_steps": 50, "start_factor": .001, "min_lr": 1e-6},
        "train": {"seed": 0, "batch_size": 8, "grad_accum": 2, "steps": 2000,
            "schedule_steps": 2000, "grad_clip_norm": 1., "log_every": 100,
            "val_every": 500, "save_every": 500, "val_max_batches": None,
            "num_workers": 0, "device": "cpu", "keep_only_best_and_last": True,
            "exact_resume": True},
        "experiment": {"scope": "Simulation-only matched-scene ACT command imitation; single bounded configuration",
            "training_examples": 32000, "real_to_sim_transfer_claimed": False,
            "frozen_panda_transfer_claimed": False}}
    (out / "plan.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    report["status"] = "training"
    write_json(out / "report.json", report)
    report["training"] = train(cfg, out / "data", out / "training")
    checkpoint = (out / "training/checkpoints/last").resolve()
    report["status"] = "evaluating"
    write_json(out / "report.json", report)
    # The final checkpoint is fixed in advance; do not choose rollout winners.
    report["development"] = evaluate(checkpoint, scene, out / "development", DEVELOPMENT_SEEDS)
    first_train = report["collection"]["split"]["train"][0]
    report["training_seed_feedback_diagnostic"] = evaluate(
        checkpoint, scene, out / "training_seed_rollout", [first_train])
    report["source_sha256_after"] = {p: sha256(p) for p in hashes}
    report["runtime_source_changes"] = [p for p in hashes if hashes[p] != report["source_sha256_after"][p]]
    report["status"] = "complete"
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("configs/v2/sim_so101.waypoint_20261026.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/v2/matched_scene_act_20261026"))
    args = parser.parse_args()
    run(args.scene, args.out)


if __name__ == "__main__":
    main()
