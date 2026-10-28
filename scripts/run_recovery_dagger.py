"""Collect teacher/DAgger labels and evaluate a separate frozen-ACT recovery MLP.

The expert is queried on a shadow environment at the learner's actual state.
Only common Cartesian corrective labels enter training. Runtime learned trials
never call the geometric expert; RGB supplies reference targets and IK executes
the predicted action. Simulator dynamic object state remains scoring only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from real2sim.artifacts import sha256_file
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo
from real2sim.policy.common_recovery import (
    ACTION_NAMES,
    ACTION_SCALE,
    OBSERVATION_NAMES,
    CommonCartesianExecutor,
    FrozenACTFeatures,
    RecoveryObservation,
    clip_common,
)
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.policy.recovery_mlp import RecoveryMLPPolicy
from real2sim.serialization import json_default, write_json

if __package__:
    from scripts.benchmark_frozen_act import SharedEpisode, runtime_hashes
    from scripts.evaluate_hf_act_20261027 import SNAPSHOT
    from scripts.evaluate_hf_recovery import MEASURED_SCENE, StaticBowlRecovery, lighting
else:
    from benchmark_frozen_act import SharedEpisode, runtime_hashes
    from evaluate_hf_act_20261027 import SNAPSHOT
    from evaluate_hf_recovery import MEASURED_SCENE, StaticBowlRecovery, lighting

ARRAY_NAMES = ("observations", "base_actions", "features", "target_actions", "robot_id", "episode_index")


class CartesianConvergenceTeacher(StaticBowlRecovery):
    """Expert phase convergence follows TCP, retaining native planning/gates.

    Panda's common Cartesian action intentionally has no null-space joint
    coordinate. Reaching an equivalent TCP must not stall the expert forever
    while waiting for its original arbitrary seven-joint branch.
    """

    def step(self, obs, env):
        from real2sim.geometry.so3 import delta_action

        # DAgger may execute a learner command instead of the queried expert.
        # Do not count an opposing executed jaw command as expert close/release
        # dwell time. This uses causal command history, never held/contact flags.
        selected = getattr(self, "executed_semantic_gripper", None)
        if selected is not None and ((self.phase == "CLOSE" and selected >= .5)
                                     or (self.phase == "RELEASE" and selected < .5)):
            self.phase_tick = max(0, self.phase_tick - 1)
        if self.target_q is not None and self.phase in (
                "PRE_GRASP", "DESCEND", "LIFT", "TRANSPORT", "LOWER", "RETREAT"):
            opened = self.phase in ("PRE_GRASP", "DESCEND", "RETREAT")
            target = self._forward(self.target_q, self._grip(opened))
            error = delta_action(env.ee_pose(), target)
            if np.linalg.norm(error[:3]) <= .002 and np.linalg.norm(error[3:]) <= .02:
                self.target_q = self._native()[:self.n].copy()
        return super().step(obs, env)


def teacher_common_action(teacher, label_result, executor):
    """Project expert phase goals into the shared action without joint nullspace.

    Panda's redundant joint branch may differ from the expert's. A native joint
    interpolation can spend most of its rate budget moving in that nullspace.
    Use the already collision-checked phase TCP endpoint as a Cartesian expert
    target; the live executor still applies all native IK/path/rate checks.
    This target exists only in training, never in learned runtime inputs.
    """
    from real2sim.geometry.so3 import delta_action

    native = label_result.detail["requested_native"]
    common = executor.native_to_common(native)
    if (teacher.robot == "panda" and teacher.target_q is not None and teacher.phase in (
            "PRE_GRASP", "DESCEND", "LIFT", "TRANSPORT", "LOWER", "RETREAT")):
        opened = teacher.phase in ("PRE_GRASP", "DESCEND", "RETREAT")
        target = teacher._forward(teacher.target_q, teacher._grip(opened))
        common[:6] = delta_action(executor.env.ee_pose(), target)
        common = clip_common(common)
    return common


def initial_references(env, robot):
    """Acquire RGB once; this detector never plans or executes a waypoint."""
    detector = StaticBowlRecovery(env, robot=robot)
    detector.bowl_memory = None
    if not detector._acquire():
        raise RuntimeError("Initial RGB ball/bowl references unavailable")
    return detector.ball.copy(), detector.bowl.copy(), detector.perception_log[-1]


def tracker_for(env):
    return TrialTracker(success_hold_ticks=math.ceil(env.task["episode"]["success_hold_s"] * env.control_hz),
        approach_distance_m=.03, table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])


def source_files(snapshot, source_scene, extra=()):
    return runtime_hashes(snapshot, source_scene, extra_paths=[Path(__file__),
        Path("src/real2sim/policy/common_recovery.py"), Path("src/real2sim/policy/recovery_mlp.py"), *extra])


def save_pack(path, rows):
    if rows:
        arrays = {name: np.stack([row[name] for row in rows]) for name in ARRAY_NAMES}
    else:
        arrays = {"observations": np.empty((0, 99), np.float32),
            "base_actions": np.empty((0, 7), np.float32), "features": np.empty((0, 256), np.float32),
            "target_actions": np.empty((0, 7), np.float32),
            "robot_id": np.empty(0, np.int64), "episode_index": np.empty(0, np.int64)}
    np.savez_compressed(path, **arrays)


class LearnedRecoveryController:
    def __init__(self, episode, head, policy=None):
        self.episode, self.env, self.robot, self.head = episode, episode.env, episode.robot, head
        self.executor = CommonCartesianExecutor(episode)
        self.use_act = bool(head.metadata["use_act"])
        self.encoder = RecoveryObservation(self.robot, self.env.control_hz)
        self.act = FrozenACTFeatures(episode, policy) if self.use_act else None
        if policy is not None:
            policy.policy.eval().requires_grad_(False)
        self.reset()

    def reset(self):
        self.tick, self.ball, self.bowl = 0, None, None
        self.encoder.reset()
        if self.act:
            self.act.reset()

    def step(self, obs, env):
        del obs
        if self.ball is None:
            self.ball, self.bowl, self.rgb_acquisition = initial_references(env, self.robot)
        observed = self.encoder.encode(env, self.ball, self.bowl, self.tick)
        if self.act:
            raw, features = self.act.next()
            try:
                base = self.executor.hf_to_common(raw)
                invalid_prior = None
            except (RuntimeError, ValueError) as exc:
                base = np.r_[np.zeros(6), np.clip(raw[5] / 90., 0., 1.)]
                invalid_prior = str(exc)
        else:
            base, features, invalid_prior = np.zeros(7), np.zeros(256, np.float32), None
        action = self.head.predict(observed, base, features)
        result = self.executor.step(action)
        self.encoder.selected_action(action, self.tick)
        self.tick += 1
        result.detail.update(uses_act=self.use_act, intervention=True,
            no_runtime_geometric_teacher=True, invalid_base_prior=invalid_prior,
            head_sha256=self.head.head_sha256)
        return result

    def close(self):
        if self.act:
            self.act.close()


def collect(robot, out, train_seeds, val_seeds, *, snapshot=SNAPSHOT,
            source_scene=MEASURED_SCENE, learner=None, beta=1., iteration=0,
            expert_execution="common"):
    if out.exists():
        raise FileExistsError(out)
    if not 0 <= beta <= 1 or set(train_seeds) & set(val_seeds):
        raise ValueError("DAgger beta must be in0..1 and source partitions disjoint")
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    policy = NativeLeRobotACT(snapshot)
    policy.policy.eval().requires_grad_(False)
    head = RecoveryMLPPolicy(learner, expected_base_sha256=policy.weights_sha256) if learner else None
    if head is None and beta < 1:
        raise ValueError("Learner-mixture DAgger requires a fitted recovery head")
    extra = ([Path(learner) / "head.json", Path(learner) / "head.pt"] if learner else [])
    before = source_files(snapshot, source_scene, extra)
    shared = SharedEpisode(robot, out / "live", source_scene)
    shadow = SharedEpisode(robot, out / "expert_shadow", source_scene)
    act = FrozenACTFeatures(shared, policy)
    executor = CommonCartesianExecutor(shared)
    encoder = RecoveryObservation(robot, shared.env.control_hz)
    report = {"schema": "recovery_mlp_dagger_collection_v1", "status": "collecting",
        "robot": robot, "base_checkpoint_sha256": policy.weights_sha256,
        "base_weights_frozen": True, "lighting_changed": False,
        "iteration": iteration, "beta_teacher_execution": beta, "learner": str(learner) if learner else None,
        "expert_execution": expert_execution, "action_names": ACTION_NAMES,
        "action_scales": ACTION_SCALE.tolist(), "observation_names": OBSERVATION_NAMES,
        "act_feature_dim": 256,
        "split": {"train": list(train_seeds), "val": list(val_seeds), "evaluation": list(range(13000, 13100))},
        "expert_queries_actual_learner_states": head is not None,
        "teacher_phase_convergence": "Measured TCP; close/release dwell pauses on opposing executed jaw commands",
        "teacher_phase_or_object_truth_in_observations": False,
        "reference_source": "Initial RGB detections plus declared camera/object geometry",
        "source_sha256_before": before, "trials": []}
    cache = {"train": [], "val": []}
    write_json(out / "manifest.json", report, atomic=True)
    try:
        for partition, seeds in (("train", train_seeds), ("val", val_seeds)):
            for seed in seeds:
                env = shared.env
                tracker = tracker_for(env)
                abort, initial_rgb, failures, learner_ticks = None, None, 0, 0
                reason_counts = {}
                trial_rows = []
                try:
                    shared.reset(seed)
                    shadow.reset(seed)
                    initial_lighting = lighting(env)
                    ball, bowl, initial_rgb = initial_references(env, robot)
                    teacher = CartesianConvergenceTeacher(shadow.env, robot=robot)
                    teacher.bowl_memory = bowl.copy()
                    encoder.reset()
                    act.reset()
                    rng = np.random.default_rng(np.random.SeedSequence([seed, iteration, 20261028]))
                    for tick in range(env.max_episode_steps):
                        observation = encoder.encode(env, ball, bowl, tick)
                        raw, features = act.next()
                        try:
                            base = executor.hf_to_common(raw)
                        except (RuntimeError, ValueError):
                            base = np.r_[np.zeros(6), np.clip(raw[5] / 90., 0., 1.)]
                            failures += 1
                        # The expert sees exactly the current physical state in
                        # its own environment. The learner never receives it.
                        mujoco.mj_copyData(shadow.env.data, shadow.env.model, env.data)
                        shadow.env.steps, shadow.env._ready = env.steps, True
                        if robot == "panda":
                            shadow.env._ee_velocity = env._ee_velocity.copy()
                        teacher.executed_semantic_gripper = float(encoder.previous_action[6]) if tick else None
                        label_result = teacher.step({}, shadow.env)
                        target = teacher_common_action(teacher, label_result, executor)
                        if teacher.phase != "FAILED":
                            row = {"observations": observation,
                                "base_actions": base.astype(np.float32), "features": features,
                                "target_actions": target.astype(np.float32),
                                "robot_id": np.int64(robot == "panda"), "episode_index": np.int64(seed)}
                            cache[partition].append(row)
                            trial_rows.append(row)
                        choose_teacher = head is None or rng.random() < beta
                        selected = target if choose_teacher else head.predict(observation, base, features)
                        if choose_teacher and expert_execution == "native":
                            action = label_result.detail["requested_native"]
                            _, _, _, truncated, info = env.step(action)
                            rejected, saturation = label_result.ik_rejected, label_result.saturation
                            reason = "teacher_native"
                        else:
                            result = executor.step(selected)
                            truncated, info = result.truncated, result.info
                            rejected, saturation = result.ik_rejected, result.saturation
                            reason = result.detail["ik_reason"]
                        learner_ticks += not choose_teacher
                        encoder.selected_action(selected, tick)
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                        done = tracker.update(info["privileged_metrics"],
                                               saturation=saturation, ik_rejected=rejected)
                        if done or truncated:
                            break
                    if lighting(env) != initial_lighting:
                        raise RuntimeError("Lighting changed during collection")
                except (ValueError, RuntimeError) as exc:
                    abort = f"{type(exc).__name__}: {exc}"
                row = tracker.row(experiment_id="RECOVERY_DAGGER_COLLECTION", controller_source="DISCLOSED_RGB_EXPERT_MIXTURE",
                    robot=robot, seed=seed, checkpoint_sha256=policy.weights_sha256, checkpoint_step=None,
                    mean_latency_ms=None, video_path="", abort=abort)
                row.update(partition=partition, supervised_rows=len(trial_rows), learner_executed_ticks=learner_ticks,
                    invalid_hf_priors=failures, command_reasons=reason_counts, initial_rgb=initial_rgb,
                    teacher_final_phase=getattr(teacher, "phase", None) if initial_rgb else None)
                report["trials"].append(row)
                report["summary"] = summarize(report["trials"])
                for name in cache:
                    save_pack(out / f"{name}.npz", cache[name])
                report["supervised_rows"] = {name: len(rows) for name, rows in cache.items()}
                write_json(out / "manifest.json", report, atomic=True)
                print(json.dumps({"robot": robot, "seed": seed, "partition": partition,
                    "placed": row["pick_place_success"], "rows": len(trial_rows),
                    "learner_ticks": learner_ticks, "ik_rejections": row["ik_rejections"], "abort": abort}), flush=True)
        after = {path: sha256_file(Path(path)) for path in before}
        report.update(status="complete", source_sha256_after=after,
            runtime_source_changes=[path for path in before if before[path] != after[path]],
            checkpoint_unchanged=sha256_file(Path(snapshot) / "model.safetensors") == policy.weights_sha256)
        write_json(out / "manifest.json", report, atomic=True)
        return report
    finally:
        act.close()
        shadow.close()
        shared.close()


def evaluate(robot, head_dir, out, seeds, *, snapshot=SNAPSHOT,
             source_scene=MEASURED_SCENE, video=False):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    frozen_hash = sha256_file(Path(snapshot) / "model.safetensors")
    head = RecoveryMLPPolicy(head_dir, expected_base_sha256=frozen_hash)
    policy = NativeLeRobotACT(snapshot) if head.metadata["use_act"] else None
    shared = SharedEpisode(robot, out / "setup", source_scene)
    controller = LearnedRecoveryController(shared, head, policy)
    before = source_files(snapshot, source_scene, [head_dir / "head.json", head_dir / "head.pt"])
    report = {"schema": "learned_recovery_mlp_closed_loop_v1", "status": "running",
        "robot": robot, "seeds": seeds, "head": str(head_dir), "head_sha256": head.head_sha256,
        "base_checkpoint_sha256": frozen_hash, "use_act": head.metadata["use_act"],
        "no_runtime_geometric_teacher": True, "lighting_changed": False,
        "source_sha256_before": before, "trials": []}
    write_json(out / "report.json", report, atomic=True)
    try:
        for seed in seeds:
            tracker = tracker_for(shared.env)
            writer, abort = None, None
            directory = out / f"seed_{seed}"
            directory.mkdir()
            try:
                obs, _ = shared.reset(seed)
                initial_lighting = lighting(shared.env)
                controller.reset()
                if video:
                    writer = CaptionedVideo(directory / "rollout.mp4", shared.env.control_hz, (360, 640))
                with (directory / "ticks.jsonl").open("w") as stream:
                    for tick in range(shared.env.max_episode_steps):
                        result = controller.step(obs, shared.env)
                        obs = result.obs
                        done = tracker.update(result.info["privileged_metrics"],
                            saturation=result.saturation, ik_rejected=result.ik_rejected)
                        stream.write(json.dumps({"tick": tick, "detail": result.detail,
                            "control": result.info["control"], "scoring_only": result.info["privileged_metrics"]},
                            default=json_default) + "\n")
                        if writer:
                            writer.write(shared.env.render("iphone")[:, :, ::-1].copy(),
                                f"{'Frozen ACT + ' if head.metadata['use_act'] else ''}learned recovery MLP; {robot}")
                        if done or result.truncated:
                            break
                if lighting(shared.env) != initial_lighting:
                    raise RuntimeError("Lighting changed during learned evaluation")
            except (ValueError, RuntimeError) as exc:
                abort = f"{type(exc).__name__}: {exc}"
            finally:
                if writer:
                    writer.close()
            row = tracker.row(experiment_id="LEARNED_RECOVERY_MLP", controller_source="FROZEN_ACT_PLUS_MLP" if head.metadata["use_act"] else "MLP_WITHOUT_ACT",
                robot=robot, seed=seed, checkpoint_sha256=frozen_hash, checkpoint_step=None,
                mean_latency_ms=None, video_path=str(directory / "rollout.mp4") if video else "", abort=abort)
            row.update(head_sha256=head.head_sha256, no_runtime_geometric_teacher=True,
                act_policy_calls=controller.act.calls if controller.act else 0,
                trace=str(directory / "ticks.jsonl"))
            report["trials"].append(row)
            report["summary"] = summarize(report["trials"])
            write_json(out / "report.json", report, atomic=True)
            print(json.dumps({"robot": robot, "seed": seed, "placed": row["pick_place_success"],
                "phase": row["phase_failed"], "ik_rejections": row["ik_rejections"], "abort": abort}), flush=True)
        after = {path: sha256_file(Path(path)) for path in before}
        report.update(status="complete", source_sha256_after=after,
            runtime_source_changes=[path for path in before if before[path] != after[path]],
            checkpoint_unchanged=sha256_file(Path(snapshot) / "model.safetensors") == frozen_hash)
        write_json(out / "report.json", report, atomic=True)
        return report
    finally:
        controller.close()
        shared.close()


def controller_factory(*, env, source, robot, snapshot, condition, options):
    """Plugin for the matched benchmark; models are instantiated independently."""
    head = RecoveryMLPPolicy(Path(options["head"]),
        expected_base_sha256=sha256_file(snapshot / "model.safetensors"))
    if (condition == "no_act") == bool(head.metadata["use_act"]):
        raise ValueError("Benchmark condition and recovery head ACT ablation differ")
    policy = NativeLeRobotACT(snapshot) if head.metadata["use_act"] else None
    shared = SimpleNamespace(env=env, source=source, robot=robot)
    return LearnedRecoveryController(shared, head, policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "evaluate"))
    parser.add_argument("--robot", choices=("so101", "panda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--source-scene", type=Path, default=MEASURED_SCENE)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(range(10000, 10012)))
    parser.add_argument("--val-seeds", type=int, nargs="+", default=list(range(11000, 11004)))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(12000, 12010)))
    parser.add_argument("--head", type=Path)
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--expert-execution", choices=("native", "common"), default="common")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.mode == "collect":
        return collect(args.robot, args.out, args.train_seeds, args.val_seeds,
            snapshot=args.snapshot, source_scene=args.source_scene, learner=args.head,
            beta=args.beta, iteration=args.iteration, expert_execution=args.expert_execution)
    if args.head is None:
        parser.error("--head is required for evaluation")
    return evaluate(args.robot, args.head, args.out, args.seeds,
        snapshot=args.snapshot, source_scene=args.source_scene, video=args.video)


if __name__ == "__main__":
    main()
