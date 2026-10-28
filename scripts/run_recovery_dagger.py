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
import shutil
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
    TRACKED_OBSERVATION_NAMES,
    CommonCartesianExecutor,
    FrozenACTFeatures,
    RecoveryObservation,
    clip_common,
    semantic_gripper,
)
from real2sim.policy.grasp_feedback import ObservableGraspMonitor, RecoveryPhaseGuard
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.policy.recovery_mlp import RecoveryMLPPolicy
from real2sim.policy.rgb_reference_tracker import RGBReferenceTracker
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
        # Require continuous executed close/release dwell. Opposing commands
        # reset it; the hypothetical expert query must not earn a dwell tick.
        # This uses causal command history, never held/contact flags.
        selected = getattr(self, "executed_semantic_gripper", None)
        if selected is not None and ((self.phase == "CLOSE" and selected >= .5)
                                     or (self.phase == "RELEASE" and selected < .5)):
            self.phase_tick = -1  # super().step increments once for this query
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
    command_phase = label_result.detail["phase"]
    if (teacher.robot == "panda" and command_phase == teacher.phase
            and teacher.target_q is not None and command_phase in (
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
        Path("src/real2sim/policy/common_recovery.py"), Path("src/real2sim/policy/recovery_mlp.py"),
        Path("src/real2sim/policy/phase_recovery_mlp.py"),
        Path("src/real2sim/policy/grasp_feedback.py"),
        Path("src/real2sim/policy/rgb_reference_tracker.py"), *extra])


def snapshot_sources(out, hashes):
    """Preserve hashed executable text; reference large binary artifacts by hash."""
    root, directory = Path.cwd().resolve(), out / "executed_sources"
    directory.mkdir()
    entries = []
    for index, (filename, digest) in enumerate(sorted(hashes.items())):
        source = Path(filename)
        entry = {"source": filename, "sha256": digest, "snapshot": None}
        if source.suffix in {".py", ".yaml", ".yml", ".xml", ".md", ".json"}:
            relative = (source.relative_to(root) if source.is_relative_to(root)
                        else Path("external") / str(index) / source.name)
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256_file(target) != digest:
                raise RuntimeError("Runtime source changed while taking its executed snapshot")
            entry["snapshot"] = str(target.relative_to(out))
        else:
            entry["storage"] = "external_artifact_reference"
        entries.append(entry)
    ledger = directory / "manifest.json"
    write_json(ledger, {"schema": "recovery_executed_sources_v1", "entries": entries}, atomic=True)
    return str(ledger)


def load_recovery_head(directory, base_hash):
    metadata = json.loads((Path(directory) / "head.json").read_text())
    if metadata.get("schema") == "frozen_act_phase_supervised_recovery_mlp_v1":
        from real2sim.policy.phase_recovery_mlp import PhaseRecoveryMLPPolicy

        return PhaseRecoveryMLPPolicy(directory, expected_base_sha256=base_hash)
    return RecoveryMLPPolicy(directory, expected_base_sha256=base_hash)


def head_reference_mode(head):
    names = head.metadata.get("observation_names") or []
    return "tracked" if any("fresh_rgb_" in name for name in names) else "initial"


def issued_gripper_opening(env, robot, native):
    """Semantic jaw intent of the issued native command, including arm holds."""
    opening = (env.mapping.gripper_sim_to_real_percent(native[-1]) / 90.
               if robot == "so101" else native[-1] / env.max_opening)
    return float(np.clip(opening, 0., 1.))


def save_pack(path, rows):
    if rows:
        arrays = {name: np.stack([row[name] for row in rows]) for name in ARRAY_NAMES}
        for name in ("expert_phase_id", "tick"):
            if name in rows[0]:
                arrays[name] = np.stack([row[name] for row in rows])
    else:
        arrays = {"observations": np.empty((0, 99), np.float32),
            "base_actions": np.empty((0, 7), np.float32), "features": np.empty((0, 256), np.float32),
            "target_actions": np.empty((0, 7), np.float32),
            "robot_id": np.empty(0, np.int64), "episode_index": np.empty(0, np.int64)}
    np.savez_compressed(path, **arrays)


class LearnedRecoveryController:
    def __init__(self, episode, head, policy=None, *, phase_guard=False):
        self.episode, self.env, self.robot, self.head = episode, episode.env, episode.robot, head
        self.use_act = bool(head.metadata["use_act"])
        self.executor = CommonCartesianExecutor(episode, require_source_mapping=self.use_act)
        if phase_guard and not hasattr(head, "last_phase_id"):
            raise ValueError("Phase guard requires a phase-supervised recovery head")
        self.grasp_monitor = ObservableGraspMonitor(self.env.control_hz)
        self.phase_guard = RecoveryPhaseGuard(self.env.control_hz) if phase_guard else None
        self.encoder = RecoveryObservation(self.robot, self.env.control_hz)
        self.reference_mode = head_reference_mode(head)
        self.act = FrozenACTFeatures(episode, policy) if self.use_act else None
        if policy is not None:
            policy.policy.eval().requires_grad_(False)
        self.reset()

    def reset(self):
        self.tick, self.ball, self.bowl = 0, None, None
        self.references = RGBReferenceTracker(self.env) if self.reference_mode == "tracked" else None
        self.grasp_anchor = None
        self.previous_issued_gripper = 1.
        self.grasp_monitor.reset()
        if self.phase_guard:
            self.phase_guard.reset()
        self.encoder.reset()
        if self.act:
            self.act.reset()

    def step(self, obs, env):
        del obs
        if self.references:
            self.references.update(self.tick, force=self.ball is None)
            metadata = self.references.observation_metadata(self.tick)
            self.ball, self.bowl = self.references.ball, self.references.bowl
            self.rgb_acquisition = self.references.log[0]
        elif self.ball is None:
            self.ball, self.bowl, self.rgb_acquisition = initial_references(env, self.robot)
        metadata = self.references.observation_metadata(self.tick) if self.references else {}
        observed = self.encoder.encode(env, self.ball, self.bowl, self.tick, **metadata)
        reference_ages = [(self.tick - seen) / env.control_hz
                          for seen in metadata.get("last_seen_ticks", (0, 0))]
        pose = env.ee_pose()
        predicted_ball = (None if self.grasp_anchor is None
                          else pose[:3, 3] + pose[:3, :3] @ self.grasp_anchor)
        fresh_ball = (self.ball if metadata.get("visibility", (False, False))[0]
                      and metadata.get("last_seen_ticks", (None, None))[0] == self.tick else None)
        feedback = self.grasp_monitor.update(self.tick, pose[:3, 3],
            semantic_gripper(env, self.robot),
            self.previous_issued_gripper,
            fresh_ball=fresh_ball, predicted_ball=predicted_ball)
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
        predicted_action = action.copy()
        guard_detail = None
        if self.phase_guard:
            from real2sim.policy.phase_recovery_mlp import PHASE_NAMES

            action, guard_detail = self.phase_guard.filter(
                action, PHASE_NAMES[self.head.last_phase_id], feedback)
        result = self.executor.step(action)
        native = result.detail.get("requested_native")
        issued = (issued_gripper_opening(env, self.robot, native) if native is not None
                  else float(action[6]))
        # Anchor only a nearby, recent RGB reference at the issued close edge.
        # Its rigid carry prediction is a hypothesis used to detect separation,
        # never a measurement of the carried ball or an input to the old head.
        if issued >= .5:
            self.grasp_anchor = None
        elif self.previous_issued_gripper >= .5:
            recent = bool(reference_ages and reference_ages[0] <= .2)
            if recent and np.linalg.norm(self.ball - pose[:3, 3]) <= .06:
                self.grasp_anchor = pose[:3, :3].T @ (self.ball - pose[:3, 3])
        self.previous_issued_gripper = issued
        self.encoder.selected_action(action, self.tick)
        self.tick += 1
        result.detail.update(uses_act=self.use_act, intervention=True,
            no_runtime_geometric_teacher=True, invalid_base_prior=invalid_prior,
            head_sha256=self.head.head_sha256, reference_mode=self.reference_mode,
            rgb_ball_reference=self.ball.tolist(), rgb_bowl_reference=self.bowl.tolist(),
            rgb_reference_visibility=list(metadata.get("visibility", (True, True))),
            rgb_reference_age_s=reference_ages,
            recovery_inputs={"observations": observed.tolist(), "base_actions": base.tolist(),
                             "features": features.tolist()},
            observable_grasp_feedback=feedback, phase_guard=guard_detail,
            phase_guard_enabled=self.phase_guard is not None,
            predicted_common_action=predicted_action.tolist(),
            issued_gripper_open_fraction=issued,
            virtual_source_projection=getattr(self.act, "last_source_projection", None))
        if hasattr(self.head, "last_phase_id"):
            result.detail["learned_phase_id"] = self.head.last_phase_id
            result.detail["learned_phase_probabilities"] = self.head.last_phase_probabilities
        return result

    def close(self):
        if self.act:
            self.act.close()


def collect(robot, out, train_seeds, val_seeds, *, snapshot=SNAPSHOT,
            source_scene=MEASURED_SCENE, learner=None, beta=1., iteration=0,
            expert_execution="common", reference_mode="initial"):
    if out.exists():
        raise FileExistsError(out)
    if not 0 <= beta <= 1 or set(train_seeds) & set(val_seeds):
        raise ValueError("DAgger beta must be in0..1 and source partitions disjoint")
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    policy = NativeLeRobotACT(snapshot)
    policy.policy.eval().requires_grad_(False)
    head = load_recovery_head(learner, policy.weights_sha256) if learner else None
    if head is not None and head_reference_mode(head) != reference_mode:
        raise ValueError("Learner and collection RGB-reference contracts differ")
    if head is None and beta < 1:
        raise ValueError("Learner-mixture DAgger requires a fitted recovery head")
    extra = ([Path(learner) / "head.json", Path(learner) / "head.pt"] if learner else [])
    before = source_files(snapshot, source_scene, extra)
    executed_sources = snapshot_sources(out, before)
    shared = SharedEpisode(robot, out / "live", source_scene)
    shadow = SharedEpisode(robot, out / "expert_shadow", source_scene)
    act = FrozenACTFeatures(shared, policy)
    executor = CommonCartesianExecutor(shared, require_source_mapping=True)
    encoder = RecoveryObservation(robot, shared.env.control_hz)
    report = {"schema": "recovery_mlp_dagger_collection_v1", "status": "collecting",
        "robot": robot, "base_checkpoint_sha256": policy.weights_sha256,
        "base_weights_frozen": True, "lighting_changed": False,
        "iteration": iteration, "beta_teacher_execution": beta, "learner": str(learner) if learner else None,
        "expert_execution": expert_execution, "action_names": ACTION_NAMES,
        "action_scales": ACTION_SCALE.tolist(),
        "observation_names": TRACKED_OBSERVATION_NAMES if reference_mode == "tracked" else OBSERVATION_NAMES,
        "act_feature_dim": 256,
        "split": {"train": list(train_seeds), "val": list(val_seeds), "evaluation": list(range(13000, 13100))},
        "expert_queries_actual_learner_states": head is not None,
        "teacher_phase_convergence": "Measured TCP; continuous close/release dwell resets on opposing issued jaw commands",
        "teacher_phase_or_object_truth_in_observations": False,
        "expert_phase_is_audit_label_only": True,
        "expert_phase_label_source": "issued_command",
        "panda_actions_require_source_mapping": robot == "panda" and expert_execution == "common",
        "native_teacher_execution_bypasses_source_mapping": robot == "panda" and expert_execution == "native",
        "expert_terminal_hold_labels_included": False,
        "reference_mode": reference_mode,
        "reference_source": "Confident RGB detections plus declared camera/object geometry; tracked mode refreshes every5ticks",
        "source_sha256_before": before, "executed_sources_manifest": executed_sources, "trials": []}
    cache = {"train": [], "val": []}
    write_json(out / "manifest.json", report, atomic=True)
    try:
        for partition, seeds in (("train", train_seeds), ("val", val_seeds)):
            for seed in seeds:
                env = shared.env
                tracker = tracker_for(env)
                abort, initial_rgb, failures, learner_ticks, replans = None, None, 0, 0, 0
                abort_projection = None
                previous_issued_gripper = None
                references = None
                reason_counts = {}
                trial_rows = []
                try:
                    shared.reset(seed)
                    shadow.reset(seed)
                    initial_lighting = lighting(env)
                    ball, bowl, initial_rgb = initial_references(env, robot)
                    if reference_mode == "tracked":
                        references = RGBReferenceTracker(env)
                        references.update(0, force=True)
                        references.observation_metadata(0)
                        ball, bowl = references.ball, references.bowl
                    teacher = CartesianConvergenceTeacher(shadow.env, robot=robot)
                    teacher.bowl_memory = bowl.copy()
                    encoder.reset()
                    act.reset()
                    rng = np.random.default_rng(np.random.SeedSequence([seed, iteration, 20261028]))
                    for tick in range(env.max_episode_steps):
                        if references:
                            update = references.update(tick)
                            ball, bowl = references.ball, references.bowl
                            if references.expert_needs_replan(teacher, update):
                                teacher.reset()
                                teacher.bowl_memory = bowl.copy()
                                replans += 1
                        metadata = references.observation_metadata(tick) if references else {}
                        observation = encoder.encode(env, ball, bowl, tick, **metadata)
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
                        teacher.executed_semantic_gripper = previous_issued_gripper
                        label_result = teacher.step({}, shadow.env)
                        command_phase = label_result.detail["phase"]
                        target = teacher_common_action(teacher, label_result, executor)
                        # Terminal expert holds from a failed physical pickup
                        # are not corrective demonstrations. Retain the attempt
                        # in scoring but avoid hundreds of idle terminal labels.
                        if command_phase not in ("FAILED", "DONE"):
                            row = {"observations": observation,
                                "base_actions": base.astype(np.float32), "features": features,
                                "target_actions": target.astype(np.float32),
                                "robot_id": np.int64(robot == "panda"), "episode_index": np.int64(seed),
                                "expert_phase_id": np.int64(teacher.phases.index(command_phase)),
                                "tick": np.int64(tick)}
                            cache[partition].append(row)
                            trial_rows.append(row)
                        choose_teacher = head is None or rng.random() < beta
                        selected = target if choose_teacher else head.predict(observation, base, features)
                        if choose_teacher and expert_execution == "native":
                            action = label_result.detail["requested_native"]
                            _, _, _, truncated, info = env.step(action)
                            rejected, saturation = label_result.ik_rejected, label_result.saturation
                            reason = "teacher_native"
                            issued_native = action
                        else:
                            result = executor.step(selected)
                            truncated, info = result.truncated, result.info
                            rejected, saturation = result.ik_rejected, result.saturation
                            reason = result.detail["ik_reason"]
                            issued_native = result.detail["requested_native"]
                        previous_issued_gripper = issued_gripper_opening(env, robot, issued_native)
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
                    if (projection := getattr(exc, "projection", None)) is not None:
                        abort_projection = projection.diagnostics()
                row = tracker.row(experiment_id="RECOVERY_DAGGER_COLLECTION", controller_source="DISCLOSED_RGB_EXPERT_MIXTURE",
                    robot=robot, seed=seed, checkpoint_sha256=policy.weights_sha256, checkpoint_step=None,
                    mean_latency_ms=None, video_path="", abort=abort)
                row.update(partition=partition, supervised_rows=len(trial_rows), learner_executed_ticks=learner_ticks,
                    invalid_hf_priors=failures, command_reasons=reason_counts, initial_rgb=initial_rgb,
                    abort_source_projection=abort_projection,
                    teacher_final_phase=getattr(teacher, "phase", None) if initial_rgb else None,
                    rgb_expert_replans=replans, rgb_reference_log=references.log if references else None)
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
             source_scene=MEASURED_SCENE, video=False, phase_guard=False):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    frozen_hash = sha256_file(Path(snapshot) / "model.safetensors")
    head = load_recovery_head(head_dir, frozen_hash)
    policy = NativeLeRobotACT(snapshot) if head.metadata["use_act"] else None
    shared = SharedEpisode(robot, out / "setup", source_scene)
    controller = LearnedRecoveryController(shared, head, policy, phase_guard=phase_guard)
    before = source_files(snapshot, source_scene, [head_dir / "head.json", head_dir / "head.pt"])
    executed_sources = snapshot_sources(out, before)
    report = {"schema": "learned_recovery_mlp_closed_loop_v1", "status": "running",
        "robot": robot, "seeds": seeds, "head": str(head_dir), "head_sha256": head.head_sha256,
        "base_checkpoint_sha256": frozen_hash, "use_act": head.metadata["use_act"],
        "reference_mode": head_reference_mode(head),
        "panda_actions_require_source_mapping": robot == "panda" and bool(head.metadata["use_act"]),
        "observable_grasp_feedback_changes_head_inputs": False,
        "phase_guard_enabled": phase_guard,
        "phase_guard_is_additional_execution_assistance": phase_guard,
        "observable_grasp_monitor_config": controller.grasp_monitor.config,
        "phase_guard_config": controller.phase_guard.config if controller.phase_guard else None,
        "no_runtime_geometric_teacher": True, "lighting_changed": False,
        "source_sha256_before": before, "executed_sources_manifest": executed_sources, "trials": []}
    write_json(out / "report.json", report, atomic=True)
    try:
        for seed in seeds:
            tracker = tracker_for(shared.env)
            writer, abort = None, None
            abort_projection = None
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
                if (projection := getattr(exc, "projection", None)) is not None:
                    abort_projection = projection.diagnostics()
            finally:
                if writer:
                    writer.close()
            row = tracker.row(experiment_id="LEARNED_RECOVERY_MLP", controller_source="FROZEN_ACT_PLUS_MLP" if head.metadata["use_act"] else "MLP_WITHOUT_ACT",
                robot=robot, seed=seed, checkpoint_sha256=frozen_hash, checkpoint_step=None,
                mean_latency_ms=None, video_path=str(directory / "rollout.mp4") if video else "", abort=abort)
            row.update(head_sha256=head.head_sha256, no_runtime_geometric_teacher=True,
                abort_source_projection=abort_projection,
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
    head = load_recovery_head(Path(options["head"]), sha256_file(snapshot / "model.safetensors"))
    if (condition == "no_act") == bool(head.metadata["use_act"]):
        raise ValueError("Benchmark condition and recovery head ACT ablation differ")
    policy = NativeLeRobotACT(snapshot) if head.metadata["use_act"] else None
    shared = SimpleNamespace(env=env, source=source, robot=robot)
    return LearnedRecoveryController(shared, head, policy,
        phase_guard=bool(options.get("phase_guard", False)))


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
    parser.add_argument("--reference-mode", choices=("initial", "tracked"), default="initial")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--phase-guard", action="store_true",
                        help="Evaluate disclosed stateful jaw/phase/motion assistance")
    args = parser.parse_args()
    if args.mode == "collect":
        if args.phase_guard:
            parser.error("--phase-guard is an evaluation assistance option")
        return collect(args.robot, args.out, args.train_seeds, args.val_seeds,
            snapshot=args.snapshot, source_scene=args.source_scene, learner=args.head,
            beta=args.beta, iteration=args.iteration, expert_execution=args.expert_execution,
            reference_mode=args.reference_mode)
    if args.head is None:
        parser.error("--head is required for evaluation")
    return evaluate(args.robot, args.head, args.out, args.seeds,
        snapshot=args.snapshot, source_scene=args.source_scene, video=args.video,
        phase_guard=args.phase_guard)


if __name__ == "__main__":
    main()
