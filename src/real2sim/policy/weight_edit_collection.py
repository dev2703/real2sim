"""Collect native ACT features and applied simulator-state teacher commands.

The training-only simulator teacher supplies every supervised action. An optional ACT
checkpoint executes an initial prefix before teacher takeover. Frozen base ACT
features observe live state/images only after takeover. Only complete successful
episodes enter supervision; every attempted episode score and trace is retained.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from real2sim.artifacts import sha256_file
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.serialization import json_default, write_json
from real2sim.sim.env import real_from_sim

ROBOT_IDS = {"so101": 0, "panda": 1}


def validate_robots(robots):
    robots = tuple(robots)
    if not robots or len(set(robots)) != len(robots) or any(robot not in ROBOT_IDS for robot in robots):
        raise ValueError("Supply unique supported robots: so101 and/or panda")
    return robots


def native_policy_state(shared):
    """Use the benchmark's live observation projection, including Panda jaw units."""
    env = shared.env
    if shared.robot == "so101":
        return env.real_state()
    state, _ = shared.source.source_state(env)
    state[5] = np.clip(env.gripper_opening() / env.max_opening, 0., 1.) * 90.
    return state


def accepted_panda_target(shared, result, *, native_command=None):
    """Invert applied Panda commands through the benchmark's robot-only adapter.

    Seven Panda joints cannot label six native SO-101 channels directly. Project
    their applied TCP target into the virtual source and invert the same endpoint
    jaw mapping used at execution. Unrepresentable or rejected targets are masked;
    a successful projection becomes the next causal command prior. No ball pose
    or task phase enters this conversion.
    """
    from scripts.diag_v2_learned_panda_retarget_20261027 import VirtualSourceCommandError

    env, source = shared.env, shared.source
    applied = np.asarray(result.info["control"]["applied"] if native_command is None
                         else native_command, dtype=np.float64)
    if applied.shape != (8,) or not np.isfinite(applied).all():
        raise ValueError("Panda teacher must return seven finite arm radians and jaw opening")
    if not 0 <= applied[7] <= env.max_opening:
        raise ValueError("Applied Panda jaw opening exceeds native endpoints")
    pose = shared.guard.forward(applied[:7], finger_opening=applied[7])
    projection = source.project_pose(pose)
    if result.ik_rejected or not projection.accepted:
        return np.zeros(6, np.float32), False
    percent = float(applied[7] / env.max_opening * 90.)
    native = np.r_[projection.arm_qpos, source.mapping.gripper_real_to_sim_rad(percent)]
    raw = real_from_sim(source.mapping, native).astype(np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Panda inverse mapping returned non-finite native targets")
    # Validate the same native limits and advance only command memory. The
    # measured source.q is refreshed by source_state at the next observation.
    try:
        source.source_action(raw)
    except VirtualSourceCommandError:
        # A bounded inverse can round beyond a strict stop when stored in the
        # checkpoint's float32 units. Mask that label without weakening the
        # execution guard or discarding all other commands in the episode.
        return np.zeros(6, np.float32), False
    return raw, True


def validate_seed_split(train_seeds, val_seeds):
    """Whole episodes must have an unambiguous, disjoint fitting role."""
    for name, seeds in (("training", train_seeds), ("validation", val_seeds)):
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"Supply nonempty unique nonnegative integer {name} seeds")
    if set(train_seeds) & set(val_seeds):
        raise ValueError("Training and validation seeds overlap")


def validate_rollin_options(rollin_snapshot, act_prefix_steps):
    if type(act_prefix_steps) is not int or act_prefix_steps < 0:
        raise ValueError("ACT prefix steps must be a nonnegative integer")
    if (rollin_snapshot is None) != (act_prefix_steps == 0):
        raise ValueError("A roll-in snapshot and positive ACT prefix steps must be supplied together")


def collect_anchor_at(tick, stride, teacher_start_tick):
    """Observe the frozen base only after handoff; query offset zero is causal."""
    return tick >= teacher_start_tick and (tick - teacher_start_tick) % stride == 0


def validate_rollin_contract(base, rollin):
    """A changed head may roll in only through the same native saved interface."""
    if base.scene_contract != rollin.scene_contract:
        raise ValueError("Roll-in checkpoint native scene contract differs from the feature base")
    base_path, rollin_path = Path(base.path), Path(rollin.path)
    base_config = json.loads((base_path / "config.json").read_text())
    rollin_config = json.loads((rollin_path / "config.json").read_text())
    for key in ("input_features", "output_features", "normalization_mapping"):
        if base_config.get(key) != rollin_config.get(key):
            raise ValueError(f"Roll-in checkpoint {key} differs from the feature base")
    # Exact saved processor equality establishes joint units, normalization and
    # camera processing without inferring them from a learned head's dimensions.
    processor_names = {path.name for path in base_path.glob("policy_*")
                       if path.suffix in (".json", ".safetensors")}
    rollin_names = {path.name for path in rollin_path.glob("policy_*")
                    if path.suffix in (".json", ".safetensors")}
    if not processor_names or processor_names != rollin_names or any(
            sha256_file(base_path / name) != sha256_file(rollin_path / name)
            for name in processor_names):
        raise ValueError("Roll-in checkpoint saved processors differ from the feature base")


def accepted_native_target(env, result, *, native_command=None):
    """Label the actual rate/limit-bounded actuator target in checkpoint units."""
    applied = np.asarray(result.info["control"]["applied_radians"] if native_command is None
                         else native_command, dtype=np.float64)
    if applied.shape != (6,) or not np.isfinite(applied).all():
        raise ValueError("Teacher must return six finite applied native radians")
    raw = real_from_sim(env.mapping, applied).astype(np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Teacher command inverse mapping returned non-finite targets")
    return raw, not bool(result.ik_rejected)


def episode_pack(anchors, commands, accepted, *, chunk_size, feature_dim, mean, std, robot="so101"):
    """Align query offsets to future commands actually executed in this episode.

    No command is extrapolated beyond the observed episode. Rejected teacher
    steps are masked even when another offset in that same chunk is accepted.
    """
    validate_robots((robot,))
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    action_dim = len(mean)
    if (mean.shape != (action_dim,) or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError("Finite action means and positive standard deviations required")
    commands = np.asarray(commands, np.float32).reshape(-1, action_dim)
    accepted = np.asarray(accepted, bool)
    if accepted.shape != (len(commands),) or not np.isfinite(commands).all():
        raise ValueError("Teacher command validity differs from executed command rows")
    rows = []
    for anchor in anchors:
        features = np.asarray(anchor["features"], np.float32)
        state = np.asarray(anchor["raw_state"], np.float32)
        tick = anchor["tick"]
        if (features.shape != (chunk_size, feature_dim) or not np.isfinite(features).all()
                or state.shape != (6,) or not np.isfinite(state).all()
                or type(tick) is not int or not 0 <= tick < len(commands)):
            raise ValueError("Invalid decoder anchor geometry or command tick")
        offsets = tick + np.arange(chunk_size)
        in_episode = offsets < len(commands)
        valid = np.zeros(chunk_size, bool)
        valid[in_episode] = accepted[offsets[in_episode]]
        targets = np.zeros((chunk_size, action_dim), np.float32)
        targets[valid] = (commands[offsets[valid]] - mean) / std
        row = {"features": features, "target_normalized": targets, "valid": valid,
                     "raw_state": state, "robot_id": np.int64(ROBOT_IDS[robot]),
                     "episode_index": np.int64(anchor["episode_index"]), "tick": np.int64(tick)}
        if "encoder_memory" in anchor or "encoder_positions" in anchor:
            for key in ("encoder_memory", "encoder_positions"):
                value = np.asarray(anchor.get(key), np.float32)
                if (value.ndim != 2 or value.shape[1] != feature_dim or not value.size
                        or not np.isfinite(value).all()):
                    raise ValueError("Invalid cached encoder memory or positions")
                row[key] = value
            if row["encoder_memory"].shape != row["encoder_positions"].shape:
                raise ValueError("Encoder memory and position shapes must match")
        rows.append(row)
    return stack_records(rows, chunk_size=chunk_size, feature_dim=feature_dim, action_dim=action_dim)


def stack_records(records, *, chunk_size, feature_dim, action_dim):
    names = ("features", "target_normalized", "valid", "raw_state", "robot_id", "episode_index", "tick")
    if records:
        if "encoder_memory" in records[0]:
            names += ("encoder_memory", "encoder_positions")
        if any(set(record) != set(names) for record in records):
            raise ValueError("Feature records must contain the same cached tensors")
        return {name: np.stack([row[name] for row in records]) for name in names}
    return {"features": np.empty((0, chunk_size, feature_dim), np.float32),
            "target_normalized": np.empty((0, chunk_size, action_dim), np.float32),
            "valid": np.empty((0, chunk_size), bool), "raw_state": np.empty((0, 6), np.float32),
            **{name: np.empty(0, np.int64) for name in names[-3:]}}


def _records(pack):
    return [{name: value[index] for name, value in pack.items()}
            for index in range(len(pack["tick"]))]


def saved_action_statistics(snapshot, action_dim):
    """Read the active saved unnormalizer rather than assuming its filename."""
    snapshot = Path(snapshot)
    steps = json.loads((snapshot / "policy_postprocessor.json").read_text()).get("steps", [])
    unnormalizers = [step for step in steps if step.get("registry_name") == "unnormalizer_processor"]
    if (len(unnormalizers) != 1 or any(step.get("registry_name") not in (
            "unnormalizer_processor", "device_processor") for step in steps)):
        raise ValueError("Collection requires a single native action unnormalizer plus optional device steps")
    step = unnormalizers[0]
    config = step.get("config", {})
    if (config.get("norm_map", {}).get("ACTION") != "MEAN_STD"
            or config.get("features", {}).get("action", {}).get("shape") != [action_dim]):
        raise ValueError("Saved postprocessor must use MEAN_STD for checkpoint action channels")
    filename = step.get("state_file", "")
    if not filename or Path(filename).name != filename:
        raise ValueError("Saved action unnormalizer requires a local state filename")
    stats = load_file(snapshot / filename)
    mean, std = stats["action.mean"].numpy(), stats["action.std"].numpy()
    if (mean.shape != (action_dim,) or std.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError("Saved action normalization differs from the checkpoint output")
    return mean, std


def collect(snapshot, out, *, train_seeds=(10090, 10091, 10092), val_seeds=(11090, 11091),
            stride=10, source_scene=None, inference_backend="vectorized",
            rollin_snapshot=None, act_prefix_steps=0, robots=("so101",),
            initial_gripper_percent=1.2, feature_layer="action_head", randomization=None,
            target_kind="applied"):
    """Collect native fitting/validation features on either or both embodiments."""
    from real2sim.evaluation.raw_act import DEFAULT_SCENE, SharedEpisode, runtime_hashes

    snapshot, out = Path(snapshot).expanduser().resolve(), Path(out)
    source_scene = Path(source_scene) if source_scene is not None else DEFAULT_SCENE
    validate_seed_split(train_seeds, val_seeds)
    validate_rollin_options(rollin_snapshot, act_prefix_steps)
    robots = validate_robots(robots)
    if type(stride) is not int or stride < 1:
        raise ValueError("Anchor stride must be a positive integer")
    if feature_layer not in ("action_head", "decoder_ffn_input", "decoder_memory"):
        raise ValueError("Unsupported native ACT feature layer")
    if target_kind not in ("applied", "planned_native_goal"):
        raise ValueError("Training target must be applied or planned_native_goal")
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    policy = NativeLeRobotACT(snapshot, inference_backend=inference_backend)
    config = policy.config
    if feature_layer == "decoder_ffn_input" and config.pre_norm:
        raise ValueError("Decoder FFN collection requires a post-norm ACT decoder")
    chunk_size, feature_dim = config.chunk_size, config.dim_model
    action_dim = config.output_features["action"].shape[0]
    saved_config = json.loads((snapshot / "config.json").read_text())
    if saved_config.get("normalization_mapping", {}).get("ACTION") != "MEAN_STD":
        raise ValueError("Weight-edit collection requires saved MEAN_STD native action processors")
    mean, std = saved_action_statistics(snapshot, action_dim)
    rollin_policy = None
    rollin_files = []
    if rollin_snapshot is not None:
        rollin_snapshot = Path(rollin_snapshot).expanduser().resolve()
        rollin_policy = NativeLeRobotACT(rollin_snapshot, inference_backend=inference_backend)
        validate_rollin_contract(policy, rollin_policy)
        rollin_files = [path for path in rollin_snapshot.iterdir()
                        if path.suffix in (".json", ".safetensors")]
    out.mkdir(parents=True)
    before = runtime_hashes(snapshot, source_scene, extra_paths=[
        Path(__file__), Path("src/real2sim/sim/training_teacher.py"),
        *([Path("src/real2sim/sim/domain_randomization.py")] if randomization is not None else []),
        Path("src/real2sim/evaluation/oracle.py"), Path("src/real2sim/policy/bounded_joint.py"),
        *rollin_files])
    report = {"schema": "native_act_weight_edit_collection_v2", "status": "collecting",
              "snapshot": str(snapshot), "weights_sha256": policy.weights_sha256,
              "checkpoint_identity": policy.checkpoint_identity,
              "training_seeds": list(train_seeds), "validation_seeds": list(val_seeds),
              "robots": list(robots), "robot_ids": ROBOT_IDS,
              "source_scene": str(source_scene), "source_scene_sha256": sha256_file(source_scene),
              "initial_gripper_percent": initial_gripper_percent, "anchor_stride": stride,
              "chunk_size": chunk_size, "feature_dim": feature_dim, "action_dim": action_dim,
              "policy_hz": 25., "native_scene_contract": policy.scene_contract,
              "runtime_options": policy.runtime_options,
              "feature_layer": feature_layer,
              "feature_source": f"Actual native ACT {feature_layer} forward prehook on live RGB/state",
              "teacher_source": "SIM_STATE_TRAINING_TEACHER",
              "target_kind": target_kind,
              "target_source": f"Training-only simulator teacher {target_kind}; native SO-101 inverse mapping or Panda virtual-source projection with endpoint jaw inverse",
              "panda_target_projection": "Same position-only 3 mm acceptance and causal command prior as native execution; failed projections masked",
              "normalization": {"mean": mean, "std": std,
                                "inverse": "(applied_command-mean)/std; exact inverse of saved MEAN_STD unnormalizer"},
              "selection": "Whole successful teacher episodes only; all attempted trials retained",
              "policy_inference_drives_actions": rollin_policy is not None,
              "base_feature_policy_drives_actions": False, "teacher_success_is_learned_success": False,
              "rollin_snapshot": str(rollin_snapshot) if rollin_snapshot is not None else None,
              "rollin_checkpoint_sha256": rollin_policy.weights_sha256 if rollin_policy is not None else None,
              "act_prefix_steps": act_prefix_steps, "teacher_start_tick": act_prefix_steps,
              "label_mask": "Accepted simulator-training-teacher commands only; ACT-prefix and rejected-command ticks masked",
              "cross_embodiment_transfer_claim": False,
              "target_arm_supervision": "panda" in robots,
              "privileged_metrics_used_only_for_scoring_and_episode_selection": False,
              "privileged_teacher_state_used_for_training_labels_only": True,
              "domain_randomization": randomization,
              "source_sha256_before": before, "trials": []}
    write_json(out / "report.json", report, atomic=True)
    cache = {"train": [], "val": []}
    captured = {}

    def hook(_module, inputs, kwargs):
        features = inputs[0]
        if feature_layer in ("decoder_ffn_input", "decoder_memory"):
            features = features.transpose(0, 1)
        if features.shape != (1, chunk_size, feature_dim) or not torch.isfinite(features).all():
            raise ValueError("Native decoder returned invalid collection features")
        captured["features"] = features.detach().cpu().numpy()[0].copy()
        if feature_layer == "decoder_memory":
            for name, value in (("encoder_memory", inputs[1]),
                                ("encoder_positions", kwargs["encoder_pos_embed"])):
                if (value.ndim != 3 or value.shape[1:] != (1, feature_dim)
                        or not torch.isfinite(value).all()):
                    raise ValueError("Native decoder returned invalid encoder tokens")
                captured[name] = value.detach().cpu().numpy()[:, 0].copy()

    feature_module = (policy.policy.model.decoder if feature_layer == "decoder_memory" else (
        policy.policy.model.action_head if feature_layer == "action_head"
        else policy.policy.model.decoder.layers[-1].linear1))
    handle = feature_module.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for robot in robots:
            shared = None
            try:
                shared = SharedEpisode(robot, out / "setups" / robot, source_scene=source_scene,
                                       native_contract=policy.scene_contract,
                                       initial_gripper_percent=initial_gripper_percent)
                report.setdefault("effective_rendering", {})[robot] = {
                    "near_clip_m": float(shared.env.model.vis.map.znear) * float(shared.env.model.stat.extent),
                    "width": shared.env.config["render"]["width"],
                    "height": shared.env.config["render"]["height"],
                }
                randomizer = None
                if randomization is not None:
                    from real2sim.sim.domain_randomization import DomainRandomizer

                    randomizer = DomainRandomizer(shared.env, randomization)
                _collect_robot(shared, policy, rollin_policy, out, report, cache, captured,
                               train_seeds=train_seeds, val_seeds=val_seeds, stride=stride,
                               act_prefix_steps=act_prefix_steps, chunk_size=chunk_size,
                               feature_dim=feature_dim, action_dim=action_dim, mean=mean, std=std,
                               randomizer=randomizer, target_kind=target_kind)
            finally:
                if shared is not None:
                    shared.close()
        for name, records in cache.items():
            np.savez_compressed(out / f"{name}.npz", **stack_records(
                records, chunk_size=chunk_size, feature_dim=feature_dim, action_dim=action_dim))
        after = {path: sha256_file(path) for path in before}
        covered = {name: set(int(row["robot_id"]) for row in rows) for name, rows in cache.items()}
        usable = all(covered[name] == {ROBOT_IDS[robot] for robot in robots} for name in cache)
        report.update(status="complete" if usable else "failed", source_sha256_after=after,
                      runtime_source_changes=[path for path in before if before[path] != after[path]],
                      checkpoint_unchanged=policy.weights_sha256 == sha256_file(snapshot / "model.safetensors"),
                      rollin_checkpoint_unchanged=rollin_policy is None or (
                          rollin_policy.weights_sha256 == sha256_file(rollin_snapshot / "model.safetensors")),
                      feature_files_sha256={name: sha256_file(out / name)
                                            for name in ("train.npz", "val.npz")},
                      failure=None if usable else "Each requested robot needs successful teacher supervision in both splits")
        write_json(out / "report.json", report, atomic=True)
        return report
    finally:
        handle.remove()


def _collect_robot(shared, policy, rollin_policy, out, report, cache, captured, *,
                   train_seeds, val_seeds, stride, act_prefix_steps, chunk_size,
                   feature_dim, action_dim, mean, std, randomizer=None, target_kind="applied"):
    from real2sim.evaluation.raw_act import FrozenDriver, initial_fingerprint
    from real2sim.sim.training_teacher import SimStateTrainingTeacher

    env, robot = shared.env, shared.robot
    if env.control_hz != 25.:
        raise ValueError("Native weight-edit teacher collection requires 25 Hz control")
    if act_prefix_steps >= env.max_episode_steps:
        raise ValueError("ACT prefix must leave at least one geometric teacher tick")
    for partition, seeds in (("train", train_seeds), ("val", val_seeds)):
        for seed in seeds:
            directory = out / f"{robot}_{seed}"
            directory.mkdir()
            tracker = TrialTracker(success_hold_ticks=math.ceil(
                env.task["episode"]["success_hold_s"] * env.control_hz),
                approach_distance_m=.03, table_top_z=env.table_top,
                ball_radius_m=env.task["ball"]["radius"])
            anchors, commands, accepted = [], [], []
            abort, initial, teacher, randomized = None, None, None, None
            started = time.perf_counter()
            try:
                randomized = randomizer.apply(seed) if randomizer is not None else None
                obs, info = shared.reset(seed, randomized["scenario"] if randomized else None)
                initial = initial_fingerprint(shared, info)
                teacher = SimStateTrainingTeacher(env, robot=robot)
                teacher.reset()
                policy.reset()
                rollin_driver = FrozenDriver(shared, rollin_policy) if rollin_policy is not None else None
                with (directory / "ticks.jsonl").open("w") as stream:
                    for tick in range(env.max_episode_steps):
                        teacher_tick = tick >= act_prefix_steps
                        if collect_anchor_at(tick, stride, act_prefix_steps):
                            state = native_policy_state(shared)
                            images = {key: env.render(env.camera_bindings[key]) for key in policy.camera_keys}
                            captured.clear()
                            policy.predict_action_chunk(state, images)
                            if "features" not in captured:
                                raise ValueError("ACT did not provide the requested cached features")
                            anchors.append({**captured, "raw_state": state,
                                            "tick": tick, "episode_index": seed})
                        result = teacher.step(obs, env) if teacher_tick else rollin_driver.step(obs, env)
                        native_goal = (result.detail["supervision_native_goal"]
                                       if teacher_tick and target_kind == "planned_native_goal" else None)
                        if not teacher_tick:
                            raw_command, valid_command = np.zeros(action_dim, np.float32), False
                        elif robot == "so101":
                            raw_command, valid_command = accepted_native_target(env, result, native_command=native_goal)
                        else:
                            raw_command, valid_command = accepted_panda_target(shared, result, native_command=native_goal)
                        commands.append(raw_command)
                        accepted.append(valid_command)
                        obs = result.obs
                        done = tracker.update(result.info["privileged_metrics"],
                                              saturation=result.saturation, ik_rejected=result.ik_rejected)
                        stream.write(json.dumps({"tick": tick, "time_s": tick / env.control_hz,
                            "uses_act": not teacher_tick, "assisted": teacher_tick,
                            "teacher_start_tick": act_prefix_steps, "detail": result.detail,
                            "target_kind": target_kind,
                            "accepted_supervision": valid_command, "target_degree_percent": raw_command,
                            "control": result.info["control"],
                            "scoring_only": result.info["privileged_metrics"]},
                            default=json_default, allow_nan=False) + "\n")
                        if done or result.truncated:
                            break
            except (RuntimeError, ValueError, OSError) as exc:
                abort = f"{type(exc).__name__}: {exc}"
            # A failed pre-step query has no command label and must not make
            # an anchor appear to have an executed future action.
            anchors = [anchor for anchor in anchors if anchor["tick"] < len(commands)]
            pack = episode_pack(anchors, commands, accepted, chunk_size=chunk_size,
                                feature_dim=feature_dim, mean=mean, std=std, robot=robot)
            np.savez_compressed(directory / "anchors.npz", **pack)
            row = tracker.row(experiment_id="ACT_WEIGHT_EDIT_TEACHER_COLLECTION",
                controller_source="ACT_ROLLIN_SIM_TRAINING_TEACHER" if rollin_policy is not None
                                  else "SIM_STATE_TRAINING_TEACHER", robot=robot, seed=seed,
                checkpoint_sha256=policy.weights_sha256, checkpoint_step=None,
                mean_latency_ms=None, video_path="", abort=abort)
            included = bool(row["pick_place_success"]) and abort is None and bool(pack["valid"].any())
            if included:
                cache[partition].extend(_records(pack))
            row.update(partition=partition, included_in_supervision=included,
                       initial_state=initial, anchors=len(anchors), domain_randomization=randomized,
                       teacher_start_tick=act_prefix_steps,
                       rollin_checkpoint_sha256=report["rollin_checkpoint_sha256"],
                       valid_command_targets=int(pack["valid"].sum()),
                       wall_s=time.perf_counter() - started,
                       trace=str(directory / "ticks.jsonl"),
                       perception_log=teacher.perception_log if teacher is not None else [])
            write_json(directory / "report.json", row, atomic=True)
            report["trials"].append(row)
            if report["feature_layer"] != "decoder_memory":
                for name, rows in cache.items():
                    np.savez_compressed(out / f"{name}.npz", **stack_records(
                        rows, chunk_size=chunk_size, feature_dim=feature_dim, action_dim=action_dim))
            report["supervised_anchors"] = {name: len(rows) for name, rows in cache.items()}
            report["summary"] = summarize(report["trials"])
            write_json(out / "report.json", report, atomic=True)
            print(json.dumps({"robot": robot, "partition": partition, "seed": seed, "teacher_success": included,
                              "anchors": len(anchors), "abort": abort}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[10090, 10091, 10092])
    parser.add_argument("--val-seeds", type=int, nargs="+", default=[11090, 11091])
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--source-scene", type=Path)
    parser.add_argument("--inference-backend", choices=("legacy", "vectorized"), default="vectorized")
    parser.add_argument("--rollin-snapshot", type=Path)
    parser.add_argument("--act-prefix-steps", type=int, default=0)
    parser.add_argument("--robots", choices=tuple(ROBOT_IDS), nargs="+", default=["so101"])
    parser.add_argument("--initial-gripper-percent", type=float, default=1.2)
    parser.add_argument("--feature-layer", choices=("action_head", "decoder_ffn_input", "decoder_memory"),
                        default="action_head")
    parser.add_argument("--domain-randomization", type=Path,
                        help="JSON randomization profile; {} enables the standard training profile")
    parser.add_argument("--target-kind", choices=("applied", "planned_native_goal"), default="applied")
    args = parser.parse_args()
    collect(args.snapshot, args.out, train_seeds=args.train_seeds, val_seeds=args.val_seeds,
            stride=args.stride, source_scene=args.source_scene, inference_backend=args.inference_backend,
            rollin_snapshot=args.rollin_snapshot, act_prefix_steps=args.act_prefix_steps,
            robots=args.robots, initial_gripper_percent=args.initial_gripper_percent,
            feature_layer=args.feature_layer,
            randomization=json.loads(args.domain_randomization.read_text()) if args.domain_randomization else None,
            target_kind=args.target_kind)


if __name__ == "__main__":
    main()
