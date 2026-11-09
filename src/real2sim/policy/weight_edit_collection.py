"""Collect native ACT decoder features and accepted RGB teacher commands.

The geometric teacher supplies every supervised action. An optional separate ACT
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


def accepted_native_target(env, result):
    """Label the actual rate/limit-bounded actuator target in checkpoint units."""
    applied = np.asarray(result.info["control"]["applied_radians"], dtype=np.float64)
    if applied.shape != (6,) or not np.isfinite(applied).all():
        raise ValueError("Teacher must return six finite applied native radians")
    raw = real_from_sim(env.mapping, applied).astype(np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Teacher command inverse mapping returned non-finite targets")
    return raw, not bool(result.ik_rejected)


def episode_pack(anchors, commands, accepted, *, chunk_size, feature_dim, mean, std):
    """Align query offsets to future commands actually executed in this episode.

    No command is extrapolated beyond the observed episode. Rejected teacher
    steps are masked even when another offset in that same chunk is accepted.
    """
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
        rows.append({"features": features, "target_normalized": targets, "valid": valid,
                     "raw_state": state, "robot_id": np.int64(0),
                     "episode_index": np.int64(anchor["episode_index"]), "tick": np.int64(tick)})
    return stack_records(rows, chunk_size=chunk_size, feature_dim=feature_dim, action_dim=action_dim)


def stack_records(records, *, chunk_size, feature_dim, action_dim):
    names = ("features", "target_normalized", "valid", "raw_state", "robot_id", "episode_index", "tick")
    if records:
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
            rollin_snapshot=None, act_prefix_steps=0):
    """Collect SO-101 fitting and validation packs without changing the scene."""
    from scripts.benchmark_frozen_act import (
        FrozenDriver,
        SharedEpisode,
        initial_fingerprint,
        runtime_hashes,
    )
    from scripts.evaluate_hf_recovery import MEASURED_SCENE, StaticBowlRecovery

    snapshot, out = Path(snapshot).expanduser().resolve(), Path(out)
    source_scene = Path(source_scene) if source_scene is not None else MEASURED_SCENE
    validate_seed_split(train_seeds, val_seeds)
    validate_rollin_options(rollin_snapshot, act_prefix_steps)
    if type(stride) is not int or stride < 1:
        raise ValueError("Anchor stride must be a positive integer")
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    policy = NativeLeRobotACT(snapshot, inference_backend=inference_backend)
    config = policy.config
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
        Path(__file__), Path("src/real2sim/perception/ball_hsv.py"),
        Path("src/real2sim/evaluation/oracle.py"), Path("src/real2sim/policy/bounded_joint.py"),
        *rollin_files])
    report = {"schema": "native_act_weight_edit_collection_v1", "status": "collecting",
              "snapshot": str(snapshot), "weights_sha256": policy.weights_sha256,
              "checkpoint_identity": policy.checkpoint_identity,
              "training_seeds": list(train_seeds), "validation_seeds": list(val_seeds),
              "robot": "so101", "source_scene": str(source_scene), "anchor_stride": stride,
              "chunk_size": chunk_size, "feature_dim": feature_dim, "action_dim": action_dim,
              "policy_hz": 25., "native_scene_contract": policy.scene_contract,
              "runtime_options": policy.runtime_options,
              "feature_source": "Actual native ACT action_head forward prehook on live RGB/state",
              "target_source": "Accepted RGB geometric teacher applied actuator target, inverse-mapped to native degree/percent",
              "normalization": {"mean": mean, "std": std,
                                "inverse": "(applied_command-mean)/std; exact inverse of saved MEAN_STD unnormalizer"},
              "selection": "Whole successful teacher episodes only; all attempted trials retained",
              "policy_inference_drives_actions": rollin_policy is not None,
              "base_feature_policy_drives_actions": False, "teacher_success_is_learned_success": False,
              "rollin_snapshot": str(rollin_snapshot) if rollin_snapshot is not None else None,
              "rollin_checkpoint_sha256": rollin_policy.weights_sha256 if rollin_policy is not None else None,
              "act_prefix_steps": act_prefix_steps, "teacher_start_tick": act_prefix_steps,
              "label_mask": "Accepted geometric-teacher commands only; all ACT-prefix and rejected-command ticks masked",
              "cross_embodiment_transfer_claim": False,
              "privileged_metrics_used_only_for_scoring_and_episode_selection": True,
              "source_sha256_before": before, "trials": []}
    write_json(out / "report.json", report, atomic=True)
    cache = {"train": [], "val": []}
    captured = {}

    def hook(_module, inputs):
        features = inputs[0]
        if features.shape != (1, chunk_size, feature_dim) or not torch.isfinite(features).all():
            raise ValueError("Native decoder returned invalid collection features")
        captured["features"] = features.detach().cpu().numpy()[0].copy()

    handle = policy.policy.model.action_head.register_forward_pre_hook(hook)
    shared = None
    try:
        shared = SharedEpisode("so101", out / "setups", source_scene=source_scene,
                               native_contract=policy.scene_contract)
        env = shared.env
        if env.control_hz != 25.:
            raise ValueError("Native weight-edit teacher collection requires 25 Hz control")
        if act_prefix_steps >= env.max_episode_steps:
            raise ValueError("ACT prefix must leave at least one geometric teacher tick")
        for partition, seeds in (("train", train_seeds), ("val", val_seeds)):
            for seed in seeds:
                directory = out / f"so101_{seed}"
                directory.mkdir()
                tracker = TrialTracker(success_hold_ticks=math.ceil(
                    env.task["episode"]["success_hold_s"] * env.control_hz),
                    approach_distance_m=.03, table_top_z=env.table_top,
                    ball_radius_m=env.task["ball"]["radius"])
                anchors, commands, accepted = [], [], []
                abort, initial, teacher = None, None, None
                started = time.perf_counter()
                try:
                    obs, info = shared.reset(seed)
                    initial = initial_fingerprint(shared, info)
                    teacher = StaticBowlRecovery(env, robot="so101")
                    teacher.bowl_memory = None
                    teacher.reset()
                    policy.reset()
                    rollin_driver = FrozenDriver(shared, rollin_policy) if rollin_policy is not None else None
                    with (directory / "ticks.jsonl").open("w") as stream:
                        for tick in range(env.max_episode_steps):
                            teacher_tick = tick >= act_prefix_steps
                            if collect_anchor_at(tick, stride, act_prefix_steps):
                                state = env.real_state()
                                images = {key: env.render(env.camera_bindings[key]) for key in policy.camera_keys}
                                captured.clear()
                                policy.predict_action_chunk(state, images)
                                if "features" not in captured:
                                    raise ValueError("ACT action_head did not provide decoder features")
                                anchors.append({"features": captured.pop("features"), "raw_state": state,
                                                "tick": tick, "episode_index": seed})
                            result = teacher.step(obs, env) if teacher_tick else rollin_driver.step(obs, env)
                            raw_command, valid_command = accepted_native_target(env, result)
                            valid_command = teacher_tick and valid_command
                            commands.append(raw_command)
                            accepted.append(valid_command)
                            obs = result.obs
                            done = tracker.update(result.info["privileged_metrics"],
                                                  saturation=result.saturation, ik_rejected=result.ik_rejected)
                            stream.write(json.dumps({"tick": tick, "time_s": tick / env.control_hz,
                                "uses_act": not teacher_tick, "assisted": teacher_tick,
                                "teacher_start_tick": act_prefix_steps, "detail": result.detail,
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
                                    feature_dim=feature_dim, mean=mean, std=std)
                np.savez_compressed(directory / "anchors.npz", **pack)
                row = tracker.row(experiment_id="ACT_WEIGHT_EDIT_TEACHER_COLLECTION",
                    controller_source="ACT_ROLLIN_RGB_GEOMETRIC_TEACHER" if rollin_policy is not None
                                      else "RGB_GEOMETRIC_TEACHER", robot="so101", seed=seed,
                    checkpoint_sha256=policy.weights_sha256, checkpoint_step=None,
                    mean_latency_ms=None, video_path="", abort=abort)
                included = bool(row["pick_place_success"]) and abort is None and bool(pack["valid"].any())
                if included:
                    cache[partition].extend(_records(pack))
                row.update(partition=partition, included_in_supervision=included,
                           initial_state=initial, anchors=len(anchors),
                           teacher_start_tick=act_prefix_steps,
                           rollin_checkpoint_sha256=report["rollin_checkpoint_sha256"],
                           valid_command_targets=int(pack["valid"].sum()),
                           wall_s=time.perf_counter() - started,
                           trace=str(directory / "ticks.jsonl"),
                           perception_log=teacher.perception_log if teacher is not None else [])
                write_json(directory / "report.json", row, atomic=True)
                report["trials"].append(row)
                for name, rows in cache.items():
                    np.savez_compressed(out / f"{name}.npz", **stack_records(
                        rows, chunk_size=chunk_size, feature_dim=feature_dim, action_dim=action_dim))
                report["supervised_anchors"] = {name: len(rows) for name, rows in cache.items()}
                report["summary"] = summarize(report["trials"])
                write_json(out / "report.json", report, atomic=True)
                print(json.dumps({"partition": partition, "seed": seed, "teacher_success": included,
                                  "anchors": len(anchors), "abort": abort}), flush=True)
        after = {path: sha256_file(path) for path in before}
        usable = all(cache.values())
        report.update(status="complete" if usable else "failed", source_sha256_after=after,
                      runtime_source_changes=[path for path in before if before[path] != after[path]],
                      checkpoint_unchanged=policy.weights_sha256 == sha256_file(snapshot / "model.safetensors"),
                      rollin_checkpoint_unchanged=rollin_policy is None or (
                          rollin_policy.weights_sha256 == sha256_file(rollin_snapshot / "model.safetensors")),
                      feature_files_sha256={name: sha256_file(out / name)
                                            for name in ("train.npz", "val.npz")},
                      failure=None if usable else "No successful teacher supervision in at least one split")
        write_json(out / "report.json", report, atomic=True)
        return report
    finally:
        handle.remove()
        if shared is not None:
            shared.close()


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
    args = parser.parse_args()
    collect(args.snapshot, args.out, train_seeds=args.train_seeds, val_seeds=args.val_seeds,
            stride=args.stride, source_scene=args.source_scene, inference_backend=args.inference_backend,
            rollin_snapshot=args.rollin_snapshot, act_prefix_steps=args.act_prefix_steps)


if __name__ == "__main__":
    main()
