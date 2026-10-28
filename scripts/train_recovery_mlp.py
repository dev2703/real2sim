"""Fit a separate recovery MLP from immutable teacher/DAgger collection packs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from real2sim.artifacts import sha256_file
from real2sim.policy.recovery_mlp import ACTION_NAMES, ACTION_SCALES, RecoveryMLP
from real2sim.serialization import write_json

SNAPSHOT = Path(".cache/hf_models_20261027/models--bottowork--so101-act-h256-b16/snapshots/e696c8a078935432e3501e2c28c1fb2598fbff7e/latest")
ARRAY_NAMES = ("observations", "base_actions", "features", "target_actions", "robot_id", "episode_index")


def validate_rows(rows, *, where):
    if set(ARRAY_NAMES) - set(rows):
        raise ValueError(f"{where}: missing recovery arrays")
    n = len(rows["observations"])
    if not n or any(len(rows[name]) != n for name in ARRAY_NAMES):
        raise ValueError(f"{where}: empty or inconsistent recovery rows")
    for name in ("observations", "features", "base_actions", "target_actions"):
        if rows[name].ndim != 2 or not np.isfinite(rows[name]).all():
            raise ValueError(f"{where}: {name} must be a finite matrix")
    if any(rows[name].shape != (n, 7) for name in ("base_actions", "target_actions")):
        raise ValueError(f"{where}: common action matrices must have seven columns")
    if any(rows[name].shape != (n,) for name in ("robot_id", "episode_index")):
        raise ValueError(f"{where}: robot/episode IDs must be per-row vectors")
    if not np.isin(rows["robot_id"], [0, 1]).all():
        raise ValueError(f"{where}: robot ID must be SO101=0 or Panda=1")
    if not np.issubdtype(rows["episode_index"].dtype, np.integer):
        raise ValueError(f"{where}: episode IDs must be integers")


def load_collections(directories, *, expected_base_sha256):
    """Aggregate iteration packs without leaking held-out episodes into training."""
    collected, manifests, sources = {"train": [], "val": []}, [], {}
    dimension = None
    names = None
    evaluation_episodes = set()
    for directory in directories:
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete" or manifest.get("runtime_source_changes"):
            raise ValueError("Recovery collection must complete with unchanged sources")
        if manifest.get("base_checkpoint_sha256") != expected_base_sha256:
            raise ValueError("Recovery features belong to a different frozen ACT checkpoint")
        if manifest.get("lighting_changed") is not False or manifest.get("base_weights_frozen") is not True:
            raise ValueError("Recovery collection must preserve lighting and frozen ACT weights")
        action_names = manifest.get("action_names", ACTION_NAMES)
        if action_names != ACTION_NAMES:
            raise ValueError("Recovery collection common action semantics differ")
        scale = manifest.get("action_scales", ACTION_SCALES)
        if not np.allclose(scale, ACTION_SCALES, rtol=0, atol=1e-8):
            raise ValueError("Recovery collection common action scales differ")
        observation_names = manifest.get("observation_names")
        if names is None:
            names = observation_names
        elif observation_names != names:
            raise ValueError("Recovery observation names differ across DAgger iterations")
        evaluation_episodes.update(manifest.get("evaluation_seeds", []))
        evaluation_episodes.update(manifest.get("split", {}).get("evaluation", []))
        sources[str(manifest_path)] = sha256_file(manifest_path)
        for partition in collected:
            path = directory / f"{partition}.npz"
            with np.load(path, allow_pickle=False) as payload:
                rows = {name: payload[name] for name in ARRAY_NAMES if name in payload.files}
            validate_rows(rows, where=str(path))
            current = (rows["observations"].shape[1], rows["features"].shape[1])
            if dimension is None:
                dimension = current
            if current != dimension:
                raise ValueError("Recovery input dimensions differ across collection packs")
            if manifest.get("act_feature_dim", current[1]) != current[1]:
                raise ValueError("Frozen ACT feature metadata differs from the feature matrix")
            if observation_names is not None and len(observation_names) != current[0]:
                raise ValueError("Recovery observation names do not match the observation matrix")
            collected[partition].append(rows)
            sources[str(path)] = sha256_file(path)
        manifests.append({"directory": str(directory), "manifest": manifest})
    if not manifests:
        raise ValueError("Supply at least one recovery collection")
    merged = {partition: {name: np.concatenate([rows[name] for rows in groups], axis=0)
                         for name in ARRAY_NAMES}
              for partition, groups in collected.items()}
    train_episodes = set(merged["train"]["episode_index"].tolist())
    val_episodes = set(merged["val"]["episode_index"].tolist())
    if train_episodes & val_episodes:
        raise ValueError("Recovery train and validation episodes overlap across DAgger iterations")
    if (train_episodes | val_episodes) & evaluation_episodes:
        raise ValueError("Reserved evaluation episodes entered recovery training/validation")
    return merged, manifests, sources


def robot_episode_weights(robots, episodes):
    """Equal robot exposure, then equal episode exposure within each robot."""
    weights = np.zeros(len(robots), np.float64)
    unique_robots = np.unique(robots)
    for robot in unique_robots:
        robot_episodes = np.unique(episodes[robots == robot])
        for episode in robot_episodes:
            mask = (robots == robot) & (episodes == episode)
            weights[mask] = 1. / (len(unique_robots) * len(robot_episodes) * mask.sum())
    return torch.as_tensor(weights, dtype=torch.float64)


def evaluate(model, tensors, robots):
    with torch.inference_mode():
        prediction = model(tensors["observations"], tensors["base_actions"], tensors["features"])
        error = (prediction - tensors["target_actions"]).abs()
        normalized = error / model.action_scales
    values, raw = normalized.cpu().numpy(), error.cpu().numpy()
    return {"mean_absolute_normalized_error": float(values.mean()),
            "translation_mae_mm": float(raw[:, :3].mean() * 1000),
            "rotation_mae_rad": float(raw[:, 3:6].mean()),
            "gripper_mae_fraction": float(raw[:, 6].mean()),
            "per_robot_mae_normalized": {str(int(robot)): float(values[robots == robot].mean())
                                        for robot in np.unique(robots)}}


def train(directories, out, *, snapshot=SNAPSHOT, use_act=True, steps=3000,
          batch_size=256, learning_rate=5e-4, seed=42):
    if steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Training steps, batch size and learning rate must be positive")
    if out.exists():
        raise FileExistsError(out)
    base_hash = sha256_file(snapshot / "model.safetensors")
    data, manifests, sources = load_collections(directories, expected_base_sha256=base_hash)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_rows, val_rows = data["train"], data["val"]
    # All statistics are fitted to training rows alone, including ACT features.
    model = RecoveryMLP(train_rows["observations"].mean(0), train_rows["observations"].std(0),
        train_rows["features"].mean(0), train_rows["features"].std(0), use_act=use_act)
    tensors = {partition: {name: torch.from_numpy(rows[name].astype(np.float32))
                          for name in ("observations", "base_actions", "features", "target_actions")}
               for partition, rows in data.items()}
    weights = robot_episode_weights(train_rows["robot_id"], train_rows["episode_index"])
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=learning_rate, weight_decay=1e-4)
    sources.update({str(path): sha256_file(path) for path in (
        Path(__file__), Path("src/real2sim/policy/recovery_mlp.py"), snapshot / "model.safetensors")})
    report = {"schema": "frozen_act_cartesian_recovery_mlp_v1", "status": "training",
              "base_checkpoint_sha256": base_hash, "base_weights_frozen": True,
              "lighting_changed": False, "use_act": use_act,
              "architecture": "Separate normalized-input MLP256->128->7; residual from ACT prior or direct no-ACT ablation",
              "model_is_recovery_mlp": True, "dagger_is_collection_algorithm": True,
              "runtime_geometric_teacher_used": False,
              "observation_dim": model.observation_dim, "act_feature_dim": model.feature_dim,
              "observation_names": manifests[0]["manifest"].get("observation_names"),
              "action_names": ACTION_NAMES, "action_scales": ACTION_SCALES,
              "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
              "train_rows": len(train_rows["observations"]), "validation_rows": len(val_rows["observations"]),
              "train_episodes": sorted(set(train_rows["episode_index"].tolist())),
              "validation_episodes": sorted(set(val_rows["episode_index"].tolist())),
              "collections": manifests,
              "sampling": "Equal robot probability then equal source-episode probability within each robot",
              "checkpoint_selection": "Minimum pooled-row normalized action MAE on held-out episodes; closed-loop task performance evaluated separately",
              "normalization_source": "Training rows only",
              "seed": seed, "steps": steps, "batch_size": batch_size,
              "learning_rate": learning_rate, "source_sha256_before": sources,
              "baseline_validation": evaluate(model, tensors["val"], val_rows["robot_id"]),
              "history": []}
    write_json(out / "report.json", report, atomic=True)
    best, best_step, started = float("inf"), 0, time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        indices = torch.multinomial(weights, min(batch_size, len(weights)), replacement=True)
        batch = {name: values[indices] for name, values in tensors["train"].items()}
        prediction = model(batch["observations"], batch["base_actions"], batch["features"])
        loss = torch.nn.functional.smooth_l1_loss(prediction / model.action_scales,
            batch["target_actions"] / model.action_scales, beta=.1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == steps:
            model.eval()
            metrics = evaluate(model, tensors["val"], val_rows["robot_id"])
            entry = {"step": step, "training_loss": float(loss.detach()), **metrics}
            report["history"].append(entry)
            if metrics["mean_absolute_normalized_error"] < best:
                best, best_step = metrics["mean_absolute_normalized_error"], step
                torch.save(model.state_dict(), out / "head.pt")
            write_json(out / "report.json", report, atomic=True)
            print(json.dumps(entry), flush=True)
    after = {path: sha256_file(Path(path)) for path in sources}
    if after != sources:
        raise RuntimeError("Recovery training inputs, source or frozen ACT weights changed")
    head_hash = sha256_file(out / "head.pt")
    best_metrics = {key: value for key, value in next(
        entry for entry in report["history"] if entry["step"] == best_step).items()
        if key not in ("step", "training_loss")}
    policy_id = hashlib.sha256(f"{base_hash}:{head_hash}:{use_act}".encode()).hexdigest()
    metadata = {"schema": report["schema"], "head_sha256": head_hash,
                "base_checkpoint_sha256": base_hash, "effective_policy_identifier": policy_id,
                "use_act": use_act, "observation_dim": model.observation_dim,
                "observation_names": manifests[0]["manifest"].get("observation_names"),
                "act_feature_dim": model.feature_dim, "hidden_sizes": list(model.hidden_sizes),
                "action_names": ACTION_NAMES, "action_scales": ACTION_SCALES,
                "best_step": best_step, "validation_mae_normalized": best,
                "validation_metrics": best_metrics,
                "collection_iterations": [item["manifest"].get("iteration", 0) for item in manifests],
                "base_weights_frozen": True, "training_report": str(out / "report.json")}
    write_json(out / "head.json", metadata, atomic=True)
    report.update(status="complete", head=metadata, best_step=best_step,
                  validation_metrics=best_metrics,
                  best_validation_mae_normalized=best, training_wall_s=time.perf_counter() - started,
                  source_sha256_after=after, runtime_source_changes=[])
    write_json(out / "report.json", report, atomic=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--use-act", action="store_true", dest="use_act", default=True)
    mode.add_argument("--no-act", action="store_false", dest="use_act")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.out, snapshot=args.snapshot, use_act=args.use_act, steps=args.steps,
          batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed)
