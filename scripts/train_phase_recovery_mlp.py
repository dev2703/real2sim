"""Fit a phase-supervised recovery pilot using actual audit labels only.

Only packs containing expert_phase_id audits are accepted. Expert phases label
the loss; model inputs stay observable history, ACT proposal and ACT features.
The learned phase mixture supplies inference actions without a teacher FSM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from real2sim.artifacts import sha256_file
from real2sim.policy.phase_recovery_mlp import (
    N_CONTROL_PHASES,
    PHASE_NAMES,
    PhaseRecoveryMLP,
    phase_training_loss,
)
from real2sim.policy.recovery_mlp import ACTION_NAMES, ACTION_SCALES
from real2sim.serialization import write_json

if __package__:
    from scripts.train_recovery_mlp import SNAPSHOT, load_collections, robot_episode_weights
else:
    from train_recovery_mlp import SNAPSHOT, load_collections, robot_episode_weights


def load_phase_collections(directories, *, expected_base_sha256):
    data, manifests, sources = load_collections(directories, expected_base_sha256=expected_base_sha256)
    phases = {"train": [], "val": []}
    for directory, item in zip(directories, manifests, strict=True):
        if item["manifest"].get("expert_phase_is_audit_label_only") is not True:
            raise ValueError("Phase recovery requires actual audit-only expert phase labels")
        observation_names = item["manifest"].get("observation_names", []) or []
        if any(any(forbidden in name.lower() for forbidden in
                   ("expert_phase", "teacher_phase", "privileged", "ball_held", "task_success"))
               for name in observation_names):
            raise ValueError("Expert phase or privileged task state cannot be a recovery observation")
        for partition in phases:
            path = Path(directory) / f"{partition}.npz"
            with np.load(path, allow_pickle=False) as payload:
                if "expert_phase_id" not in payload:
                    raise ValueError("Collection has no expert phase audit; phases must not be invented")
                values = payload["expert_phase_id"]
                if (values.shape != (len(payload["observations"]),)
                        or not np.issubdtype(values.dtype, np.integer)
                        or np.any(values < 0) or np.any(values >= N_CONTROL_PHASES)):
                    raise ValueError("Expert phase audits must be nonterminal integer IDs")
                phases[partition].append(values.astype(np.int64))
    for partition in phases:
        data[partition]["expert_phase_id"] = np.concatenate(phases[partition])
    return data, manifests, sources


def evaluate(model, tensors, robots):
    with torch.inference_mode():
        prediction, logits, _ = model(tensors["observations"], tensors["base_actions"], tensors["features"])
        raw = (prediction - tensors["target_actions"]).abs().cpu().numpy()
        normalized = raw / np.asarray(ACTION_SCALES)
        predicted_phase = logits.argmax(-1).cpu().numpy()
        phases = tensors["expert_phase_id"].cpu().numpy()
    per_robot = {str(int(robot)): float(normalized[robots == robot].mean()) for robot in np.unique(robots)}
    return {"mean_absolute_normalized_error": float(normalized.mean()),
            "robot_balanced_mae_normalized": float(np.mean(list(per_robot.values()))),
            "per_robot_mae_normalized": per_robot,
            "translation_mae_mm": float(raw[:, :3].mean() * 1000),
            "rotation_mae_rad": float(raw[:, 3:6].mean()),
            "gripper_mae_fraction": float(raw[:, 6].mean()),
            "phase_accuracy": float((predicted_phase == phases).mean()),
            "phase_confusion": [[int(np.sum((phases == actual) & (predicted_phase == predicted)))
                for predicted in range(len(PHASE_NAMES))] for actual in range(len(PHASE_NAMES))]}


def train(directories, out, *, snapshot=SNAPSHOT, steps=4000, seed=42,
          phase_weight=.2, temperature=.5):
    if steps < 1 or not np.isfinite(phase_weight) or phase_weight <= 0:
        raise ValueError("Positive training steps and phase-loss weight required")
    if out.exists():
        raise FileExistsError(out)
    base_hash = sha256_file(snapshot / "model.safetensors")
    data, manifests, sources = load_phase_collections(directories, expected_base_sha256=base_hash)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    out.mkdir(parents=True)
    training = data["train"]
    model = PhaseRecoveryMLP(training["observations"].mean(0), training["observations"].std(0),
        training["features"].mean(0), training["features"].std(0), temperature=temperature)
    tensors = {partition: {name: torch.from_numpy(rows[name].astype(
        np.int64 if name == "expert_phase_id" else np.float32))
        for name in ("observations", "base_actions", "features", "target_actions", "expert_phase_id")}
        for partition, rows in data.items()}
    sampling = robot_episode_weights(training["robot_id"], training["episode_index"])
    counts = np.bincount(training["expert_phase_id"], minlength=N_CONTROL_PHASES)
    class_weights = np.zeros(N_CONTROL_PHASES, np.float32)
    seen = counts > 0
    class_weights[seen] = 1 / np.sqrt(counts[seen])
    class_weights[seen] /= class_weights[seen].mean()
    class_weights_tensor = torch.from_numpy(class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sources.update({str(path): sha256_file(path) for path in (
        Path(__file__), Path("src/real2sim/policy/phase_recovery_mlp.py"),
        Path("src/real2sim/policy/recovery_mlp.py"), Path("scripts/train_recovery_mlp.py"),
        snapshot / "model.safetensors")})
    report = {"schema": "frozen_act_phase_supervised_recovery_mlp_v1", "status": "training",
        "base_checkpoint_sha256": base_hash, "base_weights_frozen": True, "lighting_changed": False,
        "use_act": True, "architecture": "Shared MLP256->128, 11 phase logits, 11x7 residual actions; predicted soft phase mixture",
        "phase_names": PHASE_NAMES, "phase_temperature": model.temperature,
        "phase_loss_weight": phase_weight, "expert_phase_is_runtime_input": False,
        "expert_phase_used_only_for_training_loss": True,
        "no_runtime_geometric_teacher": True, "action_names": ACTION_NAMES,
        "action_scales": ACTION_SCALES, "observation_dim": model.observation_dim,
        "act_feature_dim": model.feature_dim,
        "observation_names": manifests[0]["manifest"].get("observation_names"),
        "train_phase_counts": {PHASE_NAMES[i]: int(n) for i, n in enumerate(counts)},
        "validation_phase_counts": {PHASE_NAMES[i]: int(np.sum(data["val"]["expert_phase_id"] == i))
                                    for i in range(len(PHASE_NAMES))},
        "validation_phases_absent_from_training": sorted(set(data["val"]["expert_phase_id"].tolist())
            - set(training["expert_phase_id"].tolist())),
        "train_rows": len(training["observations"]), "validation_rows": len(data["val"]["observations"]),
        "train_episodes": sorted(set(training["episode_index"].tolist())),
        "validation_episodes": sorted(set(data["val"]["episode_index"].tolist())),
        "selection": "Minimum mean of per-robot pooled-row action MAE on held-out episodes; task success remains a separate closed-loop test",
        "normalization_source": "Training rows only", "collections": manifests,
        "seed": seed, "steps": steps, "source_sha256_before": sources,
        "baseline_validation": evaluate(model, tensors["val"], data["val"]["robot_id"]), "history": []}
    write_json(out / "report.json", report, atomic=True)
    best, best_step, started = float("inf"), 0, time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        indices = torch.multinomial(sampling, min(256, len(sampling)), replacement=True)
        batch = {name: values[indices] for name, values in tensors["train"].items()}
        loss, parts = phase_training_loss(model, batch["observations"], batch["base_actions"],
            batch["features"], batch["target_actions"], batch["expert_phase_id"],
            phase_weight=phase_weight, class_weights=class_weights_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == steps:
            model.eval()
            metrics = evaluate(model, tensors["val"], data["val"]["robot_id"])
            entry = {"step": step, "training_loss": float(loss.detach()),
                     **{key: float(value.detach()) for key, value in parts.items()}, **metrics}
            report["history"].append(entry)
            if metrics["robot_balanced_mae_normalized"] < best:
                best, best_step = metrics["robot_balanced_mae_normalized"], step
                torch.save(model.state_dict(), out / "head.pt")
            write_json(out / "report.json", report, atomic=True)
            print(json.dumps({key: value for key, value in entry.items() if key != "phase_confusion"}), flush=True)
    after = {path: sha256_file(Path(path)) for path in sources}
    if sources != after:
        raise RuntimeError("Phase-recovery packs, source or frozen ACT weights changed")
    model.load_state_dict(torch.load(out / "head.pt", map_location="cpu", weights_only=True))
    model.eval()
    metrics = evaluate(model, tensors["val"], data["val"]["robot_id"])
    head_hash = sha256_file(out / "head.pt")
    identity = hashlib.sha256(f"{base_hash}:{head_hash}:predicted_phase_mixture".encode()).hexdigest()
    metadata = {"schema": report["schema"], "head_sha256": head_hash,
        "base_checkpoint_sha256": base_hash, "effective_policy_identifier": identity,
        "use_act": True, "observation_dim": model.observation_dim,
        "act_feature_dim": model.feature_dim, "hidden_sizes": list(model.hidden_sizes),
        "action_names": ACTION_NAMES, "action_scales": ACTION_SCALES,
        "observation_names": report["observation_names"], "phase_names": PHASE_NAMES,
        "phase_temperature": model.temperature, "expert_phase_is_runtime_input": False,
        "expert_phase_used_only_for_training_loss": True, "best_step": best_step,
        "base_weights_frozen": True, "lighting_changed": False,
        "validation_metrics": metrics, "training_report": str(out / "report.json")}
    write_json(out / "head.json", metadata, atomic=True)
    report.update(status="complete", best_step=best_step, validation_metrics=metrics,
        head=metadata, source_sha256_after=after, runtime_source_changes=[],
        training_wall_s=time.perf_counter() - started)
    write_json(out / "report.json", report, atomic=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--phase-weight", type=float, default=.2)
    parser.add_argument("--temperature", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.out, snapshot=args.snapshot, steps=args.steps,
          seed=args.seed, phase_weight=args.phase_weight, temperature=args.temperature)
