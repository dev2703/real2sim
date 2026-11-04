"""Train a separate intervention gate from episode-disjoint recovery labels.

Expert disagreement labels exist only in these training packs. The gate cannot
infer when ACT is sufficient from exclusively positive labels; such a data set
is retained and reported as conservative always-on recovery, not useful gating.
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
from real2sim.policy.recovery_gate import (
    LABEL_THRESHOLDS,
    InterventionGateMLP,
    calibrate_threshold,
    classification_metrics,
    correction_needed,
)
from real2sim.policy.recovery_mlp import RecoveryMLPPolicy
from real2sim.serialization import write_json

if __package__:
    from scripts.train_recovery_mlp import SNAPSHOT, load_collections, robot_episode_weights
else:
    from train_recovery_mlp import SNAPSHOT, load_collections, robot_episode_weights


def probabilities(model, tensors):
    with torch.inference_mode():
        return torch.sigmoid(model(tensors["observations"], tensors["base_actions"],
                                   tensors["features"])).cpu().numpy()


def train(directories, recovery_head, out, *, snapshot=SNAPSHOT, steps=1000,
          minimum_recall=.995, seed=42):
    if steps < 1 or not 0 < minimum_recall <= 1:
        raise ValueError("Positive training steps and recall target in0..1 required")
    if out.exists():
        raise FileExistsError(out)
    base_hash = sha256_file(snapshot / "model.safetensors")
    recovery = RecoveryMLPPolicy(recovery_head, expected_base_sha256=base_hash)
    if not recovery.metadata["use_act"]:
        raise ValueError("ACT intervention gating requires an ACT-enabled recovery head")
    data, manifests, sources = load_collections(directories, expected_base_sha256=base_hash)
    train_rows, val_rows = data["train"], data["val"]
    if (train_rows["observations"].shape[1] != recovery.metadata["observation_dim"]
            or train_rows["features"].shape[1] != recovery.metadata["act_feature_dim"]):
        raise ValueError("Gate training inputs differ from the bound recovery head")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    out.mkdir(parents=True)
    model = InterventionGateMLP(train_rows["observations"].mean(0), train_rows["observations"].std(0),
        train_rows["features"].mean(0), train_rows["features"].std(0))
    tensors = {part: {name: torch.from_numpy(rows[name].astype(np.float32))
                     for name in ("observations", "base_actions", "features")}
               for part, rows in data.items()}
    labels = {part: correction_needed(rows["base_actions"], rows["target_actions"])
              for part, rows in data.items()}
    training_labels = torch.from_numpy(labels["train"].astype(np.float32))
    weights = robot_episode_weights(train_rows["robot_id"], train_rows["episode_index"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sources.update({str(path): sha256_file(path) for path in (
        Path(__file__), Path("src/real2sim/policy/recovery_gate.py"),
        Path("src/real2sim/policy/recovery_mlp.py"), Path("scripts/train_recovery_mlp.py"),
        recovery_head / "head.json", recovery_head / "head.pt", snapshot / "model.safetensors")})
    report = {"schema": "frozen_act_recovery_intervention_gate_v1", "status": "training",
        "base_checkpoint_sha256": base_hash, "recovery_head_sha256": recovery.head_sha256,
        "base_weights_frozen": True, "recovery_head_weights_frozen": True,
        "lighting_changed": False, "no_runtime_expert_actions": True,
        "label_rule": LABEL_THRESHOLDS,
        "label_scope": "Need for correction means expert/ACT action disagreement, not measured task failure or counterfactual proof that ACT would fail.",
        "architecture": "Observable-input MLP64->32->1; sigmoid intervention probability",
        "observation_dim": model.observation_dim, "act_feature_dim": model.feature_dim,
        "observation_names": manifests[0]["manifest"].get("observation_names"),
        "normalization_source": "Training rows only", "seed": seed, "steps": steps,
        "training_labels": {"n": len(labels["train"]), "positive": int(labels["train"].sum()),
                            "negative": int((~labels["train"]).sum())},
        "validation_labels": {"n": len(labels["val"]), "positive": int(labels["val"].sum()),
                              "negative": int((~labels["val"]).sum())},
        "train_episodes": sorted(set(train_rows["episode_index"].tolist())),
        "validation_episodes": sorted(set(val_rows["episode_index"].tolist())),
        "collections": manifests, "checkpoint_selection": "Minimum held-out episode Brier score; threshold selected for held-out positive recall per robot",
        "source_sha256_before": sources, "history": []}
    write_json(out / "report.json", report, atomic=True)
    best, best_step, started = float("inf"), 0, time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        indices = torch.multinomial(weights, min(256, len(weights)), replacement=True)
        batch = {name: values[indices] for name, values in tensors["train"].items()}
        logits = model(batch["observations"], batch["base_actions"], batch["features"])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, training_labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == steps:
            model.eval()
            probs = probabilities(model, tensors["val"])
            metrics = classification_metrics(probs, labels["val"], .5)
            entry = {"step": step, "training_loss": float(loss.detach()), **metrics}
            report["history"].append(entry)
            if metrics["brier_score"] < best:
                best, best_step = metrics["brier_score"], step
                torch.save(model.state_dict(), out / "gate.pt")
            write_json(out / "report.json", report, atomic=True)
            print(json.dumps(entry), flush=True)
    model.load_state_dict(torch.load(out / "gate.pt", map_location="cpu", weights_only=True))
    model.eval()
    validation_probabilities = probabilities(model, tensors["val"])
    calibration = calibrate_threshold(validation_probabilities, labels["val"], labels["train"],
        minimum_recall=minimum_recall, robots=val_rows["robot_id"])
    metrics = classification_metrics(validation_probabilities, labels["val"], calibration["threshold"])
    after = {path: sha256_file(Path(path)) for path in sources}
    if sources != after:
        raise RuntimeError("Gate inputs, source, ACT or recovery weights changed during fitting")
    gate_hash = sha256_file(out / "gate.pt")
    identity = hashlib.sha256(f"{base_hash}:{recovery.head_sha256}:{gate_hash}:{calibration}".encode()).hexdigest()
    metadata = {"schema": report["schema"], "gate_sha256": gate_hash,
        "base_checkpoint_sha256": base_hash, "recovery_head_sha256": recovery.head_sha256,
        "effective_policy_identifier": identity, "hidden_sizes": list(model.hidden_sizes),
        "observation_dim": model.observation_dim, "act_feature_dim": model.feature_dim,
        "observation_names": report["observation_names"], "calibration": calibration,
        "label_rule": LABEL_THRESHOLDS, "best_step": best_step,
        "validation_metrics": metrics, "training_report": str(out / "report.json")}
    write_json(out / "gate.json", metadata, atomic=True)
    report.update(status="complete", best_step=best_step, calibration=calibration,
        validation_metrics=metrics,
        per_robot_validation={str(int(robot)): classification_metrics(
            validation_probabilities[val_rows["robot_id"] == robot],
            labels["val"][val_rows["robot_id"] == robot], calibration["threshold"])
            for robot in np.unique(val_rows["robot_id"])},
        gate=metadata, source_sha256_after=after, runtime_source_changes=[],
        training_wall_s=time.perf_counter() - started)
    write_json(out / "report.json", report, atomic=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--recovery-head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--minimum-recall", type=float, default=.995)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.recovery_head, args.out, snapshot=args.snapshot, steps=args.steps,
          minimum_recall=args.minimum_recall, seed=args.seed)
