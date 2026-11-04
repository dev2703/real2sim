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
from real2sim.policy.phase_sequences import (
    consecutive_pairs,
    phase_transition_loss,
    sequence_metrics,
    temporal_phase_loss,
)
from real2sim.policy.recovery_mlp import ACTION_NAMES, ACTION_SCALES
from real2sim.serialization import write_json

if __package__:
    from scripts.run_recovery_dagger import snapshot_sources
    from scripts.train_recovery_mlp import (
        SNAPSHOT,
        load_collections,
        recovery_normalizers,
        robot_episode_weights,
    )
else:
    from run_recovery_dagger import snapshot_sources
    from train_recovery_mlp import (
        SNAPSHOT,
        load_collections,
        recovery_normalizers,
        robot_episode_weights,
    )


def load_phase_collections(directories, *, expected_base_sha256,
                           require_command_labels=False, require_sequences=False):
    data, manifests, sources = load_collections(directories, expected_base_sha256=expected_base_sha256)
    phases = {"train": [], "val": []}
    ticks, collection_ids = {"train": [], "val": []}, {"train": [], "val": []}
    for collection_id, (directory, item) in enumerate(zip(directories, manifests, strict=True)):
        if item["manifest"].get("expert_phase_is_audit_label_only") is not True:
            raise ValueError("Phase recovery requires actual audit-only expert phase labels")
        command_labels = item["manifest"].get("expert_phase_label_source") == "issued_command"
        if (require_command_labels or require_sequences) and not command_labels:
            raise ValueError("This experiment requires issued-command phase labels")
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
                saved_ticks = payload["tick"] if "tick" in payload else None
                if saved_ticks is not None and (saved_ticks.shape != values.shape
                        or not np.issubdtype(saved_ticks.dtype, np.integer)
                        or np.any(saved_ticks < 0)):
                    raise ValueError("Saved sequence ticks must be per-row nonnegative integers")
                if require_sequences and saved_ticks is None:
                    raise ValueError("Temporal supervision requires saved control ticks")
                # Missing ticks or the historical post-step phase convention
                # cannot establish temporal supervision; never invent pairs.
                ticks[partition].append(saved_ticks.astype(np.int64) if saved_ticks is not None
                                        and command_labels else np.full(len(values), -1, np.int64))
                collection_ids[partition].append(np.full(len(values), collection_id, np.int64))
    for partition in phases:
        data[partition]["expert_phase_id"] = np.concatenate(phases[partition])
        tick_values, groups = np.concatenate(ticks[partition]), np.concatenate(collection_ids[partition])
        eligible = np.flatnonzero(tick_values >= 0)
        rows = data[partition]
        pairs = consecutive_pairs(groups[eligible], rows["robot_id"][eligible].astype(np.int64),
            rows["episode_index"][eligible], tick_values[eligible])
        rows["consecutive_pairs"] = eligible[pairs]
        if require_sequences and not len(pairs):
            raise ValueError(f"No consecutive control-tick pairs in {partition}")
    return data, manifests, sources


def evaluate(model, tensors, robots, *, episodes=None, pairs=None):
    with torch.inference_mode():
        prediction, logits, components = model(tensors["observations"], tensors["base_actions"], tensors["features"])
        raw = (prediction - tensors["target_actions"]).abs().cpu().numpy()
        normalized = raw / np.asarray(ACTION_SCALES)
        predicted_phase = logits.argmax(-1).cpu().numpy()
        phases = tensors["expert_phase_id"].cpu().numpy()
        probabilities = torch.softmax(logits / model.temperature, dim=-1)
        soft = (probabilities[..., None] * components).sum(-2)
        hard = components.gather(-2, logits.argmax(-1)[..., None, None].expand(-1, 1, 7)).squeeze(-2)
        route_difference = (soft - hard).abs().cpu().numpy()
        hard_error = ((hard - tensors["target_actions"]).abs() / model.action_scales).mean().item()
    per_robot = {str(int(robot)): float(normalized[robots == robot].mean()) for robot in np.unique(robots)}
    balanced = (float(np.mean([np.mean([normalized[(robots == robot) & (episodes == episode)].mean()
        for episode in np.unique(episodes[robots == robot])]) for robot in np.unique(robots)]))
        if episodes is not None else float(np.mean(list(per_robot.values()))))
    return {"mean_absolute_normalized_error": float(normalized.mean()),
            "robot_balanced_mae_normalized": float(np.mean(list(per_robot.values()))),
            "robot_episode_balanced_mae_normalized": balanced,
            "per_robot_mae_normalized": per_robot,
            "translation_mae_mm": float(raw[:, :3].mean() * 1000),
            "rotation_mae_rad": float(raw[:, 3:6].mean()),
            "gripper_mae_fraction": float(raw[:, 6].mean()),
            "hard_component_mae_normalized": hard_error,
            "soft_hard_translation_difference_mm": float(route_difference[:, :3].mean() * 1000),
            "soft_hard_gripper_difference_fraction": float(route_difference[:, 6].mean()),
            "mean_max_phase_probability": float(probabilities.max(-1).values.mean()),
            "phase_accuracy": float((predicted_phase == phases).mean()),
            "phase_sequence_metrics": sequence_metrics(predicted_phase, phases,
                pairs if pairs is not None else np.empty((0, 2), np.int64)),
            "phase_confusion": [[int(np.sum((phases == actual) & (predicted_phase == predicted)))
                for predicted in range(len(PHASE_NAMES))] for actual in range(len(PHASE_NAMES))]}


def corrupt_action_history(observations, observation_names, *, noise=0., dropout=0., generator=None):
    """Corrupt only previous selected actions, preserving measured geometry.

    Noise is a fraction of the declared metre/radian/opening action scales.
    Entire seven-channel history frames may be dropped. This is augmentation,
    not a substitute for DAgger labels on actual learner-visited states.
    """
    if not np.isfinite([noise, dropout]).all() or noise < 0 or not 0 <= dropout < 1:
        raise ValueError("History noise must be nonnegative and dropout in [0,1)")
    if not noise and not dropout:
        return observations
    if observation_names is None or len(observation_names) != observations.shape[-1]:
        raise ValueError("Action-history augmentation requires declared observation names")
    groups = {}
    for index, name in enumerate(observation_names):
        for action_index, action_name in enumerate(ACTION_NAMES):
            suffixes = ("_previous_" + action_name,
                        "_previous_achieved_" + action_name) if action_index < 6 else (
                            "_previous_" + action_name, "_previous_issued_gripper_open_fraction")
            matched = next((suffix for suffix in suffixes if name.endswith(suffix)), None)
            if matched:
                groups.setdefault(name[:-len(matched)], {})[action_index] = index
                break
    if not groups or any(set(group) != set(range(7)) for group in groups.values()):
        raise ValueError("Action-history augmentation requires complete previous-action frames")
    corrupted = observations.clone()
    scale = torch.as_tensor(ACTION_SCALES, dtype=observations.dtype, device=observations.device)
    for frame in groups.values():
        columns = [frame[index] for index in range(7)]
        values = observations[..., columns].clone()
        if noise:
            values += torch.randn(values.shape, dtype=values.dtype, device=values.device,
                                  generator=generator) * scale * noise
        for part, limit in ((slice(0, 3), .03), (slice(3, 6), .3)):
            length = values[..., part].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            values[..., part] *= (limit / length).clamp_max(1.)
        values[..., 6] = values[..., 6].clamp(0., 1.)
        if dropout:
            keep = torch.rand((*values.shape[:-1], 1), device=values.device,
                              generator=generator) >= dropout
            values *= keep
        corrupted[..., columns] = values
    return corrupted


def drop_approach_timers(observations, observation_names, phase_ids, training_mean, *,
                         probability=0., generator=None):
    """Reduce approach-clock shortcuts without erasing close/release dwell.

    One training-only mask replaces all timer-history columns with their
    training means. Expert labels select augmentation rows, never runtime
    observations. Geometry, jaw feedback and non-approach rows stay unchanged.
    """
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Approach timer dropout must be in [0,1]")
    if not probability:
        return observations
    if observation_names is None or len(observation_names) != observations.shape[-1]:
        raise ValueError("Approach timer augmentation requires declared observation names")
    columns = [i for i, name in enumerate(observation_names)
               if name.endswith(("seconds_since_selected_gripper_switch", "seconds_since_issued_gripper_switch",
                                  "seconds_since_task_gripper_event"))]
    if len(columns) != 3 or phase_ids.shape != observations.shape[:-1]:
        raise ValueError("Approach timer augmentation requires three timers and per-row phase labels")
    selected = (phase_ids == PHASE_NAMES.index("PRE_GRASP")) | (phase_ids == PHASE_NAMES.index("DESCEND"))
    dropped = selected & (torch.rand(phase_ids.shape, device=observations.device,
                                    generator=generator) < probability)
    result = observations.clone()
    result[..., columns] = torch.where(dropped[..., None], training_mean[columns], observations[..., columns])
    return result


def train(directories, out, *, snapshot=SNAPSHOT, steps=4000, seed=42,
          phase_weight=.2, temperature=.5, temporal_weight=0., require_command_labels=False,
          use_act=True, action_routing="soft", history_noise=0., history_dropout=0.,
          transition_weight=0., residual_prior=True, approach_timer_dropout=0.):
    if (steps < 1 or not np.isfinite([phase_weight, temporal_weight, transition_weight]).all()
            or phase_weight <= 0 or min(temporal_weight, transition_weight) < 0):
        raise ValueError("Positive training steps and phase-loss weight required")
    if out.exists():
        raise FileExistsError(out)
    base_hash = sha256_file(snapshot / "model.safetensors")
    data, manifests, sources = load_phase_collections(directories, expected_base_sha256=base_hash,
        require_command_labels=require_command_labels,
        require_sequences=temporal_weight > 0 or transition_weight > 0)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    out.mkdir(parents=True)
    training = data["train"]
    observation_names = manifests[0]["manifest"].get("observation_names")
    model = PhaseRecoveryMLP(*recovery_normalizers(training, observation_names), temperature=temperature,
        use_act=use_act, action_routing=action_routing, residual_prior=residual_prior)
    tensors = {partition: {name: torch.from_numpy(rows[name].astype(
        np.int64 if name == "expert_phase_id" else np.float32))
        for name in ("observations", "base_actions", "features", "target_actions", "expert_phase_id")}
        for partition, rows in data.items()}
    # Validate augmentation before writing an apparently usable training run.
    history_generator = torch.Generator().manual_seed(seed + 2)
    corrupt_action_history(tensors["train"]["observations"][:1], observation_names,
        noise=history_noise, dropout=history_dropout, generator=history_generator)
    drop_approach_timers(tensors["train"]["observations"][:1], observation_names,
        tensors["train"]["expert_phase_id"][:1], model.observation_mean,
        probability=approach_timer_dropout, generator=history_generator)
    sampling = robot_episode_weights(training["robot_id"], training["episode_index"])
    pairs = training["consecutive_pairs"]
    pair_sampling = (robot_episode_weights(training["robot_id"][pairs[:, 0]],
        training["episode_index"][pairs[:, 0]]) if len(pairs) else None)
    pair_generator = torch.Generator().manual_seed(seed + 1)
    boundaries = pairs[training["expert_phase_id"][pairs[:, 0]] !=
                       training["expert_phase_id"][pairs[:, 1]]]
    if transition_weight and not len(boundaries):
        raise ValueError("Transition supervision requires observed command-phase boundaries")
    boundary_sampling = (robot_episode_weights(training["robot_id"][boundaries[:, 0]],
        training["episode_index"][boundaries[:, 0]]) if len(boundaries) else None)
    validation_options = {"episodes": data["val"]["episode_index"],
                          "pairs": data["val"]["consecutive_pairs"]}
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
        Path("src/real2sim/policy/phase_sequences.py"),
        Path("scripts/run_recovery_dagger.py"),
        snapshot / "model.safetensors")})
    executed_sources = snapshot_sources(out, sources)
    report = {"schema": "frozen_act_phase_supervised_recovery_mlp_v1", "status": "training",
        "base_checkpoint_sha256": base_hash, "base_weights_frozen": True, "lighting_changed": False,
        "use_act": use_act, "architecture": "Shared MLP256->128, 11 predicted phase logits and 11x7 action components; persisted soft/hard predicted routing with optional ACT residual prior",
        "phase_names": PHASE_NAMES, "phase_temperature": model.temperature,
        "phase_loss_weight": phase_weight, "expert_phase_is_runtime_input": False,
        "action_routing": model.action_routing,
        "residual_prior": model.residual_prior,
        "approach_timer_dropout": approach_timer_dropout,
        "history_action_noise_fraction": history_noise, "history_action_dropout": history_dropout,
        "history_augmentation_changes_measured_pose_or_rgb": False,
        "temporal_loss_weight": temporal_weight,
        "transition_loss_weight": transition_weight, "train_transition_pairs": len(boundaries),
        "require_issued_command_labels": require_command_labels or temporal_weight > 0 or transition_weight > 0,
        "temporal_supervision": "Within-collection/robot/episode consecutive ticks only; same-label phase stability and labelled action differences, including legitimate retries",
        "train_consecutive_pairs": len(pairs),
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
        "selection": "Minimum equal-robot then equal-episode normalized action MAE on held-out episodes; task success remains a separate closed-loop test",
        "normalization_source": "Training rows only", "collections": manifests,
        "binary_channel_normalization": "Declared robot/visibility indicators use mean 0 and scale 1",
        "training_robot_ids": sorted(set(training["robot_id"].tolist())),
        "seed": seed, "steps": steps, "source_sha256_before": sources,
        "executed_sources_manifest": executed_sources,
        "baseline_validation": evaluate(model, tensors["val"], data["val"]["robot_id"],
                                        **validation_options), "history": []}
    write_json(out / "report.json", report, atomic=True)
    best, best_step, started = float("inf"), 0, time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        indices = torch.multinomial(sampling, min(256, len(sampling)), replacement=True)
        batch = {name: values[indices] for name, values in tensors["train"].items()}
        batch["observations"] = corrupt_action_history(batch["observations"], observation_names,
            noise=history_noise, dropout=history_dropout, generator=history_generator)
        batch["observations"] = drop_approach_timers(batch["observations"], observation_names,
            batch["expert_phase_id"], model.observation_mean,
            probability=approach_timer_dropout, generator=history_generator)
        loss, parts = phase_training_loss(model, batch["observations"], batch["base_actions"],
            batch["features"], batch["target_actions"], batch["expert_phase_id"],
            phase_weight=phase_weight, class_weights=class_weights_tensor)
        if temporal_weight:
            pair_indices = torch.multinomial(pair_sampling, min(256, len(pairs)),
                                              replacement=True, generator=pair_generator)
            temporal, temporal_parts = temporal_phase_loss(model, tensors["train"],
                torch.from_numpy(pairs[pair_indices.numpy()]),
                balance_transitions=transition_weight > 0)
            loss = loss + temporal_weight * temporal
            parts.update(temporal_parts)
        if transition_weight:
            boundary_indices = torch.multinomial(boundary_sampling, min(256, len(boundaries)),
                                                  replacement=True, generator=pair_generator)
            transition = phase_transition_loss(model, tensors["train"],
                torch.from_numpy(boundaries[boundary_indices.numpy()]))
            loss = loss + transition_weight * transition
            parts["phase_transition_loss"] = transition
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == steps:
            model.eval()
            metrics = evaluate(model, tensors["val"], data["val"]["robot_id"], **validation_options)
            entry = {"step": step, "training_loss": float(loss.detach()),
                     **{key: float(value.detach()) for key, value in parts.items()}, **metrics}
            report["history"].append(entry)
            if metrics["robot_episode_balanced_mae_normalized"] < best:
                best, best_step = metrics["robot_episode_balanced_mae_normalized"], step
                torch.save(model.state_dict(), out / "head.pt")
            write_json(out / "report.json", report, atomic=True)
            print(json.dumps({key: value for key, value in entry.items() if key != "phase_confusion"}), flush=True)
    after = {path: sha256_file(Path(path)) for path in sources}
    if sources != after:
        raise RuntimeError("Phase-recovery packs, source or frozen ACT weights changed")
    model.load_state_dict(torch.load(out / "head.pt", map_location="cpu", weights_only=True))
    model.eval()
    metrics = evaluate(model, tensors["val"], data["val"]["robot_id"], **validation_options)
    head_hash = sha256_file(out / "head.pt")
    identity = hashlib.sha256(f"{base_hash}:{head_hash}:predicted_phase_mixture:{use_act}:{model.action_routing}:{model.residual_prior}".encode()).hexdigest()
    metadata = {"schema": report["schema"], "head_sha256": head_hash,
        "base_checkpoint_sha256": base_hash, "effective_policy_identifier": identity,
        "use_act": use_act, "observation_dim": model.observation_dim,
        "training_robot_ids": report["training_robot_ids"],
        "act_feature_dim": model.feature_dim, "hidden_sizes": list(model.hidden_sizes),
        "action_names": ACTION_NAMES, "action_scales": ACTION_SCALES,
        "observation_names": report["observation_names"], "phase_names": PHASE_NAMES,
        "phase_temperature": model.temperature, "expert_phase_is_runtime_input": False,
        "action_routing": model.action_routing,
        "residual_prior": model.residual_prior,
        "approach_timer_dropout": approach_timer_dropout,
        "history_action_noise_fraction": history_noise, "history_action_dropout": history_dropout,
        "temporal_loss_weight": temporal_weight,
        "transition_loss_weight": transition_weight,
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
    parser.add_argument("--temporal-weight", type=float, default=0.)
    parser.add_argument("--transition-weight", type=float, default=0.)
    parser.add_argument("--action-routing", choices=("soft", "hard"), default="soft")
    parser.add_argument("--history-noise", type=float, default=0.)
    parser.add_argument("--history-dropout", type=float, default=0.)
    parser.add_argument("--approach-timer-dropout", type=float, default=0.)
    parser.add_argument("--no-residual-prior", action="store_false", dest="residual_prior", default=True)
    parser.add_argument("--require-command-labels", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--use-act", action="store_true", dest="use_act", default=True)
    mode.add_argument("--no-act", action="store_false", dest="use_act")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.out, snapshot=args.snapshot, steps=args.steps,
          seed=args.seed, phase_weight=args.phase_weight, temperature=args.temperature,
          temporal_weight=args.temporal_weight, require_command_labels=args.require_command_labels,
          use_act=args.use_act, action_routing=args.action_routing,
          history_noise=args.history_noise, history_dropout=args.history_dropout,
          transition_weight=args.transition_weight, residual_prior=args.residual_prior,
          approach_timer_dropout=args.approach_timer_dropout)
