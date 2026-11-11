"""Edit the final native ACT decoder feedforward block using cached inputs.

Only its two linear layers and the existing action head are optimized. Upstream
attention and both final layer norms stay frozen, so post-norm ACT can replay
this subgraph exactly from the last decoder linear1 input. Exports contain the
original native architecture and processors, with no extra runtime network or teacher.
Held-out action accuracy is separate from closed-loop task completion.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from real2sim.artifacts import sha256_file
from real2sim.policy.weight_edit import _groups, _matrix, _scene_identity, _validated_cache
from real2sim.serialization import write_json

FEATURE_LAYER = "decoder_ffn_input"


class DecoderTail(nn.Module):
    """Native post-norm final FFN, residual, two norms and action projection."""

    def __init__(self, config, state):
        super().__init__()
        if config.get("type") != "act" or config.get("pre_norm", False) is not False:
            raise ValueError("Decoder replay requires native ACT with pre_norm=False")
        layers = config.get("n_decoder_layers", 1)
        if type(layers) is not int or layers < 1:
            raise ValueError("ACT needs a positive number of decoder layers")
        activation = config.get("feedforward_activation", "relu")
        if activation not in ("relu", "gelu"):
            raise ValueError("Decoder replay supports relu and gelu activation")
        self.activation = activation
        prefix = f"model.decoder.layers.{layers - 1}"
        self.checkpoint_keys = {
            **{f"{layer}.{suffix}": f"{prefix}.{layer}.{suffix}"
               for layer in ("linear1", "linear2", "norm3") for suffix in ("weight", "bias")},
            **{f"decoder_norm.{suffix}": f"model.decoder.norm.{suffix}" for suffix in ("weight", "bias")},
            **{f"action_head.{suffix}": f"model.action_head.{suffix}" for suffix in ("weight", "bias")}}
        if any(key not in state for key in self.checkpoint_keys.values()):
            raise ValueError("Checkpoint is missing native decoder FFN, norms or action head")
        head = state["model.action_head.weight"]
        first = state[f"{prefix}.linear1.weight"]
        if head.ndim != 2 or first.ndim != 2 or head.shape[1] != first.shape[1]:
            raise ValueError("Native decoder and action head dimensions differ")
        actions, dimension = head.shape
        hidden = first.shape[0]
        if config.get("dim_model", dimension) != dimension or config.get("dim_feedforward", hidden) != hidden:
            raise ValueError("Saved ACT dimensions differ from native decoder tensors")
        if head.dtype not in (torch.float32, torch.float64):
            raise ValueError("Decoder fitting requires float32 or float64 checkpoint tensors")
        # Loading saved parameters immediately replaces these initial values.
        # Preserve callers' random state; fitting uses a separate local generator.
        with torch.random.fork_rng(devices=[]):
            self.linear1 = nn.Linear(dimension, hidden)
            self.linear2 = nn.Linear(hidden, dimension)
            self.norm3 = nn.LayerNorm(dimension)
            self.decoder_norm = nn.LayerNorm(dimension)
            self.action_head = nn.Linear(dimension, actions)
        self.to(dtype=head.dtype, device="cpu")
        local = {}
        for name, key in self.checkpoint_keys.items():
            tensor = state[key].detach().cpu()
            if tensor.dtype != head.dtype or not torch.isfinite(tensor).all():
                raise ValueError("Decoder replay parameters must share a finite checkpoint dtype")
            local[name] = tensor
        self.load_state_dict(local, strict=True)
        for module in (self.norm3, self.decoder_norm):
            module.requires_grad_(False)
        self.eval()

    @property
    def trainable_checkpoint_keys(self):
        return tuple(self.checkpoint_keys[name] for name, parameter in self.named_parameters()
                     if parameter.requires_grad)

    def forward(self, features):
        hidden = self.linear1(features)
        hidden = F.relu(hidden) if self.activation == "relu" else F.gelu(hidden)
        decoded = self.norm3(features + self.linear2(hidden))
        return self.action_head(self.decoder_norm(decoded))

    def edited_tensors(self):
        return {self.checkpoint_keys[name]: parameter.detach().cpu().clone()
                for name, parameter in self.named_parameters() if parameter.requires_grad}


def _metrics(model, features, targets, groups):
    with torch.inference_mode():
        errors = torch.cat([(model(features[start:start + 1024]) - targets[start:start + 1024]).abs()
                            for start in range(0, len(features), 1024)])
    result = {"mae_normalized": float(errors.mean()),
              "mse_normalized": float(errors.square().mean()),
              "per_action_mae_normalized": errors.mean(0).tolist(), "per_robot": {}}
    for robot in np.unique(groups):
        subset = errors[torch.from_numpy(groups == robot)]
        result["per_robot"][str(robot)] = {"rows": len(subset), "mae_normalized": float(subset.mean()),
            "mse_normalized": float(subset.square().mean()),
            "per_action_mae_normalized": subset.mean(0).tolist()}
    result["robot_balanced_mae_normalized"] = float(np.mean([
        value["mae_normalized"] for value in result["per_robot"].values()]))
    return result


def fit_decoder_edit(state, config, train_features, train_targets,
                     validation_features, validation_targets, *, train_groups=None,
                     validation_groups=None, steps=600, batch_size=128, lr=3e-4,
                     l2=1., validation_every=25, patience=10, min_improvement=1e-6, seed=0,
                     loss="mse"):
    """Fit native tensors with balanced robots and baseline-preserving selection.

    Dropout is disabled exactly as in native inference. Training examples alone
    determine Adam updates; validation selects the checkpoint, including step 0.
    A robot's validation MAE may never regress against that original checkpoint.
    """
    for name, value in (("steps", steps), ("batch_size", batch_size),
                        ("validation_every", validation_every), ("patience", patience)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(lr) or lr <= 0 or not np.isfinite(l2) or l2 < 0:
        raise ValueError("Learning rate must be finite/positive and L2 finite/nonnegative")
    if not np.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("minimum improvement must be finite and nonnegative")
    if loss not in ("mse", "l1"):
        raise ValueError("loss must be mse or l1")
    x, y = _matrix(train_features, "training features"), _matrix(train_targets, "training targets")
    vx, vy = _matrix(validation_features, "validation features"), _matrix(validation_targets, "validation targets")
    model = DecoderTail(config, state)
    dimension, actions = model.linear1.in_features, model.action_head.out_features
    if (x.shape != (len(y), dimension) or vx.shape != (len(vy), dimension)
            or y.shape[1] != actions or vy.shape[1] != actions):
        raise ValueError("Cached decoder feature and target dimensions differ from native tensors")
    if (train_groups is None) != (validation_groups is None):
        raise ValueError("Supply training and validation robot groups together")
    train_groups = _groups(np.zeros(len(x), np.int64) if train_groups is None else train_groups,
                           len(x), "training groups")
    validation_groups = _groups(np.zeros(len(vx), np.int64) if validation_groups is None else validation_groups,
                                len(vx), "validation groups")
    robots = np.unique(train_groups)
    if not np.array_equal(robots, np.unique(validation_groups)):
        raise ValueError("Training and validation must cover the same robot IDs")
    dtype = model.action_head.weight.dtype
    x, y, vx, vy = (torch.as_tensor(array, dtype=dtype) for array in (x, y, vx, vy))
    baseline = _metrics(model, vx, vy, validation_groups)
    baseline_train = _metrics(model, x, y, train_groups)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    originals = [parameter.detach().clone() for parameter in parameters]
    parameter_count = sum(parameter.numel() for parameter in parameters)
    best_tensors = model.edited_tensors()
    best_score = baseline["robot_balanced_mae_normalized"]
    best_step, stale, stopped_step = 0, 0, 0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probabilities = np.zeros(len(x), np.float64)
    for robot in robots:
        mask = train_groups == robot
        probabilities[mask] = 1. / (len(robots) * int(mask.sum()))
    probabilities = torch.from_numpy(probabilities)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    history = [{"step": 0, "no_robot_regression": True, **baseline}]
    for step in range(1, steps + 1):
        indices = torch.multinomial(probabilities, batch_size, replacement=True, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x[indices])
        fit_loss = (F.mse_loss(prediction, y[indices]) if loss == "mse"
                    else F.l1_loss(prediction, y[indices]))
        penalty = sum((parameter - original).square().sum()
                      for parameter, original in zip(parameters, originals, strict=True)) / parameter_count
        objective = fit_loss + l2 * penalty
        if not torch.isfinite(objective):
            raise ValueError("Decoder optimization produced a non-finite loss")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
        optimizer.step()
        stopped_step = step
        if step % validation_every and step != steps:
            continue
        metrics = _metrics(model, vx, vy, validation_groups)
        no_regression = all(value["mae_normalized"] <= baseline["per_robot"][robot]["mae_normalized"]
                            for robot, value in metrics["per_robot"].items())
        history.append({"step": step, "training_batch_loss": float(fit_loss.detach()),
            "delta_mean_square": float(penalty.detach()), "no_robot_regression": no_regression, **metrics})
        score = metrics["robot_balanced_mae_normalized"]
        if no_regression and score < best_score - min_improvement:
            best_score, best_step, stale = score, step, 0
            best_tensors = model.edited_tensors()
        else:
            stale += 1
        if stale >= patience:
            break
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_tensors[model.checkpoint_keys[name]])
    deltas = torch.cat([(best_tensors[key] - state[key].cpu()).reshape(-1)
                        for key in model.trainable_checkpoint_keys])
    original_flat = torch.cat([state[key].cpu().reshape(-1) for key in model.trainable_checkpoint_keys])
    report = {"schema": "native_act_decoder_weight_edit_v1", "accepted": best_step > 0,
        "feature_layer": FEATURE_LAYER, "selection": "minimum robot-balanced held-out action MAE with no robot baseline regression; original tensors included",
        "objective": f"robot-balanced {'mean squared' if loss == 'mse' else 'mean absolute'} action error + L2 times mean squared native tensor delta",
        "robot_ids": robots.tolist(), "train_rows": len(x), "validation_rows": len(vx),
        "baseline_validation": baseline, "edited_validation": _metrics(model, vx, vy, validation_groups),
        "baseline_train": baseline_train, "edited_train": _metrics(model, x, y, train_groups),
        "selected_step": best_step, "completed_steps": stopped_step, "requested_steps": steps,
        "batch_size": batch_size, "lr": lr, "l2": l2, "loss": loss, "validation_every": validation_every,
        "patience": patience, "seed": seed, "min_improvement": min_improvement,
        "edited_parameter_count": parameter_count, "trainable_tensor_names": list(model.trainable_checkpoint_keys),
        "delta_l2": float(deltas.norm()), "delta_max_abs": float(deltas.abs().max()),
        "delta_relative_l2": float(deltas.norm() / original_flat.norm().clamp_min(1e-12)),
        "history": history, "runtime_teacher": False, "task_completion_established": False}
    return best_tensors, report


def fit_from_cache(data, out, snapshot, *, prefix=10, **options):
    """Validate immutable decoder caches and export only changed native tensors."""
    paths = [Path(data)] if isinstance(data, (str, Path)) else [Path(path) for path in data]
    if not paths:
        raise ValueError("Supply at least one decoder feature cache")
    snapshot, out = Path(snapshot).expanduser().resolve(), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if any(path.resolve() in out.resolve().parents for path in [snapshot, *paths]):
        raise ValueError("Export directory must be separate from checkpoint and features")
    config = json.loads((snapshot / "config.json").read_text())
    if type(prefix) is not int or not 1 <= prefix <= config["chunk_size"]:
        raise ValueError("Prefix must fit the native action chunk")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if not (snapshot / name).is_file():
            raise ValueError(f"Native checkpoint is missing {name}")
    source_files = sorted(set(snapshot.glob("*.json")) | set(snapshot.glob("*.safetensors")))
    watched = [Path(__file__), *source_files,
               *(path / name for path in paths for name in ("report.json", "train.npz", "val.npz"))]
    before = {str(path.resolve()): sha256_file(path) for path in watched}
    base_sha = sha256_file(snapshot / "model.safetensors")
    caches = [_validated_cache(path, source_files, base_sha, prefix) for path in paths]
    if any(cache[6].get("feature_layer") != FEATURE_LAYER for cache in caches):
        raise ValueError("Decoder fitting requires feature_layer=decoder_ffn_input caches")
    if len({_scene_identity(cache[6]) for cache in caches}) > 1:
        raise ValueError("Feature caches must declare the same source scene")
    train_episodes = set().union(*(cache[4] for cache in caches))
    val_episodes = set().union(*(cache[5] for cache in caches))
    if train_episodes & val_episodes:
        raise ValueError("Training and validation robot/episode identities overlap")
    x, y, vx, vy, groups, vgroups = (np.concatenate([cache[index] for cache in caches])
                                    for index in (0, 1, 2, 3, 7, 8))
    state = load_file(snapshot / "model.safetensors", device="cpu")
    edits, report = fit_decoder_edit(state, config, x, y, vx, vy,
                                     train_groups=groups, validation_groups=vgroups, **options)
    changed = [key for key, tensor in edits.items() if not torch.equal(tensor, state[key])]
    if not set(changed) <= set(report["trainable_tensor_names"]):
        raise RuntimeError("Decoder edit changed tensors outside its declared whitelist")
    after = {path: sha256_file(path) for path in before}
    if before != after:
        raise RuntimeError("Decoder edit inputs changed during fitting")
    report.update(status="complete", prefix=prefix, base_checkpoint=str(snapshot),
        base_checkpoint_sha256=base_sha, base_checkpoint_unchanged=True,
        feature_collections=[str(path.resolve()) for path in paths],
        train_episodes=[list(pair) for pair in sorted(train_episodes)],
        validation_episodes=[list(pair) for pair in sorted(val_episodes)],
        source_sha256_before=before, source_sha256_after=after,
        changed_tensor_names=changed, hf_upload_performed=False)
    out.mkdir(parents=True)
    if report["accepted"]:
        for path in source_files:
            if path.name != "model.safetensors":
                shutil.copy2(path, out / path.name)
        exported = {**state, **edits}
        with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        save_file(exported, out / "model.safetensors", metadata=metadata)
        reloaded = load_file(out / "model.safetensors", device="cpu")
        actual = [key for key in state if not torch.equal(state[key], reloaded[key])]
        if set(actual) != set(changed) or set(reloaded) != set(state):
            raise RuntimeError("Native decoder export changed unexpected checkpoint tensors")
        report["edited_checkpoint_sha256"] = sha256_file(out / "model.safetensors")
        report["export_files_sha256"] = {path.name: sha256_file(path)
            for path in sorted(out.iterdir()) if path.is_file()}
    else:
        report["edited_checkpoint_sha256"] = None
    write_json(out / "decoder_weight_edit_report.json", report, atomic=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", type=int, default=10)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2", type=float, default=1.)
    parser.add_argument("--loss", choices=("mse", "l1"), default="mse")
    parser.add_argument("--validation-every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = fit_from_cache(args.data, args.out, args.snapshot, prefix=args.prefix,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, l2=args.l2,
        validation_every=args.validation_every, patience=args.patience, seed=args.seed, loss=args.loss)
    print(json.dumps({key: report[key] for key in (
        "accepted", "baseline_validation", "edited_validation", "selected_step")}))


if __name__ == "__main__":
    main()
