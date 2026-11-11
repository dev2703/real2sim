"""Adapt native ACT attention and action weights from frozen encoder memories.

Only the existing decoder, its query embeddings and action head change. Cached
memories come from live training images through the unchanged backbone/encoder.
Inference remains the native ACT graph with no teacher or additional network.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTDecoder
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from real2sim.artifacts import sha256_file
from real2sim.policy.weight_edit import _scene_identity
from real2sim.serialization import write_json

FEATURE_LAYER = "decoder_memory"
PACK_NAMES = ("features", "encoder_memory", "encoder_positions", "target_normalized",
              "valid", "robot_id", "episode_index", "tick")


class NativeDecoder(nn.Module):
    """Replay the original native decoder using cached encoder output tokens."""

    def __init__(self, config, state):
        super().__init__()
        if config.get("type") != "act" or config.get("n_decoder_layers", 1) != 1:
            raise ValueError("Cached adaptation requires native ACT with one decoder layer")
        defaults = dict(dim_model=512, n_heads=8, dim_feedforward=3200,
                        feedforward_activation="relu", dropout=.1, pre_norm=False,
                        n_decoder_layers=1)
        native_config = SimpleNamespace(**{key: config.get(key, value) for key, value in defaults.items()})
        if native_config.feedforward_activation not in ("relu", "gelu"):
            raise ValueError("Native decoder adaptation supports relu and gelu")
        head = state.get("model.action_head.weight")
        query = state.get("model.decoder_pos_embed.weight")
        if (head is None or query is None or head.ndim != 2 or query.ndim != 2
                or head.shape[1] != native_config.dim_model
                or query.shape != (config["chunk_size"], native_config.dim_model)):
            raise ValueError("Checkpoint native decoder/head/query dimensions differ")
        if head.dtype not in (torch.float32, torch.float64):
            raise ValueError("Native decoder adaptation requires float32 or float64 tensors")
        with torch.random.fork_rng(devices=[]):
            self.decoder = ACTDecoder(native_config)
            self.decoder_pos_embed = nn.Embedding(config["chunk_size"], native_config.dim_model)
            self.action_head = nn.Linear(native_config.dim_model, head.shape[0])
        self.to(device="cpu", dtype=head.dtype)
        local = {}
        for name in self.state_dict():
            key = f"model.{name}"
            if key not in state or state[key].dtype != head.dtype or not torch.isfinite(state[key]).all():
                raise ValueError(f"Checkpoint is missing a finite native tensor: {key}")
            local[name] = state[key].detach().cpu()
        self.load_state_dict(local, strict=True)
        # Cached observations correspond to inference, so attention/FFN dropout
        # stays disabled while gradients still update the native parameters.
        self.eval()

    def forward(self, features, memory, positions):
        decoded = self.decoder(features.transpose(0, 1), memory.transpose(0, 1),
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            encoder_pos_embed=positions.transpose(0, 1))
        return self.action_head(decoded.transpose(0, 1))

    def edited_tensors(self):
        return {f"model.{name}": value.detach().cpu().clone()
                for name, value in self.state_dict().items()}


def validate_pack(pack, *, prefix=None):
    """Validate whole anchor memories and masks without flattening their queries."""
    arrays = {key: np.asarray(pack[key]) for key in PACK_NAMES}
    x, memory, positions, targets, valid = (arrays[key] for key in PACK_NAMES[:5])
    if (x.ndim != 3 or memory.ndim != 3 or positions.shape != memory.shape
            or targets.ndim != 3 or valid.shape != x.shape[:2] or valid.dtype != np.bool_
            or memory.shape[0] != len(x) or memory.shape[2] != x.shape[2]
            or targets.shape[:2] != x.shape[:2] or not len(x) or not memory.shape[1]):
        raise ValueError("Decoder cache needs matching anchor/query/token arrays and boolean masks")
    for key in PACK_NAMES[:4]:
        value = arrays[key]
        if value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"Cached {key} must be finite floating-point arrays")
    for key in PACK_NAMES[5:]:
        value = arrays[key]
        if value.shape != (len(x),) or value.dtype.kind not in "iu" or np.any(value < 0):
            raise ValueError(f"Cached {key} must be one nonnegative integer per anchor")
    if prefix is not None:
        if type(prefix) is not int or not 1 <= prefix <= x.shape[1]:
            raise ValueError("Prefix must fit the cached action chunk")
        arrays["valid"] = valid.copy()
        arrays["valid"][:, prefix:] = False
    usable = arrays["valid"].any(axis=1)
    if not usable.any():
        raise ValueError("Decoder cache has no valid supervised queries")
    return {key: value[usable] for key, value in arrays.items()}


def _tensors(pack, dtype):
    return {key: torch.as_tensor(value, dtype=(torch.bool if key == "valid" else dtype))
            for key, value in pack.items() if key in PACK_NAMES[:5]}


def _forward(model, pack, indices):
    return model(pack["features"][indices], pack["encoder_memory"][indices],
                 pack["encoder_positions"][indices])


def _metrics(model, pack, groups, batch_size):
    totals = {int(robot): [0., 0., 0] for robot in np.unique(groups)}
    action_sums = torch.zeros(model.action_head.out_features, dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, len(groups), batch_size):
            stop = min(start + batch_size, len(groups))
            error = _forward(model, pack, slice(start, stop)) - pack["target_normalized"][start:stop]
            mask = pack["valid"][start:stop]
            action_sums += error[mask].abs().double().sum(0)
            for robot in np.unique(groups[start:stop]):
                local = torch.from_numpy(groups[start:stop] == robot)
                values = error[local][mask[local]]
                totals[int(robot)][0] += float(values.abs().double().sum())
                totals[int(robot)][1] += float(values.square().double().sum())
                totals[int(robot)][2] += values.numel()
    per_robot = {str(robot): {"valid_queries": count // model.action_head.out_features,
                  "mae_normalized": absolute / count, "mse_normalized": square / count}
                 for robot, (absolute, square, count) in totals.items()}
    count = sum(row[2] for row in totals.values())
    return {"mae_normalized": sum(row[0] for row in totals.values()) / count,
        "mse_normalized": sum(row[1] for row in totals.values()) / count,
        "per_action_mae_normalized": (action_sums / (count / model.action_head.out_features)).tolist(),
        "per_robot": per_robot,
        "robot_balanced_mae_normalized": float(np.mean([row["mae_normalized"] for row in per_robot.values()]))}


def fit_decoder(state, config, train, validation, *, steps=1500, batch_size=8, lr=1e-4,
                l2=1., loss="l1", validation_every=50, patience=10, min_improvement=1e-6,
                seed=0, progress=False):
    """Train native decoder attention using robot-balanced, masked whole chunks."""
    for name, value in (("steps", steps), ("batch_size", batch_size),
                        ("validation_every", validation_every), ("patience", patience)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0 or loss not in ("l1", "mse"):
        raise ValueError("Use a nonnegative seed and l1 or mse action loss")
    if (not np.isfinite(lr) or lr <= 0 or not np.isfinite(l2) or l2 < 0
            or not np.isfinite(min_improvement) or min_improvement < 0):
        raise ValueError("Learning rate, L2 and improvement threshold must be finite and valid")
    train, validation = validate_pack(train), validate_pack(validation)
    groups, vgroups = train["robot_id"], validation["robot_id"]
    if not np.array_equal(np.unique(groups), np.unique(vgroups)):
        raise ValueError("Training and validation must cover the same robot IDs")
    model = NativeDecoder(config, state)
    expected = (config["chunk_size"], model.action_head.in_features)
    if any(pack["features"].shape[1:] != expected or pack["target_normalized"].shape[2] != model.action_head.out_features
           for pack in (train, validation)):
        raise ValueError("Cached feature/action dimensions differ from native checkpoint")
    dtype = model.action_head.weight.dtype
    train_data, val_data = _tensors(train, dtype), _tensors(validation, dtype)
    baseline = _metrics(model, val_data, vgroups, batch_size)
    baseline_train = _metrics(model, train_data, groups, batch_size)
    originals = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    count = sum(value.numel() for value in originals.values())
    probabilities = np.zeros(len(groups), np.float64)
    for robot in np.unique(groups):
        mask = groups == robot
        probabilities[mask] = 1. / (len(np.unique(groups)) * int(mask.sum()))
    probabilities = torch.from_numpy(probabilities)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_step, stale = model.edited_tensors(), 0, 0
    score = baseline["robot_balanced_mae_normalized"]
    history = [{"step": 0, "no_robot_regression": True, **baseline}]
    for step in range(1, steps + 1):
        indices = torch.multinomial(probabilities, batch_size, replacement=True, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        error = _forward(model, train_data, indices) - train_data["target_normalized"][indices]
        mask = train_data["valid"][indices]
        error = error.masked_fill(~mask.unsqueeze(-1), 0.)
        values = error.abs() if loss == "l1" else error.square()
        # Average each sampled anchor separately so long surviving chunks do
        # not change the declared equal-robot sampling weight.
        fit_loss = ((values * mask.unsqueeze(-1)).sum((1, 2))
                    / (mask.sum(1) * values.shape[2])).mean()
        penalty = sum((parameter - originals[name]).square().sum()
                      for name, parameter in model.named_parameters()) / count
        objective = fit_loss + l2 * penalty
        if not torch.isfinite(objective):
            raise ValueError("Native decoder adaptation produced non-finite loss")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        if step % validation_every and step != steps:
            continue
        metrics = _metrics(model, val_data, vgroups, batch_size)
        no_regression = all(row["mae_normalized"] <= baseline["per_robot"][robot]["mae_normalized"]
                            for robot, row in metrics["per_robot"].items())
        history.append({"step": step, "training_batch_loss": float(fit_loss.detach()),
                        "no_robot_regression": no_regression, **metrics})
        if no_regression and metrics["robot_balanced_mae_normalized"] < score - min_improvement:
            best, best_step, stale = model.edited_tensors(), step, 0
            score = metrics["robot_balanced_mae_normalized"]
        else:
            stale += 1
        if progress:
            print(json.dumps({"step": step, "best_step": best_step,
                "validation_mae": metrics["robot_balanced_mae_normalized"],
                "per_robot_validation_mae": {robot: row["mae_normalized"]
                    for robot, row in metrics["per_robot"].items()},
                "no_robot_regression": no_regression}), flush=True)
        if stale >= patience:
            break
    model.load_state_dict({key.removeprefix("model."): value for key, value in best.items()}, strict=True)
    delta = torch.cat([(tensor - state[key].cpu()).reshape(-1) for key, tensor in best.items()])
    norm = torch.cat([state[key].cpu().reshape(-1) for key in best]).norm()
    report = {"schema": "native_act_decoder_adaptation_v1", "accepted": best_step > 0,
        "feature_layer": FEATURE_LAYER, "robot_ids": np.unique(groups).tolist(),
        "selection": "minimum robot-balanced validation MAE with no robot baseline regression; baseline included",
        "train_anchors": len(groups), "validation_anchors": len(vgroups),
        "train_rows": int(train["valid"].sum()), "validation_rows": int(validation["valid"].sum()),
        "baseline_validation": baseline, "edited_validation": _metrics(model, val_data, vgroups, batch_size),
        "baseline_train": baseline_train, "edited_train": _metrics(model, train_data, groups, batch_size),
        "selected_step": best_step, "completed_steps": step, "requested_steps": steps,
        "batch_size": batch_size, "lr": lr, "l2": l2, "loss": loss,
        "validation_every": validation_every, "patience": patience, "seed": seed,
        "min_improvement": min_improvement, "edited_parameter_count": count,
        "delta_l2": float(delta.norm()), "delta_relative_l2": float(delta.norm() / norm.clamp_min(1e-12)),
        "trainable_tensor_names": sorted(best), "history": history,
        "runtime_teacher": False, "task_completion_established": False}
    return best, report


def _load_collection(path, source_files, base_sha, prefix):
    provenance = json.loads((path / "report.json").read_text())
    before, after = provenance.get("source_sha256_before"), provenance.get("source_sha256_after")
    if (provenance.get("status") != "complete" or provenance.get("checkpoint_unchanged") is not True
            or provenance.get("runtime_source_changes") != [] or not before or before != after):
        raise ValueError("Collection must be complete with unchanged checkpoint/runtime inputs")
    if provenance.get("weights_sha256") != base_sha or provenance.get("feature_layer") != FEATURE_LAYER:
        raise ValueError("Collection needs decoder_memory from this original checkpoint")
    for source in source_files:
        if before.get(str(source.resolve())) != sha256_file(source):
            raise ValueError(f"Checkpoint differs from feature collection: {source.name}")
    hashes = provenance.get("feature_files_sha256", {})
    if set(hashes) != {"train.npz", "val.npz"}:
        raise ValueError("Collection must checksum both feature packs")
    packs = []
    for name in ("train.npz", "val.npz"):
        if sha256_file(path / name) != hashes[name]:
            raise ValueError(f"Feature checksum mismatch: {name}")
        with np.load(path / name, allow_pickle=False) as data:
            packs.append(validate_pack({key: data[key] for key in PACK_NAMES}, prefix=prefix))
    return *packs, provenance


def fit_from_cache(data, out, snapshot, *, prefix=50, **options):
    """Fit immutable memory caches and export only the native decoder tensors."""
    paths = [Path(data)] if isinstance(data, (str, Path)) else [Path(path) for path in data]
    snapshot, out = Path(snapshot).expanduser().resolve(), Path(out)
    if not paths or out.exists():
        raise ValueError("Supply feature caches and a fresh output directory")
    if any(path.resolve() in out.resolve().parents for path in [snapshot, *paths]):
        raise ValueError("Export must be separate from checkpoint and feature caches")
    config = json.loads((snapshot / "config.json").read_text())
    if type(prefix) is not int or not 1 <= prefix <= config["chunk_size"]:
        raise ValueError("Prefix must fit the native action chunk")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if not (snapshot / name).is_file():
            raise ValueError(f"Native checkpoint is missing {name}")
    source_files = sorted(set(snapshot.glob("*.json")) | set(snapshot.glob("*.safetensors")))
    watched = [Path(__file__), Path(__file__).with_name("weight_edit.py"), *source_files,
               *(path / name for path in paths for name in ("report.json", "train.npz", "val.npz"))]
    before = {str(path.resolve()): sha256_file(path) for path in watched}
    base_sha = sha256_file(snapshot / "model.safetensors")
    caches = [_load_collection(path, source_files, base_sha, prefix) for path in paths]
    identities = {_scene_identity(cache[2]) for cache in caches}
    if len(identities) != 1:
        raise ValueError("Decoder collections must share the same source scene and render code")
    profiles = {json.dumps(cache[2].get("domain_randomization"), sort_keys=True) for cache in caches}
    if len(profiles) != 1:
        raise ValueError("Decoder collections must share the same domain randomization profile")
    target_kinds = {cache[2].get("target_kind", "applied") for cache in caches}
    if len(target_kinds) != 1:
        raise ValueError("Decoder collections must share the same target_kind supervision semantics")
    train, validation = ({key: np.concatenate([cache[index][key] for cache in caches], axis=0)
                          for key in PACK_NAMES} for index in (0, 1))
    episodes = [set(zip(pack["robot_id"].tolist(), pack["episode_index"].tolist(), strict=True))
                for pack in (train, validation)]
    if episodes[0] & episodes[1]:
        raise ValueError("Training and validation robot/episode identities overlap")
    state = load_file(snapshot / "model.safetensors", device="cpu")
    edits, report = fit_decoder(state, config, train, validation, **options)
    changed = [key for key, value in edits.items() if not torch.equal(value, state[key])]
    after = {path: sha256_file(path) for path in before}
    if before != after:
        raise RuntimeError("Adaptation inputs changed during fitting")
    report.update(status="complete", prefix=prefix, base_checkpoint=str(snapshot),
        base_checkpoint_sha256=base_sha, base_checkpoint_unchanged=True,
        source_sha256_before=before, source_sha256_after=after,
        source_scene=caches[0][2].get("source_scene"),
        domain_randomization=caches[0][2].get("domain_randomization"),
        target_kind=next(iter(target_kinds)),
        supervision_source=[cache[2].get("target_source") for cache in caches],
        feature_collections=[str(path.resolve()) for path in paths],
        train_episodes=[list(value) for value in sorted(episodes[0])],
        validation_episodes=[list(value) for value in sorted(episodes[1])],
        training_seeds=sorted({value[1] for value in episodes[0]}),
        validation_seeds=sorted({value[1] for value in episodes[1]}),
        changed_tensor_names=changed, hf_upload_performed=False)
    out.mkdir(parents=True)
    if report["accepted"]:
        for source in source_files:
            if source.name != "model.safetensors":
                shutil.copy2(source, out / source.name)
        with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        save_file({**state, **edits}, out / "model.safetensors", metadata=metadata)
        exported = load_file(out / "model.safetensors", device="cpu")
        actual = [key for key in state if not torch.equal(state[key], exported[key])]
        if set(state) != set(exported) or set(actual) != set(changed):
            raise RuntimeError("Native export changed tensors outside its declared decoder edit")
        report["edited_checkpoint_sha256"] = sha256_file(out / "model.safetensors")
        report["export_files_sha256"] = {path.name: sha256_file(path) for path in out.iterdir() if path.is_file()}
    else:
        report["edited_checkpoint_sha256"] = None
    write_json(out / "decoder_adaptation_report.json", report, atomic=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=1.)
    parser.add_argument("--loss", choices=("l1", "mse"), default="l1")
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="Print only the final adaptation report")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = fit_from_cache(args.data, args.out, args.snapshot, prefix=args.prefix,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, l2=args.l2, loss=args.loss,
        validation_every=args.validation_every, patience=args.patience, seed=args.seed,
        progress=not args.quiet)
    print(json.dumps({key: report[key] for key in (
        "accepted", "selected_step", "baseline_validation", "edited_validation")}))


if __name__ == "__main__":
    main()
