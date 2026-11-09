"""Fit a supervised edit directly into a native ACT action-head matrix.

Decoder features and desired actions come from a separately recorded simulation
teacher. Only training episodes determine the ridge solutions; held-out episodes
select their strength. The exported native checkpoint needs no runtime teacher
or residual network. A lower validation error is not evidence of task success.
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

from real2sim.artifacts import sha256_file
from real2sim.serialization import write_json

WEIGHT_KEY = "model.action_head.weight"
BIAS_KEY = "model.action_head.bias"
DEFAULT_RIDGES = (.01, .1, 1., 10.)
DEFAULT_ALPHAS = (0., .25, .5, .75, 1.)


def _matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return value


def _metric(x, y, weight, bias):
    error = x @ weight.T + bias - y
    return {"mae_normalized": float(np.abs(error).mean()),
            "per_action_mae_normalized": np.abs(error).mean(axis=0).tolist(),
            "mse_normalized": float(np.square(error).mean())}


def fit_weight_edit(weight, bias, train_features, train_targets,
                    validation_features, validation_targets, *, ridges=DEFAULT_RIDGES,
                    alphas=DEFAULT_ALPHAS, min_improvement=1e-6):
    """Return edited ``(weight, bias, report)``; rejected fits return the originals.

    Minimize mean squared residual error plus ``ridge * ||delta||²`` in
    training-standardized feature coordinates, including a regularized intercept.
    Candidate metrics use the export dtype so selection reflects saved weights.
    """
    original_weight = np.asarray(weight)
    original_bias = np.asarray(bias)
    if original_weight.dtype not in (np.float32, np.float64) or original_bias.dtype != original_weight.dtype:
        raise ValueError("head weight and bias must share float32 or float64 dtype")
    w = _matrix(original_weight, "weight")
    b = np.asarray(bias, dtype=np.float64)
    if b.shape != (w.shape[0],) or not np.isfinite(b).all():
        raise ValueError("bias must be finite and match the action dimension")
    x, y = _matrix(train_features, "train features"), _matrix(train_targets, "train targets")
    vx, vy = _matrix(validation_features, "validation features"), _matrix(validation_targets, "validation targets")
    if (x.shape[0] != y.shape[0] or vx.shape[0] != vy.shape[0]
            or x.shape[1] != w.shape[1] or vx.shape[1] != w.shape[1]
            or y.shape[1] != w.shape[0] or vy.shape[1] != w.shape[0]):
        raise ValueError("feature and target dimensions must match the action head")
    ridges, alphas = tuple(float(r) for r in ridges), tuple(float(a) for a in alphas)
    if not ridges or any(not np.isfinite(r) or r <= 0 for r in ridges):
        raise ValueError("ridge values must be finite and positive")
    if (not alphas or 0. not in alphas
            or any(not np.isfinite(a) or not 0 <= a <= 1 for a in alphas)):
        raise ValueError("edit strengths must include zero and lie in [0,1]")
    if not np.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("minimum improvement must be finite and nonnegative")
    mean, scale = x.mean(0), x.std(0).clip(.01)
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    residual = y - (x @ w.T + b)
    gram, rhs = design.T @ design / len(x), design.T @ residual / len(x)
    identity = np.eye(design.shape[1])
    baseline = _metric(vx, vy, w, b)
    best = baseline["mae_normalized"]
    selected_w, selected_b = original_weight.copy(), original_bias.copy()
    selected_ridge, selected_alpha = None, 0.
    candidates = []
    for ridge in ridges:
        solution = np.linalg.solve(gram + ridge * identity, rhs)
        delta_w = (solution[:-1] / scale[:, None]).T
        delta_b = solution[-1] - mean @ delta_w.T
        for alpha in alphas:
            candidate_w = (w + alpha * delta_w).astype(original_weight.dtype)
            candidate_b = (b + alpha * delta_b).astype(original_bias.dtype)
            if not np.isfinite(candidate_w).all() or not np.isfinite(candidate_b).all():
                raise ValueError("weight edit exceeds the checkpoint dtype")
            metrics = _metric(vx, vy, candidate_w, candidate_b)
            candidates.append({"ridge": ridge, "alpha": alpha, **metrics})
            if metrics["mae_normalized"] < best:
                best = metrics["mae_normalized"]
                selected_w, selected_b = candidate_w, candidate_b
                selected_ridge, selected_alpha = ridge, alpha
    improvement = baseline["mae_normalized"] - best
    accepted = selected_alpha > 0 and improvement > min_improvement
    if not accepted:
        selected_w, selected_b = original_weight.copy(), original_bias.copy()
        selected_ridge, selected_alpha = None, 0.
    delta = np.r_[(selected_w.astype(np.float64) - w).ravel(),
                  selected_b.astype(np.float64) - b]
    original_norm = float(np.linalg.norm(np.r_[w.ravel(), b]))
    report = {"accepted": accepted, "selection": "minimum held-out episode normalized action MAE; baseline included",
        "objective": "mean squared training residual + ridge times squared standardized head delta (including bias)",
        "train_rows": len(x), "validation_rows": len(vx),
        "feature_dimension": w.shape[1], "action_dimension": w.shape[0],
        "train_feature_std_floor": .01, "min_improvement": min_improvement,
        "baseline_validation": baseline,
        "edited_validation": _metric(vx, vy, selected_w, selected_b),
        "baseline_train": _metric(x, y, w, b),
        "edited_train": _metric(x, y, selected_w, selected_b),
        "selected_ridge": selected_ridge, "selected_alpha": selected_alpha,
        "validation_mae_improvement": baseline["mae_normalized"] - _metric(vx, vy, selected_w, selected_b)["mae_normalized"],
        "delta_l2": float(np.linalg.norm(delta)), "delta_max_abs": float(np.abs(delta).max()),
        "delta_relative_l2": float(np.linalg.norm(delta) / max(original_norm, 1e-12)),
        "edited_parameter_count": int(w.size + b.size), "candidates": candidates,
        "task_completion_established": False}
    return selected_w, selected_b, report


def load_feature_rows(path, prefix):
    """Flatten valid prefix queries and retain episode identities for split checks."""
    if type(prefix) is not int or prefix < 1:
        raise ValueError("prefix must be a positive integer")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in (
            "features", "target_normalized", "valid", "robot_id", "episode_index", "tick")}
    x, y, mask = arrays["features"], arrays["target_normalized"], arrays["valid"]
    if (x.ndim != 3 or y.ndim != 3 or x.shape[:2] != y.shape[:2]
            or mask.shape != x.shape[:2] or mask.dtype != np.bool_
            or not x.size or prefix > x.shape[1]):
        raise ValueError("feature pack requires matching nonempty [anchor,query,dimension] arrays and bool valid mask")
    if x.dtype.kind != "f" or y.dtype.kind != "f" or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("cached features and targets must be finite floating arrays")
    for key in ("robot_id", "episode_index", "tick"):
        values = arrays[key]
        if values.shape != (len(x),) or values.dtype.kind not in "iu":
            raise ValueError(f"{key} must be an integer per anchor")
    mask = mask[:, :prefix]
    if not mask.any():
        raise ValueError("feature pack contains no valid supervised prefix queries")
    identities = set(zip(arrays["robot_id"].tolist(), arrays["episode_index"].tolist(), strict=True))
    return x[:, :prefix][mask], y[:, :prefix][mask], identities


def _validated_cache(data, source_files, base_sha, prefix):
    """Check one immutable cache before aggregating any of its rows."""
    provenance = json.loads((data / "report.json").read_text())
    if (provenance.get("status") != "complete"
            or provenance.get("checkpoint_unchanged") is not True
            or provenance.get("runtime_source_changes") != []):
        raise ValueError("feature collection must be complete with unchanged checkpoint and runtime sources")
    if provenance.get("weights_sha256") != base_sha:
        raise ValueError("cached features belong to a different ACT checkpoint")
    recorded_before = provenance.get("source_sha256_before")
    recorded_after = provenance.get("source_sha256_after")
    if (not isinstance(recorded_before, dict) or not recorded_before
            or recorded_before != recorded_after):
        raise ValueError("collection must record identical before/after source checksums")
    for path in source_files:
        recorded = recorded_before.get(str(path.resolve()))
        if recorded != sha256_file(path):
            raise ValueError(f"checkpoint file differs from feature collection: {path.name}")
    feature_hashes = provenance.get("feature_files_sha256")
    if not isinstance(feature_hashes, dict) or set(feature_hashes) != {"train.npz", "val.npz"}:
        raise ValueError("collection must checksum exactly the training and validation feature packs")
    for name, expected in feature_hashes.items():
        if sha256_file(data / name) != expected:
            raise ValueError(f"feature pack checksum mismatch: {name}")
    x, y, train_episodes = load_feature_rows(data / "train.npz", prefix)
    vx, vy, val_episodes = load_feature_rows(data / "val.npz", prefix)
    return x, y, vx, vy, train_episodes, val_episodes, provenance


def fit_from_cache(data, out, snapshot, *, prefix=10, ridges=DEFAULT_RIDGES,
                   alphas=DEFAULT_ALPHAS, min_improvement=1e-6):
    """Fit one cache or aggregate a sequence of independently validated caches.

    Repeated robot/episode identities may appear across caches within the same
    role, for example after a corrective collection. Training and validation
    identities must remain disjoint across the entire aggregate.
    """
    data_paths = [Path(data)] if isinstance(data, (str, Path)) else [Path(path) for path in data]
    if not data_paths:
        raise ValueError("provide at least one feature cache")
    out, snapshot = Path(out), Path(snapshot)
    if out.exists():
        raise FileExistsError(out)
    if any(path.resolve() in out.resolve().parents for path in [snapshot, *data_paths]):
        raise ValueError("export directory must be separate from source checkpoint and features")
    weights_path = snapshot / "model.safetensors"
    base_sha = sha256_file(weights_path)
    required = ("config.json", "policy_preprocessor.json", "policy_postprocessor.json")
    for name in required:
        if not (snapshot / name).is_file():
            raise ValueError(f"native checkpoint is missing {name}")
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("type") != "act":
        raise ValueError("weight editing requires a native ACT checkpoint")
    if type(prefix) is not int or not 1 <= prefix <= config["chunk_size"]:
        raise ValueError("prefix must fit the native action chunk")
    source_files = sorted(set(snapshot.glob("*.json")) | set(snapshot.glob("*.safetensors")))
    watched = [Path(__file__), *source_files, *(path / name for path in data_paths
        for name in ("report.json", "train.npz", "val.npz"))]
    before = {str(path.resolve()): sha256_file(path) for path in watched}
    caches = [_validated_cache(path, source_files, base_sha, prefix) for path in data_paths]
    train_episodes = set().union(*(cache[4] for cache in caches))
    val_episodes = set().union(*(cache[5] for cache in caches))
    if train_episodes & val_episodes:
        raise ValueError("training and validation robot/episode identities overlap")
    if any(cache[index].shape[1] != caches[0][index].shape[1]
           for cache in caches for index in range(4)):
        raise ValueError("feature cache dimensions differ across collections")
    x, y, vx, vy = (np.concatenate([cache[index] for cache in caches], axis=0)
        for index in range(4))
    state = load_file(weights_path, device="cpu")
    if WEIGHT_KEY not in state or BIAS_KEY not in state:
        raise ValueError("checkpoint lacks the native linear ACT action head")
    weight, bias = state[WEIGHT_KEY], state[BIAS_KEY]
    edited_w, edited_b, report = fit_weight_edit(weight.numpy(), bias.numpy(), x, y, vx, vy,
        ridges=ridges, alphas=alphas, min_improvement=min_improvement)
    report.update(schema="native_act_direct_weight_edit_v1", status="complete",
        base_checkpoint=str(snapshot.resolve()), base_checkpoint_sha256=base_sha,
        data=str(data_paths[0].resolve()) if len(data_paths) == 1 else [str(path.resolve()) for path in data_paths],
        prefix=prefix,
        train_episodes=[list(pair) for pair in sorted(train_episodes)],
        validation_episodes=[list(pair) for pair in sorted(val_episodes)],
        source_sha256_before=before,
        supervision_source=caches[0][6].get("target_source", caches[0][6].get("supervision_source"))
            if len(caches) == 1 else [cache[6].get("target_source", cache[6].get("supervision_source"))
                for cache in caches],
        feature_collections=[{"data": str(path.resolve()), "train_rows": len(cache[0]),
            "validation_rows": len(cache[2]),
            "train_episodes": [list(pair) for pair in sorted(cache[4])],
            "validation_episodes": [list(pair) for pair in sorted(cache[5])],
            "feature_files_sha256": cache[6]["feature_files_sha256"]}
            for path, cache in zip(data_paths, caches, strict=True)],
        changed_tensor_names=[WEIGHT_KEY, BIAS_KEY] if report["accepted"] else [],
        runtime_teacher=False, hf_upload_performed=False)
    after = {path: sha256_file(path) for path in before}
    if before != after:
        raise RuntimeError("weight-edit inputs changed during fitting")
    report["source_sha256_after"] = after
    report["base_checkpoint_unchanged"] = base_sha == sha256_file(weights_path)
    out.mkdir(parents=True)
    if report["accepted"]:
        for path in source_files:
            if path.name != "model.safetensors":
                shutil.copy2(path, out / path.name)
        state[WEIGHT_KEY] = torch.from_numpy(edited_w.copy())
        state[BIAS_KEY] = torch.from_numpy(edited_b.copy())
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        save_file(state, out / "model.safetensors", metadata=metadata)
        report["edited_checkpoint_sha256"] = sha256_file(out / "model.safetensors")
        report["export_files_sha256"] = {path.name: sha256_file(path)
            for path in sorted(out.iterdir()) if path.is_file()}
    else:
        report["edited_checkpoint_sha256"] = None
    write_json(out / "weight_edit_report.json", report, atomic=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prefix", type=int, default=10)
    parser.add_argument("--ridges", type=float, nargs="+", default=DEFAULT_RIDGES)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    args = parser.parse_args()
    report = fit_from_cache(args.data, args.out, args.snapshot,
        prefix=args.prefix, ridges=args.ridges, alphas=args.alphas)
    print(json.dumps({key: report[key] for key in (
        "accepted", "baseline_validation", "edited_validation", "selected_ridge", "selected_alpha")}))


if __name__ == "__main__":
    main()
