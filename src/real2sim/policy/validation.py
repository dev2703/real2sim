"""Common teacher-forced metrics at the execution cadence, never task-success evidence.

Predictions are raw, before controller clipping. Only train/validation episodes may
be consumed. Every valid action must occur once in a nonoverlapping executed prefix.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from real2sim.geometry.so3 import exp_so3, log_so3, matrix_from_rot6d

VERSION = "executed_prefix_motion_v1"
DELTA_KINDS = ("cartesian", "cartesian_tool")
MOTION_MM = .4
MOTION_RAD = .01
MOTION_DEG = .5


def _canonical(arrays):
    return arrays.get("canonical_state", arrays["state"])[:, :13]


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def action_metrics(pred, label, *, action_kind, std=None):
    """Physical errors; absolute poses are never treated as delta motion."""
    pred, label = np.asarray(pred), np.asarray(label)
    if pred.shape != label.shape or pred.ndim != 2 or not np.isfinite(pred).all() or not np.isfinite(label).all():
        raise ValueError("finite aligned action matrices required")
    if not len(pred):
        return {"n": 0}
    error = pred - label
    result = {"n": len(pred), "mae_per_dimension": np.abs(error).mean(0).tolist(),
              "gripper_mae": float(np.abs(error[:, -1]).mean() / (100 if action_kind == "canonical_joint" else 1)),
              "gripper_unit": "fraction of calibrated range"}
    if std is not None:
        result["l1"] = float(np.abs(error / std).mean())
    if action_kind == "canonical_joint":
        result.update(arm_joint_mae_deg=np.abs(error[:, :5]).mean(0).tolist(),
                      arm_joint_vector_error_mean_deg=float(np.linalg.norm(error[:, :5], axis=1).mean()),
                      raw_dimension_units=["deg"] * 5 + ["percentage points"])
        return result
    xyz = np.linalg.norm(error[:, :3], axis=1) * 1000
    result.update(translation_mean_mm=float(xyz.mean()), translation_p95_mm=float(np.quantile(xyz, .95)),
                  translation_bias_xyz_mm=(error[:, :3].mean(0) * 1000).tolist(),
                  raw_dimension_units=(["m"] * 3 + (["unitless rotation columns"] * 6 if action_kind == "cartesian_absolute" else ["rad"] * 3) + ["fraction"]))
    if action_kind in DELTA_KINDS:
        target_norm = np.linalg.norm(label[:, :3], axis=1)
        predicted_norm = np.linalg.norm(pred[:, :3], axis=1)
        moving = target_norm * 1000 > MOTION_MM
        zero_error = float(target_norm.mean() * 1000)
        result.update(zero_delta_translation_mean_mm=zero_error,
                      translation_skill_vs_zero=1 - float(xyz.mean()) / zero_error if zero_error else None,
                      translation_motion={
                          "label_step_threshold_mm": MOTION_MM, "n": int(moving.sum()),
                          "translation_mean_mm": _mean(xyz[moving]),
                          "predicted_step_mean_mm": _mean(predicted_norm[moving] * 1000),
                          "label_step_mean_mm": _mean(target_norm[moving] * 1000),
                          "median_predicted_to_label_step_ratio": float(np.median(predicted_norm[moving] / target_norm[moving])) if moving.any() else None,
                          "positive_direction_fraction": _mean(np.sum(pred[moving, :3] * label[moving, :3], axis=1) > 0),
                      })
    rotation, invalid = [], 0
    for p, target in zip(pred, label, strict=True):
        try:
            rp = matrix_from_rot6d(p[3:9]) if action_kind == "cartesian_absolute" else exp_so3(p[3:6])
            rt = matrix_from_rot6d(target[3:9]) if action_kind == "cartesian_absolute" else exp_so3(target[3:6])
            rotation.append(float(np.linalg.norm(log_so3(rp @ rt.T))))
        except ValueError:
            invalid += 1
    result.update(rotation_mean_rad=_mean(rotation), invalid_rotations=invalid)
    return result


def _motions(values, arrays, ticks, kind):
    """Express Cartesian vectors in the base frame before integrating drift."""
    state = _canonical(arrays)
    if kind == "canonical_joint":
        return values[:, :5] - arrays["joint_state"][ticks, :5], None, MOTION_DEG, "deg"
    if kind == "cartesian_absolute":
        rotation = np.array([np.linalg.norm(log_so3(matrix_from_rot6d(target[3:9]) @ matrix_from_rot6d(current[3:9]).T))
                             for target, current in zip(values, state[ticks], strict=True)])
        return (values[:, :3] - state[ticks, :3]) * 1000, rotation, MOTION_MM, "mm"
    vectors = values[:, :3] * 1000
    if kind == "cartesian_tool":
        rotations = np.stack([matrix_from_rot6d(row[3:9]) for row in state[ticks]])
        vectors = np.einsum("nij,nj->ni", rotations, vectors)
    return vectors, np.linalg.norm(values[:, 3:6], axis=1), MOTION_MM, "mm"


def _baselines(arrays, ticks, anchors, kind):
    state = _canonical(arrays)
    hold = np.zeros_like(arrays["action"][ticks])
    if kind == "canonical_joint":
        hold[:] = arrays["joint_state"][anchors]
    elif kind == "cartesian_absolute":
        hold[:, :9] = state[anchors, :9]
        hold[:, -1] = state[anchors, -1]
    else:
        hold[:, -1] = state[anchors, -1]
    persistence = hold.copy()
    previous = anchors > arrays["ep_start"][anchors]
    persistence[previous] = arrays["action"][anchors[previous] - 1]
    if kind == "cartesian_tool" and previous.any():
        # Carry the previous measured tool delta into the current anchor's frame.
        for i in np.flatnonzero(previous):
            before = matrix_from_rot6d(state[anchors[i] - 1, 3:9])
            current = matrix_from_rot6d(state[anchors[i], 3:9])
            for part in (slice(0, 3), slice(3, 6)):
                persistence[i, part] = current.T @ before @ persistence[i, part]
    # Both actions hold the gripper visible at the call, without a future state.
    persistence[:, -1] = hold[:, -1]
    return {"hold": hold, "persistence": persistence}


def _drift(pred_motion, label_motion, stationary, ticks, episodes, unit):
    """Integrate within contiguous stationary runs, never across gaps/episodes."""
    runs, current = [], []
    for i in range(len(ticks)):
        if not stationary[i] or (current and (episodes[i] != episodes[current[-1]] or ticks[i] != ticks[current[-1]] + 1)):
            if current:
                runs.append(current)
                current = []
        if stationary[i]:
            current.append(i)
    if current:
        runs.append(current)
    errors = [float(np.linalg.norm((pred_motion - label_motion)[run].sum(0))) for run in runs]
    return {"n": int(stationary.sum()), "contiguous_runs": len(runs), "unit": unit,
            "predicted_step_mean": _mean(np.linalg.norm(pred_motion[stationary], axis=1)),
            "summed_error_per_run_mean": _mean(errors),
            "summed_error_per_run_max": max(errors) if errors else None,
            "interpretation": "teacher-forced displacement/target-error sum proxy; no rollout feedback"}


def executed_prefix_report(predictions, ticks, anchors, arrays, manifest, std, episodes, *, n_action_steps=2):
    """Validate coverage and report one common protocol for every model family."""
    episodes = sorted(int(e) for e in episodes)
    allowed = set(manifest["split"]["train"]) | set(manifest["split"]["val"])
    reserved = set(manifest["split"].get("reserved", []))
    if not episodes or not set(episodes) <= allowed or set(episodes) & reserved:
        raise ValueError("offline selection may consume train/validation episodes only")
    ticks, anchors = np.asarray(ticks, dtype=np.int64), np.asarray(anchors, dtype=np.int64)
    extrapolated = arrays.get("label_is_extrapolated", np.zeros(len(arrays["action"]), bool))
    expected = np.flatnonzero(np.isin(arrays["episode_index"], episodes) & ~extrapolated)
    if not len(expected):
        raise ValueError("selected episodes have no nonextrapolated actions")
    if n_action_steps < 1:
        raise ValueError("positive execution prefix required")
    expected_anchors = arrays["ep_start"][expected] + ((expected - arrays["ep_start"][expected]) // n_action_steps) * n_action_steps
    if (not np.array_equal(ticks, expected) or len(np.unique(ticks)) != len(ticks)
            or anchors.shape != ticks.shape or np.any(anchors > ticks)
            or not np.array_equal(anchors, expected_anchors)):
        raise ValueError("executed prefixes must cover each nonpadding/nonextrapolated action exactly once")
    kind = manifest.get("action_representation", "cartesian")
    if kind not in (*DELTA_KINDS, "cartesian_absolute", "canonical_joint"):
        raise ValueError(f"unsupported action representation: {kind}")
    label = arrays["action"][ticks]
    predictions = np.asarray(predictions)
    model = action_metrics(predictions, label, action_kind=kind, std=std)
    label_motion, label_rotation, threshold, unit = _motions(label, arrays, ticks, kind)
    magnitude = np.linalg.norm(label_motion, axis=1)
    moving = magnitude > threshold
    stationary = ~moving
    if label_rotation is not None:
        stationary &= label_rotation <= MOTION_RAD
    baselines = _baselines(arrays, ticks, anchors, kind)
    reports = {}
    for name, values in {"model": predictions, **baselines}.items():
        motion, _, _, _ = _motions(values, arrays, ticks, kind)
        predicted_magnitude = np.linalg.norm(motion, axis=1)
        error = np.linalg.norm(motion - label_motion, axis=1)
        dot = np.sum(motion[moving] * label_motion[moving], axis=1)
        directional = moving & (predicted_magnitude > 1e-8)
        cosine = np.sum(motion[directional] * label_motion[directional], axis=1) / (predicted_magnitude[directional] * magnitude[directional])
        reports[name] = {
            "all": model if name == "model" else action_metrics(values, label, action_kind=kind, std=std),
            "movement": {"n": int(moving.sum()), "unit": unit, "label_step_threshold": threshold,
                         "vector_error_mean": _mean(error[moving]),
                         "positive_direction_fraction": _mean(dot > 0),
                         "direction_cosine_mean_nonzero_predictions": _mean(cosine),
                         "zero_prediction_fraction": _mean(predicted_magnitude[moving] <= 1e-8),
                         "predicted_to_label_step_ratio_median": float(np.median(predicted_magnitude[moving] / magnitude[moving])) if moving.any() else None},
            "stationary_drift": _drift(motion, label_motion, stationary, ticks, arrays["episode_index"][ticks], unit),
            "per_episode": {},
        }
        for ep in episodes:
            mask = arrays["episode_index"][ticks] == ep
            reports[name]["per_episode"][str(ep)] = {
                **action_metrics(values[mask], label[mask], action_kind=kind, std=std),
                "summed_motion_error": float(np.linalg.norm((motion - label_motion)[mask].sum(0))),
                "summed_motion_error_unit": unit,
            }
    result = {"version": VERSION, "action_representation": kind, "episodes": episodes,
              "valid_actions": len(ticks), "excluded_extrapolated_actions": int((np.isin(arrays["episode_index"], episodes) & extrapolated).sum()),
              "inference_calls": len(np.unique(anchors)), "scores": reports,
              "thresholds": {"translation_mm_per_tick": MOTION_MM, "rotation_rad_per_tick": MOTION_RAD, "joint_vector_deg_per_tick": MOTION_DEG},
              "protocol": "Raw teacher-forced predictions; episode-local nonoverlapping prefixes; each valid action once; training episodes measure fitting only.",
              "baseline_contract": "Hold uses observed pose/joints and gripper at the inference anchor; persistence repeats the prior achieved delta/recorded target and holds the anchor gripper. Tool deltas are transported between observed tool frames. Neither uses future observations.",
              "geometry_caution": "Cartesian metrics inherit the prepared FK calibration; joint metrics are degrees, not Cartesian task error.",
              "physical_grasp_success": "not_evaluated"}
    result["offline_selection"] = offline_selection(result)
    return result


def offline_selection(report):
    """A fixed descriptive shortlist score, comparable only inside one data contract."""
    model, hold = report["scores"]["model"], report["scores"]["hold"]
    components = {"executed_prefix_normalized_l1": model["all"]["l1"]}
    motion = model["movement"]
    if motion["n"]:
        denominator = max(hold["movement"]["vector_error_mean"], motion["label_step_threshold"])
        components["moving_error_vs_hold"] = motion["vector_error_mean"] / denominator
        components["movement_direction_failure_fraction"] = 1 - motion["positive_direction_fraction"]
    drift = model["stationary_drift"]
    if drift["n"]:
        components["stationary_step_over_motion_threshold"] = drift["predicted_step_mean"] / motion["label_step_threshold"]
    return {"version": VERSION, "score": float(np.mean(list(components.values()))),
            "components": components, "weights": "equal mean of available components; thresholds fixed before training",
            "scope": "offline shortlist only; same split, targets, normalization and component availability required for comparison",
            "promotion": "requires bounded common development rollouts, then frozen reserved evaluation; offline score establishes no task success"}


@torch.inference_mode()
def evaluate_executed_prefix(policy, dataset, device, batch_size=16, *, seed=0, sampling_steps=4):
    """Predict at the declared execution cadence on a canonical or feature dataset."""
    n = int(policy.config.n_action_steps)
    if n < 1 or n > int(policy.config.chunk_size):
        raise ValueError("valid execution prefix required")
    arrays = dataset.arrays
    episodes = np.unique(arrays["episode_index"][dataset.indices]).tolist()
    # Validate the partition before fetching an image or making a policy call.
    split = dataset.manifest["split"]
    allowed = set(split["train"]) | set(split["val"])
    if not set(episodes) <= allowed or set(episodes) & set(split.get("reserved", [])):
        raise ValueError("offline selection may consume train/validation episodes only")
    selected = [i for i, tick in enumerate(dataset.indices) if (tick - arrays["ep_start"][tick]) % n == 0]
    was_training = policy.training
    policy.eval()
    predictions, ticks, anchors = [], [], []
    try:
        for batch in DataLoader(Subset(dataset, selected), batch_size=batch_size, shuffle=False):
            obs = {k: v.to(device) for k, v in batch.items() if k.startswith("observation.")}
            kwargs = {"sampling_seed": [int(seed) * 1_000_003 + int(t) for t in batch["tick"]], "sampling_steps": sampling_steps} if getattr(policy.config, "type", None) == "small_chunk_head" else {}
            raw = dataset.normalizer.unaction(policy.predict_action_chunk(obs, **kwargs)[:, :n]).cpu().numpy()
            valid = ~batch["action_is_pad"][:, :n].numpy()
            anchor = np.broadcast_to(batch["tick"].numpy()[:, None], valid.shape)
            ticks.extend((anchor + np.arange(n)[None])[valid].tolist())
            anchors.extend(anchor[valid].tolist())
            predictions.append(raw[valid])
        pred = np.concatenate(predictions)
        report = executed_prefix_report(pred, ticks, anchors, arrays, dataset.manifest, dataset.normalizer.action_std, episodes, n_action_steps=n)
        report["executed_actions_per_call"] = n
        return report, {"tick": np.asarray(ticks), "anchor_tick": np.asarray(anchors), "label": arrays["action"][ticks], "model": pred}
    finally:
        policy.train(was_training)
