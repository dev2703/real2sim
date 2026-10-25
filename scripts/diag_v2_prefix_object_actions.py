"""Apply the frozen matched-object interventions to canonical 13-state ACTs.

The original pair selection and measured masks remain unchanged. The full-chunk
and executed-prefix capacity checkpoints can be scored separately under this same
probe. Donor labels remain observational comparators, not counterfactual targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diag_v2_matched_object_actions import (
    CAMERA,
    RULES,
    contrast_metrics,
    object_composite,
    object_support,
    verify_plan,
)

from real2sim.data.raw import sha256_file
from real2sim.policy.act import load_checkpoint

CANONICAL_STATE_NAMES = ["ee_x", "ee_y", "ee_z", *[f"ee_rot6d_{i}" for i in range(6)],
                         "ee_vx", "ee_vy", "ee_vz", "gripper_obs"]
SHARED_FIELDS = ("camera_keys", "action_names", "action_representation", "split", "letterbox",
                 "image_shape_hwc", "calibration_sha256", "controller_semantics", "timing_basis",
                 "camera_transforms", "policy_hz", "label_semantics")


def validate_canonical_manifest(manifest):
    if (manifest.get("camera_keys") != [CAMERA]
            or manifest.get("state_names") != CANONICAL_STATE_NAMES
            or manifest.get("action_representation") != "cartesian"
            or manifest.get("visual_grounding") is not None
            or manifest.get("observation_variant") is not None):
        raise ValueError("requires canonical 13-state single-iPhone Cartesian data")


def verify_payload(root, manifest):
    for name, expected in manifest["files_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"data payload changed: {root / name}")


def verify_common_source(canonical, grounded):
    validate_canonical_manifest(canonical)
    for key in SHARED_FIELDS:
        if canonical.get(key) != grounded.get(key):
            raise ValueError(f"frozen object measurements/data differ: {key}")
    for name in ("arrays.npz", canonical["image_files"][CAMERA]):
        other_name = grounded["image_files"][CAMERA] if name != "arrays.npz" else name
        if canonical["files_sha256"][name] != grounded["files_sha256"][other_name]:
            raise ValueError(f"frozen probe source payload differs: {name}")


def verify_checkpoint_contract(contract, manifest, checkpoint_stats, source_stats):
    validate_canonical_manifest(manifest)
    for key in ("camera_keys", "state_names", "action_names", "split", "action_representation",
                "visual_grounding", "observation_variant", "calibration_sha256", "controller_semantics",
                "timing_basis", "camera_transforms", "letterbox", "label_semantics", "policy_hz"):
        if contract.get(key) != manifest.get(key):
            raise ValueError(f"checkpoint/data contract differs: {key}")
    if contract.get("derived_files_sha256") != manifest["files_sha256"]:
        raise ValueError("checkpoint/data payload map differs")
    if checkpoint_stats != source_stats:
        raise ValueError("checkpoint/source normalization differs")


def grouped_scores(rows):
    groups = []
    for partition in ("train", "val"):
        for name in ("ball", "bowl"):
            for context in ("stationary_start", "active"):
                selected = [r for r in rows if (r["partition"], r["object"], r["context"])
                            == (partition, name, context)]
                if not selected:
                    continue
                for mode in ("paired_full_image", "paired_object_region"):
                    scores = [r["scores"][mode] for r in selected]
                    informative = [s for s in scores if s["informative_target_contrast"]]
                    def mean(values):
                        values = [v for v in values if v is not None]
                        return float(np.mean(values)) if values else None
                    groups.append({"partition": partition, "object": name, "context": context, "mode": mode,
                        "pairs": len({r["pair_index"] for r in selected}), "reciprocal_directions": len(selected),
                        "informative_pairs": len({r["pair_index"] for r in selected
                                                   if r["scores"][mode]["informative_target_contrast"]}),
                        "mean_swap_effect_prefix_norm_mm": mean([
                            s["fixed_state_swap_effect"]["prefix_vector_norm_mm"] for s in scores]),
                        "mean_informative_amplitude_ratio": mean([
                            s["fixed_state_swap_effect"]["target_contrast_amplitude_ratio"] for s in informative]),
                        "mean_informative_cosine": mean([
                            s["fixed_state_swap_effect"]["target_contrast_cosine"] for s in informative]),
                        "mean_target_contrast_prefix_norm_mm": mean([
                            s["target_contrast_prefix_norm_mm"] for s in scores]),
                        "mean_original_prediction_step_mm": mean([r["original_prediction_step_mean_mm"] for r in selected]),
                        "mean_original_target_step_mm": mean([r["anchor_target_step_mean_mm"] for r in selected]),
                        "mean_natural_prediction_amplitude_ratio": mean([
                            s["natural_prediction_contrast"]["target_contrast_amplitude_ratio"] for s in informative])})
    return groups


def run(checkpoint, data, frozen_root, out, name):
    if out.exists():
        raise FileExistsError(out)
    frozen, selection = verify_plan(frozen_root)
    objects_path = frozen_root / "objects.json"
    objects = json.loads(objects_path.read_text())
    if objects["plan_sha256"] != sha256_file(frozen_root / "plan.json"):
        raise ValueError("object measurement plan differs")
    frames = {r["tick"]: r for r in objects["frames"]}
    if len(frames) != len(objects["frames"]) or set(frames) != {r["tick"] for r in selection["images"]}:
        raise ValueError("frozen object measurement coverage differs")
    for frame in frames.values():
        if sha256_file(Path(frame["mask_path"])) != frame["mask_sha256"]:
            raise ValueError("frozen object mask changed")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    grounded_root = Path(frozen["data"]) / "ball_input"
    grounded = json.loads((grounded_root / "manifest.json").read_text())
    verify_common_source(manifest, grounded)
    verify_payload(data, manifest)
    # The old frozen plan hashes grounded payloads and all selected image files.
    policy, norm, contract = load_checkpoint(checkpoint, "cpu")
    source_stats = json.loads((data / "stats.json").read_text())
    checkpoint_stats = json.loads((checkpoint / "stats.json").read_text())
    verify_checkpoint_contract(contract, manifest, checkpoint_stats, source_stats)
    if (policy.config.n_action_steps != RULES["executed_prefix"]
            or len(norm.state_mean) != 13 or list(policy.config.image_features) != [CAMERA]):
        raise ValueError("checkpoint requires different state/camera/execution inputs")
    info = json.loads((checkpoint / "run_info.json").read_text())
    training_episodes = set(info.get("diagnostic_episodes") or manifest["split"]["train"])
    pairs = [{**pair, "object_support": object_support(pair, frames)} for pair in frozen["pairs"]]
    arrays = dict(np.load(data / "arrays.npz", allow_pickle=False))
    images = np.load(data / manifest["image_files"][CAMERA], mmap_mode="r")
    selected_ticks = {p[k] for p in pairs for k in ("anchor_tick", "donor_tick")}
    if any(int(arrays["episode_index"][t]) in manifest["split"]["reserved"] for t in selected_ticks):
        raise ValueError("reserved example cannot enter object probing")
    for pair in pairs:
        for side in ("anchor", "donor"):
            tick, episode = pair[f"{side}_tick"], pair[f"{side}_episode"]
            if (int(arrays["episode_index"][tick]) != episode
                    or not np.all(arrays["episode_index"][tick:tick + 2] == episode)
                    or np.any(arrays["label_is_extrapolated"][tick:tick + 2])):
                raise ValueError("frozen pair tick/real-prefix identity differs")
        pair["actual_training_endpoint_count"] = sum(
            pair[f"{side}_episode"] in training_episodes for side in ("anchor", "donor"))
    out.parent.mkdir(parents=True, exist_ok=True)
    plan_path = out.with_suffix(".plan.json")
    if plan_path.exists():
        raise FileExistsError(plan_path)
    dependencies = [Path(__file__), Path("scripts/diag_v2_matched_object_actions.py"),
                    Path("src/real2sim/policy/act.py"), Path("src/real2sim/policy/dataset.py"),
                    Path("src/real2sim/policy/attention.py"),
                    frozen_root / "plan.json", frozen_root / "selection.json", objects_path,
                    manifest_path, data / "stats.json", checkpoint / "model.safetensors",
                    checkpoint / "config.json", checkpoint / "real2sim_contract.json", checkpoint / "run_info.json"]
    plan = {"schema": "canonical_matched_object_probe_plan_v1", "name": name,
            "checkpoint": str(checkpoint.resolve()), "data": str(data.resolve()), "rules": RULES,
            "pairs": pairs, "actual_training_episodes": sorted(training_episodes),
            "source_sha256": {str(p.resolve()): sha256_file(p) for p in dependencies},
            "reserved_episodes_used": [],
            "interpretation": "Original train/val labels describe the frozen source partition. Endpoints outside this checkpoint's diagnostic training episodes are not training-fit examples; per-pair overlap is explicit. Reciprocal directions are correlated. Donor actions are observational comparators. Mask unions replace some background/occlusion pixels; full-image swaps also change arm/background. No physical counterfactual or closed-loop success is established."}
    # The exact pairs, endpoint overlap, inputs, checkpoint and thresholds precede inference.
    plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    cache = {}
    def predict(tick, image):
        batch = {"observation.state": torch.from_numpy(norm.state(arrays["state"][tick]).astype(np.float32))[None],
                 CAMERA: torch.from_numpy(np.ascontiguousarray(norm.image(image)))[None]}
        with torch.inference_mode():
            value = norm.unaction(policy.predict_action_chunk(batch)).cpu().numpy()[0, :2]
        if not np.isfinite(value).all():
            raise ValueError("nonfinite policy prediction")
        return value
    for tick in sorted(selected_ticks):
        cache[tick] = predict(tick, np.asarray(images[tick]))
    rows = []
    for pair_index, pair in enumerate(pairs):
        if not pair["object_support"]["supported"]:
            continue
        for reverse in (False, True):
            ta, tb = pair["anchor_tick"], pair["donor_tick"]
            if reverse:
                ta, tb = tb, ta
            anchor, donor = np.asarray(images[ta]), np.asarray(images[tb])
            ma, mb = (np.load(frames[t]["mask_path"])[pair["object"]] for t in (ta, tb))
            if ma.shape != anchor.shape[:2] or mb.shape != donor.shape[:2]:
                raise ValueError("frozen object mask resolution differs")
            composite, mask = object_composite(anchor, donor, ma, mb, RULES["mask_dilation_policy_px"])
            predictions = {"paired_full_image": predict(ta, donor),
                           "paired_object_region": predict(ta, composite)}
            scores = {mode: contrast_metrics(arrays["action"][ta:ta + 2], arrays["action"][tb:tb + 2],
                                             cache[ta], cache[tb], value) for mode, value in predictions.items()}
            rows.append({"pair_index": pair_index, "partition": pair["partition"], "object": pair["object"],
                "context": pair["context"], "anchor_tick": ta, "donor_tick": tb, "reverse": reverse,
                "actual_training_endpoint_count": pair["actual_training_endpoint_count"],
                "object_swap_pixel_count": int(mask.sum()), "scores": scores,
                "original_prediction_step_mean_mm": float(np.linalg.norm(cache[ta][:, :3], axis=-1).mean() * 1000),
                "anchor_target_step_mean_mm": float(np.linalg.norm(arrays["action"][ta:ta + 2, :3], axis=-1).mean() * 1000),
                "anchor_target": arrays["action"][ta:ta + 2].tolist(),
                "donor_target": arrays["action"][tb:tb + 2].tolist(),
                "original_prediction": cache[ta].tolist(), "natural_donor_prediction": cache[tb].tolist(),
                "intervention_predictions": {mode: value.tolist() for mode, value in predictions.items()}})
    for path, expected in plan["source_sha256"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"probe source changed during inference: {path}")
    report = {"schema": "canonical_matched_object_probe_v1", "name": name, "plan": plan,
              "plan_sha256": sha256_file(plan_path), "physical_calibration_accepted": False,
              "checkpoint_promoted": False, "reserved_episodes_used": [],
              "selected_pairs": len(pairs), "supported_pairs": sum(p["object_support"]["supported"] for p in pairs),
              "rows": rows, "groups": grouped_scores(rows)}
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone"))
    parser.add_argument("--frozen-root", type=Path, default=Path("outputs/v2/matched_object_actions_20261024"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run(args.checkpoint, args.data, args.frozen_root, args.out, args.name)
    print(json.dumps({"selected_pairs": result["selected_pairs"], "supported_pairs": result["supported_pairs"],
                      "groups": [g for g in result["groups"] if g["partition"] == "val"]}, indent=2))
