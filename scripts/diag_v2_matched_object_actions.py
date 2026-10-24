"""Matched-arm object interventions with frozen selection and explicit support counts.

Image coordinates are uncalibrated. Recorded donor actions are observational
comparators, not counterfactual labels for a synthetic object swap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from real2sim.data.raw import sha256_file
from real2sim.policy.act import load_checkpoint
from real2sim.policy.visual_grounding import BallImageFeatures, append_ball_features

CAMERA = "observation.images.iphone"
RULES = {
    "minimum_episode": 7,
    "joint_max_difference_deg": 5., "joint_rms_difference_deg": 3.,
    "gripper_max_difference_percent": 5., "velocity_difference_m_s": .03,
    "stationary_history_ticks": 4, "stationary_joint_span_deg": .5,
    "stationary_speed_m_s": .005, "stationary_start_ticks": 25,
    "active_speed_m_s": .02, "candidate_stride_ticks": 2,
    "pairs_per_partition_object_context": 3,
    "object_center_separation_policy_px": 8., "mask_dilation_policy_px": 2,
    "minimum_ball_mask_area_px": 18, "minimum_bowl_mask_area_px": 80,
    "minimum_informative_target_contrast_mm": 1.,
    "detector_threshold": .22, "detector_prompt": "a tennis ball. a bowl.",
    "executed_prefix": 2,
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def candidate_ticks(arrays, episode, context, rules=RULES):
    ticks = np.flatnonzero(arrays["episode_index"] == episode)
    start = int(ticks[0])
    valid = []
    for tick in ticks[::rules["candidate_stride_ticks"]]:
        # Both scored actions need real successors, and velocity needs real history.
        if tick < start + rules["stationary_history_ticks"] - 1 or tick + 2 > ticks[-1]:
            continue
        if np.any(arrays["label_is_extrapolated"][tick:tick + 2]):
            continue
        speed = np.linalg.norm(arrays["state"][tick, 9:12])
        if context == "stationary_start":
            history = arrays["joint_state"][tick - rules["stationary_history_ticks"] + 1:tick + 1, :5]
            if (tick - start >= rules["stationary_start_ticks"]
                    or speed > rules["stationary_speed_m_s"]
                    or np.max(np.ptp(history, axis=0)) > rules["stationary_joint_span_deg"]):
                continue
        elif context == "active":
            if speed < rules["active_speed_m_s"]:
                continue
        else:
            raise ValueError("unknown context")
        valid.append(int(tick))
    return np.asarray(valid, dtype=int)


def select_pairs(arrays, split, rules=RULES):
    if (set(split["train"]) & set(split["val"])
            or (set(split["train"]) | set(split["val"])) & set(split["reserved"])):
        raise ValueError("partitions overlap")
    pairs, support = [], []
    for partition in ("train", "val"):
        allowed = [e for e in split[partition] if e >= rules["minimum_episode"]]
        baseline = [e for e in allowed if split["conditions"][str(e)] == "baseline"]
        for object_name in ("ball", "bowl"):
            shifted = [e for e in allowed if split["conditions"][str(e)] == object_name + "_shift"]
            for context in ("stationary_start", "active"):
                options, candidate_pairs = [], 0
                for a in baseline:
                    ta = candidate_ticks(arrays, a, context, rules)
                    if not len(ta):
                        continue
                    for b in shifted:
                        tb = candidate_ticks(arrays, b, context, rules)
                        if not len(tb):
                            continue
                        delta = arrays["joint_state"][ta, None, :5] - arrays["joint_state"][tb, :5]
                        rms = np.sqrt(np.mean(delta ** 2, axis=-1))
                        maximum = np.max(np.abs(delta), axis=-1)
                        gripper = np.abs(arrays["joint_state"][ta, None, 5] - arrays["joint_state"][tb, 5])
                        velocity = np.linalg.norm(arrays["state"][ta, None, 9:12] - arrays["state"][tb, 9:12], axis=-1)
                        valid = ((maximum <= rules["joint_max_difference_deg"])
                                 & (rms <= rules["joint_rms_difference_deg"])
                                 & (gripper <= rules["gripper_max_difference_percent"])
                                 & (velocity <= rules["velocity_difference_m_s"]))
                        candidate_pairs += int(valid.sum())
                        ia, ib = np.nonzero(valid)
                        if not len(ia):
                            continue
                        # One pair per episode pair; deterministic ordering never sees actions/images.
                        order = np.lexsort((tb[ib], ta[ia], velocity[ia, ib], rms[ia, ib]))
                        i, j = ia[order[0]], ib[order[0]]
                        options.append({"partition": partition, "object": object_name, "context": context,
                                        "anchor_episode": a, "donor_episode": b,
                                        "anchor_tick": int(ta[i]), "donor_tick": int(tb[j]),
                                        "joint_rms_difference_deg": float(rms[i, j]),
                                        "joint_max_difference_deg": float(maximum[i, j]),
                                        "gripper_difference_percent": float(gripper[i, j]),
                                        "velocity_difference_m_s": float(velocity[i, j])})
                options.sort(key=lambda p: (p["joint_rms_difference_deg"], p["velocity_difference_m_s"],
                                            p["anchor_tick"], p["donor_tick"]))
                chosen = options[:rules["pairs_per_partition_object_context"]]
                support.append({"partition": partition, "object": object_name, "context": context,
                                "candidate_tick_pairs": candidate_pairs, "episode_pairs_supported": len(options),
                                "selected_pairs": len(chosen)})
                pairs.extend(chosen)
    return pairs, support


def object_composite(anchor, donor, anchor_mask, donor_mask, radius=2):
    if anchor.shape != donor.shape or anchor_mask.shape != anchor.shape[:2] or donor_mask.shape != anchor.shape[:2]:
        raise ValueError("image/mask dimensions differ")
    union = np.logical_or(anchor_mask, donor_mask).astype(np.uint8)
    union = cv2.dilate(union, np.ones((2 * radius + 1,) * 2, np.uint8)).astype(bool)
    composite = anchor.copy()
    composite[union] = donor[union]
    return composite, union


def contrast_metrics(anchor, donor, prediction_anchor, prediction_donor, swapped):
    target = (np.asarray(donor) - anchor)[:, :3]
    prediction = (np.asarray(prediction_donor) - prediction_anchor)[:, :3]
    effect = (np.asarray(swapped) - prediction_anchor)[:, :3]
    tnorm = float(np.linalg.norm(target))
    informative = tnorm * 1000 >= RULES["minimum_informative_target_contrast_mm"]

    def compare(value):
        vnorm = float(np.linalg.norm(value))
        return {"prefix_vector_norm_mm": vnorm * 1000,
                "target_contrast_amplitude_ratio": vnorm / tnorm if informative else None,
                "target_contrast_cosine": float(np.sum(value * target) / (vnorm * tnorm)) if informative and vnorm > 1e-12 else None,
                "target_contrast_error_mm": float(np.linalg.norm(value - target) * 1000)}

    return {"target_contrast_prefix_norm_mm": tnorm * 1000, "informative_target_contrast": informative,
            "natural_prediction_contrast": compare(prediction), "fixed_state_swap_effect": compare(effect),
            "original_to_anchor_target_mean_mm": float(np.linalg.norm(np.asarray(prediction_anchor)[:, :3] - np.asarray(anchor)[:, :3], axis=-1).mean() * 1000),
            "original_to_donor_target_mean_mm": float(np.linalg.norm(np.asarray(prediction_anchor)[:, :3] - np.asarray(donor)[:, :3], axis=-1).mean() * 1000),
            "swap_to_donor_target_mean_mm": float(np.linalg.norm(np.asarray(swapped)[:, :3] - np.asarray(donor)[:, :3], axis=-1).mean() * 1000)}


def prepare(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    source = args.data / "ball_input"
    manifest = json.loads((source / "manifest.json").read_text())
    arrays = dict(np.load(source / "arrays.npz", allow_pickle=False))
    if manifest["camera_keys"] != [CAMERA] or manifest["action_representation"] != "cartesian":
        raise ValueError("requires single iPhone Cartesian pilot")
    pairs, support = select_pairs(arrays, manifest["split"])
    args.out.mkdir(parents=True)
    inputs = [Path(__file__), source / "manifest.json", source / "arrays.npz", source / "stats.json",
              source / manifest["image_files"][CAMERA]]
    for name in ("disabled_control", "ball_input"):
        inputs.extend(args.pilot / name / "checkpoints/best" / f for f in ("model.safetensors", "real2sim_contract.json", "config.json"))
    plan = {"schema": "matched_object_actions_v1", "rules": RULES,
            "data": str(args.data.resolve()), "pilot": str(args.pilot.resolve()),
            "pairs": pairs, "support": support, "reserved_episodes_used": [],
            "source_sha256": {str(p.resolve()): sha256_file(p) for p in inputs},
            "selection": "Raw arm/gripper readbacks and causal FK velocity only; same late image regime, separate train/development pairs; minimum-distance pair per episode pair; no labels or policy outputs select pairs.",
            "acceptance": "All selected pairs retained, including missing/insufficient object separation; object support is checked before policy inference without retries.",
            "limitations": ["Matching approximate joint states and provisional FK velocity does not remove teleoperator timing/phase or scene confounds.",
                            "Zero-action stationary starts cannot establish directional object use; report these separately.",
                            "Donor actions are observational comparisons, not counterfactual targets after an image intervention."]}
    # Persist before reading any selected images, running object models, or policy inference.
    write_json(args.out / "plan.json", plan)
    images = np.load(source / manifest["image_files"][CAMERA], mmap_mode="r")
    selections = []
    for tick in sorted({p[k] for p in pairs for k in ("anchor_tick", "donor_tick")}):
        path = args.out / f"tick_{tick:05d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(images[tick], cv2.COLOR_RGB2BGR))
        selections.append({"tick": tick, "episode": int(arrays["episode_index"][tick]),
                           "frame": int(arrays["frame_index"][tick]), "path": str(path.resolve()),
                           "sha256": sha256_file(path)})
    write_json(args.out / "selection.json", {"plan_sha256": sha256_file(args.out / "plan.json"), "images": selections})
    return {"support": support, "selected_frames": len(selections)}


def verify_plan(out):
    plan = json.loads((out / "plan.json").read_text())
    if plan["rules"] != RULES:
        raise ValueError("frozen rules differ")
    for path, digest in plan["source_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"frozen input changed: {path}")
    selection = json.loads((out / "selection.json").read_text())
    if selection["plan_sha256"] != sha256_file(out / "plan.json"):
        raise ValueError("selection plan changed")
    for row in selection["images"]:
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise ValueError("selected image changed")
    return plan, selection


def measure(args):
    from PIL import Image
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    _, selection = verify_plan(args.out)
    report_path = args.out / "objects.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    detector_path = Path(".cache/pretrained/IDEA-Research--grounding-dino-tiny-a2bb814dd30d")
    sam_path = Path(".cache/pretrained/facebook--sam2.1-hiera-tiny-de431c404385")
    processor = AutoProcessor.from_pretrained(detector_path, local_files_only=True)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_path, local_files_only=True).eval()
    sam_processor = Sam2Processor.from_pretrained(sam_path, local_files_only=True)
    sam = Sam2Model.from_pretrained(sam_path, local_files_only=True).eval()
    report = {"plan_sha256": sha256_file(args.out / "plan.json"),
              "weights_sha256": {str(p / "model.safetensors"): sha256_file(p / "model.safetensors") for p in (detector_path, sam_path)},
              "pixel_space": "256x192 policy letterbox, not source pixels or metric coordinates",
              "frames": [], "uncertainty": "Model confidence/IoU are not calibrated localization error; mask occlusion, shadows and background can confound object crops."}
    for source in selection["images"]:
        image = Image.open(source["path"]).convert("RGB")
        inputs = processor(images=image, text=RULES["detector_prompt"], return_tensors="pt")
        with torch.inference_mode():
            outputs = detector(**inputs)
        detections = processor.post_process_grounded_object_detection(outputs, inputs.input_ids,
            threshold=RULES["detector_threshold"], text_threshold=RULES["detector_threshold"], target_sizes=[image.size[::-1]])[0]
        best = {}
        for label, score, box in zip(detections["text_labels"], detections["scores"], detections["boxes"], strict=True):
            for name, text in (("ball", "tennis ball"), ("bowl", "bowl")):
                if text in label and (name not in best or float(score) > best[name]["score"]):
                    best[name] = {"score": float(score), "box_xyxy": box.tolist()}
        masks = {}
        preview = np.asarray(image).copy()
        if best:
            values = sam_processor(images=image, input_boxes=[[v["box_xyxy"] for v in best.values()]], return_tensors="pt")
            with torch.inference_mode():
                result = sam(**values, multimask_output=False)
            output = sam_processor.post_process_masks(result.pred_masks.cpu(), values["original_sizes"].cpu())[0]
            for j, name in enumerate(best):
                mask = output[j, 0].numpy().astype(np.uint8)
                y, x = np.nonzero(mask)
                masks[name] = mask
                best[name].update(mask_area_px=int(mask.sum()),
                                  center_xy=[float(x.mean()), float(y.mean())] if len(x) else None,
                                  predicted_iou=float(result.iou_scores[0, j, 0]))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(preview, contours, -1, (255, 0, 0) if name == "ball" else (0, 255, 255), 1)
        mask_path = args.out / f"tick_{source['tick']:05d}_masks.npz"
        np.savez_compressed(mask_path, **masks)
        cv2.imwrite(str(args.out / f"tick_{source['tick']:05d}_objects.png"), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        report["frames"].append({**source, "objects": best, "mask_path": str(mask_path.resolve()), "mask_sha256": sha256_file(mask_path)})
        print(json.dumps({"tick": source["tick"], "objects": best}), flush=True)
    write_json(report_path, report)
    return {"frames": len(report["frames"])}


def object_support(pair, frames):
    name = pair["object"]
    a, b = (frames[pair[k]]["objects"].get(name) for k in ("anchor_tick", "donor_tick"))
    if not a or not b or a["center_xy"] is None or b["center_xy"] is None:
        return {"supported": False, "reason": "object_missing"}
    displacement = float(np.linalg.norm(np.asarray(a["center_xy"]) - b["center_xy"]))
    min_area = RULES[f"minimum_{name}_mask_area_px"]
    supported = min(a["mask_area_px"], b["mask_area_px"]) >= min_area and displacement >= RULES["object_center_separation_policy_px"]
    return {"supported": supported, "reason": "supported" if supported else "insufficient_area_or_separation",
            "center_separation_policy_px": displacement, "anchor": a, "donor": b}


def score(args):
    plan, _ = verify_plan(args.out)
    report_path = args.out / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    objects = json.loads((args.out / "objects.json").read_text())
    if objects["plan_sha256"] != sha256_file(args.out / "plan.json"):
        raise ValueError("object measurements used another plan")
    frames = {r["tick"]: r for r in objects["frames"]}
    for row in frames.values():
        if sha256_file(Path(row["mask_path"])) != row["mask_sha256"]:
            raise ValueError("object mask changed")
    source = Path(plan["data"]) / "ball_input"
    arrays = dict(np.load(source / "arrays.npz", allow_pickle=False))
    manifests = {name: json.loads((Path(plan["data"]) / name / "manifest.json").read_text()) for name in ("disabled_control", "ball_input")}
    images = np.load(source / manifests["ball_input"]["image_files"][CAMERA], mmap_mode="r")
    report = {"schema": "matched_object_actions_v1", "plan": plan,
              "plan_sha256": sha256_file(args.out / "plan.json"), "objects_sha256": sha256_file(args.out / "objects.json"),
              "physical_calibration_accepted": False, "reserved_episodes_used": [], "checkpoint_promoted": False,
              "models": {}, "pairs": [{**p, "object_support": object_support(p, frames)} for p in plan["pairs"]],
              "interpretation": "Natural action contrast and fixed-arm-state input sensitivity are separate. Donor action is an observational comparator only. Full-image swaps also replace unmatched background/arm pixels; object-region swaps retain union-mask background/occlusion confounds. Features are recomputed from the intervened image except the deliberately inconsistent feature-only isolation. Stationary starts are offline observation probes, not simulation rollouts."}
    for name in ("disabled_control", "ball_input"):
        path = Path(plan["pilot"]) / name / "checkpoints/best"
        policy, norm, contract = load_checkpoint(path, "cpu")
        manifest = manifests[name]
        for key in ("camera_keys", "state_names", "action_names", "split", "visual_grounding", "calibration_sha256", "controller_semantics"):
            if contract.get(key) != manifest.get(key):
                raise ValueError(f"checkpoint/data contract mismatch: {key}")
        if contract["derived_files_sha256"] != manifest["files_sha256"]:
            raise ValueError("checkpoint/data file contract differs")
        for key, digest in manifest["files_sha256"].items():
            if sha256_file(Path(plan["data"]) / name / key) != digest:
                raise ValueError("data payload changed")
        if contract["action_representation"] != "cartesian" or policy.config.n_action_steps != 2:
            raise ValueError("two-step Cartesian action contract required")
        spec = BallImageFeatures.from_dict(contract["visual_grounding"])

        def predict(tick, image, feature_image=None, norm=norm, policy=policy, spec=spec):
            state = append_ball_features(arrays["state"][tick], image if feature_image is None else feature_image, spec)
            obs = {"observation.state": torch.from_numpy(norm.state(state))[None], CAMERA: torch.from_numpy(norm.image(image))[None]}
            with torch.inference_mode():
                return norm.unaction(policy.predict_action_chunk(obs)).cpu().numpy()[0, :2]

        model_rows = []
        for pair in report["pairs"]:
            if not pair["object_support"]["supported"]:
                continue
            # Score both directions, retaining approximate observed state and exact intervened state.
            for reverse in (False, True):
                ta, tb = pair["anchor_tick"], pair["donor_tick"]
                if reverse:
                    ta, tb = tb, ta
                anchor, donor = np.asarray(images[ta]), np.asarray(images[tb])
                ma, mb = (np.load(frames[t]["mask_path"])[pair["object"]] for t in (ta, tb))
                composite, mask = object_composite(anchor, donor, ma, mb, RULES["mask_dilation_policy_px"])
                original, natural_donor = predict(ta, anchor), predict(tb, donor)
                predictions = {"paired_full_image_and_features": predict(ta, donor),
                               "paired_object_region_and_features": predict(ta, composite),
                               "features_only": predict(ta, anchor, donor)}
                scores = {mode: contrast_metrics(arrays["action"][ta:ta + 2], arrays["action"][tb:tb + 2], original, natural_donor, value)
                          for mode, value in predictions.items()}
                row = {"partition": pair["partition"], "object": pair["object"], "context": pair["context"],
                       "anchor_tick": ta, "donor_tick": tb, "reverse": reverse,
                       "object_swap_pixel_count": int(mask.sum()), "scores": scores,
                       "anchor_target": arrays["action"][ta:ta + 2].tolist(), "donor_target": arrays["action"][tb:tb + 2].tolist(),
                       "original_prediction": original.tolist(), "natural_donor_prediction": natural_donor.tolist(),
                       "intervention_predictions": {k: v.tolist() for k, v in predictions.items()}}
                model_rows.append(row)
                if name == "ball_input" and not reverse:
                    panel = np.concatenate([anchor, donor, composite], axis=1)
                    cv2.imwrite(str(args.out / f"pair_{ta:05d}_{tb:05d}_{pair['object']}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        report["models"][name] = {"checkpoint": str(path), "rows": model_rows}
        print(name, "scored directions", len(model_rows), flush=True)
    write_json(report_path, report)
    return {"selected_pairs": len(plan["pairs"]), "supported_pairs": sum(p["object_support"]["supported"] for p in report["pairs"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "measure", "score"])
    parser.add_argument("--pilot", type=Path, default=Path("outputs/v2/ball_grounding_pilot_20261024"))
    parser.add_argument("--data", type=Path, default=Path(".cache/v2_ball_grounding_pilot_20261024"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps({"prepare": prepare, "measure": measure, "score": score}[args.phase](args), indent=2))
