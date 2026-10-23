"""Recorded-image feature parity and controlled sensitivity of ball-input checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from real2sim.data.raw import sha256_file
from real2sim.policy.act import load_checkpoint
from real2sim.policy.adapters import ObservationAdapter
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.validation import action_metrics
from real2sim.policy.visual_grounding import BallImageFeatures, append_ball_features, ball_features


def diagnose(checkpoint: Path, data: Path, per_episode: int = 4) -> dict:
    policy, norm, contract = load_checkpoint(checkpoint)
    ds = DerivedTickDataset(data, contract["split"]["val"], policy.config.chunk_size, norm)
    if per_episode < 1 or int(policy.config.n_action_steps) != 2:
        raise ValueError("positive anchor count and two-action prefixes required")
    for name in ("camera_keys", "state_names", "action_names", "split", "visual_grounding"):
        if contract.get(name) != ds.manifest.get(name):
            raise ValueError(f"checkpoint/data mismatch: {name}")
    if contract["derived_files_sha256"] != ds.manifest["files_sha256"]:
        raise ValueError("checkpoint/data payload hashes differ")
    for name, digest in ds.manifest["files_sha256"].items():
        if sha256_file(data / name) != digest:
            raise ValueError(f"data payload mismatch: {name}")
    spec = BallImageFeatures.from_dict(contract["visual_grounding"])
    indices = []
    for episode in contract["split"]["val"]:
        available = [i for i, tick in enumerate(ds.indices)
                     if ds.arrays["episode_index"][tick] == episode
                     and (tick - ds.arrays["ep_start"][tick]) % 2 == 0
                     and not ds.arrays["label_is_extrapolated"][tick]]
        indices.extend(available[j] for j in np.unique(np.linspace(0, len(available) - 1,
                       min(per_episode, len(available))).astype(int)))
    adapter = ObservationAdapter(contract, norm)
    predictions = {name: [] for name in ("original", "paired_image_and_features_swap",
                                         "image_only_swap", "features_only_swap")}
    labels, parity, selections, detections = [], [], [], []
    stride = len(indices) // len(contract["split"]["val"])
    for j, index in enumerate(indices):
        tick = int(ds.indices[index])
        image = np.asarray(ds.images[spec.camera_key][tick])

        class Env:
            def __init__(self, state):
                self.state = state

            def canonical_proprio(self, previous):
                return self.state.copy()

        env_obs = {key: np.asarray(ds.images[key][tick]).transpose(2, 0, 1).astype(np.float32) / 255
                   for key in contract["camera_keys"]}
        expected = ds[index]
        live = adapter.batch(env_obs, Env(ds.arrays["state"][tick]))
        parity.append(float(torch.max(torch.abs(live["observation.state"][0] - expected["observation.state"]))))
        batch = default_collate([expected])
        donor_index = indices[(j + stride) % len(indices)]
        donor_tick = int(ds.indices[donor_index])
        donor = ds[donor_index]
        feature_state = append_ball_features(ds.arrays["state"][tick],
                                            np.asarray(ds.images[spec.camera_key][donor_tick]), spec)
        swapped_features = torch.from_numpy(norm.state(feature_state).astype(np.float32))[None]
        for mode in predictions:
            obs = {k: v for k, v in batch.items() if k.startswith("observation.")}
            if mode in ("paired_image_and_features_swap", "image_only_swap"):
                for key in contract["camera_keys"]:
                    obs[key] = donor[key][None]
            if mode in ("paired_image_and_features_swap", "features_only_swap"):
                obs["observation.state"] = swapped_features
            with torch.inference_mode():
                raw = norm.unaction(policy.predict_action_chunk(obs)).cpu().numpy()[0, :2]
            predictions[mode].extend(raw[~batch["action_is_pad"][0, :2].numpy()])
        offsets = np.arange(2)[~batch["action_is_pad"][0, :2].numpy()]
        labels.extend(ds.arrays["action"][tick + offsets])
        detections.append(ball_features(image, spec).tolist())
        selections.append({"tick": tick, "episode": int(ds.arrays["episode_index"][tick]),
                           "frame": int(ds.arrays["frame_index"][tick]), "donor_tick": donor_tick})
    label = np.asarray(labels)
    original = np.asarray(predictions["original"])
    scores = {}
    for mode, values in predictions.items():
        values = np.asarray(values)
        scores[mode] = {**action_metrics(values, label, action_kind="cartesian", std=norm.action_std),
                        "action_translation_change_mean_mm": float(np.linalg.norm(values[:, :3] - original[:, :3], axis=1).mean() * 1000)}
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint / "model.safetensors"),
            "data_manifest_sha256": sha256_file(data / "manifest.json"),
            "feature_spec": spec.to_dict(), "validation_anchors": selections,
            "raw_detections": detections, "feature_train_live_max_abs_difference": max(parity),
            "scores": scores, "reserved_episodes_used": [],
            "limitations": "Sparse teacher-forced sensitivity on validation images. Image-only and features-only swaps deliberately break the feature/image relation to isolate input dependence; only joint swaps preserve the input contract. These interventions cannot establish correct localization, causal task improvement or rollout success."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, default=Path("outputs/v2/ball_grounding_pilot_20261024"))
    p.add_argument("--data", type=Path, default=Path(".cache/v2_ball_grounding_pilot_20261024"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    report = {"kind": "BALL_IMAGE_INPUT_PARITY_AND_SENSITIVITY", "models": {}}
    for name in ("disabled_control", "ball_input"):
        report["models"][name] = diagnose(args.pilot / name / "checkpoints/best", args.data / name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: value["scores"] for name, value in report["models"].items()}, indent=2))
