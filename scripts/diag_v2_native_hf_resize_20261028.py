"""Compare native HF predictions under recorded-frame resize kernels only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

from real2sim.artifacts import sha256_file
from real2sim.data.raw import RawDataset
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.serialization import write_json
from scripts.evaluate_hf_act_20261027 import SNAPSHOT
from scripts.evaluate_v2_native_hf_start_20261028 import DATA


def run(out, snapshot=SNAPSHOT):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    native = NativeLeRobotACT(snapshot)
    native.policy.eval().requires_grad_(False)
    raw = RawDataset(DATA)
    experiment = json.loads((snapshot / "experiment.json").read_text())
    before = sha256_file(snapshot / "model.safetensors")
    report = {"schema": "native_hf_resize_probe_20261028_v1", "status": "running",
        "checkpoint_sha256": before, "episodes": experiment["val_episodes"],
        "sampling": "Source frame0 plus20frames before the strongest negative gripper-action transition, one pair per validation episode. Label-derived anchor selection is offline diagnostics only.",
        "source_training_resize_evidence": "/Users/deveshvyas/Downloads/lerobot/act03_so101_debug/offline_act03.py: torchvision.Resize antialias=True (local diagnostic states this matches training; original training notebook unavailable)",
        "no_simulator_rollouts": True, "no_calibration_or_weight_change": True,
        "conditions": ["opencv_linear", "opencv_area", "torchvision_antialias_float"], "cases": []}
    for episode in experiment["val_episodes"]:
        rows = raw.episode_rows(episode)
        states, actions = (raw.stack(rows, key) for key in ("observation.state", "action"))
        closing = int(np.argmin(np.diff(actions[:, 5]))) + 1
        frames = sorted({0, max(0, closing - 20)})
        decoded = {key: {} for key in native.camera_keys}
        for key in native.camera_keys:
            for frame, _pts, image in raw.iter_frames(key, episode):
                if frame in frames:
                    decoded[key][frame] = image.to_ndarray(format="rgb24")
                if frame >= max(frames):
                    break
        for frame in frames:
            condition_images = {name: {} for name in report["conditions"]}
            for key in native.camera_keys:
                rgb = decoded[key][frame]
                condition_images["opencv_linear"][key] = torch.from_numpy(cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1).copy()).float() / 255.
                condition_images["opencv_area"][key] = torch.from_numpy(cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA).transpose(2, 0, 1).copy()).float() / 255.
                condition_images["torchvision_antialias_float"][key] = TF.resize(torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255., [360, 640], antialias=True)
            predictions = {}
            for condition in report["conditions"]:
                native.reset()
                batch = {"observation.state": torch.tensor(states[frame], dtype=torch.float32), **condition_images[condition]}
                with torch.inference_mode():
                    predictions[condition] = native.postprocessor(native.policy.predict_action_chunk(native.preprocessor(batch)))[0].numpy()
            reference = actions[frame:frame + 10]
            detail = {"episode": episode, "frame": frame, "source_time_s": frame / raw.fps,
                "condition_prefix_mae_degree_percent": {name: np.abs(value[:len(reference)] - reference).mean(0).tolist() for name, value in predictions.items()},
                "condition_action_change_vs_linear_degree_percent": {name: np.abs(value[:10] - predictions["opencv_linear"][:10]).mean(0).tolist() for name, value in predictions.items()},
                "mean_rgb_difference_vs_linear": {name: float(np.mean([torch.abs(condition_images[name][key] - condition_images["opencv_linear"][key]).mean().item() for key in native.camera_keys])) for name in predictions}}
            report["cases"].append(detail)
            np.savez_compressed(out / f"episode_{episode:03d}_frame_{frame:04d}.npz", state=states[frame], reference=reference, **predictions)
            print(json.dumps({"episode": episode, "frame": frame, "aa_action_change": detail["condition_action_change_vs_linear_degree_percent"]["torchvision_antialias_float"]}), flush=True)
            write_json(out / "report.json", report, atomic=True)
    report["summary"] = {condition: {
        "mean_five_arm_joint_prefix_mae_degrees": float(np.mean([np.mean(case["condition_prefix_mae_degree_percent"][condition][:5]) for case in report["cases"]])),
        "mean_gripper_prefix_mae_percent": float(np.mean([case["condition_prefix_mae_degree_percent"][condition][5] for case in report["cases"]])),
        "mean_five_arm_joint_action_change_vs_linear_degrees": float(np.mean([np.mean(case["condition_action_change_vs_linear_degree_percent"][condition][:5]) for case in report["cases"]]))}
        for condition in report["conditions"]}
    report.update(status="complete", checkpoint_unchanged=before == sha256_file(snapshot / "model.safetensors"))
    write_json(out / "report.json", report, atomic=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    run(args.out, args.snapshot)
