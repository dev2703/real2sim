"""Separate image and reset-velocity effects at one recorded validation anchor.

This is a teacher-forced diagnostic, not a learned rollout or calibration fit.
The simulator uses its existing provisional scene without moving objects to fit
recorded pixels. No training inputs or checkpoint contracts are modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from real2sim.data.paired_approach import camera_parameters
from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.paired import file_sha256
from real2sim.sim.env import So101PickPlaceEnv


def yellow_ball(image):
    """Image-only centroid on the actual letterboxed policy input, with no fit."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([20, 80, 60]), np.array([50, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(cv2.convexHull(c), True)
        if area > 30 and perimeter and 4 * np.pi * area / perimeter**2 > .5:
            m = cv2.moments(c)
            candidates.append((area, [m["m10"] / m["m00"], m["m01"] / m["m00"]]))
    if not candidates:
        return None
    area, center = max(candidates, key=lambda x: x[0])
    return {"policy_pixel": center, "area_policy_pixels": area}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone"))
    p.add_argument("--config", type=Path, default=Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml"))
    p.add_argument("--tick", type=int, default=2839)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    policy, norm, contract = load_checkpoint(a.checkpoint, "cpu")
    manifest = json.loads((a.data / "manifest.json").read_text())
    ds = DerivedTickDataset(a.data, manifest["split"]["val"], policy.config.chunk_size, norm)
    tick = a.tick
    index = np.flatnonzero(ds.indices == tick)
    if len(index) != 1:
        raise ValueError("anchor must belong to validation, never reserved")
    key = contract["camera_keys"][0]
    if contract["camera_keys"] != [key] or len(contract["state_names"]) != 13:
        raise ValueError("single-camera canonical Cartesian checkpoint required")
    if contract["derived_files_sha256"] != manifest["files_sha256"]:
        raise ValueError("checkpoint and recorded arrays differ")
    recorded_state = ds.arrays["state"][tick]
    recorded_image = np.asarray(ds.images[key][tick])
    with So101PickPlaceEnv(a.config, render_images=True) as env:
        obs, info = env.reset(seed=0)
        if not np.allclose(env.real_state(), ds.arrays["joint_state"][tick], atol=1e-6):
            raise ValueError("config reset is not this recorded joint anchor")
        simulated_state = env.canonical_proprio()
        rendered_image = np.clip(obs[key].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        cases = {}
        for image_name, image in (("recorded_image", recorded_image), ("rendered_image", rendered_image)):
            for state_name, state in (("recorded_state", recorded_state), ("reset_zero_velocity_state", simulated_state)):
                batch = {key: torch.from_numpy(norm.image(image)[None]),
                         "observation.state": torch.from_numpy(norm.state(state)[None])}
                policy.reset()
                with torch.inference_mode():
                    pred = norm.unaction(policy.predict_action_chunk(batch))[0, :2].cpu().numpy()
                label = ds.arrays["action"][tick:tick + 2]
                ball = np.asarray(info["privileged_metrics"]["ball_xyz_task"])
                ee = env.ee_pose()[:3, 3]
                toward = (ball - ee) / np.linalg.norm(ball - ee)
                cases[f"{image_name}__{state_name}"] = {
                    "first_two_actions": pred.tolist(),
                    "translation_error_mm": (1000 * np.linalg.norm(pred[:, :3] - label[:, :3], axis=1)).tolist(),
                    "translation_toward_simulator_ball_mm": (1000 * (pred[:, :3] @ toward)).tolist(),
                    "gripper_error_fraction": np.abs(pred[:, -1] - label[:, -1]).tolist(),
                }
        rvec, tvec, intrinsic, _, _ = camera_parameters(env)
        ball = np.asarray(info["privileged_metrics"]["ball_xyz_task"])
        projected = cv2.projectPoints(ball.reshape(1, 3), rvec, tvec, intrinsic, None)[0].reshape(2)
        metrics = info["privileged_metrics"]
        task_hash = file_sha256(Path(env.config["task_config"]))
        mapping = env.task["robot_so101"]["real_to_sim"]
    z_train = norm.state(ds.arrays["state"][np.isin(ds.arrays["episode_index"], manifest["split"]["train"])])
    z_anchor, z_reset = norm.state(recorded_state), norm.state(simulated_state)
    image_pair_path = a.out.with_suffix(".jpg")
    pair = np.concatenate([recorded_image, rendered_image], axis=1)
    cv2.imwrite(str(image_pair_path), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
    episode = int(ds.arrays["episode_index"][tick])
    detections = {"recorded": yellow_ball(recorded_image), "rendered": yellow_ball(rendered_image)}
    displacement = None
    if all(detections.values()):
        displacement = float(np.linalg.norm(np.asarray(detections["recorded"]["policy_pixel"])
                                             - detections["rendered"]["policy_pixel"]))
    source_paths = [Path(__file__), a.config, a.data / "manifest.json", a.checkpoint / "model.safetensors",
                    Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/base_env.py"), Path("src/real2sim/sim/model.py"), Path("src/real2sim/artifacts.py"),
                    Path("src/real2sim/policy/adapters.py"), Path("src/real2sim/sim/cameras.py")]
    report = {
        "kind": "ONE_VALIDATION_ANCHOR_IMAGE_VELOCITY_FACTORIAL_DIAGNOSTIC",
        "learned_task_success": None, "physical_calibration_accepted": False,
        "anchor": {"tick": tick, "episode": episode,
                   "source_frame": int(ds.arrays["frame_index"][tick]),
                   "condition": manifest["split"]["conditions"][str(episode)]},
        "checkpoint": str(a.checkpoint), "data": str(a.data), "config": str(a.config),
        "source_sha256": {str(p): file_sha256(p) for p in source_paths},
        "task_sha256": task_hash, "checkpoint_task_sha256": contract["calibration_sha256"],
        "joint_mapping": mapping, "reserved_episodes_used": [],
        "recorded_state": recorded_state.tolist(), "simulator_reset_state": simulated_state.tolist(),
        "reset_minus_recorded_state": (simulated_state - recorded_state).tolist(),
        "reset_minus_recorded_standardized_state": (z_reset - z_anchor).tolist(),
        "recorded_state_train_feature_percentiles": (100 * (z_train <= z_anchor).mean(axis=0)).tolist(),
        "reset_state_train_feature_percentiles": (100 * (z_train <= z_reset).mean(axis=0)).tolist(),
        "recorded_state_max_abs_z": float(np.abs(z_anchor).max()),
        "reset_state_max_abs_z": float(np.abs(z_reset).max()),
        "first_two_recorded_actions": ds.arrays["action"][tick:tick + 2].tolist(),
        "state_and_image_cases": cases,
        "simulator_ball_xyz_native_m": ball.tolist(), "simulator_ee_xyz_native_m": ee.tolist(),
        "simulator_ee_ball_distance_m": metrics["ee_ball_distance_m"],
        "simulator_camera_intrinsic_source_pixels": intrinsic.tolist(),
        "simulator_ball_projected_source_pixel": projected.tolist(),
        "yellow_ball_detections": detections,
        "yellow_ball_centroid_displacement_policy_pixels": displacement,
        "image_pair_recorded_left_simulated_right": str(image_pair_path),
        "limitations": "One teacher-forced validation anchor. No fit or object relocation. The recorded episode has ball_shift condition while simulator uses baseline task region. Image sensitivity and standardized feature overlap are not task success or calibration acceptance.",
    }
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"anchor": report["anchor"], "cases": cases,
                      "centroid_displacement": displacement, "max_state_z": report["recorded_state_max_abs_z"]}, indent=2))


if __name__ == "__main__":
    main()
