"""Audit the real-data ACT bridge without inventing a physical registration.

Scores the HF checkpoint's declared validation episodes, recomputes existing
camera confirmation errors, and records exactly what the available measurements
can identify. No fit, joint clipping, scene correction or policy training occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from real2sim.config import load_yaml
from real2sim.data.raw import RawDataset, sha256_file
from real2sim.geometry.so101_fk import ARM_JOINTS
from real2sim.policy.visual_targets import camera_pose_task
from real2sim.sim.cameras import camera_intrinsics_px
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

SNAPSHOT = Path(".cache/hf_models_20261027/models--bottowork--so101-act-h256-b16/snapshots/e696c8a078935432e3501e2c28c1fb2598fbff7e")
SCENE = Path("configs/v2/sim_so101.waypoint_20261026.yaml")
CONFIRMATION = Path("results/v2/fresh_model_camera_confirmation_20261024.json")
CAMERA = Path("outputs/v2/model_multiframe_camera_20261024_v4/report.json")


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def project(points, intrinsic, transform):
    points, intrinsic, transform = map(np.asarray, (points, intrinsic, transform))
    return cv2.projectPoints(points.astype(float), cv2.Rodrigues(transform[:3, :3])[0],
                             transform[:3, 3], intrinsic.astype(float), None)[0].reshape(-1, 2)


def errors(predicted, observed):
    error = np.linalg.norm(np.asarray(predicted) - np.asarray(observed), axis=1)
    return {"per_corner_error_px": error.tolist(), "rms_px": float(np.sqrt(np.mean(error**2))),
            "p95_px": float(np.percentile(error, 95))}


def lattice(values, calibration):
    result = {}
    for j, name in enumerate(ARM_JOINTS):
        lo, hi = (calibration[name][key] for key in ("range_min", "range_max"))
        tick = values[:, j] * 4095 / 360 + (lo + hi) / 2
        alternative = (values[:, j] + 100) / 200 * (hi - lo) + lo
        result[name] = {
            "degree_inverse_tick_residual_p95": float(np.percentile(abs(tick - np.rint(tick)), 95)),
            "alternative_pm100_tick_residual_p95": float(np.percentile(abs(alternative - np.rint(alternative)), 95)),
            "degree_at_stored_homing_pose_2047": (2047 - (lo + hi) / 2) * 360 / 4095,
            "homing_pose_is_CAD_neutral_established": False,
        }
    return result


def audit(root, calibration_path, out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    raw = RawDataset(root)
    experiment = json.loads((SNAPSHOT / "experiment.json").read_text())
    calibration = json.loads(calibration_path.read_text())
    geometry = json.loads(Path("configs/v2/geometry_landmarks.template.json").read_text())
    inventory = json.loads(Path("results/v2/fixed_base_multilevel_inventory_20261024.json").read_text())
    task_path = Path(load_yaml(SCENE)["task_config"])
    inputs = [Path(__file__), SCENE, task_path, calibration_path, CONFIRMATION, CAMERA,
              SNAPSHOT / "experiment.json", SNAPSHOT / "model.safetensors",
              SNAPSHOT / "latest/model.safetensors", root / "meta/info.json",
              Path("configs/v2/geometry_landmarks.template.json"),
              Path("results/v2/fixed_base_multilevel_inventory_20261024.json"),
              Path("configs/v2/measurements.operator_clarified_20261026.json"),
              Path("physical_measurements.md"),
              Path("assets/menagerie/robotstudio_so101/so101.xml")]
    inputs += sorted((root / "data").rglob("*.parquet"))
    inputs += sorted((root / "meta/episodes").rglob("*.parquet"))
    inputs += sorted({raw.video_segment(key, ep).path
                      for ep in experiment["val_episodes"] for key in raw.image_keys})
    hashes = {str(path): sha256_file(path) for path in inputs}
    with So101PickPlaceEnv(SCENE, render_images=False) as env:
        env.reset(seed=0)
        limits = env.limits.copy()
        names = list(ARM_JOINTS) + ["gripper"]
        episodes, all_state = [], []
        for ep in experiment["val_episodes"]:
            rows = raw.episode_rows(ep)
            state, action = (raw.stack(rows, key) for key in ("observation.state", "action"))
            all_state.append(state)
            record = {"episode": ep, "rows": len(rows), "fps": raw.fps,
                      "first_follower_state_deg_percent": state[0].tolist()}
            for key, values in (("follower_state", state), ("recorded_action", action)):
                native = np.stack([sim_from_real(env.mapping, value) for value in values])
                outside = (native < limits[:, 0]) | (native > limits[:, 1])
                compatible = np.flatnonzero(~outside.any(axis=1))
                first = int(compatible[0]) if len(compatible) else None
                record[key] = {
                    "rows_outside_any_native_limit": int(outside.any(axis=1).sum()),
                    "outside_by_joint": {name: int(outside[:, j].sum()) for j, name in enumerate(names)},
                    "min_deg_percent": values.min(axis=0).tolist(),
                    "max_deg_percent": values.max(axis=0).tolist(),
                    "first_native_compatible_frame": first,
                    "first_native_compatible_timestamp_s": None if first is None else float(rows.iloc[first]["timestamp"]),
                    "first_compatible_is_approved_task_start": False,
                }
            episodes.append(record)
        states = np.concatenate(all_state)
        lower, upper = np.rad2deg(limits[:5, 0]), np.rad2deg(limits[:5, 1])
        offset_intervals = {}
        for j, name in enumerate(ARM_JOINTS):
            lo = float(lower[j] - states[:, j].min())
            hi = float(upper[j] - states[:, j].max())
            offset_intervals[name] = {"additive_offset_interval_deg": [lo, hi],
                                      "interval_nonempty": lo <= hi,
                                      "physical_offset_identified": False}
        cid = env.model.camera("iphone").id
        current_K = camera_intrinsics_px(env.model, cid, 1280, 720)
        current_T = np.linalg.inv(camera_pose_task(env.data.cam_xpos[cid], env.data.cam_xmat[cid], env.offset))
        mat = env.task["table"]["mat"]
        center, half = np.asarray(mat["center_task_xy"]), np.asarray(mat["size_xy"]) / 2
        # Declared geometric corner order is top-left, top-right, bottom-right,
        # bottom-left for the existing camera and task axes. No best permutation fit.
        scene_corners = np.c_[center + half * [[-1, 1], [1, 1], [1, -1], [-1, -1]],
                             np.full(4, env.table_top - env.offset[2])]
        current_prediction = project(scene_corners, current_K, current_T)
        mapping = {"sign": list(env.mapping.sign), "offset_deg": list(env.mapping.offset_deg),
                   "native_limits_rad": limits.tolist(), "native_arm_limits_deg": np.rad2deg(limits[:5]).tolist()}
    confirmation, camera = (json.loads(p.read_text()) for p in (CONFIRMATION, CAMERA))
    plane_points = [[0., 0., 0.], [.305, 0., 0.], [.305, .406, 0.], [0., .406, 0.]]
    camera_rows, overlays = [], []
    for row in confirmation["frames"]:
        candidate = camera["regimes"][row["camera_regime"]]["baseline"]
        prediction = project(plane_points, candidate["K"], candidate["T_optical_plane"])
        # Regression: our independent rescore must reproduce the frozen report.
        score = errors(prediction, row["corners_source_px"])
        if not np.isclose(score["rms_px"], row["rms_px"], atol=1e-8, rtol=0):
            raise ValueError("confirmation source or projection contract changed")
        source = Path(row["path"])
        if sha256_file(source) != row["sha256"]:
            raise ValueError("source confirmation pixels changed")
        hashes[str(source)] = row["sha256"]
        current_score = errors(current_prediction, row["corners_source_px"])
        camera_rows.append({"episode": row["episode"], "frame": row["frame"], "regime": row["camera_regime"],
                            "source": str(source), "source_sha256": row["sha256"],
                            "current_scene_mat_projection": current_score,
                            "frozen_conditional_mat_camera": score,
                            "candidate_passes_existing_3px_gate": score["rms_px"] <= 3.,
                            "current_scene_prediction_source_px": current_prediction.tolist(),
                            "conditional_candidate_prediction_source_px": prediction.tolist()})
        image = cv2.imread(str(source))
        for corners, color in ((row["corners_source_px"], (255, 255, 255)),
                               (current_prediction, (80, 80, 255)), (prediction, (0, 220, 220))):
            cv2.polylines(image, [np.rint(corners).astype(np.int32)], True, color, 3)
        image = cv2.resize(image, (640, 360))
        banner = np.zeros((56, 640, 3), np.uint8)
        cv2.putText(banner, f"ep{row['episode']}: scene RMS {current_score['rms_px']:.1f}px; candidate {score['rms_px']:.1f}px", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.putText(banner, "White: measured edges | Red: scene | Yellow: conditional fit", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        overlays.append(np.vstack([image, banner]))
    cv2.imwrite(str(out / "camera_projection_comparison.jpg"), np.hstack(overlays))
    arm_lattice = lattice(states, calibration)
    encoder_slope = (calibration["gripper"]["range_max"] - calibration["gripper"]["range_min"]) * (2 * np.pi / 4095) / 100
    task_mapping = load_yaml(task_path)["robot_so101"]["gripper"]
    gripper_slope = np.diff(task_mapping["sim_range_rad"])[0] / 100
    report = {
        "schema": "real_data_ACT_registration_bridge_20261027_v1", "status": "complete",
        "physical_calibration_accepted": False, "configuration_correction_adopted": False,
        "optimization_performed": False, "reference_clipping": False,
        "scope": {"dataset_root": str(root), "HF_validation_episodes": experiment["val_episodes"],
                  "validation_rows": len(states), "split_source": str(SNAPSHOT / "experiment.json"),
                  "HF_split_differs_from_historical_project_reserved_split": True,
                  "model_weights_changed": False, "task_success_used_for_parameter_selection": False},
        "current_mapping": mapping, "validation_episode_compatibility": episodes,
        "arm_degree_unit_checks": arm_lattice,
        "limit_only_offset_feasibility_not_calibration": offset_intervals,
        "gripper_scale": {"current_rad_per_percent": float(gripper_slope),
                          "encoder_rad_per_percent_if_direct_1to1_horn": float(encoder_slope),
                          "current_scale_below_encoder_fraction": float(1 - gripper_slope / encoder_slope),
                          "closed_zero_physically_measured": False, "scale_correction_adopted": False},
        "camera_confirmation_rescore": camera_rows,
        "camera_interpretation": "Current scene errors combine camera and mat/base registration. Conditional candidates operate in their own mat frame and cannot replace only the camera in the current task. Both fail the existing 3px frozen confirmation gate; comparison does not identify a physical correction.",
        "evidence_inventory": {
            "verified_static_CAD_landmark_identities": inventory["existing_verified_identity_count"],
            "candidate_static_landmarks": inventory["existing_inventory_count"],
            "complete_camera_registration_rows": sum(all(r.get(k) is not None for k in ("capture_group", "source_frame_reference", "landmark_id", "point_task_m", "source_pixel_xy", "sigma_px")) for r in geometry["camera_registration"]["observations"]),
            "complete_joint_registration_rows": sum(all(r.get(k) is not None for k in ("capture_group", "source_frame_reference", "landmark_id", "q_nominal_rad", "gripper_joint_rad", "body_name", "point_body_m", "source_pixel_xy", "sigma_px")) for r in geometry["joint_zero_registration"]["observations"]),
            "source_resolution_intrinsics_measured": geometry["camera"]["K_source_pixels"] is not None,
            "wrist_intrinsics_measured": geometry["wrist_camera"]["K_source_pixels"] is not None,
            "physical_signs_and_zeros_verified": False,
        },
        "decision": "Retain the declared provisional mapping for controlled HF retests. Degree units are supported, but no independently justified physical neutral/sign/camera correction can be adopted from the available files.",
        "specific_missing_evidence": [
            "An independently known shoulder-lift physical pose relative to the base, paired with follower readback; the servo range midpoint and stored homing offset do not establish CAD neutral.",
            "Source-resolution intrinsics/distortion for each camera capture regime, plus a target-to-base transform independent of the arm FK under test and named fixed landmarks with verification captures.",
            "A measured jaw gap at known gripper percentages, including a closed-zero reference; current encoder span alone does not establish physical closure/TCP.",
        ],
        "input_sha256_before": hashes,
    }
    after = {path: sha256_file(Path(path)) for path in hashes}
    report.update(input_sha256_after=after, changed_inputs=[path for path in hashes if hashes[path] != after[path]])
    if report["changed_inputs"]:
        raise ValueError("source evidence changed while audit ran")
    write(out / "report.json", report)
    (out / "executed_audit.py").write_bytes(Path(__file__).read_bytes())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("~/.cache/huggingface/lerobot/local/so101_act_pickplace_v2").expanduser())
    parser.add_argument("--follower-calibration", type=Path, default=Path("~/.cache/huggingface/lerobot/calibration/robots/so_follower/aligned_follower.json").expanduser())
    parser.add_argument("--out", type=Path, default=Path("outputs/v2/registration_bridge_20261027"))
    args = parser.parse_args()
    result = audit(args.root, args.follower_calibration, args.out)
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "camera": result["camera_confirmation_rescore"],
                      "decision": result["decision"]}, indent=2))
