"""Fit constrained joint/jaw correspondence from image tips, without task outcomes.

Camera, meshes, link lengths and table remain fixed. Two near/far identity hypotheses
avoid assuming which real finger is moving. Only annotated training episodes enter
optimization. Historical validation is reported; no candidate is automatically adopted.
"""

import argparse
import copy
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml
from diag_camera_landmark_refit import Model, load_rows
from scipy.optimize import least_squares

from real2sim.data.paired_approach import camera_parameters
from real2sim.policy.paired import file_sha256
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "results/metrics/dinov2_act_pilot_20261015/configs/sim_so101.yaml"))
    parser.add_argument("--annotations", type=Path, default=Path(
        "results/audit/geometry_control_20261016/g1_jaw_review/annotations.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-arm-landmarks", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    annotations = json.loads(args.annotations.read_text())
    arrays = np.load(".cache/derived_joint/arrays.npz")
    rows = annotations["rows"]
    states = []
    for row in rows:
        index = np.flatnonzero((arrays["episode_index"] == row["episode"]) &
                              (arrays["frame_index"] == row["frame"]))
        if len(index) != 1:
            raise ValueError(row)
        states.append(arrays["state"][index[0]].astype(float))
    train = [i for i, row in enumerate(rows) if row["episode"] in annotations["train_episodes"]]
    val = [i for i in range(len(rows)) if i not in train]
    report = {"kind": "IMAGE_ONLY_JAW_REGISTRATION_FIT_NOT_ADOPTED",
              "annotations_sha256": file_sha256(args.annotations),
              "config_sha256": file_sha256(args.config),
              "selection": "Optimize/select by training jaw image residual only; no contact/success feedback",
              "limits": "Arm first four zero offsets bounded +/-10deg; wrist-roll +/-180deg is an identity hypothesis, not verified calibration; gripper affine gain0.8..1.2 and zero+/-20deg; native joint limits unchanged",
              "models": {}}
    with So101PickPlaceEnv(args.config, render_images=False) as env:
        env.reset(seed=0)
        rv, tv, K, _, _ = camera_parameters(env)
        geom_ids = [env.model.geom(name).id for name in ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1")]
        old_landmarks = json.loads(Path("results/audit/camera_landmark_refit_float64_20261014/landmarks.json").read_text())
        old_report = json.loads(Path("results/audit/camera_landmark_refit_float64_20261014/report.json").read_text())
        old_fit = old_report["models"]["M1_refit_camera"]
        old_rows = load_rows(env, old_landmarks, arrays)
        arm_model = Model(env, old_rows, old_landmarks["mat_corners_px"],
                          old_report["train_episodes"], old_report["camera_prior"])

        def arm_errors(params):
            points = arm_model.landmark_points(params[:4], np.array(old_fit["horn_axis_offsets_m"]),
                                               np.array(old_fit["roll_servo_offset_wrist_frame_m"]))
            projected = cv2.projectPoints(points, rv, tv, K, None)[0].reshape(-1, 2)
            return projected - np.array([r["px"] for r in old_rows])

        def predictions(params):
            outputs = []
            for state in states:
                q = sim_from_real(env.mapping, state)
                q[:5] += np.deg2rad(params[:5])
                q[5] = params[5] * q[5] + np.deg2rad(params[6])
                env.data.qpos[env.qadr] = q
                mujoco.mj_kinematics(env.model, env.data)
                xyz = env.data.geom_xpos[geom_ids] - env.offset
                outputs.append(cv2.projectPoints(xyz, rv, tv, K, None)[0].reshape(2, 2))
            return np.array(outputs)

        def errors(params, near_index, indices):
            points = predictions(params)
            residuals = []
            for i in indices:
                residuals.append(points[i, near_index] - rows[i]["near"])
                if "far" in rows[i]:
                    residuals.append(points[i, 1 - near_index] - rows[i]["far"])
            return np.array(residuals)

        def residual(params, near_index):
            image = errors(params, near_index, train).ravel() / annotations["sigma_px"]
            priors = np.r_[params[:4] / 5.0, params[4] / 90.0,
                           (params[5] - 1.0) / .1, params[6] / 10.0]
            parts = [image, priors]
            if args.include_arm_landmarks:
                res = arm_errors(params) / np.array([r["sigma_px"] for r in old_rows])[:, None]
                parts.append(res[arm_model.train].ravel())
            return np.concatenate(parts)

        def score(params, near_index):
            by_split = {}
            for name, indices in (("train", train), ("historical_validation", val)):
                distance = np.linalg.norm(errors(params, near_index, indices), axis=1)
                by_split[name] = {"landmarks": len(distance),
                                  "rms_px": float(np.sqrt(np.mean(distance ** 2))),
                                  "median_px": float(np.median(distance)),
                                  "max_px": float(np.max(distance))}
            projected = predictions(params)
            old_error = np.linalg.norm(arm_errors(params), axis=1)
            return {"parameters": params.tolist(), "near_is": ["fixed", "moving"][near_index],
                    "old_arm_landmark_rms_px": {
                        "train": float(np.sqrt(np.mean(old_error[arm_model.train] ** 2))),
                        "historical_validation": float(np.sqrt(np.mean(old_error[~arm_model.train] ** 2)))},
                    **by_split, "rows": [dict(row, projected_fixed_px=projected[i, 0].tolist(),
                                             projected_moving_px=projected[i, 1].tolist())
                                         for i, row in enumerate(rows)]}

        neutral = np.r_[np.zeros(5), 1., 0.]
        for near_index, name in enumerate(("near_fixed", "near_moving")):
            report["models"][f"baseline_{name}"] = score(neutral, near_index)
            runs = []
            for roll in (-150., -75., 0., 75., 150.):
                initial = neutral.copy()
                initial[4] = roll
                fit = least_squares(residual, initial, args=(near_index,),
                                    bounds=([-10] * 4 + [-180, .8, -20],
                                            [10] * 4 + [180, 1.2, 20]),
                                    loss="soft_l1", max_nfev=500, diff_step=1e-4)
                runs.append(fit)
            best = min(runs, key=lambda value: value.cost)
            report["models"][f"fit_{name}"] = score(best.x, near_index) | {
                "training_objective": float(best.cost), "optimizer_success": bool(best.success)}

        best_name = min(("fit_near_fixed", "fit_near_moving"),
                        key=lambda name: report["models"][name]["training_objective"])
        report["train_selected_candidate"] = best_name
        report["old_arm_landmarks_in_training_objective"] = args.include_arm_landmarks
        params = np.array(report["models"][best_name]["parameters"])
        cfg = yaml.safe_load(args.config.read_text())
        task = yaml.safe_load(Path(cfg["task_config"]).read_text())
        candidate = copy.deepcopy(task)
        candidate["robot_so101"]["real_to_sim"]["offset_deg"] = (
            np.array(task["robot_so101"]["real_to_sim"]["offset_deg"]) + params[:5]).tolist()
        candidate["robot_so101"]["real_to_sim"]["provenance"] = "DIAGNOSTIC image-only jaw registration; not physically verified"
        candidate["robot_so101"]["gripper"]["sim_range_rad"] = (
            np.array(task["robot_so101"]["gripper"]["sim_range_rad"]) * params[5] +
            np.deg2rad(params[6])).tolist()
        task_path = args.out / "task_candidate.yaml"
        task_path.write_text(yaml.safe_dump(candidate, sort_keys=False))
        cfg["task_config"] = str(task_path.resolve())
        q0 = sim_from_real(env.mapping, np.array(cfg["initial_qpos_real_deg"]))
        q0[:5] += np.deg2rad(params[:5])
        q0[5] = params[5] * q0[5] + np.deg2rad(params[6])
        cfg["initial_qpos"] = np.clip(q0, *env.limits.T).tolist()
        (args.out / "sim_candidate.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {key: value[key] for key in ("near_is", "parameters", "train", "historical_validation")}
                      for name, value in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
