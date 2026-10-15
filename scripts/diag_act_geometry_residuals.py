"""Post-hoc geometry consistency diagnostic; never changes calibration or policy inputs."""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares

from real2sim.data.paired_approach import camera_parameters, pose_render
from real2sim.sim.env import So101PickPlaceEnv


def main(config, out):
    if out.exists():
        raise FileExistsError(out)
    audit = json.loads(Path("results/audit/paired_approach_3988577/report.json").read_text())
    response = json.loads(Path("results/audit/follower_response_approach_20261013/model.json").read_text())
    env = So101PickPlaceEnv(config, render_images=False)
    try:
        env.reset(seed=0)
        rv, tv, K, camera, _ = camera_parameters(env)
        Rcam = cv2.Rodrigues(rv)[0]
        rows, transforms = [], []
        for c in audit["correspondences"]:
            if c["kind"] != "held":
                continue
            pose_render(env, c["state_deg"], env.task["ball"]["start_region"]["center_task_xy"])
            T = env.ee_pose()
            ray = Rcam.T @ np.linalg.solve(K, np.r_[c["pixel"], 1.0])
            ray /= np.linalg.norm(ray)
            delta = T[:3, 3] - camera
            minimum = np.linalg.norm(delta - ray * np.dot(delta, ray))
            rows.append({"episode": c["episode"], "frame": c["frame"], "pixel": c["pixel"],
                         "split": "train" if c["episode"] in response["fit_episodes"] else "val",
                         "minimum_tool_to_ball_offset_m": float(minimum),
                         "table_contact_at_recorded_pose": env._robot_table_penetration()})
            transforms.append(T)
        transforms = np.asarray(transforms)
        observed = np.array([r["pixel"] for r in rows])
        train = np.array([r["split"] == "train" for r in rows])

        def project(offset):
            xyz = transforms[:, :3, 3] + transforms[:, :3, :3] @ offset
            return cv2.projectPoints(xyz, rv, tv, K, None)[0].reshape(-1, 2)

        # Nuisance-parameter audit only: fit a common ball-in-gripper offset on
        # existing TRAIN correspondences, never refit the camera or joint mapping.
        fit = least_squares(lambda x: (project(x)[train] - observed[train]).ravel(), np.zeros(3))
        offsets = {"tool_site_proxy": np.zeros(3), "oracle_grasp_offset": np.array([-.01, 0, .03]),
                   "train_fitted_constant_offset": fit.x}
        scores = {}
        for name, offset in offsets.items():
            errors = np.linalg.norm(project(offset) - observed, axis=1)
            scores[name] = {"offset_m": offset.tolist(), "norm_m": float(np.linalg.norm(offset)),
                            "train_rms_px": float(np.sqrt(np.mean(errors[train] ** 2))),
                            "val_rms_px": float(np.sqrt(np.mean(errors[~train] ** 2)))}
            for row, err in zip(rows, errors, strict=True):
                row[name + "_error_px"] = float(err)
        report = {"kind": "POSTHOC_GEOMETRY_CONSISTENCY_NOT_CALIBRATION",
                  "config": str(config),
                  "source": "results/audit/paired_approach_3988577/report.json",
                  "n_train": int(train.sum()), "n_val": int((~train).sum()),
                  "scores": scores, "rows": rows,
                  "caveats": ["Held ball centre is not the gripper site; original residual is a proxy.",
                              "Observed pixels/held labels, joint mapping, image timing and camera are all uncertain.",
                              "Fitted offset is an exploratory nuisance test, not a physical measurement or config change.",
                              "Repeated held frames from each episode are correlated; validation has been reused."]}
        # Reuse the existing hand-read robot landmarks, independently of ball
        # location. These observations have roughly 15 px annotation uncertainty.
        # Source: scripts/diag_camera_fit_robot.py; no new anchors selected here.
        observations = {
            0: {"shoulder_lift": (305, 235), "elbow_flex": (75, 210), "wrist_flex": (335, 118)},
            150: {"shoulder_lift": (315, 220), "elbow_flex": (60, 180), "wrist_flex": (355, 95)},
            240: {"shoulder_lift": (330, 215), "elbow_flex": (545, 55)},
            330: {"shoulder_lift": (330, 215), "elbow_flex": (535, 50), "wrist_flex": (690, 140)},
        }
        anchors = []
        with np.load(".cache/derived_joint/arrays.npz") as arrays:
            for frame, points in observations.items():
                idx = np.flatnonzero((arrays["episode_index"] == 5) & (arrays["frame_index"] == frame))
                if len(idx) != 1:
                    raise ValueError("expected one matching recorded state")
                pose_render(env, arrays["state"][idx[0]], env.task["ball"]["start_region"]["center_task_xy"])
                for joint, pixel in points.items():
                    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    xyz = env.data.xanchor[jid] - env.offset
                    projected = cv2.projectPoints(xyz.reshape(1, 3), rv, tv, K, None)[0].reshape(2)
                    anchors.append({"episode": 5, "frame": frame, "joint": joint,
                                    "observed_pixel": pixel, "projected_pixel": projected.tolist(),
                                    "error_px": float(np.linalg.norm(projected - pixel))})
        report["robot_landmarks"] = {
            "source": "scripts/diag_camera_fit_robot.py OBS, excluding ambiguous tip landmark",
            "caveat": "Hand-read train-episode landmarks, approximately 15 px uncertainty; no image-lag correction.",
            "rms_px": float(np.sqrt(np.mean([r["error_px"] ** 2 for r in anchors]))),
            "rows": anchors,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(scores, indent=2))
        print("minimum offset median m", np.median([r["minimum_tool_to_ball_offset_m"] for r in rows]))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("results/metrics/geometry_3988577/configs/sim_so101.yaml"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/audit/act_root_cause_20261014/geometry_residuals.json"))
    args = parser.parse_args()
    main(args.config, args.out)
