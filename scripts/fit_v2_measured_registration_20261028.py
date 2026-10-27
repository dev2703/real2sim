"""Fit an explicitly provisional measured-layout camera/joint hypothesis.

Only recorded image landmarks and operator geometry choose parameters. ACT
outputs and simulated task success are excluded. Lighting is never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml
from diag_v2_conditional_joint_zeros import CAPTURES
from scipy.optimize import least_squares

from real2sim.artifacts import sha256_file
from real2sim.geometry.so101_fk import ARM_JOINTS
from real2sim.serialization import write_json

OUT = Path("configs/v2/measured_registration_20261028")
DATA = Path(".cache/v2_exploratory_20261022/cartesian_iphone_top/arrays.npz")
PHOTO = Path("outputs/v2/iphone_stability_20261021/source_frames/ep00_f0200.png")
MAT_ANNOTATIONS = Path("configs/v2/mat_camera.annotations_20261022.json")
STATIC = Path("outputs/v2/static_base_camera_20261023/report.json")


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def look_at(position, target):
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return cv2.Rodrigues(np.stack([right, down, forward]))[0].ravel()


class Fit:
    def __init__(self):
        from diag_v2_mat_camera import fit_edges

        annotations = json.loads(MAT_ANNOTATIONS.read_text())
        self.mat_pixels, _ = fit_edges(annotations["edge_samples_source_px"])
        self.arrays = np.load(DATA)
        self.model = mujoco.MjModel.from_xml_path("assets/menagerie/robotstudio_so101/so101.xml")
        self.data = mujoco.MjData(self.model)
        self.qadr = self.model.jnt_qposadr[
            [self.model.joint(n).id for n in (*ARM_JOINTS, "gripper")]
        ]
        self.late = np.asarray(
            json.loads(STATIC.read_text())["image_similarity_models"]["without_fixed_base_foot"][
                "matrix_reference_to_late"
            ]
        )
        self.captures = []
        for capture in CAPTURES:
            ids = np.flatnonzero(
                (self.arrays["episode_index"] == capture["episode"])
                & (self.arrays["frame_index"] == capture["frame"])
            )
            self.captures.append(
                {**capture, "state": np.asarray(self.arrays["joint_state"][ids[0]], float)}
            )

    @staticmethod
    def geometry(x):
        yaw = x[0]
        R2 = rotation(yaw)
        ball = R2 @ np.array([0.0508, 0.3302])
        camera = np.r_[R2 @ np.array([0.8255, 0.4445]), 0.3944]
        bowl = ball + 0.3048 * np.array([np.cos(x[14]), np.sin(x[14])])
        return ball, camera, bowl

    @staticmethod
    def project(points, x, camera):
        matrix = cv2.Rodrigues(x[1:4])[0]
        optical = np.asarray(points) @ matrix.T - matrix @ camera
        # Smoothly finite even when a poor initial guess puts a point behind
        # the camera; all selected solutions report minimum depth separately.
        depth = np.maximum(optical[:, 2], 0.02)
        return x[4] * optical[:, :2] / depth[:, None] + [640.0, 360.0], optical[:, 2]

    def components(self, x):
        ball, camera, bowl = self.geometry(x)
        mat_local = np.array(
            [
                [-0.3048 / 2, 0.4064 / 2],
                [0.3048 / 2, 0.4064 / 2],
                [0.3048 / 2, -0.4064 / 2],
                [-0.3048 / 2, -0.4064 / 2],
            ]
        )
        mat = np.c_[mat_local @ rotation(x[7]).T + x[5:7], np.full(4, -0.012)]
        mat_projection, depths = self.project(mat, x, camera)
        horn_rows = []
        for capture in self.captures:
            state = capture["state"]
            q = np.r_[
                np.deg2rad(state[:5] + np.r_[x[11:14], 0.0, 0.0]),
                -0.1745 + state[5] / 100 * (1.7454 + 0.1745),
            ]
            self.data.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            for name, observed in capture["horns"].items():
                jid = self.model.joint(name).id
                horn = ("shoulder_lift", "elbow_flex", "wrist_flex").index(name)
                position = self.data.xanchor[jid] + self.data.xaxis[jid] * x[8 + horn]
                prediction, depth = self.project(position[None], x, camera)
                prediction = prediction[0]
                if capture["episode"] >= 7:
                    prediction = self.late @ np.r_[prediction, 1.0]
                horn_rows.append(
                    {
                        "episode": capture["episode"],
                        "frame": capture["frame"],
                        "split": capture["split"],
                        "joint": name,
                        "observed_px": observed,
                        "predicted_px": prediction.tolist(),
                        "error_xy_px": (prediction - observed).tolist(),
                        "depth_m": float(depth[0]),
                    }
                )
        radius = 0.2032 / (2 * np.pi)
        ball_point = np.r_[ball, -0.012 + radius + x[15]]
        ball_pixels, ball_depth = self.project(ball_point[None], x, camera)
        ball_projected_radius = x[4] * radius / max(float(ball_depth[0]), 0.02)
        bowl_pixels, bowl_depth = self.project(np.r_[bowl, 0.030 - 0.012][None], x, camera)
        return {
            "horn_rows": horn_rows,
            "mat_prediction": mat_projection,
            "ball_prediction": ball_pixels[0],
            "bowl_prediction": bowl_pixels[0],
            "ball_projected_radius": ball_projected_radius,
            "minimum_depth_m": float(
                min(
                    *depths,
                    float(ball_depth[0]),
                    float(bowl_depth[0]),
                    *(r["depth_m"] for r in horn_rows),
                )
            ),
        }

    def residual(self, x):
        parts = self.components(x)
        horns = np.array(
            [row["error_xy_px"] for row in parts["horn_rows"] if row["split"] == "train"]
        )
        # Centres/radius read on original ep0:200 before any candidate render.
        # Bowl's approximate rim centre is deliberately given wider uncertainty.
        return np.r_[
            horns.ravel() / 8,
            (parts["mat_prediction"] - self.mat_pixels).ravel() / 3,
            (parts["ball_prediction"] - [743.0, 237.0]) / 5,
            (parts["bowl_prediction"] - [846.0, 499.0]) / 8,
            (parts["ball_projected_radius"] - 51.0) / 4,
            x[11:14] / 20.0,
            x[8:11] / 0.04,
        ]

    def score(self, x):
        parts = self.components(x)
        return {
            "horns": {
                split: {
                    "n": len(rows),
                    "rms_px": float(
                        np.sqrt(
                            np.mean(
                                [np.dot(row["error_xy_px"], row["error_xy_px"]) for row in rows]
                            )
                        )
                    ),
                }
                for split in ("train", "val")
                if (rows := [r for r in parts["horn_rows"] if r["split"] == split])
            },
            "mat_rms_px": float(
                np.sqrt(np.mean(np.sum((parts["mat_prediction"] - self.mat_pixels) ** 2, axis=1)))
            ),
            **parts,
        }


def run(report_path):
    if report_path.exists():
        raise FileExistsError(report_path)
    fit = Fit()
    solutions = []
    bounds = (
        np.r_[
            -4 * np.pi,
            np.full(3, -4 * np.pi),
            500.0,
            -0.6,
            -0.6,
            -4 * np.pi,
            np.full(3, -0.04),
            -80.0,
            -30.0,
            -30.0,
            -4 * np.pi,
            0.0,
        ],
        np.r_[
            4 * np.pi,
            np.full(3, 4 * np.pi),
            3000.0,
            0.8,
            0.8,
            4 * np.pi,
            np.full(3, 0.04),
            80.0,
            30.0,
            30.0,
            4 * np.pi,
            0.03,
        ],
    )
    for degrees in (-50.0, -80.0, -110.0, -140.0):
        yaw = np.deg2rad(degrees)
        ball = rotation(yaw) @ [0.0508, 0.3302]
        bowl = ball + 0.3048 * (rotation(yaw) @ [0.674248302, -0.738504724])
        camera = np.r_[rotation(yaw) @ [0.8255, 0.4445], 0.3944]
        target = np.r_[(ball + bowl) / 2, 0.02]
        start = np.r_[
            yaw,
            look_at(camera, target),
            1400.0,
            (ball + bowl) / 2,
            yaw + np.pi / 2,
            np.zeros(3),
            0.0,
            5.0,
            -5.0,
            np.arctan2(*(bowl - ball)[::-1]),
            0.005,
        ]
        result = least_squares(
            fit.residual,
            start,
            bounds=bounds,
            loss="soft_l1",
            f_scale=2.0,
            x_scale="jac",
            max_nfev=1200,
            diff_step=1e-4,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        solutions.append(result)
        print(
            json.dumps(
                {
                    "start_operator_yaw_deg": degrees,
                    "cost": result.cost,
                    "parameters": result.x.tolist(),
                    "nfev": result.nfev,
                }
            ),
            flush=True,
        )
    best = min(solutions, key=lambda result: result.cost)
    x = best.x
    ball, camera, bowl = fit.geometry(x)
    R = cv2.Rodrigues(x[1:4])[0]
    task_path = OUT / "task_positive_all_scalars.yaml"
    task = yaml.safe_load(task_path.read_text())
    task["ball"]["start_region"]["center_task_xy"] = ball.tolist()
    task["bowl"]["center_task_xy"] = bowl.tolist()
    task["table"]["mat"].update(size_xy=[.3048, .4064], center_task_xy=x[5:7].tolist(), yaw_deg=float(np.rad2deg(x[7])))
    task["camera_iphone"].update(
        approx_position_task=camera.tolist(),
        intrinsics_px={"fx": float(x[4]), "fy": float(x[4]), "cx": 640.0, "cy": 360.0},
        fovy_deg=float(np.rad2deg(2 * np.arctan(360 / x[4]))),
        xyaxes_task=np.r_[R[0], -R[1]].tolist(),
        look_at_task=(camera + R[2]).tolist(),
        extrinsics_provenance="Provisional joint image+operator measurement fit20261028; fixed scalargeometry, freeoperator-to-CADyaw/opticalrotation/focal; not physicalcalibration",
    )
    task["robot_so101"]["real_to_sim"].update(
        offset_deg=np.r_[x[11:14], 0.0, 0.0].tolist(),
        provenance="Provisional camera+horn fit20261028 from sourceimages only; no tasksuccess objective",
    )
    task["geometry_status"] = "provisional_measured_geometry_source_landmark_fit_20261028"
    task["measured_registration_hypothesis"] = {
        "operator_to_task_yaw_deg": float(np.rad2deg(x[0])),
        "ball_support_height_estimate_m": float(x[15]),
        "support_geometry_applied": False,
        "lighting_changed": False,
        "task_success_used_for_selection": False,
    }
    candidate = OUT / "task_landmark_fit.yaml"
    candidate.write_text(yaml.safe_dump(task, sort_keys=False))
    for robot in ("so101", "panda"):
        sim = yaml.safe_load((OUT / f"sim_{robot}_positive_all_scalars.yaml").read_text())
        sim["task_config"] = str(candidate.resolve())
        (OUT / f"sim_{robot}_landmark_fit.yaml").write_text(yaml.safe_dump(sim, sort_keys=False))
    score = fit.score(x)
    report = {
        "schema": "measured_landmark_registration_hypothesis_20261028_v1",
        "physical_calibration_accepted": False,
        "lighting_changed": False,
        "policy_inference_or_task_success_used": False,
        "source_sha256": {
            str(p): sha256_file(p)
            for p in (
                Path(__file__),
                DATA,
                PHOTO,
                MAT_ANNOTATIONS,
                STATIC,
                Path("physical_measurements.md"),
            )
        },
        "source_annotation": {
            "ball_center_px": [743.0, 237.0],
            "ball_radius_px": 51.0,
            "bowl_rim_center_px": [846.0, 499.0],
            "image": str(PHOTO),
            "note": "Manual approximate sourcepixelreadings, not physical pointidentities.",
        },
        "parameters": x,
        "parameter_order": [
            "operator_yaw_rad",
            "optical_rvec_x",
            "optical_rvec_y",
            "optical_rvec_z",
            "focal_px",
            "mat_x",
            "mat_y",
            "mat_yaw",
            "horn_axial_lift",
            "horn_axial_elbow",
            "horn_axial_wrist",
            "offset_pan_deg",
            "offset_lift_deg",
            "offset_elbow_deg",
            "bowl_bearing_rad",
            "ball_support_height_m",
        ],
        "multistart_costs": [float(s.cost) for s in solutions],
        "train_objective_cost": float(best.cost),
        "score": score,
        "task_candidate": str(candidate),
        "joint_offsets_deg": np.r_[x[11:14], 0.0, 0.0],
        "operator_to_task_yaw_deg": float(np.rad2deg(x[0])),
        "ball_xy": ball,
        "bowl_xy": bowl,
        "camera_position": camera,
        "limitations": [
            "Joint/optical/base/model registration remains conditional and underconstrained; cannot claim independently calibrated transforms.",
            "Measured scalars fixed to centralvalues; operator axisrotation fitted, base reference assumed CADorigin.",
            "Bowl diameter4.5in assumedouter; bowlheight retained provisional30mm.",
            "Sourcepaperheight estimated toprojectball but supportgeometry notapplied in emitted candidate; report reveals mismatch.",
            "Validationhorns scored after selection; allsourcecapture identities previously explored; no untouched physicalacceptance claim.",
        ],
    }
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "task": str(candidate),
                "joint_offsets_deg": report["joint_offsets_deg"].tolist(),
                "mat_rms_px": score["mat_rms_px"],
                "horns": score["horns"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("results/v2/measured_registration_fit_20261028.json")
    )
    run(parser.parse_args().report)
