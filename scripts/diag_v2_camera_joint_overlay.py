"""Project unclipped v2 arm CAD through arm-independent candidate camera hypotheses.

Manual visible horn centres were read before producing these overlays. They are
exploratory identities, not measured joint-zero calibration. No parameter is fitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from real2sim.config import load_yaml
from real2sim.data.raw import sha256_file
from real2sim.geometry.so101_fk import ARM_JOINTS, JointMapping

CAPTURES = [
    {"episode": 0, "frame": 200, "horn_centres_px": {"shoulder_lift": [322., 198.]}},
    {"episode": 31, "frame": 400, "horn_centres_px": {
        "shoulder_lift": [298., 224.], "elbow_flex": [397., 98.],
        "wrist_flex": [653., 190.]}},
]


def base_transform(x):
    c, s = np.cos(x[2]), np.sin(x[2])
    transform = np.eye(4)
    transform[:3, :3] = [[c, s, 0], [s, -c, 0], [0, 0, -1]]
    transform[:3, 3] = [x[0], x[1], -.0024]
    return transform


def legacy_camera(task):
    camera = task["camera_iphone"]
    axes = np.asarray(camera["xyaxes_task"], float)
    x = axes[:3] / np.linalg.norm(axes[:3])
    y = axes[3:] - x * np.dot(axes[3:], x)
    y /= np.linalg.norm(y)
    transform = np.eye(4)
    transform[:3, :3] = np.stack([x, -y, -np.cross(x, y)])
    transform[:3, 3] = -transform[:3, :3] @ np.asarray(camera["approx_position_task"])
    k = camera["intrinsics_px"]
    return np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.]]), transform


def project(points, k, transform, affine=None):
    optical = np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]
    if np.any(optical[:, 2] <= 0):
        raise ValueError("CAD behind candidate camera")
    pixels = optical[:, :2] / optical[:, 2, None]
    pixels = pixels @ k[:2, :2].T + k[:2, 2]
    if affine is not None:
        pixels = np.c_[pixels, np.ones(len(pixels))] @ affine.T
    return pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    camera_report_path = Path("outputs/v2/static_base_camera_20261023/report.json")
    camera_report = json.loads(camera_report_path.read_text())
    foot = camera_report["base_foot"]
    k_candidate = np.asarray(foot["candidate_K_early"])
    transform_candidate = np.asarray(foot["candidate_T_optical_base_early"])
    transform_plane = transform_candidate @ np.linalg.inv(np.asarray(foot["candidate_T_mat_base"]))
    alternative = foot["distinct_alignment_basins"][1]["xy_yaw_radians"]
    transform_alternative = transform_plane @ base_transform(alternative)
    task_path = Path("configs/v2/task.exploratory_20261022.yaml")
    task = load_yaml(task_path)
    mapping = JointMapping.from_task_config(task)
    k_legacy, transform_legacy = legacy_camera(task)
    cameras = {"legacy": (k_legacy, transform_legacy),
               "static_foot_candidate": (k_candidate, transform_candidate),
               "alternate_foot_basin": (k_candidate, transform_alternative)}
    late_affine = np.asarray(camera_report["image_similarity_models"][
        "without_fixed_base_foot"]["matrix_reference_to_late"])
    model_path = Path("assets/menagerie/robotstudio_so101/so101.xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joints = [model.joint(n).id for n in (*ARM_JOINTS, "gripper")]
    qadr = model.jnt_qposadr[joints]
    arrays_path = Path(".cache/v2_exploratory_20261022/cartesian_iphone_top/arrays.npz")
    arrays = np.load(arrays_path)
    args.out.mkdir(parents=True)
    report = {"schema": "v2_camera_joint_overlay_v1", "physical_calibration_accepted": False,
              "parameters_fitted": False, "reserved_episodes_used": False,
              "manual_annotation_protocol": "Visible camera-facing servo horn centres read at native"
              " resolution before generating overlays. Approximate uncertainty +/-8px."
              " Shoulder/elbow/wrist identities are tentative nominal-CAD correspondences.",
              "captures": [], "input_sha256": {
                  str(p): sha256_file(p) for p in (camera_report_path, task_path, model_path, arrays_path)},
              "script_sha256": sha256_file(Path(__file__)),
              "limitations": [
                  "No zero/sign, camera, horn axial position or landmark geometry optimization.",
                  "Distance to projected hinge axis removes unspecified horn along-axis position;"
                  " small axis distance alone cannot establish correct 3D correspondence.",
                  "Convex contours of individual projected visual meshes show shape extents,"
                  " not visibility-correct silhouettes, depth occlusion or measured pixel correspondences.",
                  "Legacy uses configured 1786px pinhole intrinsics and normalized configured xyaxes."
                  " It receives no late-regime affine because its original source regime is unresolved.",
                  "Candidate late-regime affine comes from fixed cabinet/clamp training correspondences."
                  " It is applied only to projections, never to policy images.",
                  "Static-foot candidates assume nominal CAD, planar mat and 2.4mm base support."
                  " Repeated hole correspondences are not independently verified.",
              ]}
    for capture in CAPTURES:
        ep, frame = capture["episode"], capture["frame"]
        index = np.flatnonzero((arrays["episode_index"] == ep) & (arrays["frame_index"] == frame))
        if len(index) != 1:
            raise ValueError("no unique source state")
        real = np.asarray(arrays["joint_state"][index[0]], float)
        native = np.r_[mapping.arm_real_to_sim_rad(real), mapping.gripper_real_to_sim_rad(real[5])]
        data.qpos[qadr] = native
        mujoco.mj_kinematics(model, data)
        photo_path = Path(f"outputs/v2/geometry_alignment_20261021/ep{ep:02d}_f{frame:04d}.jpg")
        photo = cv2.imread(str(photo_path))
        panels, scores = [], {}
        for name, (k, transform) in cameras.items():
            image = photo.copy()
            affine = late_affine if ep >= 7 and name != "legacy" else None
            for geom in range(model.ngeom):
                if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 2:
                    continue
                mesh = model.geom_dataid[geom]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                points = model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T
                points = points + data.geom_xpos[geom]
                pixels = project(points, k, transform, affine)
                hull = cv2.convexHull(pixels.astype(np.float32)).astype(np.int32)
                cv2.polylines(image, [hull], True, (0, 145, 255), 1, cv2.LINE_AA)
            scores[name] = {}
            for joint, observed in capture["horn_centres_px"].items():
                jid = model.joint(joint).id
                anchor, axis = data.xanchor[jid], data.xaxis[jid]
                segment = project(np.array([anchor - .04 * axis, anchor + .04 * axis]),
                                  k, transform, affine)
                direction = segment[1] - segment[0]
                observed = np.asarray(observed)
                relative = observed - segment[0]
                distance = abs(float(direction[0] * relative[1] - direction[1] * relative[0]))
                distance /= np.linalg.norm(direction)
                anchor_px = project(anchor[None], k, transform, affine)[0]
                scores[name][joint] = {
                    "observed_horn_centre_px": observed.tolist(),
                    "projected_joint_anchor_px": anchor_px.tolist(),
                    "distance_to_projected_axis_px": float(distance),
                    "distance_to_joint_anchor_px": float(np.linalg.norm(anchor_px - observed)),
                }
                cv2.line(image, tuple(segment[0].astype(int)), tuple(segment[1].astype(int)),
                         (255, 80, 40), 2, cv2.LINE_AA)
                cv2.drawMarker(image, tuple(observed.astype(int)), (0, 255, 0),
                               cv2.MARKER_CROSS, 14, 2)
                cv2.putText(image, joint, tuple(observed.astype(int) + [10, 10]),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, name, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 4)
            cv2.putText(image, name, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 1)
            panels.append(cv2.resize(image, (640, 360)))
        overlay_path = args.out / f"ep{ep:02d}_f{frame:04d}.jpg"
        cv2.imwrite(str(overlay_path), np.concatenate(panels, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 88])
        report["captures"].append({**capture, "source_path": str(photo_path),
                                   "source_sha256": sha256_file(photo_path),
                                   "real_state_deg_percent": real.tolist(),
                                   "native_unclipped_rad": native.tolist(), "scores": scores,
                                   "overlay": str(overlay_path)})
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({f"{r['episode']}:{r['frame']}": r["scores"] for r in report["captures"]}, indent=2))


if __name__ == "__main__":
    main()
