"""Apply the clarified tabletop camera height and compare rendered detections.

This diagnostic uses simulator object state only to score camera projection and
RGB localization. It runs no control actions or policy success evaluation. The
new camera height is measured, while its identity, horizontal position, optics
and orientation are explicitly provisional.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from real2sim.config import load_yaml
from real2sim.policy.visual_targets import camera_pose_task, locate_objects
from real2sim.sim.cameras import camera_intrinsics_px
from real2sim.sim.env import So101PickPlaceEnv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera_height_candidate(task, clarification):
    """Camera task Z accounts for the task origin above the tabletop."""
    interpretation = clarification["interpretation"]
    height = float(interpretation["camera_height_above_table_m"])
    interval = np.asarray(interpretation["camera_height_interval_m"], float)
    base_offset = float(task["robot_so101"]["base_z_offset"])
    if (not np.isfinite([height, base_offset, *interval]).all()
            or interval.shape != (2,) or not 0 < interval[0] <= height <= interval[1]):
        raise ValueError("positive finite height and ordered uncertainty interval required")
    candidate = copy.deepcopy(task)
    camera = candidate["camera_iphone"]
    camera["approx_position_task"][2] = height - base_offset
    camera["height_above_table_m"] = height
    camera["height_above_table_interval_m"] = interval.tolist()
    camera["extrinsics_provenance"] = (
        "Operator clarification 2026-10-26: camera is 16 inches above tabletop. "
        "Applied to global iPhone as an explicit camera-identity assumption. "
        "Historical camera x/y, xyaxes orientation, look_at and intrinsics retained provisionally; "
        "this is a camera-height-only candidate, not a fitted or accepted registration. "
        "The xyaxes field controls orientation; look_at is retained as historical metadata.")
    candidate["geometry_status"] = "measured_ball_radius_and_camera_height_other_registration_unverified"
    candidate["physical_calibration_accepted"] = False
    return candidate


def project(point, K, T):
    camera = T[:3, :3].T @ (np.asarray(point) - T[:3, 3])
    pixel = K @ camera
    return pixel[:2] / pixel[2]


def inspect(config, out, seeds):
    rows = []
    with So101PickPlaceEnv(config, render_images=False) as env:
        for seed in seeds:
            _, info = env.reset(seed=seed)
            image = env.render("iphone")
            camera = env.model.camera("iphone").id
            K = camera_intrinsics_px(env.model, camera, image.shape[1], image.shape[0])
            T = camera_pose_task(env.data.cam_xpos[camera], env.data.cam_xmat[camera], env.offset)
            table_z = env.table_top - env.offset[2]
            ball_z = table_z + env.task["ball"]["radius"]
            bowl_z = table_z + env.bowl["height"]
            found = locate_objects(image, K, T, ball_radius_m=env.task["ball"]["radius"],
                ball_center_height_m=ball_z, bowl_rim_height_m=bowl_z,
                bowl_outer_radius_m=env.bowl["rim_radius"] + env.bowl["wall"],
                workspace_xy=[env.config["ik"]["workspace_min"][:2], env.config["ik"]["workspace_max"][:2]])
            # Scoring-only targets: compare ball centre plane and bowl rim plane.
            pm = info["privileged_metrics"]
            ball = np.array(pm["ball_xyz_task"])
            ball[2] = ball_z
            bowl = np.array(pm["bowl_xyz_task"])
            bowl[2] = bowl_z
            row = {"seed": seed, "camera_height_above_table_m": float(env.data.cam_xpos[camera, 2] - env.table_top),
                   "scoring_only": {}, "detections": {}}
            for name, reference in (("ball", ball), ("bowl", bowl)):
                pixel = project(reference, K, T)
                row["scoring_only"][name] = {"reference_task_m": reference.tolist(), "projected_pixel_xy": pixel.tolist(),
                    "center_within_image": bool(0 <= pixel[0] < image.shape[1] and 0 <= pixel[1] < image.shape[0])}
                target = found[name]
                row["detections"][name] = None if target is None else {**target.to_dict(),
                    "scored_xy_error_m": float(np.linalg.norm(target.center_task[:2] - reference[:2]))}
            cv2.imwrite(str(out / f"seed_{seed:03d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=Path("configs/v2/task.scalar_measured_20261026.yaml"))
    parser.add_argument("--measurements", type=Path, default=Path("configs/v2/measurements.operator_20261026.json"))
    parser.add_argument("--clarification", type=Path, default=Path("configs/v2/measurements.clarification_20261026.json"))
    parser.add_argument("--sim", type=Path, default=Path("configs/v2/sim_so101.waypoint_20261026.yaml"))
    parser.add_argument("--task-out", type=Path, required=True)
    parser.add_argument("--measurements-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    if any(path.exists() for path in (args.task_out, args.measurements_out, args.out)):
        raise FileExistsError("fresh output paths required")
    clarification = json.loads(args.clarification.read_text())
    candidate = camera_height_candidate(load_yaml(args.task), clarification)
    candidate.update(source_task=str(args.task), source_task_sha256=digest(args.task),
                     operator_clarification={"path": str(args.clarification), "sha256": digest(args.clarification),
                                             "applied_fields": ["camera_iphone.approx_position_task[2]"]})
    measurements = json.loads(args.measurements.read_text())
    measurements.update(version="v2_operator_measurements_clarified_20261026",
                        parent_measurements={"path": str(args.measurements), "sha256": digest(args.measurements)},
                        operator_clarification=clarification)
    iphone = measurements["cameras"]["observation.images.iphone"]
    iphone.update(height_above_table_m=clarification["interpretation"]["camera_height_above_table_m"],
                  height_above_table_interval_m=clarification["interpretation"]["camera_height_interval_m"],
                  camera_identity_is_assumption=True)
    args.task_out.write_text(yaml.safe_dump(candidate, sort_keys=False))
    args.measurements_out.write_text(json.dumps(measurements, indent=2, allow_nan=False) + "\n")
    args.out.mkdir(parents=True)
    config = load_yaml(args.sim)
    report = {"schema": "camera_height_only_detection_diagnostic_v1", "physical_calibration_accepted": False,
              "no_actions_or_task_success_evaluated": True, "seeds": args.seeds,
              "camera_identity_is_assumption": True, "camera_orientation_retained_provisionally": True,
              "source_sha256": {str(path): digest(path) for path in
                  (Path(__file__), args.task, args.measurements, args.clarification, args.task_out,
                   args.measurements_out, args.sim, Path("src/real2sim/policy/visual_targets.py"))},
              "cases": {}}
    with tempfile.TemporaryDirectory() as temporary:
        for name, task in (("previous_height", args.task), ("clarified_height_only", args.task_out)):
            directory = args.out / name
            directory.mkdir()
            config["task_config"] = str(task.resolve())
            path = Path(temporary) / f"{name}.yaml"
            path.write_text(yaml.safe_dump(config))
            report["cases"][name] = inspect(path, directory, args.seeds)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, rows in report["cases"].items():
        print(json.dumps({"case": name, "ball_detections": sum(r["detections"]["ball"] is not None for r in rows),
                          "bowl_detections": sum(r["detections"]["bowl"] is not None for r in rows),
                          "camera_height_m": rows[0]["camera_height_above_table_m"]}))


if __name__ == "__main__":
    main()
