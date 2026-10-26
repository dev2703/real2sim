"""Apply positive camera offsets and the clarified ball/bowl total distance.

The measured distance supplies a circle, not a bowl bearing. This candidate uses
the previous image-derived relative layout as a bearing prior, rotated so its
base-to-ball direction agrees with the operator's (+2,+13) inch convention.
Neither that yaw nor the camera look-at/mat placement is independent metrology.
Historical +x-only bowl candidates and their rollout evidence remain untouched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from real2sim.config import load_yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def total_distance_setup(task, measurements, *, height_m=.4064):
    entries = measurements["reported_setup"]
    ball = np.asarray(entries["ball from robot base"]["value_m"], float)
    offset = np.asarray(entries["camera from ball"]["value_m"][:2], float)
    spacing = float(entries["ball & bowl"]["value_m"][0])
    mat_size = np.asarray(entries["mat"]["value_m"], float)
    old_ball = np.asarray(task["ball"]["start_region"]["center_task_xy"], float)
    old_bowl = np.asarray(task["bowl"]["center_task_xy"], float)
    if (any(value.shape != (2,) for value in (ball, offset, mat_size, old_ball, old_bowl))
            or not np.isfinite(np.r_[ball, offset, mat_size, old_ball, old_bowl,
                                     spacing, height_m]).all()
            or np.any(np.r_[ball, offset, mat_size, spacing, height_m] <= 0)
            or np.linalg.norm(old_ball) == 0 or np.linalg.norm(old_bowl - old_ball) == 0):
        raise ValueError("finite nondegenerate layout and positive measured offsets/sizes required")
    yaw = float(np.arctan2(ball[1], ball[0]) - np.arctan2(old_ball[1], old_ball[0]))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    relative = rotation @ (old_bowl - old_ball)
    direction = relative / np.linalg.norm(relative)
    bowl = ball + spacing * direction
    midpoint = (ball + bowl) / 2
    result = copy.deepcopy(task)
    base_height = float(task["robot_so101"]["base_z_offset"])
    result["ball"]["start_region"].update(
        center_task_xy=ball.tolist(),
        provenance="Operator (+2,+13) in base components; task/operator axes follow the declared working convention")
    result["bowl"].update(
        center_task_xy=bowl.tolist(),
        center_provenance="Operator 12 in TOTAL ball/bowl distance. Bearing from prior image-derived layout, yaw aligned to new base/ball bearing; not an independent physical bearing measurement")
    result["table"]["mat"].update(size_xy=mat_size.tolist(), center_task_xy=midpoint.tolist())
    camera = result["camera_iphone"]
    camera.update(
        approx_position_task=(ball + offset).tolist() + [height_m - base_height],
        look_at_task=midpoint.tolist() + [float(task["ball"]["radius"]) - base_height],
        height_above_table_m=height_m,
        height_above_table_interval_m=[height_m * .95, height_m * 1.05],
        extrinsics_provenance="Positive camera-minus-ball horizontal offsets assigned by operator; height16in above tabletop. Roll-free look-at midpoint and old intrinsics are hypotheses, not a measured camera rotation/optical calibration.")
    camera.pop("xyaxes_task", None)
    assumptions = {
        "camera_minus_ball_xy_signs": [1, 1],
        "ball_bowl_measurement_kind": "total planar centre-to-centre distance assumed; not x-only",
        "bowl_direction_method": "prior image-derived relative layout rotated by new-minus-old base/ball bearing",
        "inferred_layout_yaw_rad": yaw,
        "bowl_unit_direction_xy": direction.tolist(),
        "source_old_ball_xy": old_ball.tolist(), "source_old_bowl_xy": old_bowl.tolist(),
        "operator_axes_assumed_task_axes": True,
        "base_reference_assumed_task_origin_xy": True,
        "camera_identity": "global observation.images.iphone",
        "camera_rotation": "roll-free look-at nominal ball/bowl midpoint; not measured",
        "mat_center": "nominal ball/bowl midpoint; not measured",
        "intrinsics": "previous provisional focal length/principal point retained",
        "independently_verified_registration": False,
    }
    result.update(
        geometry_status="positive_camera_total_distance_image_bearing_prior_unverified",
        physical_calibration_accepted=False, total_distance_setup_assumptions=assumptions)
    return result


def prepare(out, task_path, measurement_path):
    if out.exists():
        raise FileExistsError(out)
    task = load_yaml(task_path)
    measured = json.loads(measurement_path.read_text())
    candidate = total_distance_setup(task, measured)
    out.mkdir(parents=True)
    clarification = {
        "schema": "operator_total_distance_clarification_v1", "work_date": "2026-10-27",
        "source": "operator replies in this conversation",
        "camera_sign_instruction_verbatim": "assign +ve sign",
        "spacing_clarification_verbatim": "12 inches is total distance, not an x-only offset",
        "ball_bowl_distance_m": .3048, "ball_bowl_distance_interval_m": [.28956, .32004],
        "supersedes_bowl_bearing_only": "configs/v2/positive_setup_20261026/task.yaml",
        "camera_horizontal_signs_assigned": True, "physical_calibration_accepted": False,
        "working_assumptions": candidate["total_distance_setup_assumptions"],
        "source_sha256": {str(p): digest(p) for p in (task_path, measurement_path)},
    }
    clarification_path = out / "operator_clarification.json"
    clarification_path.write_text(json.dumps(clarification, indent=2) + "\n")
    candidate["operator_clarification"] = {"path": str(clarification_path),
                                           "sha256": digest(clarification_path)}
    task_target = out / "task.yaml"
    task_target.write_text(yaml.safe_dump(candidate, sort_keys=False))
    sources = [Path(__file__), task_path, measurement_path]
    for robot in ("so101", "panda"):
        source = Path(f"configs/v2/sim_{robot}.waypoint_20261026.yaml")
        sim = load_yaml(source)
        sim["task_config"] = str(task_target.resolve())
        (out / f"sim_{robot}.yaml").write_text(yaml.safe_dump(sim, sort_keys=False))
        sources.append(source)
    report = {
        "schema": "positive_camera_total_distance_candidate_v1",
        "physical_calibration_accepted": False,
        "source_sha256": {str(p): digest(p) for p in sources},
        "files_sha256": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
        "ball_xy_task_m": candidate["ball"]["start_region"]["center_task_xy"],
        "bowl_xy_task_m": candidate["bowl"]["center_task_xy"],
        "camera_position_task_m": candidate["camera_iphone"]["approx_position_task"],
        "ball_bowl_distance_m": float(np.linalg.norm(
            np.subtract(candidate["bowl"]["center_task_xy"],
                        candidate["ball"]["start_region"]["center_task_xy"]))),
        "working_assumptions": candidate["total_distance_setup_assumptions"],
    }
    (out / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", type=Path, default=Path("configs/v2/task.scalar_measured_20261026.yaml"))
    parser.add_argument("--measurements", type=Path, default=Path("configs/v2/measurements.operator_20261026.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.task, args.measurements), indent=2))
