"""Make an explicit positive-axis setup candidate from operator measurements.

Positive camera/ball offsets are the operator's assigned convention. Aligning
operator axes to task axes, choosing a bowl bearing, mat centre, and a look-at
orientation are declared working assumptions, not an independently fitted pose.
Existing successful scene/checkpoint configurations are preserved.
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


def positive_setup(task, measurements, height_m=.4064):
    entries = measurements["reported_setup"]
    ball_xy = np.asarray(entries["ball from robot base"]["value_m"], float)
    camera_offset = np.asarray(entries["camera from ball"]["value_m"][:2], float)
    spacing = float(entries["ball & bowl"]["value_m"][0])
    mat_size = np.asarray(entries["mat"]["value_m"], float)
    if (ball_xy.shape != (2,) or camera_offset.shape != (2,) or mat_size.shape != (2,)
            or not np.isfinite(np.r_[ball_xy, camera_offset, mat_size, spacing, height_m]).all()
            or np.any(np.r_[ball_xy, camera_offset, mat_size, spacing, height_m] <= 0)):
        raise ValueError("positive finite measured offsets, dimensions and camera height required")
    result = copy.deepcopy(task)
    bowl_xy = ball_xy + [spacing, 0.]
    midpoint = (ball_xy + bowl_xy) / 2
    result["ball"]["start_region"].update(center_task_xy=ball_xy.tolist(),
        provenance="Operator (2,13) in with positive components; operator/task axes aligned as a working convention")
    result["bowl"].update(center_task_xy=bowl_xy.tolist(),
        center_provenance="Operator 12 in spacing, source single-number x-axis rule; +x bowl bearing assigned as a working convention")
    result["table"]["mat"].update(size_xy=mat_size.tolist(), center_task_xy=midpoint.tolist())
    base_height = float(result["robot_so101"]["base_z_offset"])
    camera = result["camera_iphone"]
    camera.update(approx_position_task=(ball_xy + camera_offset).tolist() + [height_m - base_height],
        look_at_task=midpoint.tolist() + [float(result["ball"]["radius"]) - base_height],
        height_above_table_m=height_m,
        height_above_table_interval_m=[height_m * .95, height_m * 1.05],
        extrinsics_provenance="Operator assigned positive camera-minus-ball x/y; 16 in above table. Roll-free look-at aims at nominal task midpoint as a hypothesis; old focal length retained, no physical camera rotation fit.")
    # The old fitted axes refer to the old camera side and cannot follow this move.
    camera.pop("xyaxes_task", None)
    result.update(geometry_status="operator_positive_setup_convention_not_independently_registered",
        physical_calibration_accepted=False,
        positive_setup_assumptions={
            "camera_minus_ball_xy_signs": [1, 1],
            "operator_axes_assumed_parallel_to_task_axes": True,
            "base_reference_assumed_task_origin_xy": True,
            "camera_identity": "global observation.images.iphone",
            "bowl_bearing": "+task_x from ball; source declares single-number x-axis measurements",
            "mat_center": "nominal ball/bowl midpoint; mat placement not measured",
            "camera_rotation": "roll-free look-at nominal ball/bowl midpoint; not measured",
            "intrinsics": "previous provisional focal length/principal point retained",
            "independently_verified_registration": False,
        })
    return result


def prepare(out, task_path, measurement_path):
    if out.exists():
        raise FileExistsError(out)
    task = load_yaml(task_path)
    measured = json.loads(measurement_path.read_text())
    candidate = positive_setup(task, measured)
    out.mkdir(parents=True)
    clarification = {
        "schema": "operator_positive_camera_convention_v1", "date": "2026-10-26",
        "operator_instruction": "assign +ve sign",
        "interpretation": "camera-minus-ball horizontal components are positive",
        "camera_minus_ball_xy_m": [.7747, .1143],
        "camera_height_above_table_m": .4064,
        "camera_from_base_in_operator_frame_xyz_m": [.8255, .4445, .4064],
        "vertical_reference": "tabletop; x/y referenced to reported robot base",
        "horizontal_signs_assigned": True, "physical_calibration_accepted": False,
        "working_assumptions": candidate["positive_setup_assumptions"],
        "parent_sources_sha256": {str(path): digest(path) for path in (task_path, measurement_path)},
    }
    clarification_path = out / "positive_convention.json"
    clarification_path.write_text(json.dumps(clarification, indent=2) + "\n")
    candidate["operator_positive_convention"] = {"path": str(clarification_path),
                                               "sha256": digest(clarification_path)}
    task_target = out / "task.yaml"
    task_target.write_text(yaml.safe_dump(candidate, sort_keys=False))
    for robot in ("so101", "panda"):
        source = Path(f"configs/v2/sim_{robot}.waypoint_20261026.yaml")
        sim = load_yaml(source)
        sim["task_config"] = str(task_target.resolve())
        (out / f"sim_{robot}.yaml").write_text(yaml.safe_dump(sim, sort_keys=False))
    report = {"schema": "positive_setup_candidate_v1", "physical_calibration_accepted": False,
        "source_sha256": {str(path): digest(path) for path in
            (Path(__file__), task_path, measurement_path)},
        "files_sha256": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
        "ball_xy_task_m": candidate["ball"]["start_region"]["center_task_xy"],
        "bowl_xy_task_m": candidate["bowl"]["center_task_xy"],
        "camera_position_task_m": candidate["camera_iphone"]["approx_position_task"],
        "working_assumptions": candidate["positive_setup_assumptions"]}
    (out / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", type=Path, default=Path("configs/v2/task.scalar_measured_20261026.yaml"))
    parser.add_argument("--measurements", type=Path, default=Path("configs/v2/measurements.operator_20261026.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.task, args.measurements), indent=2))
