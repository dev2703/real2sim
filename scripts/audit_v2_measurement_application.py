"""Compare executable scenes with the supplied measurements and original lighting.

Reported dimensions are usable inputs. This audit distinguishes their application
from camera rotations, optical intrinsics and robot joint zeros, which a list of
distances cannot determine. It does not gate simulation on new physical captures.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from real2sim.artifacts import sha256_file
from real2sim.config import load_yaml, project_path
from real2sim.serialization import write_json

if __package__:
    from scripts.import_v2_operator_measurements import parse_measurements
else:
    from import_v2_operator_measurements import parse_measurements


def compare(value, measured, uncertainty, *, interpretation):
    value, measured = np.atleast_1d(value).astype(float), np.atleast_1d(measured).astype(float)
    if value.shape != measured.shape or not np.isfinite(np.r_[value, measured]).all():
        raise ValueError("comparison requires finite measurements with matching dimensions")
    bound = np.abs(measured) * uncertainty
    return {
        "configured_m": value.tolist(),
        "reported_central_m": measured.tolist(),
        "difference_mm": ((value - measured) * 1000).tolist(),
        "within_reported_bound": bool(np.all(np.abs(value - measured) <= bound + 1e-10)),
        "interpretation": interpretation,
    }


def audit(source, configs):
    parsed = parse_measurements(source.read_text())
    entries, uncertainty = parsed["entries"], parsed["relative_uncertainty"]
    digest = sha256_file(source)
    report = {
        "schema": "operator_measurement_application_audit_v1",
        "measurement_file_present": True,
        "measurement_source": str(source),
        "measurement_source_sha256": digest,
        "reported_relative_uncertainty": uncertainty,
        "reported_setup": entries,
        "missing_measurement_file": False,
        "simulation_requires_new_capture": False,
        "unmeasured_quantities": [
            "Global camera optical orientation, focal lengths, principal point and lens distortion.",
            "Wrist camera optical pose relative to its link and its lens intrinsics.",
            "Robot joint neutral offsets relative to the nominal CAD model.",
            "Loaded gripper opening, contact response and TCP location.",
            "Bowl diameter inner/outer convention, support height and exact bowl bearing.",
        ],
        "scenes": [],
    }
    for path in configs:
        sim = load_yaml(path)
        task_path = project_path(sim["task_config"])
        task = load_yaml(task_path)
        reference = project_path(f"configs/v2/sim_{sim['robot']}.waypoint_20261026.yaml")
        original_lighting = load_yaml(reference).get("appearance", {}).get("lighting")
        lighting = sim.get("appearance", {}).get("lighting")
        ball = np.asarray(task["ball"]["start_region"]["center_task_xy"], float)
        bowl = np.asarray(task["bowl"]["center_task_xy"], float)
        camera = np.asarray(task["camera_iphone"]["approx_position_task"], float)
        radius = entries["ball"]["value_m"][0] / (2 * math.pi)
        comparisons = {
            "ball_radius": compare(task["ball"]["radius"], radius, uncertainty,
                interpretation="Circumference divided by 2*pi."),
            "base_ball_planar_distance": compare(np.linalg.norm(ball),
                np.linalg.norm(entries["ball from robot base"]["value_m"]), uncertainty,
                interpretation="Rotation invariant; assumes the same base reference point."),
            "ball_bowl_total_distance": compare(np.linalg.norm(bowl - ball),
                entries["ball & bowl"]["value_m"][0], uncertainty,
                interpretation="Uses total centre-to-centre planar distance; bearing remains a prior."),
            "mat_side_lengths_sorted": compare(sorted(task["table"]["mat"]["size_xy"]),
                sorted(entries["mat"]["value_m"]), uncertainty,
                interpretation="Lengths compared without assuming operator-to-CAD axis alignment."),
            "camera_height_above_table": compare(camera[2] + task["robot_so101"]["base_z_offset"],
                entries["camera from ball"]["value_m"][2], uncertainty,
                interpretation="Operator clarified Z above tabletop; accounts for task/base origin height."),
            "camera_ball_horizontal_distance": compare(np.linalg.norm(camera[:2] - ball),
                np.linalg.norm(entries["camera from ball"]["value_m"][:2]), uncertainty,
                interpretation="Rotation invariant; camera identity is assumed to be global iPhone."),
            "bowl_diameter_if_outer": compare(
                2 * (task["bowl"]["inner_rim_radius"] + task["bowl"]["wall_thickness"]),
                entries["bowl"]["value_m"][0], uncertainty,
                interpretation="Conditional on 4.5-inch measurement describing the outer diameter."),
        }
        report["scenes"].append({
            "robot": sim["robot"],
            "sim_config": str(path),
            "task_config": str(task_path),
            "source_hash_matches_task": task.get("operator_measurements", {}).get("sha256") == digest,
            "applied_fields_declared": task.get("operator_measurements", {}).get("applied_fields"),
            "comparisons": comparisons,
            "all_reported_scalars_within_bound": all(v["within_reported_bound"] for v in comparisons.values()),
            "lighting_config_unchanged": lighting == original_lighting,
            "lighting_reference": str(reference),
            "lighting": lighting,
            "joint_mapping": task["robot_so101"]["real_to_sim"],
            "geometry_status": task.get("geometry_status"),
            "source_sha256": {str(p): sha256_file(p) for p in (path, task_path, reference)},
        })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("physical_measurements.md"))
    parser.add_argument("--configs", type=Path, nargs="+", default=[
        Path(f"configs/v2/measured_registration_20261028/sim_{r}_positive_all_scalars.yaml")
        for r in ("so101", "panda")
    ])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    write_json(args.out, audit(args.source, args.configs))
