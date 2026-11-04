"""Convert the operator's inch measurements without inventing frame registration.

Produces a source-hashed measurement record, a task candidate using only the
unambiguous sphere radius, and comparisons with the previous scene. Signed axes,
camera orientation, inner/outer bowl reference and recording applicability remain
explicitly unresolved. The candidate can be used by simulation and training tools
that already accept a task YAML; it does not establish physical calibration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml

from real2sim.config import load_yaml

INCH_M = 0.0254
FIELDS = {"camera from ball": 3, "ball & bowl": 1, "ball from robot base": 2,
          "mat": 2, "bowl": 1, "ball": 1}


def parse_measurements(text: str) -> dict:
    """Read the supplied compact format; reject missing units or ambiguous repeats."""
    if "inches" not in text.casefold():
        raise ValueError("source must explicitly declare inches")
    uncertainty = re.search(r"(?:\+-|\+/-|±)\s*([\d.]+)\s*%", text)
    if uncertainty is None:
        raise ValueError("source must declare percentage uncertainty")
    relative = float(uncertainty.group(1)) / 100
    if not 0 < relative < 1:
        raise ValueError("percentage uncertainty must be between 0 and 100")
    entries = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = label.strip().casefold()
        if key not in FIELDS:
            continue
        if key in entries:
            raise ValueError(f"duplicate measurement: {key}")
        # Match only the numeric prefix; descriptors distinguish circumference
        # from diameter rather than allowing a silently incorrect conversion.
        match = re.match(r"\s*\(?\s*([-+\d.,\s]+)\)?", value)
        if match is None:
            raise ValueError(f"missing numeric measurement: {key}")
        values = [float(item.strip()) for item in match.group(1).split(",")]
        is_offset = key in {"camera from ball", "ball from robot base"}
        if (len(values) != FIELDS[key] or not np.isfinite(values).all()
                or (not is_offset and min(values) <= 0)):
            raise ValueError(f"invalid dimensions or positive sizes for {key}")
        if key == "ball" and "circumference" not in value.casefold():
            raise ValueError("ball measurement must identify circumference")
        if key == "bowl" and "diameter" not in value.casefold():
            raise ValueError("bowl measurement must identify diameter")
        si = np.asarray(values) * INCH_M
        entries[key] = {"source_text": line.strip(), "source_inches": values,
                        "value_m": si.tolist(),
                        "interval_m": np.column_stack([si - abs(si) * relative,
                                                       si + abs(si) * relative]).tolist()}
    if missing := FIELDS.keys() - entries.keys():
        raise ValueError(f"missing measurements: {sorted(missing)}")
    return {"relative_uncertainty": relative, "entries": entries}


def scalar_comparison(value: float, interval, existing: float, *, interpretation: str) -> dict:
    low, high = map(float, interval)
    return {"measured_m": float(value), "interval_m": [low, high],
            "existing_m": float(existing), "existing_minus_measured_m": float(existing - value),
            "outside_reported_interval": bool(existing < low or existing > high),
            "interpretation": interpretation}


def import_measurements(source: Path, task_path: Path, template_path: Path) -> tuple[dict, dict, dict]:
    raw = source.read_bytes()
    parsed = parse_measurements(raw.decode())
    entries, uncertainty = parsed["entries"], parsed["relative_uncertainty"]
    source_hash = hashlib.sha256(raw).hexdigest()
    previous = load_yaml(task_path)
    radius = entries["ball"]["value_m"][0] / (2 * math.pi)
    radius_interval = np.asarray(entries["ball"]["interval_m"][0]) / (2 * math.pi)
    measurements = json.loads(template_path.read_text())
    measurements.update(
        version="v2_operator_scalar_measurements_20261026",
        status="operator_scalars_available_registration_and_loaded_response_unverified",
        source={"path": str(source), "sha256": source_hash, "reported_unit": "inches",
                "relative_uncertainty": uncertainty, "uncertainty_kind": "operator_estimated_bound"},
        ball_diameter_m=2 * radius,
        ball_diameter_interval_m=(2 * radius_interval).tolist(),
        reported_setup=entries,
        operator_frame={"axes": "x lengthwise, y widthwise, z heightwise",
                        "signed_directions_established": False,
                        "T_task_operator": None,
                        "recording_applicability_established": False},
        unresolved_definitions=[
            "Which camera and lens/reference point does 'camera from ball' describe?",
            "Base reference point, signed axis directions, and relation to the simulator's task frame.",
            "Whether the bowl diameter is the inner opening or outer rim; bowl height/depth.",
            "Whether ball/bowl spacing is centre-to-centre and which recorded setup it describes.",
            "Support height, jaw gap/TCP and loaded timing/retention/release response.",
        ],
    )
    # Canonical inner/outer diameter fields intentionally stay null. A number
    # labelled only 'diameter' cannot establish either physical surface.
    measurements["bowl"]["reported_unspecified_diameter_m"] = entries["bowl"]["value_m"][0]
    measurements["bowl"]["reported_unspecified_diameter_interval_m"] = entries["bowl"]["interval_m"][0]
    task = copy.deepcopy(previous)
    task.update(
        source_task=str(task_path),
        source_task_sha256=hashlib.sha256(task_path.read_bytes()).hexdigest(),
        geometry_status="operator_ball_size_updated_camera_base_jaw_registration_unverified",
        physical_calibration_accepted=False,
        operator_measurements={"source": str(source), "sha256": source_hash,
                               "applied_fields": ["ball.radius"],
                               "relative_uncertainty": uncertainty},
    )
    circumference_inches = entries["ball"]["source_inches"][0]
    task["ball"].update(radius=radius, radius_interval_m=radius_interval.tolist(),
                        radius_provenance=f"OPERATOR-MEASURED circumference {circumference_inches:g} in; {source}; +-{100 * uncertainty:g}%; radius=C/(2*pi)")
    ball_xy = np.asarray(previous["ball"]["start_region"]["center_task_xy"])
    bowl_xy = np.asarray(previous["bowl"]["center_task_xy"])
    spacing = entries["ball & bowl"]["value_m"][0]
    base_distance = float(np.linalg.norm(entries["ball from robot base"]["value_m"]))
    comparisons = {
        "ball_radius": scalar_comparison(radius, radius_interval, previous["ball"]["radius"],
            interpretation="Unambiguous sphere radius inferred from measured circumference; applied."),
        "ball_bowl_spacing_if_center_to_center": scalar_comparison(
            spacing, entries["ball & bowl"]["interval_m"][0], np.linalg.norm(ball_xy - bowl_xy),
            interpretation="Comparison is conditional on centre-to-centre spacing and matching recording setup; no position correction inferred."),
        "base_ball_planar_distance_if_same_reference": scalar_comparison(
            base_distance, [base_distance * (1 - uncertainty), base_distance * (1 + uncertainty)],
            np.linalg.norm(ball_xy), interpretation="Planar norm is axis-rotation invariant; same physical base reference and recording applicability still require confirmation."),
    }
    for name, diameter in (("inner", 2 * previous["bowl"]["inner_rim_radius"]),
                           ("approx_outer", 2 * (previous["bowl"]["inner_rim_radius"] + previous["bowl"]["wall_thickness"]))):
        comparisons[f"bowl_{name}_diameter_conditional"] = scalar_comparison(
            entries["bowl"]["value_m"][0], entries["bowl"]["interval_m"][0], diameter,
            interpretation="Reported inner/outer reference is unspecified. Outer comparison approximates the wall envelope; neither field is automatically replaced.")
    report = {"schema": "v2_operator_measurement_comparison_v1",
              "source_sha256": source_hash, "source_task": str(task_path),
              "comparisons": comparisons,
              "mat_axis_order_differs": bool(np.argmax(entries["mat"]["value_m"]) != np.argmax(previous["table"]["mat"]["size_xy"])),
              "applied_fields": ["ball.radius"],
              "physical_calibration_accepted": False,
              "fk_label_changes_required": False,
              "reason_no_fk_change": "Ball radius does not change arm kinematics, joint mapping or recorded images.",
              "camera_transform_inferred": False}
    return measurements, task, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("physical_measurements.md"))
    parser.add_argument("--task", type=Path, default=Path("configs/v2/task.exploratory_20261022.yaml"))
    parser.add_argument("--template", type=Path, default=Path("configs/v2/measurements.template.json"))
    parser.add_argument("--measurements-out", type=Path, required=True)
    parser.add_argument("--task-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.measurements_out, args.task_out, args.audit_out]
    if any(path.exists() for path in paths):
        raise FileExistsError("choose fresh output paths to preserve measurement lineage")
    measurements, task, report = import_measurements(args.source, args.task, args.template)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.measurements_out.write_text(json.dumps(measurements, indent=2, allow_nan=False) + "\n")
    args.task_out.write_text(yaml.safe_dump(task, sort_keys=False))
    args.audit_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
