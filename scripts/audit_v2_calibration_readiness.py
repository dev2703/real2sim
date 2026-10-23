"""Read-only independent-evidence inventory and degree-lattice check for cached v2.

The encoder lattice verifies numerical units, not physical signs/zeros. Template
fields, CAD optics and operator setup attestations never count as measured optical,
joint or jaw correspondence. No reserved joint coordinates are scored here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.config import load_yaml
from real2sim.data.raw import RawDataset, sha256_file
from real2sim.geometry.so101_fk import ARM_JOINTS, DEFAULT_MJCF, JointMapping, So101FK
from real2sim.sim.cameras import nominal_so101_wrist_camera


def complete_rows(rows, fields):
    return [row for row in rows if all(row.get(key) is not None for key in fields)]


def encoder_lattice(real, calibration):
    # Inverse arithmetic must not add fresh float32 rounding to stored source values.
    real = np.asarray(real, dtype=np.float64)
    result = {}
    for j, name in enumerate(ARM_JOINTS):
        lo, hi = (calibration[name][key] for key in ("range_min", "range_max"))
        ticks = real[:, j] * 4095 / 360 + (lo + hi) / 2
        residual = np.abs(ticks - np.rint(ticks))
        alternative = (real[:, j] + 100) / 200 * (hi - lo) + lo
        result[name] = {
            "degree_inverse_tick_residual_p95": float(np.percentile(residual, 95)),
            "degree_inverse_tick_residual_max": float(residual.max()),
            "alternative_pm100_tick_residual_p95": float(np.percentile(np.abs(alternative - np.rint(alternative)), 95)),
        }
    return result


def audit(root: Path, split_path: Path, geometry_path: Path, measurements_path: Path,
          calibration_path: Path, leader_calibration_path: Path, task_path: Path) -> dict:
    ds = RawDataset(root)
    split, geometry, measurements, calibration = (
        json.loads(path.read_text()) for path in (split_path, geometry_path, measurements_path, calibration_path))
    task = load_yaml(task_path)
    selected = set(split["train"] + split["val"])
    raw_rows = {"observation.state": [], "action": []}
    for ep in sorted(selected):
        g = ds.episode_rows(ep)
        for key in raw_rows:
            raw_rows[key].append(ds.stack(g, key))
    raw_rows = {key: np.concatenate(rows) for key, rows in raw_rows.items()}
    real = np.concatenate(list(raw_rows.values()))
    lattice = {
        "follower_observation.state": encoder_lattice(raw_rows["observation.state"], calibration),
        "leader_action": encoder_lattice(raw_rows["action"], json.loads(leader_calibration_path.read_text())),
    }
    fk = So101FK(mapping=JointMapping.from_task_config(task))
    measured_geometry = geometry.get("evidence_kind") == "measured"
    camera_rows = complete_rows(geometry.get("camera_registration", {}).get("observations", []),
                                ("capture_group", "source_frame_reference", "landmark_id", "point_task_m", "source_pixel_xy", "sigma_px"))
    joint_rows = complete_rows(geometry.get("joint_zero_registration", {}).get("observations", []),
                               ("capture_group", "source_frame_reference", "landmark_id", "q_nominal_rad", "gripper_joint_rad", "body_name", "point_body_m", "source_pixel_xy", "sigma_px"))
    jaw_rows = complete_rows(measurements.get("jaw_measurements", []), ("measured_joint", "gap_m"))
    loaded = [row for row in measurements.get("recordings", []) if row.get("role", "").startswith("loaded")
              and row.get("trace_session") is not None and row.get("hold_source_frames")]
    paths = [Path(__file__), split_path, geometry_path, measurements_path, calibration_path, leader_calibration_path, task_path,
             DEFAULT_MJCF, root / "meta/info.json", Path("src/real2sim/sim/cameras.py")]
    source_hashes = {str(p.relative_to(root)): sha256_file(p)
                     for directory in ("meta", "data", "videos")
                     for p in sorted((root / directory).rglob("*")) if p.is_file()}
    return {
        "schema": "v2_calibration_readiness_audit_v1",
        "dataset": {"root": str(root), "episodes": len(ds.episode_indices), "raw_fps": ds.fps,
                    "metadata_total_frames": ds.info["total_frames"], "camera_keys": ds.image_keys,
                    "source_files_sha256": source_hashes,
                    "upstream_repo_requested": "bottowork/so101_act_pickplace_v2",
                    "upstream_local_identity_verified": False,
                    "identity_note": "Local source is content-addressed. Hashes do not establish equivalence with an inaccessible upstream revision."},
        "scope": {"train_and_validation_episodes": sorted(selected),
                  "reserved_episodes_not_scored": split.get("reserved", []),
                  "state_plus_action_raw_rows_scored": len(real),
                  "optimization_performed": False, "reference_clipping": False},
        "numerical_units": {"arm_degree_lattice": lattice,
                            "interpretation": "Degree lattice supports recorded units; it cannot verify signs, physical zeros, link geometry or historical timing."},
        "current_mapping": {"sign": list(fk.mapping.sign), "offset_deg": list(fk.mapping.offset_deg),
                            "state_plus_action_rows_outside_native_arm_limits": int(fk.out_of_range(real).any(axis=1).sum()),
                            "native_arm_limits_rad": fk.joint_range.tolist(),
                            "independent_signs_and_zeros_established": False},
        "independent_evidence_inventory": {
            "selected_geometry_document": str(geometry_path),
            "geometry_document_evidence_kind": geometry.get("evidence_kind"),
            "fixed_camera_intrinsics_present": geometry.get("camera", {}).get("K_source_pixels") is not None,
            "fixed_camera_distortion_present": geometry.get("camera", {}).get("distortion_coefficients") is not None,
            "wrist_intrinsics_present": geometry.get("wrist_camera", {}).get("K_source_pixels") is not None,
            "complete_measured_static_landmark_rows": len(camera_rows) if measured_geometry else 0,
            "complete_measured_joint_landmark_rows": len(joint_rows) if measured_geometry else 0,
            "complete_jaw_measurement_rows": len(jaw_rows),
            "complete_verification_only_jaw_rows": sum(bool(r.get("verification_only")) for r in jaw_rows),
            "loaded_trace_measurement_records": len(loaded),
            "jaw_gap_definition": measurements.get("jaw_gap_definition"),
            "jaw_measurement_uncertainty_m": measurements.get("jaw_measurement_uncertainty_m"),
        },
        "nominal_camera_mount": nominal_so101_wrist_camera(DEFAULT_MJCF),
        "physical_acceptance_established": False,
        "label_repair_status": "Safe regeneration tool available; no independently justified arm mapping correction is available to adopt from these inputs.",
        "remaining_measurement_dependencies": [
            "Independent source-resolution intrinsics/distortion applicable to each capture regime",
            "Unique static landmark identities and independently registered base/table frame",
            "Varied joint landmarks with independently fixed camera/base/sign conventions and held-out captures",
            "Jaw registration with gap definition, uncertainty and held-out openings",
            "Loaded hold/release measurements with observed relative ball/jaw motion and physical timing",
        ],
        "input_sha256": {str(p): sha256_file(p) for p in paths},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("~/.cache/huggingface/lerobot/local/so101_act_pickplace_v2").expanduser())
    parser.add_argument("--split", type=Path, default=Path("results/v2/split.json"))
    parser.add_argument("--geometry", type=Path, default=Path("configs/v2/geometry_landmarks.template.json"))
    parser.add_argument("--measurements", type=Path, default=Path("configs/v2/measurements.template.json"))
    parser.add_argument("--follower-calibration", type=Path, default=Path("~/.cache/huggingface/lerobot/calibration/robots/so_follower/aligned_follower.json").expanduser())
    parser.add_argument("--leader-calibration", type=Path, default=Path("~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/aligned_leader.json").expanduser())
    parser.add_argument("--task", type=Path, default=Path("configs/v2/task.exploratory_20261022.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = audit(args.root, args.split, args.geometry, args.measurements, args.follower_calibration, args.leader_calibration, args.task)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("scope", "numerical_units", "independent_evidence_inventory")}, indent=2))
