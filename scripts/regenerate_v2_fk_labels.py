"""Regenerate FK-dependent states, labels and training statistics into a fresh dataset.

Consumes immutable sampled joints/images; does not decode videos, change split/ticks,
clip joint references, fit geometry, or assert physical calibration acceptance. A new
contract version is mandatory. Inference must use the saved task and matching new
checkpoint contract. Camera/base scene changes alone can leave native-base labels
identical; only an independently justified mapping correction repairs incorrect FK.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np

from real2sim.config import load_yaml, project_path
from real2sim.data.labels import (
    ABSOLUTE_ACTION_NAMES,
    ACTION_NAMES,
    PROPRIO_NAMES,
    TOOL_ACTION_NAMES,
    absolute_pose_actions,
    episode_labels,
    tool_frame_actions,
)
from real2sim.data.prepare import compute_stats
from real2sim.data.raw import sha256_file
from real2sim.geometry.so101_fk import DEFAULT_MJCF, JointMapping, So101FK
from real2sim.sim.cameras import shared_task_hash

MODES = ("cartesian", "cartesian_absolute", "cartesian_tool", "canonical_joint")


def regenerate_arrays(arrays: dict, cfg: dict, mode: str) -> tuple[dict, So101FK]:
    """Recompute every FK-dependent array without crossing episode boundaries."""
    if mode not in MODES:
        raise ValueError(f"FK regeneration requires one of {MODES}")
    fk = So101FK(mapping=JointMapping.from_task_config(load_yaml(project_path(cfg["task_config"]))))
    result = {key: value.copy() for key, value in arrays.items()}
    N = len(arrays["episode_index"])
    for key in ("joint_state", "joint_action"):
        if arrays[key].shape != (N, 6) or not np.isfinite(arrays[key]).all():
            raise ValueError("finite sampled observation and command joints [N,6] are required")
    result["state"] = np.empty((N, 13), np.float32)
    result["action"] = np.empty((N, 10 if mode == "cartesian_absolute" else 6 if mode == "canonical_joint" else 7), np.float32)
    for ep in np.unique(arrays["episode_index"]):
        indices = np.flatnonzero(arrays["episode_index"] == ep)
        if np.any(np.diff(indices) != 1):
            raise ValueError(f"episode {ep} rows must remain contiguous")
        S, A = (arrays[key][indices] for key in ("joint_state", "joint_action"))
        poses = fk.fk_real(S)
        lab = episode_labels(
            poses, S[:, 5], A[:, 5], hz=cfg["policy_hz"],
            gripper_source=cfg["labels"]["cartesian_gripper_source"],
            lead=cfg["labels"]["cartesian_lead_ticks"], leader_poses=fk.fk_real(A),
        )
        if not np.array_equal(lab.label_is_extrapolated, arrays["label_is_extrapolated"][indices]):
            raise ValueError("terminal masks differ; regeneration must preserve label timing")
        action = lab.action
        if mode == "cartesian_absolute":
            if cfg["labels"]["cartesian_gripper_source"] != "leader_action":
                raise ValueError("absolute representation requires its existing leader_action gripper contract")
            action, terminal = absolute_pose_actions(poses, A[:, 5], lead=cfg["labels"]["cartesian_lead_ticks"])
            np.testing.assert_array_equal(terminal, lab.label_is_extrapolated)
        elif mode == "cartesian_tool":
            action = tool_frame_actions(poses, lab.action)
        elif mode == "canonical_joint":
            action = A
        result["state"][indices], result["action"][indices] = lab.proprio, action
        result["leader_follower_pos_err"][indices] = lab.leader_follower_pos_err
        result["jump_flag"][indices] = lab.jump_flags
    return result, fk


def regenerate(source: Path, config: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite any existing dataset/directory: {out}")
    cfg = load_yaml(config)
    manifest = json.loads((source / "manifest.json").read_text())
    if not cfg.get("contract_version") or cfg["contract_version"] == manifest.get("contract_version"):
        raise ValueError("a new explicit contract_version is required for regenerated FK lineage")
    if cfg["policy_hz"] != manifest["policy_hz"] or cfg["camera_keys"] != manifest["camera_keys"]:
        raise ValueError("regeneration preserves the source policy clock and ordered camera inputs")
    fixed_split = json.loads(project_path(cfg["split"]["fixed_manifest"]).read_text())
    if fixed_split != manifest["split"]:
        raise ValueError("regeneration must preserve the complete fixed train/validation/reserved split")
    # Validate content before creating an output, including images that will be shared.
    for name, digest in manifest["files_sha256"].items():
        if sha256_file(source / name) != digest:
            raise ValueError(f"source checksum mismatch: {name}")
    with np.load(source / "arrays.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    mode = manifest["mode"]
    result, fk = regenerate_arrays(arrays, cfg, mode)
    train = np.isin(result["episode_index"], manifest["split"]["train"])
    valid = train & ~result["label_is_extrapolated"]
    stats = json.loads((source / "stats.json").read_text())
    stats["observation.state"] = compute_stats(result["state"][train])
    stats["action"] = compute_stats(result["action"][valid])
    task_path = project_path(cfg["task_config"])
    task = load_yaml(task_path)
    dependencies = [Path(__file__), DEFAULT_MJCF, Path("src/real2sim/geometry/so101_fk.py"), Path("src/real2sim/data/labels.py")]
    lineage = {
        "schema": "v2_fk_label_regeneration_v1", "source_dataset": str(source.resolve()),
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "data_config_sha256": sha256_file(config), "task_config_sha256": sha256_file(task_path),
        "dependencies_sha256": {str(path): sha256_file(path) for path in dependencies},
        "arm_sign": list(fk.mapping.sign), "arm_offset_deg": list(fk.mapping.offset_deg),
        "coordinate_frame": "SO101 native base; scene camera/base placement does not transform these labels",
        "physical_calibration_accepted": False,
        "calibration_acceptance_note": "Regeneration proves lineage only. Independent metrology and reference replay acceptance are separate requirements.",
        "split_ticks_images_and_gripper_commands_preserved": True,
        "reference_joint_clipping": False, "native_joint_limits_rad": fk.joint_range.tolist(),
        "partitions": {},
    }
    for partition in ("train", "val"):
        mask = np.isin(arrays["episode_index"], manifest["split"][partition])
        motion = mask & ~arrays["label_is_extrapolated"]
        changed_state = np.linalg.norm(result["state"][mask, :3] - arrays["state"][mask, :3], axis=1) * 1000
        lineage["partitions"][partition] = {
            "ticks": int(mask.sum()), "out_of_native_arm_limits_ticks": int(fk.out_of_range(result["joint_state"][mask]).any(axis=1).sum()),
            "state_position_change_mm_p95": float(np.percentile(changed_state, 95)),
            "state_position_change_mm_max": float(changed_state.max()),
            "action_change_abs_max_per_dimension": np.abs(result["action"][motion] - arrays["action"][motion]).max(axis=0).tolist(),
        }
    lineage["reserved_ticks_preserved_but_not_scored"] = int(np.isin(result["episode_index"], manifest["split"].get("reserved", [])).sum())
    output_manifest = copy.deepcopy(manifest)
    output_manifest.update(
        created_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        contract_version=cfg["contract_version"], calibration_sha256=sha256_file(task_path),
        transfer_shared_task_sha256=shared_task_hash(task), physical_calibration_accepted=False,
        geometry_status=task.get("geometry_status", "unverified_geometry"),
        fk_coordinate_frame=lineage["coordinate_frame"], fk_label_regeneration=lineage,
        source_manifest_sha256=lineage["source_manifest_sha256"],
        preparation_config_sha256=sha256_file(config),
        camera_transforms={key: task[key] for key in ("camera_iphone", "additional_cameras") if key in task},
        label_semantics=f"Regenerated {mode}: achieved FK under explicitly hashed mapping; native SO101 base axes; recorded gripper command semantics preserved",
        label_diagnostics={
            "jump_flags": int(result["jump_flag"].sum()),
            "extrapolated_ticks": int(result["label_is_extrapolated"].sum()),
            "train_leader_follower_position_error_mm_p95": float(np.percentile(result["leader_follower_pos_err"][train], 95) * 1000),
        },
    )
    output_manifest["state_names"] = PROPRIO_NAMES
    output_manifest["action_names"] = {"cartesian": ACTION_NAMES, "cartesian_tool": TOOL_ACTION_NAMES,
                                       "cartesian_absolute": ABSOLUTE_ACTION_NAMES,
                                       "canonical_joint": manifest["action_names"]}[mode]
    out.mkdir(parents=True)
    np.savez(out / "arrays.npz", **result)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    shutil.copyfile(task_path, out / "task.yaml")
    shutil.copyfile(config, out / "data_config.yaml")
    for filename in output_manifest["image_files"].values():
        (out / filename).symlink_to((source / filename).resolve())
    output_manifest["files_sha256"] = {
        name: manifest["files_sha256"][name] if name in output_manifest["image_files"].values() else sha256_file(out / name)
        for name in ("arrays.npz", "stats.json", "task.yaml", "data_config.yaml", *output_manifest["image_files"].values())
    }
    (out / "manifest.json").write_text(json.dumps(output_manifest, indent=2, allow_nan=False) + "\n")
    (out / "fk_regeneration.json").write_text(json.dumps(lineage, indent=2, allow_nan=False) + "\n")
    return lineage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(regenerate(args.source, args.config, args.out), indent=2))
