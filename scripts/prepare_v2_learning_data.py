"""Prepare the v2 learning screen while physical geometry remains provisional.

Decode each raw view once, then derive camera/action ablations with identical ticks,
splits, normalization rules and shared immutable image files.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from real2sim.config import load_yaml
from real2sim.data.labels import (
    ABSOLUTE_ACTION_NAMES,
    TOOL_ACTION_NAMES,
    absolute_pose_actions,
    episode_labels,
    tool_frame_actions,
)
from real2sim.data.prepare import compute_stats, prepare
from real2sim.data.raw import sha256_file
from real2sim.geometry.so101_fk import JointMapping, So101FK

GEOMETRY_STATUS = "provisional_joint_zeros_and_camera_alignment_unverified"
SEMANTICS = {
    "cartesian": "Next achieved provisional FK pose minus current observed provisional FK pose in SO101 base axes; recorded current gripper command",
    "canonical_joint": "Canonical observed provisional FK TCP state; recorded joint command at the sampled row",
    "cartesian_absolute": "Next achieved provisional FK position/rot6d in SO101 base axes; recorded current gripper command",
    "cartesian_tool": "Next achieved provisional FK pose increment in current observed tool axes; recorded current gripper command",
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def annotate(manifest: dict) -> dict:
    manifest.update(
        geometry_status=GEOMETRY_STATUS,
        physical_calibration_accepted=False,
        learning_experiment_scope="Offline learning comparison authorized before geometry acceptance; physical/simulation success remains unverified",
        fk_coordinate_frame="SO101 native base frame; task registration is the existing provisional identity",
        camera_calibration_status={key: "unverified; recorded images used directly" for key in manifest["camera_keys"]},
    )
    manifest["label_semantics"] = SEMANTICS[manifest["mode"]]
    return manifest


def validate_source(source: Path, cfg: dict) -> tuple[dict, dict, dict]:
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest["calibration_sha256"] != sha256_file(Path(cfg["task_config"]))
            or manifest["contract_version"] != cfg["contract_version"]
            or manifest["camera_keys"] != cfg["camera_keys"]):
        raise ValueError("existing dataset configuration differs from requested exploratory data")
    for name, expected in manifest["files_sha256"].items():
        if sha256_file(source / name) != expected:
            raise ValueError(f"source checksum mismatch: {name}")
    with np.load(source / "arrays.npz") as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not all(np.isfinite(arrays[key]).all() for key in ("state", "action", "joint_state", "joint_action")):
        raise ValueError("source arrays contain non-finite values")
    for key in manifest["image_files"].values():
        image = np.load(source / key, mmap_mode="r")
        if image.shape != (manifest["n_ticks"], *manifest["image_shape_hwc"]) or image.dtype != np.uint8:
            raise ValueError(f"invalid image array {key}")
    return manifest, arrays, json.loads((source / "stats.json").read_text())


def prepare_learning_data(config: Path, raw_root: Path, out: Path) -> dict:
    started = time.monotonic()
    cfg = load_yaml(config)
    source = out / "cartesian_iphone_top"
    if not (source / "manifest.json").exists():
        if source.exists():
            raise FileExistsError(f"incomplete preparation exists; inspect before restarting: {source}")
        prepare(raw_root, cfg, source, cfg["source"]["repo_id"], cfg["source"]["revision"], mode="cartesian")
    manifest, arrays, stats = validate_source(source, cfg)
    manifest = annotate(manifest)
    manifest["preparation_config_sha256"] = sha256_file(config)
    write_json(source / "manifest.json", manifest)

    fk = So101FK(mapping=JointMapping.from_task_config(load_yaml(cfg["task_config"])))
    poses = fk.fk_real(arrays["joint_state"])
    absolute_actions = np.zeros((len(poses), len(ABSOLUTE_ACTION_NAMES)), np.float32)
    task_actions = np.zeros((len(poses), 7), np.float64)
    for ep in np.unique(arrays["episode_index"]):
        mask = arrays["episode_index"] == ep
        # Match direct preparation: rotate float64 pose deltas before final serialization.
        task_actions[mask] = episode_labels(
            poses[mask], arrays["joint_state"][mask, 5], arrays["joint_action"][mask, 5],
            hz=cfg["policy_hz"], gripper_source=cfg["labels"]["cartesian_gripper_source"],
            lead=cfg["labels"]["cartesian_lead_ticks"],
        ).action
        action, terminal = absolute_pose_actions(poses[mask], arrays["joint_action"][mask, 5], lead=cfg["labels"]["cartesian_lead_ticks"])
        if not np.array_equal(terminal, arrays["label_is_extrapolated"][mask]):
            raise ValueError(f"absolute labels disagree about terminal rows in episode {ep}")
        absolute_actions[mask] = action
    actions = {
        "canonical_joint": (arrays["joint_action"], [f"{name}.pos" for name in cfg["state"]["joint_names"]]),
        "cartesian_absolute": (absolute_actions, ABSOLUTE_ACTION_NAMES),
        "cartesian_tool": (tool_frame_actions(poses, task_actions).astype(np.float32), TOOL_ACTION_NAMES),
    }
    source_manifest_hash = sha256_file(source / "manifest.json")
    variants = [("cartesian", [key]) for key in manifest["camera_keys"]]
    variants += [(mode, manifest["camera_keys"]) for mode in actions]
    report = {
        "contract_version": cfg["contract_version"],
        "geometry_status": GEOMETRY_STATUS,
        "physical_calibration_accepted": False,
        "split": manifest["split"],
        "source": str(raw_root.resolve()),
        "n_ticks": manifest["n_ticks"],
        "source_manifest_sha256": source_manifest_hash,
        "datasets": {source.name: {"path": str(source.resolve()), "manifest_sha256": source_manifest_hash}},
    }
    for mode, cameras in variants:
        target = out / f"{mode}_{'_'.join(key.rsplit('.', 1)[-1] for key in cameras)}"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite an existing variant: {target}")
        target.mkdir()
        output_manifest = annotate(copy.deepcopy(manifest))
        output_manifest.update(mode=mode, action_representation=mode,
                               controller_semantics="accepted_target" if mode in ("cartesian", "cartesian_tool") else "current_pose",
                               camera_keys=cameras, source_manifest_sha256=source_manifest_hash,
                               label_semantics=SEMANTICS[mode])
        output_manifest["image_files"] = {key: manifest["image_files"][key] for key in cameras}
        output_manifest["camera_calibration_status"] = {key: "unverified; recorded images used directly" for key in cameras}
        output_stats = copy.deepcopy(stats)
        if mode == "cartesian":
            for name in ("arrays.npz", "stats.json"):
                (target / name).symlink_to(Path("..") / source.name / name)
        else:
            action, names = actions[mode]
            if not np.isfinite(action).all():
                raise ValueError(f"non-finite labels in {mode}")
            output_arrays = {**arrays, "action": action}
            np.savez(target / "arrays.npz", **output_arrays)
            train = np.isin(arrays["episode_index"], manifest["split"]["train"]) & ~arrays["label_is_extrapolated"]
            output_stats["action"] = compute_stats(action[train])
            write_json(target / "stats.json", output_stats)
            output_manifest["action_names"] = names
        for name in output_manifest["image_files"].values():
            (target / name).symlink_to(Path("..") / source.name / name)
        output_manifest["files_sha256"] = {
            name: manifest["files_sha256"][name] if name.startswith("images_") else sha256_file(target / name)
            for name in ("arrays.npz", "stats.json", *output_manifest["image_files"].values())
        }
        write_json(target / "manifest.json", output_manifest)
        report["datasets"][target.name] = {"path": str(target.resolve()), "manifest_sha256": sha256_file(target / "manifest.json")}
        print(f"derived {target}: action_dim={len(output_manifest['action_names'])}, cameras={len(cameras)}", flush=True)
    report.update(elapsed_seconds=time.monotonic() - started,
                  unique_image_bytes=sum((source / filename).stat().st_size for filename in manifest["image_files"].values()),
                  data_config=str(config), data_config_sha256=sha256_file(config))
    write_json(out / "preparation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v2/data.exploratory_20261022.yaml"))
    parser.add_argument("--local-root", type=Path, default=Path("~/.cache/huggingface/lerobot/local/so101_act_pickplace_v2").expanduser())
    parser.add_argument("--out", type=Path, default=Path(".cache/v2_exploratory_20261022"))
    args = parser.parse_args()
    report = prepare_learning_data(args.config, args.local_root, args.out)
    print(json.dumps({"datasets": list(report["datasets"]), "n_ticks": report["n_ticks"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2))
