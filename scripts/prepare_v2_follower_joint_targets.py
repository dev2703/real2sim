"""Prepare matched follower-next and leader-command targets without altering raw evidence."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from real2sim.config import load_yaml
from real2sim.data.follower_joint_targets import JOINT_ACTION_NAMES, VERSION, make_matched_targets
from real2sim.data.prepare import compute_stats
from real2sim.data.raw import sha256_file
from real2sim.geometry.so101_fk import JointMapping, So101FK


def prepare(source: Path, task: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    source = source.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if any(key in manifest for key in ("visual_grounding", "observation_variant", "joint_target_semantics")):
        raise ValueError("prepare from original untransformed canonical source")
    if sha256_file(task) != manifest["calibration_sha256"]:
        raise ValueError("task mapping differs from the source FK contract")
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or sha256_file(source / name) != expected:
            raise ValueError(f"source checksum mismatch: {name}")
    with np.load(source / "arrays.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    mapping = JointMapping.from_task_config(load_yaml(task))
    fk = So101FK(mapping=mapping)
    joint_names = [name.removesuffix(".pos") for name in JOINT_ACTION_NAMES]
    limits = np.asarray([fk.model.jnt_range[fk.model.joint(name).id] for name in joint_names])
    split = manifest["split"]
    if (set(split["train"]) & set(split["val"]) or
            (set(split["train"]) | set(split["val"])) & set(split["reserved"])):
        raise ValueError("train/validation/reserved partitions overlap")
    variants, summary = make_matched_targets(arrays, mapping, limits, split["train"] + split["val"])
    valid = ~next(iter(variants.values()))["action_is_invalid"]
    training = np.isin(arrays["episode_index"], split["train"]) & valid
    # Same coordinate units and same train-only action normalizer for a fair paired pilot.
    paired_train = np.concatenate([value["action"][training] for value in variants.values()])
    base_stats = json.loads((source / "stats.json").read_text())
    base_stats.update({"observation.state": compute_stats(arrays["state"][training]),
                       "action": compute_stats(paired_train), "train_episodes": split["train"],
                       "computed_on": "Shared-valid training observations; pooled paired training targets only, identical action statistics across both target definitions"})
    out.mkdir(parents=True)
    summary.update(source=str(source), source_manifest_sha256=sha256_file(source / "manifest.json"),
                   task_path=str(task), task_sha256=sha256_file(task),
                   physical_calibration_accepted=False, reserved_episodes_scored=[], variants={})
    for partition in ("train", "val"):
        mask = np.isin(arrays["episode_index"], split[partition])
        summary[partition] = {"source_ticks": int(mask.sum()), "shared_valid_ticks": int((mask & valid).sum()),
                              "fraction_retained": float(valid[mask].mean())}
    for name, values in variants.items():
        target = out / name
        target.mkdir()
        np.savez(target / "arrays.npz", **values)
        (target / "stats.json").write_text(json.dumps(base_stats, indent=2) + "\n")
        for image_file in manifest["image_files"].values():
            (target / image_file).symlink_to(source / image_file)
        output = copy.deepcopy(manifest)
        semantics = {"version": VERSION, "variant": name,
                     "arm_target": "next_observed_follower" if name.startswith("follower") else "current_recorded_leader_command",
                     "arm_units": "degrees", "gripper_target": "current_recorded_leader_command_percent",
                     "lead_policy_ticks": 1 if name.startswith("follower") else 0,
                     "inference": "guarded_absolute_native_joint_target_v1",
                     "collision_path_samples": 17,
                     "validity": "shared_native_current_and_both_targets_in_range; no clipping; contiguous valid chunks"}
        output.update(mode="canonical_joint", action_representation="canonical_joint",
                      prepare_version=VERSION, contract_version=VERSION + "_" + name,
                      action_names=JOINT_ACTION_NAMES, joint_target_semantics=semantics,
                      controller_semantics="current_pose", source_manifest_sha256=summary["source_manifest_sha256"],
                      label_semantics=("Next observed follower arm joint angles plus current recorded leader gripper command" if name.startswith("follower") else "Current recorded leader arm and gripper command, matched to follower target validity mask"),
                      target_validity=summary["mask_contract"], physical_calibration_accepted=False,
                      n_ticks_train=int(training.sum()), n_ticks_val=int((np.isin(arrays["episode_index"], split["val"]) & valid).sum()))
        output["files_sha256"] = {**manifest["files_sha256"],
                                  "arrays.npz": sha256_file(target / "arrays.npz"),
                                  "stats.json": sha256_file(target / "stats.json")}
        (target / "manifest.json").write_text(json.dumps(output, indent=2) + "\n")
        summary["variants"][name] = {"path": str(target.resolve()),
                                    "manifest_sha256": sha256_file(target / "manifest.json")}
    (out / "preparation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone"))
    p.add_argument("--task", type=Path, default=Path("configs/v2/task.exploratory_20261022.yaml"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.task, args.out), indent=2))
