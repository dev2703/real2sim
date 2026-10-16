"""Read-only FK/proprio/action-label impact of diagnostic SO-101 mapping changes.

Reconstructs canonical labels from the recorded 10 Hz follower joint samples,
verifies the existing labels, then changes only one mapping family at a time.
No source data, experiment datasets, calibration defaults or limits are changed.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from real2sim.data.labels import episode_labels
from real2sim.geometry.so3 import inv_T, log_so3
from real2sim.geometry.so101_fk import ARM_JOINTS, DEFAULT_MJCF, JointMapping, So101FK
from real2sim.policy.paired import file_sha256


def describe(values):
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"n": 0}
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)), "max": float(x.max())}


def reconstruct(arrays, task, hz):
    mapping = JointMapping.from_task_config(task)
    fk = So101FK(mapping=mapping)
    poses = fk.fk_real(arrays["joint_state"])
    state = np.empty((len(poses), 13), np.float64)
    action = np.empty((len(poses), 7), np.float64)
    extrapolated = np.zeros(len(poses), dtype=bool)
    for ep in np.unique(arrays["episode_index"]):
        rows = np.flatnonzero(arrays["episode_index"] == ep)
        labels = episode_labels(poses[rows], arrays["joint_state"][rows, 5],
                                arrays["joint_action"][rows, 5], hz, "leader_action")
        state[rows], action[rows], extrapolated[rows] = labels.proprio, labels.action, labels.label_is_extrapolated
    q = mapping.arm_real_to_sim_rad(arrays["joint_state"])
    lo, hi = fk.joint_range.T
    violations = (q < lo) | (q > hi)
    lower_excess = np.maximum(lo - q, 0)
    upper_excess = np.maximum(q - hi, 0)
    gripper_joint = fk.model.joint("gripper").id
    grip_limits = fk.model.jnt_range[gripper_joint]
    native_grip = {name: mapping.gripper_real_to_sim_rad(arrays[key][:, 5])
                   for name, key in (("follower", "joint_state"), ("leader", "joint_action"))}
    return {"poses": poses, "state": state, "action": action, "extrapolated": extrapolated,
            "mapping": asdict(mapping), "fk": fk,
            "range": {"arm_violating_ticks": int(violations.any(1).sum()),
                      "arm_violations_per_joint": dict(zip(ARM_JOINTS, violations.sum(0).tolist(), strict=True)),
                      "arm_max_excess_deg_per_joint": dict(zip(ARM_JOINTS, np.rad2deg(
                          np.maximum(lower_excess, upper_excess).max(0)).tolist(), strict=True)),
                      "native_gripper_limits_rad": grip_limits.tolist(),
                      "gripper": {name: {"violating_ticks": int(((qg < grip_limits[0]) | (qg > grip_limits[1])).sum()),
                                          "min_rad": float(qg.min()), "max_rad": float(qg.max())}
                                  for name, qg in native_grip.items()}}, "native_grip": native_grip}


def compare(candidate, baseline, arrays, rows, normalizer):
    pose_delta = candidate["poses"][rows, :3, 3] - baseline["poses"][rows, :3, 3]
    rotations = candidate["poses"][rows, :3, :3] @ baseline["poses"][rows, :3, :3].transpose(0, 2, 1)
    orientation = np.array([np.linalg.norm(log_so3(r)) for r in rotations])
    valid = rows[~baseline["extrapolated"][rows]]
    state_error = candidate["state"][rows] - baseline["state"][rows]
    action_error = candidate["action"][valid] - baseline["action"][valid]
    original_action = baseline["action"][valid]
    new_action = candidate["action"][valid]
    old_active = (np.linalg.norm(original_action[:, :3], axis=1) > .001) | (np.linalg.norm(original_action[:, 3:6], axis=1) > .01)
    new_active = (np.linalg.norm(new_action[:, :3], axis=1) > .001) | (np.linalg.norm(new_action[:, 3:6], axis=1) > .01)
    delta_norm = np.linalg.norm(action_error[:, :3], axis=1)
    rotation_delta_norm = np.linalg.norm(action_error[:, 3:6], axis=1)
    per_dim = np.abs(state_error).mean(0)
    return {
        "pose_translation_difference_m": describe(np.linalg.norm(pose_delta, axis=1)),
        "pose_orientation_difference_rad": describe(orientation),
        "proprio_abs_difference_per_dim": per_dim.tolist(),
        "proprio_abs_difference_old_std_units_per_dim": (per_dim / np.asarray(normalizer["observation.state"]["std"])).tolist(),
        "velocity_difference_m_per_s": describe(np.linalg.norm(state_error[:, 9:12], axis=1)),
        "action_translation_difference_m": describe(delta_norm),
        "action_rotation_difference_rad": describe(rotation_delta_norm),
        "action_abs_difference_per_dim": np.abs(action_error).mean(0).tolist(),
        "action_abs_difference_old_std_units_per_dim": (
            np.abs(action_error).mean(0) / np.asarray(normalizer["action"]["std"])).tolist(),
        "action_translation_difference_above_0_1mm": int((delta_norm > .0001).sum()),
        "action_translation_difference_above_1mm": int((delta_norm > .001).sum()),
        "action_rotation_difference_above_0_01rad": int((rotation_delta_norm > .01).sum()),
        "action_gripper_max_abs_difference": float(np.abs(action_error[:, 6]).max()),
        "active_partition_changed_actions": int((old_active != new_active).sum()),
        "old_active_action_count": int(old_active.sum()), "new_active_action_count": int(new_active.sum()),
        "active_actions": {"translation_difference_m": describe(delta_norm[old_active]),
                           "rotation_difference_rad": describe(rotation_delta_norm[old_active])},
        "stationary_actions": {"translation_difference_m": describe(delta_norm[~old_active]),
                               "rotation_difference_rad": describe(rotation_delta_norm[~old_active])},
        "native_gripper_difference_rad": {
            name: describe(np.abs(candidate["native_grip"][name][rows] - baseline["native_grip"][name][rows]))
            for name in ("follower", "leader")},
    }


def constant_tool_transform_check(candidate, baseline, arrays):
    old, new = baseline["poses"], candidate["poses"]
    transform = inv_T(old[0]) @ new[0]
    right_multiplied = old @ transform
    rotation_residual = np.linalg.norm(right_multiplied[:, :3, :3] - new[:, :3, :3], axis=(1, 2))
    position_residual = np.linalg.norm(right_multiplied[:, :3, 3] - new[:, :3, 3], axis=1)
    valid = np.flatnonzero(~baseline["extrapolated"])
    r = transform[:3, 3]
    predicted_change = (old[valid + 1, :3, :3] - old[valid, :3, :3]) @ r
    actual_change = candidate["action"][valid, :3] - baseline["action"][valid, :3]
    return {"T_oldtool_newtool_from_first_pose": transform.tolist(),
            "translation_offset_norm_m": float(np.linalg.norm(r)),
            "orientation_offset_rad": float(np.linalg.norm(log_so3(transform[:3, :3]))),
            "max_pose_position_residual_m": float(position_residual.max()),
            "max_pose_rotation_matrix_residual": float(rotation_residual.max()),
            "max_translation_delta_formula_residual_m": float(np.linalg.norm(predicted_change - actual_change, axis=1).max()),
            "is_constant_right_transform_to_1e_10": bool(position_residual.max() < 1e-10 and rotation_residual.max() < 1e-10)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".cache/derived_cartesian"))
    parser.add_argument("--baseline-task", type=Path, default=Path("configs/task.yaml"))
    parser.add_argument("--candidate-task", type=Path,
                        default=Path("results/audit/geometry_control_20261016/g1_image_fit/task_candidate.yaml"))
    parser.add_argument("--experiment-manifest", type=Path,
                        default=Path(".cache/experiments_20261016/canonical/manifest.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.data / "manifest.json").read_text())
    stats = json.loads((args.data / "stats.json").read_text())
    if file_sha256(args.data / "arrays.npz") != manifest["files_sha256"]["arrays.npz"]:
        raise ValueError("derived arrays hash mismatch")
    arrays = dict(np.load(args.data / "arrays.npz", allow_pickle=False))
    baseline_task = yaml.safe_load(args.baseline_task.read_text())
    candidate_task = yaml.safe_load(args.candidate_task.read_text())
    for task in (baseline_task, candidate_task):
        frame = task["frames"]["T_task_so101base"]
        if not np.allclose([*frame["xyz"], *frame["rpy"]], 0):
            raise ValueError("this audit uses the existing identity task/base frame contract")
    baseline = reconstruct(arrays, baseline_task, manifest["policy_hz"])
    baseline_check = {
        "state_max_abs_difference_per_dim": np.abs(baseline["state"] - arrays["state"]).max(0).tolist(),
        "action_max_abs_difference_per_dim": np.abs(baseline["action"] - arrays["action"]).max(0).tolist(),
        "extrapolated_masks_equal": bool(np.array_equal(baseline["extrapolated"], arrays["label_is_extrapolated"])),
    }
    if (np.abs(baseline["state"] - arrays["state"]).max() > 1e-6
            or np.abs(baseline["action"] - arrays["action"]).max() > 1e-6
            or not baseline_check["extrapolated_masks_equal"]):
        raise ValueError("baseline reconstruction does not match existing dataset")
    full_offsets = candidate_task["robot_so101"]["real_to_sim"]["offset_deg"]
    configurations = {}
    for name in ("wrist_roll_exact_minus90", "wrist_roll_fitted", "gripper_only", "first_four_arm_zeros", "full_candidate"):
        task = copy.deepcopy(baseline_task)
        if name == "full_candidate":
            task = copy.deepcopy(candidate_task)
        elif name == "wrist_roll_exact_minus90":
            task["robot_so101"]["real_to_sim"]["offset_deg"][4] = -90.
        elif name == "wrist_roll_fitted":
            task["robot_so101"]["real_to_sim"]["offset_deg"][4] = full_offsets[4]
        elif name == "first_four_arm_zeros":
            task["robot_so101"]["real_to_sim"]["offset_deg"][:4] = full_offsets[:4]
        elif name == "gripper_only":
            task["robot_so101"]["gripper"] = copy.deepcopy(candidate_task["robot_so101"]["gripper"])
        configurations[name] = task
    groups = {"all_recorded": np.arange(len(arrays["state"])),
              "original_train": np.flatnonzero(np.isin(arrays["episode_index"], manifest["split"]["train"])),
              "historical_validation": np.flatnonzero(np.isin(arrays["episode_index"], manifest["split"]["val"]))}
    if args.experiment_manifest.exists():
        experiment = json.loads(args.experiment_manifest.read_text())
        for split in ("train", "val"):
            groups[f"experiment_inner_{split}"] = np.flatnonzero(np.isin(arrays["episode_index"], experiment["split"][split]))
    report = {
        "kind": "DIAGNOSTIC_FK_LABEL_MAPPING_IMPACT_NO_DATA_MUTATION",
        "source": manifest["source"], "derived_arrays_sha256": file_sha256(args.data / "arrays.npz"),
        "baseline_task": str(args.baseline_task), "baseline_task_sha256": file_sha256(args.baseline_task),
        "candidate_task": str(args.candidate_task), "candidate_task_sha256": file_sha256(args.candidate_task),
        "mjcf": str(DEFAULT_MJCF), "mjcf_sha256": file_sha256(DEFAULT_MJCF),
        "script_sha256": file_sha256(Path(__file__)), "baseline_reconstruction": baseline_check,
        "n_episodes": len(np.unique(arrays["episode_index"])), "n_ticks": len(arrays["state"]),
        "n_valid_actions": int((~baseline["extrapolated"]).sum()), "baseline_range": baseline["range"],
        "protocol": {
            "joint_inputs": "Unmodified recorded follower degrees and gripper percent at the original10Hz ticks; no simulated tracking states, clipping or smoothing.",
            "action": "Original achieved-motion FK delta contract: task-frame position difference and log(R_next R_current^T); recorded leader gripper percent/100 unchanged.",
            "tool": "Menagerie gripperframe lies off wrist-roll axis; no tool-site relocation.",
            "limits": "Count existing native range violations; do not clip FK input or change joint/safety limits.",
            "base_height": "Task origin is robot base; scene base_z_offset changes table/support location relative to task, not canonical base-frame FK. Camera changes do not alter recorded images or FK labels.",
        }, "variants": {},
    }
    for name, task in configurations.items():
        reconstructed = reconstruct(arrays, task, manifest["policy_hz"])
        result = {"mapping": reconstructed["mapping"], "range": reconstructed["range"],
                  "constant_tool_transform": constant_tool_transform_check(reconstructed, baseline, arrays),
                  "groups": {group: compare(reconstructed, baseline, arrays, rows, stats) for group, rows in groups.items()},
                  "episodes": {str(int(ep)): compare(reconstructed, baseline, arrays,
                               np.flatnonzero(arrays["episode_index"] == ep), stats)
                               for ep in np.unique(arrays["episode_index"])}}
        report["variants"][name] = result
        score = result["groups"]["all_recorded"]
        print(json.dumps({"variant": name,
                          "pose_shift_mm_mean": score["pose_translation_difference_m"]["mean"] * 1000,
                          "translation_label_shift_mm_mean": score["action_translation_difference_m"]["mean"] * 1000,
                          "rotation_label_shift_rad_mean": score["action_rotation_difference_rad"]["mean"]}), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "executed_script.txt").write_text(Path(__file__).read_text())
    lines = ["# FK label impact of diagnostic joint mapping", "",
             "All recorded episodes; unchanged follower joint samples, leader gripper targets and tool site. No candidate is adopted by this audit.", "",
             "| Variant | Mean tool shift (mm) | Mean / p95 translation label change (mm) | Mean / p95 rotation label change (rad) | Active partition changes |",
             "|---|---:|---:|---:|---:|"]
    for name, value in report["variants"].items():
        score = value["groups"]["all_recorded"]
        p, dp, dr = (score[key] for key in ("pose_translation_difference_m", "action_translation_difference_m", "action_rotation_difference_rad"))
        lines.append(f"| {name} | {p['mean'] * 1000:.4f} | {dp['mean'] * 1000:.4f} / {dp['p95'] * 1000:.4f} | {dr['mean']:.6f} / {dr['p95']:.6f} | {score['active_partition_changed_actions']} |")
    lines += ["", "For a roll-only zero correction, T_new = T_old X with constant X. Thus R_new = R_old C and p_new = p_old + R_old r. The world-frame rotation increment is unchanged, but the new translation increment is dp_old + (R_next - R_current) r. The tool site is about12mm off the roll axis, so r is nonzero.",
              "", "A gripper affine map changes native squeezing/opening commands and range clipping, but the current tool site is independent of gripper angle. Percent-valued gripper observations and targets retain their semantics; no arm FK labels change from that map alone.",
              "", "Existing architecture/proprioception runs remain valid comparisons of the original provisional data contract, not evidence for the corrected physical contract. Before final training on an adopted arm/roll calibration, regenerate canonical pose, velocity, Cartesian targets, train-only normalizers and dependent observation variants together. Raw joints, images, splits, causal filter logic and frozen DINO image cache can be reused. The new task-specific normalizers must accompany regenerated states/actions.",
              "", "If only wrist roll is adopted, an explicit fixed-tool-transform compatibility adapter can map calibrated current poses to the old virtual tool frame and transform requested deltas back exactly. That is an additional controller experiment requiring matching velocity/history transformations and collision checks; applying old Cartesian predictions unchanged is not equivalent. Full first-four-joint zero changes generally cannot use one constant tool transform."]
    (args.out / "review.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
