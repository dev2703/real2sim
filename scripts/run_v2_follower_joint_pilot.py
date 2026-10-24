"""Bounded matched target-semantics ACT pilot, followed by common achieved-FK scoring."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from real2sim.config import load_yaml
from real2sim.data.follower_joint_targets import VARIANTS, native_joint_rows
from real2sim.data.raw import sha256_file
from real2sim.geometry.so3 import log_so3
from real2sim.geometry.so101_fk import JointMapping, So101FK
from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.train import train
from real2sim.policy.validation import evaluate_executed_prefix


def common_fk_score(checkpoint: Path, data: Path, task: Path) -> dict:
    policy, norm, contract = load_checkpoint(checkpoint, "cpu")
    ds = DerivedTickDataset(data, contract["split"]["val"], policy.config.chunk_size, norm)
    _, output = evaluate_executed_prefix(policy, ds, "cpu", 16)
    ticks, prediction = output["tick"], output["model"]
    a = ds.arrays
    if np.any(a["episode_index"][ticks] != a["episode_index"][ticks + 1]):
        raise ValueError("common next-follower scoring crossed an episode")
    mapping = JointMapping.from_task_config(load_yaml(task))
    fk = So101FK(mapping=mapping)
    measured = fk.fk_real(a["joint_state"][ticks])
    target = fk.fk_real(a["joint_state"][ticks + 1])
    poses = fk.fk_real(prediction)
    error = poses[:, :3, 3] - target[:, :3, 3]
    moving_label = target[:, :3, 3] - measured[:, :3, 3]
    moving_prediction = poses[:, :3, 3] - measured[:, :3, 3]
    active = np.linalg.norm(moving_label, axis=1) > .0004
    rotation = np.asarray([np.linalg.norm(log_so3(p[:3, :3] @ t[:3, :3].T))
                           for p, t in zip(poses, target, strict=True)])
    next_rotation = np.asarray([np.linalg.norm(log_so3(t[:3, :3] @ m[:3, :3].T))
                               for t, m in zip(target, measured, strict=True)])
    stationary = ~active & (next_rotation <= .01)
    native = native_joint_rows(prediction, mapping)
    limits = np.asarray([fk.model.jnt_range[fk.model.joint(name).id] for name in
                         ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")])
    return {"n": len(ticks), "common_reference": "Recorded next achieved follower arm FK under unchanged provisional mapping; current leader gripper command",
            "translation_mean_mm": float(np.linalg.norm(error, axis=1).mean() * 1000),
            "translation_p95_mm": float(np.quantile(np.linalg.norm(error, axis=1), .95) * 1000),
            "rotation_mean_rad": float(rotation.mean()),
            "signed_position_bias_xyz_mm": (error.mean(0) * 1000).tolist(),
            "moving_actions": int(active.sum()),
            "moving_positive_direction_fraction": float((np.sum(moving_prediction[active] * moving_label[active], axis=1) > 0).mean()),
            "moving_step_to_label_ratio_median": float(np.median(np.linalg.norm(moving_prediction[active], axis=1) / np.linalg.norm(moving_label[active], axis=1))),
            "stationary_actions": int(stationary.sum()),
            "stationary_target_displacement_mean_mm": float(np.linalg.norm(moving_prediction[stationary], axis=1).mean() * 1000) if stationary.any() else None,
            "raw_predicted_native_out_of_range_actions": int(((native < limits[:, 0]) | (native > limits[:, 1])).any(axis=1).sum()),
            "leader_gripper_mae_percentage_points": float(np.abs(prediction[:, 5] - a["joint_action"][ticks, 5]).mean()),
            "raw_prediction_clipping": False,
            "coverage": "Shared-valid validation mask, identical source tick IDs for the paired target definitions; not full original validation coverage",
            "limitations": "Raw command/FK proxy before guarded rejection, rate limiting and physics. Provisional FK and selected valid contexts; no task success or collision-path certification."}


def run(data_root: Path, out: Path, steps: int):
    if out.exists() or not 1 <= steps <= 512:
        raise ValueError("fresh output directory and bounded1..512 updates required")
    torch.set_num_threads(2)
    task = Path("configs/v2/task.exploratory_20261022.yaml")
    cfg = load_yaml("outputs/v2/learning_screen_20261022/configs/L1_resnet_iphone.yaml")
    cfg["scheduler"]["warmup_steps"] = max(1, steps // 8)
    cfg["train"].update(steps=steps, schedule_steps=steps, val_every=steps, save_every=steps,
                         log_every=32, val_max_batches=None, device="cpu", keep_only_best_and_last=True)
    out.mkdir(parents=True)
    report = {"kind": "MATCHED_FOLLOWER_JOINT_TARGET_PILOT", "training_seed": 0,
              "updates_per_model": steps, "examples_per_model": steps * 16,
              "physical_calibration_accepted": False, "reserved_episodes_used": [],
              "checkpoint_promoted": False, "source_preparation": str(data_root / "preparation.json"),
              "source_preparation_sha256": sha256_file(data_root / "preparation.json"),
              "script_sha256": sha256_file(Path(__file__)), "models": {},
              "limitations": "One short seed with shared native-valid context selection and unchanged provisional FK; different target semantics are ranked only by common achieved-FK proxies, not own-label normalized loss."}
    for name in reversed(VARIANTS):
        model_cfg = copy.deepcopy(cfg)
        data = data_root / name
        model_cfg.update(name=name, data=str(data.resolve()))
        model_cfg["experiment"] = {"kind": report["kind"], "joint_target_semantics": json.loads((data / "manifest.json").read_text())["joint_target_semantics"]}
        config_path = out / (name + ".yaml")
        config_path.write_text(yaml.safe_dump(model_cfg, sort_keys=False))
        result = train(model_cfg, data, out / name)
        checkpoint = out / name / "checkpoints/best"
        common = common_fk_score(checkpoint, data, task)
        report["models"][name] = {"training": result, "common_next_follower_fk": common,
                                  "checkpoint": str(checkpoint),
                                  "checkpoint_sha256": sha256_file(checkpoint / "model.safetensors"),
                                  "config_sha256": sha256_file(config_path)}
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"name": name, "common_next_follower_fk": common}), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path(".cache/v2_follower_joint_20261024"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=256)
    args = p.parse_args()
    run(args.data, args.out, args.steps)
