"""Test completed delta predictions against the provisional five-joint action manifold.

Teacher-forced diagnosis only: no privileged object inputs, geometry fitting, simulator
rollouts, or policy changes. A feasible FK step does not certify collision-free execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from real2sim.config import load_yaml
from real2sim.geometry.so3 import delta_action, exp_so3, log_so3
from real2sim.geometry.so101_fk import JointMapping, So101FK
from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.kinematic_actions import bounded_joint_step
from real2sim.policy.paired import file_sha256


def task_jacobian(fk, q):
    """World/base-axis spatial velocity Jacobian, with arm columns only."""
    pose = fk.fk_sim(q)
    # So101FK only updates positions. mj_jacSite also needs joint motion axes/CDOF.
    mujoco.mj_comPos(fk.model, fk.data)
    jp, jr = np.zeros((3, fk.model.nv)), np.zeros((3, fk.model.nv))
    mujoco.mj_jacSite(fk.model, fk.data, jp, jr, fk.site_id)
    vadr = np.array([fk.model.jnt_dofadr[fk.model.joint(j).id] for j in (
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
    )])
    return pose, np.vstack([jp, jr])[:, vadr]


def local_residual(jacobian, action, rotation_weight):
    weights = np.r_[np.ones(3), np.full(3, rotation_weight)]
    matrix, target = weights[:, None] * jacobian, weights * action
    step = np.linalg.lstsq(matrix, target, rcond=1e-8)[0]
    return float(np.linalg.norm(target - matrix @ step) * 1000)


def rotation_error(prediction, label):
    return float(np.linalg.norm(log_so3(exp_so3(prediction) @ exp_so3(label).T)))


def select_anchors(dataset, per_episode):
    if per_episode < 1:
        raise ValueError("per_episode must be positive")
    a = dataset.arrays
    chosen = []
    for ep in np.unique(a["episode_index"][dataset.indices]):
        candidates = [i for i, t in enumerate(dataset.indices)
                      if a["episode_index"][t] == ep
                      and (t - a["ep_start"][t]) % 2 == 0
                      and not a["label_is_extrapolated"][t]]
        offsets = np.unique(np.linspace(0, len(candidates) - 1,
                                       min(per_episode, len(candidates))).astype(int))
        chosen.extend(candidates[i] for i in offsets)
    return chosen


def summary(rows):
    result = {"actions": len(rows)}
    for key in ("label_normal_residual_mm_equivalent", "prediction_normal_residual_mm_equivalent",
                "prediction_translation_error_mm", "projected_translation_error_mm",
                "prediction_rotation_error_rad", "projected_rotation_error_rad",
                "projection_translation_change_mm"):
        values = np.array([r[key] for r in rows])
        result[key] = {"mean": float(values.mean()), "p95": float(np.quantile(values, .95))}
    result["projection_improved_translation_fraction"] = float(np.mean([
        r["projected_translation_error_mm"] < r["prediction_translation_error_mm"] for r in rows
    ]))
    return result


def diagnose(checkpoint, data, sim_config, task_config, per_episode=16):
    policy, norm, contract = load_checkpoint(checkpoint, "cpu")
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["action_representation"] != "cartesian":
        raise ValueError("diagnostic requires task-frame delta actions")
    for key in ("camera_keys", "state_names", "action_names", "split", "policy_hz"):
        if contract[key] != manifest[key]:
            raise ValueError(f"checkpoint/data mismatch: {key}")
    if contract["derived_files_sha256"] != manifest["files_sha256"]:
        raise ValueError("checkpoint/data payload hashes differ")
    if file_sha256(data / "arrays.npz") != manifest["files_sha256"]["arrays.npz"]:
        raise ValueError("actual array contents differ from the prepared manifest")
    if policy.config.n_action_steps != 2:
        raise ValueError("expected two executed-prefix actions")
    ds = DerivedTickDataset(data, manifest["split"]["val"], policy.config.chunk_size, norm)
    if file_sha256(task_config) != manifest["calibration_sha256"]:
        raise ValueError("task mapping differs from the prepared-data calibration")
    fk = So101FK(mapping=JointMapping.from_task_config(load_yaml(task_config)))
    sim = load_yaml(sim_config)
    cap = np.asarray(sim["control"]["max_velocity"][:5]) / float(manifest["policy_hz"])
    weight = float(sim["ik"]["orientation_weight_m_per_rad"])
    rows, excluded = [], []
    for batch in DataLoader(Subset(ds, select_anchors(ds, per_episode)), batch_size=8):
        obs = {k: v for k, v in batch.items() if k.startswith("observation.")}
        kwargs = ({"sampling_seed": batch["tick"].tolist()}
                  if getattr(policy.config, "type", None) == "small_chunk_head" else {})
        with torch.inference_mode():
            pred = norm.unaction(policy.predict_action_chunk(obs, **kwargs)[:, :2]).cpu().numpy()
        for i, tick in enumerate(batch["tick"].tolist()):
            for offset in range(2):
                if batch["action_is_pad"][i, offset]:
                    continue
                t = tick + offset
                # Prepared joint_state uses dataset degrees and gripper percent, not radians.
                q = fk.mapping.arm_real_to_sim_rad(ds.arrays["joint_state"][t])
                if np.any(q < fk.joint_range[:, 0]) or np.any(q > fk.joint_range[:, 1]):
                    excluded.append(t)
                    continue
                pose, jac = task_jacobian(fk, q)
                action, label = pred[i, offset, :6], ds.arrays["action"][t, :6]
                step = bounded_joint_step(jac, action, q, fk.joint_range, cap,
                                          rotation_weight=weight)
                projected = delta_action(pose, fk.fk_sim(q + step))
                rows.append({
                    "tick": t, "episode": int(ds.arrays["episode_index"][t]),
                    "source_frame": int(ds.arrays["frame_index"][t]),
                    "label_normal_residual_mm_equivalent": local_residual(jac, label, weight),
                    "prediction_normal_residual_mm_equivalent": local_residual(jac, action, weight),
                    "prediction_translation_error_mm": float(np.linalg.norm(action[:3] - label[:3]) * 1000),
                    "projected_translation_error_mm": float(np.linalg.norm(projected[:3] - label[:3]) * 1000),
                    "prediction_rotation_error_rad": rotation_error(action[3:], label[3:]),
                    "projected_rotation_error_rad": rotation_error(projected[3:], label[3:]),
                    "projection_translation_change_mm": float(np.linalg.norm(projected[:3] - action[:3]) * 1000),
                })
    if not rows:
        raise ValueError("no in-limit validation actions")
    return {
        "kind": "TEACHER_FORCED_PROVISIONAL_ACTION_MANIFOLD_DIAGNOSTIC",
        "learned_task_success": None, "physical_calibration_accepted": False,
        "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
        "model_sha256": file_sha256(fk.mjcf),
        "script_sha256": file_sha256(Path(__file__)),
        "manifest_sha256": file_sha256(data / "manifest.json"),
        "sim_config_sha256": file_sha256(sim_config),
        "task_config_sha256": file_sha256(task_config),
        "dependency_sha256": {str(p): file_sha256(p) for p in (
            data / "arrays.npz", checkpoint / "config.json", checkpoint / "stats.json",
            checkpoint / "real2sim_contract.json", Path("src/real2sim/policy/kinematic_actions.py"),
        )},
        "episodes": manifest["split"]["val"], "per_episode_anchors": per_episode,
        "rotation_weight_m_per_rad": weight, "joint_step_cap_rad": cap.tolist(),
        "excluded_out_of_limit_ticks": excluded, "summary": summary(rows),
        "per_episode": {str(ep): summary([r for r in rows if r["episode"] == ep])
                        for ep in sorted({r["episode"] for r in rows})},
        "limitations": "Five-joint local tangent and bounded FK projection under the frozen provisional mapping. "
        "Second prefix action is tested at its recorded successor state, not a rolled-out state. "
        "Projection can alter intended motion; this test does not check collisions, contacts, camera geometry, "
        "closed-loop stability or placement. No geometry, model or controller changes are adopted.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone_top"))
    parser.add_argument("--sim-config", type=Path, default=Path("configs/sim_so101.yaml"))
    parser.add_argument("--task-config", type=Path, default=Path("configs/v2/task.exploratory_20261022.yaml"))
    parser.add_argument("--per-episode", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    report = diagnose(args.checkpoint, args.data, args.sim_config, args.task_config, args.per_episode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
