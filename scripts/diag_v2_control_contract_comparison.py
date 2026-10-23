"""Bounded comparison of joint targets and Cartesian position-priority candidates.

Uses recorded development images/states and two executed actions per chunk. No
physics rollout, geometry fit, object-relative guidance, or reserved evaluation.
Every candidate retains native joint, workspace, pose and endpoint collision gates.
Reported FK motion is a command-space proxy, not measured actuator tracking.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from real2sim.config import load_yaml
from real2sim.geometry.so3 import apply_delta, delta_action
from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.kinematic_actions import bounded_joint_step
from real2sim.policy.paired import file_sha256
from real2sim.sim.contact_solver import contact_solver_settings
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics


def native_q(mapping, real):
    return np.r_[mapping.arm_real_to_sim_rad(real),
                 mapping.gripper_real_to_sim_rad(real[5])]


def bounded_segment(seed, target, caps):
    """Common progress keeps a checked joint segment's direction under rate caps."""
    seed, target, caps = map(lambda x: np.asarray(x, float), (seed, target, caps))
    if seed.shape != target.shape or seed.shape != caps.shape or np.any(caps <= 0):
        raise ValueError("matching joint vectors and positive caps required")
    if not all(np.isfinite(a).all() for a in (seed, target, caps)):
        raise ValueError("joint vectors and caps must be finite")
    step = target - seed
    scale = min(1., float(np.min(caps / np.maximum(np.abs(step), 1e-15))))
    return seed + scale * step, scale


def candidate_gate(ik, q, requested_pose=None):
    """Check the original endpoint gates without clipping or re-solving a joint target."""
    q = np.asarray(q, float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("candidate needs six finite native radians")
    if np.any(q < ik.limits[:, 0]) or np.any(q > ik.limits[:, 1]):
        return {"accepted": False, "reason": "native_joint_limit", "position_error_m": None,
                "orientation_error_rad": None}
    pose = ik.forward(q)
    error = np.zeros(6) if requested_pose is None else delta_action(pose, requested_pose)
    pe, re = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
    cfg = ik.config
    workspace_pose = pose if requested_pose is None else requested_pose
    outside = (np.any(workspace_pose[:3, 3] < cfg["workspace_min"])
               or np.any(workspace_pose[:3, 3] > cfg["workspace_max"]))
    if outside:
        reason = "outside_workspace"
    elif pe > cfg["position_tolerance_m"] or re > cfg["orientation_tolerance_rad"]:
        reason = "residual_rejected"
    elif ik._collision():
        reason = "collision"
    else:
        reason = "accepted"
    return {"accepted": reason == "accepted", "reason": reason,
            "position_error_m": pe, "orientation_error_rad": re}


def sample_anchors(dataset, per_episode):
    if per_episode < 1:
        raise ValueError("per_episode must be positive")
    a, result = dataset.arrays, []
    for episode in dataset.manifest["split"]["val"]:
        choices = [i for i, t in enumerate(dataset.indices)
                   if a["episode_index"][t] == episode
                   and (t - a["ep_start"][t]) % 2 == 0
                   and not a["label_is_extrapolated"][t]]
        offsets = np.unique(np.linspace(0, len(choices) - 1,
                                       min(per_episode, len(choices))).astype(int))
        result.extend(choices[i] for i in offsets)
    return result


def predictions(checkpoint, data, per_episode):
    policy, norm, contract = load_checkpoint(checkpoint, "cpu")
    manifest = json.loads((data / "manifest.json").read_text())
    for key in ("camera_keys", "state_names", "action_names", "split", "policy_hz"):
        if manifest[key] != contract[key]:
            raise ValueError(f"checkpoint/data mismatch: {key}")
    if manifest["files_sha256"] != contract["derived_files_sha256"]:
        raise ValueError("checkpoint/data payload contract differs")
    if file_sha256(data / "arrays.npz") != manifest["files_sha256"]["arrays.npz"]:
        raise ValueError("actual arrays differ from manifest")
    if policy.config.n_action_steps != 2:
        raise ValueError("comparison requires two-action execution prefixes")
    split = manifest["split"]
    if set(split["val"]) & set(split["reserved"]):
        raise ValueError("development and reserved episodes overlap")
    ds = DerivedTickDataset(data, split["val"], policy.config.chunk_size, norm)
    selected = sample_anchors(ds, per_episode)
    outputs = {}
    for batch in DataLoader(Subset(ds, selected), batch_size=8):
        obs = {k: v for k, v in batch.items() if k.startswith("observation.")}
        with torch.inference_mode():
            pred = norm.unaction(policy.predict_action_chunk(obs)[:, :2]).cpu().numpy()
        if not np.isfinite(pred).all():
            raise ValueError("nonfinite prediction")
        for i, tick in enumerate(batch["tick"].tolist()):
            outputs[tick] = pred[i]
    # Compare exactly the images consumed by both models, independent of filenames/hardlinks.
    digest = hashlib.sha256()
    for index in selected:
        for key in manifest["camera_keys"]:
            digest.update(np.asarray(ds.images[key][ds.indices[index]]).tobytes())
    meta = {"checkpoint": str(checkpoint), "manifest_sha256": file_sha256(data / "manifest.json"),
            "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
            "sampled_image_bytes_sha256": digest.hexdigest(),
            "metadata_sha256": {name: file_sha256(checkpoint / name) for name in (
                "real2sim_contract.json", "config.json", "stats.json")}}
    return ds, outputs, meta


def quantiles(values):
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95))} if len(a) else None


def summarize(rows, mode):
    selected = [r["modes"][mode] for r in rows]
    active = [m for r, m in zip(rows, selected, strict=True) if r["label_step_mm"] > .4]
    stationary = [m for r, m in zip(rows, selected, strict=True) if r["label_step_mm"] <= .4]
    return {"n": len(selected), "accepted": sum(s["accepted"] for s in selected),
            "reasons": dict(Counter(s["reason"] for s in selected)),
            "proposed_next_pose_error_mm": quantiles([s["proposed_next_pose_error_mm"] for s in selected]),
            "bounded_command_next_pose_error_mm": quantiles([s["bounded_command_next_pose_error_mm"] for s in selected]),
            "bounded_command_orientation_error_rad": quantiles([s["bounded_command_orientation_error_rad"] for s in selected]),
            "bounded_command_gripper_error_pp": quantiles([s["bounded_command_gripper_error_pp"] for s in selected]),
            "bounded_command_signed_translation_bias_mm": np.mean([
                s["bounded_command_translation_error_mm"] for s in selected], axis=0).tolist(),
            "active_actions": len(active),
            "active_positive_direction_fraction": float(np.mean([
                s["direction_cosine"] > 0 for s in active])) if active else None,
            "active_command_label_magnitude_ratio": quantiles([s["magnitude_ratio"] for s in active]),
            "stationary_actions": len(stationary),
            "stationary_bounded_command_step_mm": quantiles([s["bounded_command_step_mm"] for s in stationary]),
            "bounded_command_endpoint_collision_actions": sum(s["bounded_command_endpoint_collision"] for s in selected)}


def run(args):
    cart, cp, cmeta = predictions(args.cart_checkpoint, args.cart_data, args.per_episode)
    joint, jp, jmeta = predictions(args.joint_checkpoint, args.joint_data, args.per_episode)
    if cart.manifest["action_representation"] != "cartesian" or joint.manifest["action_representation"] != "canonical_joint":
        raise ValueError("expected Cartesian deltas and canonical joint commands")
    for key in ("split", "camera_keys", "policy_hz", "state_names", "calibration_sha256"):
        if cart.manifest[key] != joint.manifest[key]:
            raise ValueError(f"joint/Cartesian comparison differs: {key}")
    if cmeta["sampled_image_bytes_sha256"] != jmeta["sampled_image_bytes_sha256"]:
        raise ValueError("models did not consume identical sampled images")
    if set(cp) != set(jp):
        raise ValueError("models used different anchors")
    for name in ("state", "joint_state", "joint_action", "episode_index", "frame_index", "label_is_extrapolated"):
        np.testing.assert_array_equal(cart.arrays[name], joint.arrays[name])
    task = Path(load_yaml(args.sim_config)["task_config"])
    if file_sha256(task) != cart.manifest["calibration_sha256"]:
        raise ValueError("simulator task mapping differs from the training labels")
    eval_cfg = load_yaml(args.eval_config)["cartesian"]
    rows, excluded = [], []
    with So101PickPlaceEnv(args.sim_config, render_images=False) as env:
        if env.control_hz != cart.manifest["policy_hz"]:
            raise ValueError("control rate differs from checkpoint")
        ik = NativeKinematics(env.model, env.config)
        cfg = copy.deepcopy(env.config)
        cfg["ik"]["position_priority_fallback"] = True
        fallback = NativeKinematics(env.model, cfg)
        cap = env.max_velocity / env.control_hz
        weights = [env.config["ik"]["orientation_weight_m_per_rad"] * s for s in (1., .0625)]
        names = ["cartesian_ik", "cartesian_position_priority", "joint_prediction",
                 "recorded_joint_command", "recorded_follower_next_joint", "hold",
                 *[f"cartesian_bounded_joint_weight_{w:g}" for w in weights]]
        for anchor in sorted(cp):
            references = {n: None for n in names}
            for offset in range(2):
                t = anchor + offset
                if (cart.arrays["label_is_extrapolated"][t]
                        or cart.arrays["episode_index"][t] != cart.arrays["episode_index"][anchor]):
                    continue
                q = native_q(env.mapping, cart.arrays["joint_state"][t])
                if np.any(q < ik.limits[:, 0]) or np.any(q > ik.limits[:, 1]):
                    excluded.append({"anchor": anchor, "offset": offset, "tick": t,
                                     "episode": int(cart.arrays["episode_index"][t]),
                                     "reason": "observed_native_joint_limit"})
                    continue
                current = ik.forward(q)
                np.testing.assert_allclose(current[:3, 3], cart.arrays["state"][t, :3], atol=1e-7)
                nxt = native_q(env.mapping, cart.arrays["joint_state"][t + 1])
                next_pose = ik.forward(nxt)
                label = delta_action(current, next_pose)
                np.testing.assert_allclose(label, cart.arrays["action"][t, :6], atol=1e-7)
                row = {"anchor": anchor, "prefix_offset": offset, "tick": t,
                       "episode": int(cart.arrays["episode_index"][t]),
                       "source_frame": int(cart.arrays["frame_index"][t]),
                       "q_native_rad": q.tolist(), "label_translation_m": label[:3].tolist(),
                       "label_step_mm": float(np.linalg.norm(label[:3]) * 1000),
                       "observed_endpoint_collision": candidate_gate(ik, q)["reason"] == "collision",
                       "modes": {}}
                for mode in names:
                    a = np.asarray(cp[anchor][offset], float).copy()
                    for part, maximum in ((slice(0, 3), eval_cfg["max_step_m"]),
                                          (slice(3, 6), eval_cfg["max_step_rad"])):
                        a[part] *= min(1., maximum / max(np.linalg.norm(a[part]), 1e-15))
                    reference = references[mode]
                    if reference is None or eval_cfg["target_reference"] == "current_pose":
                        reference = current
                    requested = apply_delta(reference, a[:6])
                    feedback = delta_action(current, requested)
                    for part, maximum in ((slice(0, 3), eval_cfg["max_step_m"]),
                                          (slice(3, 6), eval_cfg["max_step_rad"])):
                        feedback[part] *= min(1., maximum / max(np.linalg.norm(feedback[part]), 1e-15))
                    requested = apply_delta(current, feedback)
                    grip = float(np.clip(env.mapping.gripper_real_to_sim_rad(
                        np.clip(a[6], 0, 1) * 100), *ik.limits[5]))
                    candidate = q.copy()
                    if mode in ("cartesian_ik", "cartesian_position_priority"):
                        solver = ik if mode == "cartesian_ik" else fallback
                        result = solver.solve(requested, q, gripper=grip)
                        gate = {"accepted": result.accepted, "reason": result.reason,
                                "position_error_m": result.position_error_m,
                                "orientation_error_rad": result.orientation_error_rad}
                        candidate = result.qpos.copy()
                        proposed_pose = requested
                    elif mode.startswith("cartesian_bounded_joint_weight_"):
                        weight = float(mode.removeprefix("cartesian_bounded_joint_weight_"))
                        ik.forward(q)
                        jp_arm = np.zeros((3, env.model.nv))
                        jr_arm = np.zeros_like(jp_arm)
                        mujoco.mj_jacSite(env.model, ik.data, jp_arm, jr_arm, ik.site)
                        jac = np.vstack([jp_arm, jr_arm])[:, ik.vadr[:5]]
                        candidate[:5] += bounded_joint_step(jac, feedback, q[:5], ik.limits[:5],
                                                           cap[:5], rotation_weight=weight)
                        candidate[5] = grip
                        gate = candidate_gate(ik, candidate, requested)
                        proposed_pose = ik.forward(candidate)
                    elif mode in ("joint_prediction", "recorded_joint_command", "recorded_follower_next_joint"):
                        raw = (jp[anchor][offset] if mode == "joint_prediction"
                               else joint.arrays["action"][t] if mode == "recorded_joint_command"
                               else cart.arrays["joint_state"][t + 1])
                        candidate = native_q(env.mapping, raw)
                        proposed_pose = ik.forward(candidate)
                        gate = candidate_gate(ik, candidate)
                    else:
                        proposed_pose = current
                        gate = candidate_gate(ik, q)
                    if gate["accepted"] and mode.startswith("cartesian"):
                        references[mode] = requested.copy()
                    if not gate["accepted"]:
                        candidate = q.copy()
                        candidate[5] = grip if mode.startswith("cartesian") else q[5]
                    bounded, scale = bounded_segment(q, candidate, cap)
                    bounded_pose = ik.forward(bounded)
                    bounded_error = delta_action(bounded_pose, next_pose)
                    motion = delta_action(current, bounded_pose)[:3]
                    norm_product = np.linalg.norm(motion) * np.linalg.norm(label[:3])
                    row["modes"][mode] = {**gate, "joint_rate_progress_fraction": scale,
                        "bounded_command_native_rad": bounded.tolist(),
                        "proposed_next_pose_error_mm": float(np.linalg.norm(proposed_pose[:3, 3] - next_pose[:3, 3]) * 1000),
                        "bounded_command_next_pose_error_mm": float(np.linalg.norm(bounded_error[:3]) * 1000),
                        "bounded_command_translation_error_mm": ((bounded_pose[:3, 3] - next_pose[:3, 3]) * 1000).tolist(),
                        "bounded_command_orientation_error_rad": float(np.linalg.norm(bounded_error[3:])),
                        "bounded_command_gripper_error_pp": float(abs(env.mapping.gripper_sim_to_real_percent(bounded[5])
                                                                 - cart.arrays["joint_state"][t + 1, 5])),
                        "bounded_command_step_mm": float(np.linalg.norm(motion) * 1000),
                        "direction_cosine": float(np.dot(motion, label[:3]) / norm_product) if norm_product > 1e-15 else 0.,
                        "magnitude_ratio": float(np.linalg.norm(motion) / np.linalg.norm(label[:3])) if np.linalg.norm(label[:3]) > 1e-12 else 0.,
                        "bounded_command_endpoint_collision": candidate_gate(ik, bounded)["reason"] == "collision"}
                rows.append(row)
        if not rows:
            raise ValueError("no sampled actions have an in-range observed joint state")
        report = {"kind": "DEVELOPMENT_TEACHER_FORCED_CONTROL_CONTRACT_COMPARISON",
            "physical_calibration_accepted": False, "learned_task_success": None,
            "protocol": {"per_episode_anchors": args.per_episode, "actions_per_chunk": 2,
                "development_episodes": cart.manifest["split"]["val"], "reserved_episodes_used": [],
                "sampled_anchors": sorted(cp), "scored_actions": len(rows), "excluded_actions": excluded,
                "nonpadding_sampled_prefix_actions": len(rows) + len(excluded),
                "target_reference": eval_cfg["target_reference"], "target_rebased_at_each_sampled_anchor": True,
                "second_prefix_action_uses_recorded_successor_observation": True,
                "physics_steps": 0, "common_joint_progress_cap_rad": cap.tolist(),
                "orientation_costs_m_per_rad": weights,
                "joint_targets_outside_native_limits_rejected_without_clipping": True,
                "hard_gates_unchanged": {key: env.config["ik"][key] for key in (
                    "position_tolerance_m", "orientation_tolerance_rad", "penetration_tolerance_m",
                    "workspace_min", "workspace_max")},
                "native_joint_limits_rad": ik.limits.tolist(),
                "contact_solver_options_unchanged": contact_solver_settings(env.model)},
            "checkpoints": {"cartesian": cmeta, "joint": jmeta},
            "source_sha256": {str(p): file_sha256(p) for p in (
                Path(__file__), args.sim_config, args.eval_config, task, Path(env.config["model"]),
                args.cart_data / "arrays.npz", args.joint_data / "arrays.npz",
                Path("src/real2sim/policy/kinematic_actions.py"), Path("src/real2sim/sim/kinematics.py"))},
            "summary": {mode: summarize(rows, mode) for mode in names},
            "per_prefix_offset": {str(offset): {m: summarize([r for r in rows if r["prefix_offset"] == offset], m) for m in names}
                                  for offset in (0, 1)},
            "per_episode": {str(ep): {m: summarize([r for r in rows if r["episode"] == ep], m) for m in names}
                            for ep in sorted({r["episode"] for r in rows})},
            "limitations": "Provisional FK supplies all physical-space labels, not independently measured tool motion. "
                "Joint supervision is recorded leader commands; Cartesian supervision is achieved follower motion. "
                "The recorded-follower-next-joint mode is a label feasibility reference, not a learned policy. "
                "Different target meanings prevent interpreting this as an isolated architecture comparison. "
                "Native-incompatible observed starts are excluded and retained explicitly. Full joint commands "
                "are endpoint-gated before common-progress rate bounding. Bounded FK endpoints and collision "
                "flags do not certify swept paths or physical tracking. No simulation steps, loaded contact test, "
                "hardware, policy promotion or reserved evaluation. No action representation or default adopted.",
            "rows": rows}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    screen = Path("outputs/v2/learning_screen_20261022/runs")
    cache = Path(".cache/v2_exploratory_20261022")
    p.add_argument("--cart-checkpoint", type=Path, default=screen / "L1_resnet_iphone_top/attempt-001/checkpoints/004000")
    p.add_argument("--joint-checkpoint", type=Path, default=screen / "L5_canonical_joint/attempt-001/checkpoints/004000")
    p.add_argument("--cart-data", type=Path, default=cache / "cartesian_iphone_top")
    p.add_argument("--joint-data", type=Path, default=cache / "canonical_joint_iphone_top")
    p.add_argument("--sim-config", type=Path, default=Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml"))
    p.add_argument("--eval-config", type=Path, default=Path("configs/v2/eval_so101.yaml"))
    p.add_argument("--per-episode", type=int, default=12)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
