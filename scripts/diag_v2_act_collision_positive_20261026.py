"""Isolate ACT/table rejection, reference accumulation and image-domain effects.

All evidence is developmental, with frozen checkpoint and original provisional
geometry. The reference-mode ablation is explicit; it never edits the saved
checkpoint contract. Static probes and teacher-forced images are not task success.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from real2sim.artifacts import sha256_file as digest
from real2sim.config import load_yaml
from real2sim.data.transforms import letterbox_rgb
from real2sim.evaluation.metrics import TrialTracker
from real2sim.evaluation.run import build_learned_controller
from real2sim.policy.act import load_checkpoint
from real2sim.policy.adapters import CartesianActionAdapter
from real2sim.serialization import json_default, write_json
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics

CHECKPOINT = Path("outputs/v2/three_episode_history4_20261026/training/checkpoints/002000")
SCENE = Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml")
HISTORY = Path(".cache/v2_history4_20261026")
TRACES = Path("outputs/v2/history4_execution_20261026")


def contact_probe(env):
    ik = NativeKinematics(env.model, env.config)
    result = {}
    for mode in ("queue2", "queue1", "ensemble_0p01"):
        path = TRACES / mode / "ticks.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        contacts, reasons, details = collections.Counter(), collections.Counter(), []
        feedback = []
        first_rejection = None
        for row in rows:
            detail = row["detail"]
            reasons[detail["ik_reason"]] += 1
            feedback.append(np.linalg.norm(detail["requested_feedback"][:3]) * 1000)
            if not row["ik_rejected"]:
                continue
            seed = np.clip(row["readback_before_native_radians"], *env.limits.T)
            solved = ik.solve(np.array(detail["bounded_target_pose"]), seed,
                              gripper=detail["bounded_gripper_rad"])
            pairs = []
            for contact in ik.data.contact:
                if contact.dist >= -ik.config["penetration_tolerance_m"]:
                    continue
                bodies = sorted(env.model.body(int(env.model.geom_bodyid[int(g)])).name
                                for g in (contact.geom1, contact.geom2))
                name = "/".join(bodies)
                contacts[name] += 1
                pairs.append({"bodies": bodies, "penetration_mm": -float(contact.dist) * 1000,
                              "position_task_m": contact.pos - env.offset})
            record = {"tick": row["tick"], "replayed_reason": solved.reason,
                      "original_reason": detail["ik_reason"], "contacts_beyond_gate": pairs}
            details.append(record)
            if first_rejection is None:
                first_rejection = record
        initial = np.array(rows[0]["detail"]["reference_pose"])[:3, 3]
        final = np.array(rows[-1]["achieved_tool_pose"])[:3, 3]
        ball = np.array(rows[-1]["scoring_only"]["ball_xyz_task"])
        result[mode] = {"trace_sha256": digest(path), "reasons": dict(reasons),
                        "candidate_contacts_beyond_gate": dict(contacts),
                        "initial_tool_task_m": initial, "final_tool_task_m": final,
                        "final_ball_scoring_only_m": ball,
                        "final_horizontal_tool_ball_error_mm": np.linalg.norm((final - ball)[:2]) * 1000,
                        "max_requested_feedback_mm": max(feedback),
                        "first_rejection": first_rejection, "rejection_probes": details}
    return result


def image_domain_probe(env, out):
    """Keep each recorded state/history fixed; replace only its camera image."""
    policy, norm, contract = load_checkpoint(CHECKPOINT, "cpu")
    manifest = json.loads((HISTORY / "manifest.json").read_text())
    if contract["derived_files_sha256"] != manifest["files_sha256"]:
        raise ValueError("history data differs from checkpoint")
    if 9 not in manifest["split"]["val"] or 9 in manifest["split"]["reserved"]:
        raise ValueError("episode9 must remain a development episode")
    arrays = dict(np.load(HISTORY / "arrays.npz", allow_pickle=False))
    key = contract["camera_keys"][0]
    images = np.load(HISTORY / manifest["image_files"][key], mmap_mode="r")
    selected = np.flatnonzero((arrays["episode_index"] == 9)
                             & (arrays["frame_index"] >= 100)
                             & (arrays["frame_index"] < 200))
    real_inputs, sim_inputs, states, labels, rows, excluded = [], [], [], [], [], []
    for tick in selected:
        native = sim_from_real(env.mapping, arrays["joint_state"][tick])
        if np.any(native < env.limits[:, 0]) or np.any(native > env.limits[:, 1]):
            excluded.append(int(tick))
            continue
        env.reset(seed=0, scenario={"joint_positions": native})
        simulated, _ = letterbox_rgb(env.render("iphone"), 192, 256, 114)
        recorded = np.asarray(images[tick])
        if not rows:
            cv2.imwrite(str(out / "image_probe_recorded.png"), cv2.cvtColor(recorded, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "image_probe_simulated.png"), cv2.cvtColor(simulated, cv2.COLOR_RGB2BGR))
        real_inputs.append(norm.image(recorded))
        sim_inputs.append(norm.image(simulated))
        states.append(norm.state(arrays["state"][tick]))
        labels.append(arrays["action"][tick])
        rows.append({"tick": int(tick), "source_frame": int(arrays["frame_index"][tick])})
    state = torch.as_tensor(np.stack(states), dtype=torch.float32)
    predictions = {}
    for name, values in (("recorded_images", real_inputs), ("simulated_images", sim_inputs)):
        batches = []
        with torch.inference_mode():
            for start in range(0, len(rows), 8):
                batch = {"observation.state": state[start:start + 8],
                         key: torch.as_tensor(np.stack(values[start:start + 8]), dtype=torch.float32)}
                batches.append(norm.unaction(policy.predict_action_chunk(batch)[:, 0]).numpy())
        predictions[name] = np.concatenate(batches)
    labels = np.stack(labels)
    summary = {}
    for name, prediction in predictions.items():
        error = np.linalg.norm(prediction[:, :3] - labels[:, :3], axis=1) * 1000
        moving = np.linalg.norm(labels[:, :3], axis=1) > .0004
        summary[name] = {"translation_error_mean_mm": float(error.mean()),
                         "translation_error_p95_mm": float(np.quantile(error, .95)),
                         "positive_moving_direction_fraction": float(np.mean(
                             np.sum(prediction[moving, :3] * labels[moving, :3], axis=1) > 0)),
                         "moving_rows": int(moving.sum())}
    for i, row in enumerate(rows):
        row.update(label=labels[i], recorded_image_action=predictions["recorded_images"][i],
                   simulated_image_action=predictions["simulated_images"][i])
    return {"episode": 9, "source_frame_range": [100, 198], "rows": len(rows),
            "native_invalid_rows_excluded_without_clipping": excluded,
            "same_recorded_causal_state_history_for_both_images": True,
            "simulated_objects": "original provisional seeded scene; real object locations/camera calibration are not established",
            "scope": "teacher-forced image-domain intervention; not a causal counterfactual task label or closed-loop result",
            "summary": summary, "predictions": rows,
            "history_manifest_sha256": digest(HISTORY / "manifest.json")}


def current_pose_rollout(out):
    config = load_yaml("configs/v2/eval_so101.yaml")
    with So101PickPlaceEnv(SCENE) as env:
        obs, _ = env.reset(seed=0)
        ctrl, _, _ = build_learned_controller(env, CHECKPOINT, config, "cpu")
        ctrl.action_adapter = CartesianActionAdapter(env, NativeKinematics(env.model, env.config),
            max_step_m=.024, max_step_rad=.24, target_reference="current_pose")
        ctrl.reset()
        tracker = TrialTracker(success_hold_ticks=13, approach_distance_m=.03,
                               table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
        reasons = collections.Counter()
        with (out / "current_pose_ticks.jsonl").open("w") as stream:
            for tick in range(env.max_episode_steps):
                result = ctrl.step(obs, env)
                obs = result.obs
                reasons[result.detail["ik_reason"]] += 1
                pm = result.info["privileged_metrics"]
                done = tracker.update(pm, saturation=result.saturation, ik_rejected=result.ik_rejected)
                stream.write(json.dumps({"tick": tick + 1, "detail": result.detail,
                    "control": result.info["control"], "achieved_tool_pose": env.ee_pose(),
                    "scoring_only": pm}, default=json_default, allow_nan=False) + "\n")
                if done or result.truncated:
                    break
        return {"execution_semantics_ablation": "current_pose replaces saved accepted_target at runtime only",
                "checkpoint_contract_modified": False, "seed": 0, "ticks": tracker.ticks,
                "success": tracker.pick_place_success, "held": tracker.held_ever,
                "lifted": tracker.lifted, "phase_failed": tracker.phase_failed(),
                "closest_tool_ball_mm": tracker.min_ee_ball_distance_m * 1000,
                "reasons": dict(reasons), "max_requested_step_m": .024}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    sources = [Path(__file__), Path("src/real2sim/artifacts.py"), Path("src/real2sim/serialization.py"),
               SCENE, CHECKPOINT / "model.safetensors",
               Path("src/real2sim/policy/adapters.py"), Path("src/real2sim/sim/kinematics.py")]
    before = {str(p): digest(p) for p in sources}
    report = {"schema": "act_collision_and_image_domain_20261026_positive_v1",
              "physical_calibration_accepted": False, "reserved_episodes_used": [],
              "learned_task_success_claimed": False, "source_sha256_before": before}
    with So101PickPlaceEnv(SCENE, render_images=False) as env:
        env.reset(seed=0)
        report["replayed_collision_probes"] = contact_probe(env)
        write_json(args.out / "report.json", report)
        report["paired_image_domain_probe"] = image_domain_probe(env, args.out)
        write_json(args.out / "report.json", report)
    report["current_pose_rollout"] = current_pose_rollout(args.out)
    report["source_sha256_after"] = {str(p): digest(p) for p in sources}
    report["runtime_source_changes"] = [p for p in before if before[p] != report["source_sha256_after"][p]]
    write_json(args.out / "report.json", report)
    print(json.dumps({"image_domain": report["paired_image_domain_probe"]["summary"],
                      "current_pose": report["current_pose_rollout"]}, default=json_default, indent=2))


if __name__ == "__main__":
    main()
