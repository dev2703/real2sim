"""SO-only ACT over RGB semantic goals and grasp-point feedback, frozen on Panda.

No raw robot image is passed to ACT. Initial ball/bowl centres are measured by the
existing RGB frontend. Live feedback is robot FK at an explicitly defined grasp
point plus causal prior gripper intent/age. ACT predicts absolute grasp XYZ and
open/close intent; robot-specific IK never supplies task translation goals.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

from real2sim.artifacts import sha256_file as digest
from real2sim.config import load_yaml
from real2sim.data.prepare import compute_stats
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.oracle import grasp_orientation
from real2sim.evaluation.video import CaptionedVideo
from real2sim.geometry.so101_fk import JointMapping, So101FK
from real2sim.panda.adapter import PandaIK
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.panda.model import POINTING_DOWN
from real2sim.policy.follower_joint import check_joint_request
from real2sim.policy.visual_targets import camera_pose_task, locate_objects
from real2sim.serialization import json_default as serial
from real2sim.serialization import write_json as save
from real2sim.sim.cameras import camera_intrinsics_px
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics

SOURCE = Path("outputs/v2/matched_scene_act_20261026/data")
SO_SCENE = Path("configs/v2/sim_so101.waypoint_20261026.yaml")
PANDA_SCENE = Path("configs/v2/sim_panda.waypoint_20261026.yaml")
SO_OFFSET = np.array([-.010, 0., .030])
PANDA_OFFSET = np.array([-.008, 0., 0.])
SEEDS = [300, 301, 302]


def point(pose, offset):
    return pose[:3, 3] + pose[:3, :3] @ offset


def acquire(env):
    image = env.render("iphone")
    camera = env.model.camera("iphone").id
    K = camera_intrinsics_px(env.model, camera, image.shape[1], image.shape[0])
    T = camera_pose_task(env.data.cam_xpos[camera], env.data.cam_xmat[camera], env.offset)
    table = env.table_top - env.offset[2]
    found = locate_objects(image, K, T, ball_radius_m=env.task["ball"]["radius"],
        ball_center_height_m=table + env.task["ball"]["radius"],
        bowl_rim_height_m=table + env.bowl["height"],
        bowl_outer_radius_m=env.bowl["rim_radius"] + env.bowl["wall"],
        workspace_xy=[env.config["ik"]["workspace_min"][:2], env.config["ik"]["workspace_max"][:2]])
    if any(found[k] is None for k in ("ball", "bowl")):
        return None
    return np.r_[found["ball"].center_task[:2], found["bowl"].center_task[:2]].astype(np.float32)


def features(points, intents, hz):
    """Only past command intent enters the current state; episode reset is explicit."""
    velocity = np.zeros_like(points)
    velocity[1:] = np.diff(points, axis=0) * hz
    previous, ages, last, age = [], [], 1., 0.
    for intent in intents:
        previous.append(last)
        ages.append(min(age, 10.))
        age = age + 1 / hz if intent == last else 0.
        last = float(intent)
    return np.c_[points, velocity, previous, ages].astype(np.float32)


def prepare(out):
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    arrays = dict(np.load(SOURCE / "arrays.npz", allow_pickle=False))
    task = load_yaml(load_yaml(SO_SCENE)["task_config"])
    mapping = JointMapping.from_task_config(task)
    fk = So101FK(mapping=mapping)
    N = len(arrays["state"])
    x, env_features, y = np.zeros((N, 8), np.float32), np.zeros((N, 4), np.float32), np.zeros((N, 4), np.float32)
    detections = {}
    with So101PickPlaceEnv(SO_SCENE, render_images=False) as env:
        for episode in manifest["split"]["train"] + manifest["split"]["val"]:
            ids = np.flatnonzero(arrays["episode_index"] == episode)
            env.reset(seed=episode)
            objects = acquire(env)
            if objects is None:
                raise RuntimeError(f"source teacher episode {episode} cannot reacquire RGB goals")
            observed = fk.fk_real(arrays["state"][ids])
            commanded = fk.fk_real(arrays["action"][ids])
            p = observed[:, :3, 3] + np.einsum("nij,j->ni", observed[:, :3, :3], SO_OFFSET)
            target = commanded[:, :3, 3] + np.einsum("nij,j->ni", commanded[:, :3, :3], SO_OFFSET)
            raw_percent = mapping.gripper_sim_to_real_percent(arrays["requested_native_command_rad"][ids, 5])
            intent = (raw_percent >= 45.).astype(np.float32)
            x[ids], env_features[ids], y[ids] = features(p, intent, 12.5), objects, np.c_[target, intent]
            detections[str(episode)] = objects.tolist()
    train_mask = np.isin(arrays["episode_index"], manifest["split"]["train"])
    stats = {"state": compute_stats(x[train_mask]), "objects": compute_stats(env_features[train_mask]),
             "action": compute_stats(y[train_mask])}
    # Fixed 3 cm physical scaling prevents a constant training bowl coordinate
    # amplifying submillimetre cross-robot RGB localization differences.
    stats["objects"]["std"] = np.maximum(stats["objects"]["std"], .03).tolist()
    np.savez(out / "semantic_data.npz", state=x, objects=env_features, action=y,
        episode_index=arrays["episode_index"], ep_start=arrays["ep_start"], ep_len=arrays["ep_len"])
    save(out / "normalization.json", stats)
    save(out / "source.json", {"source_manifest": str(SOURCE / "manifest.json"),
        "source_manifest_sha256": digest(SOURCE / "manifest.json"), "split": manifest["split"],
        "rgb_initial_detections": detections, "source": "SO101 simulation teacher only",
        "inputs": "FK grasp point/velocity, prior command intent/age, initial RGB ball/bowl XY",
        "labels": "accepted command FK grasp XYZ; current raw requested binary gripper intent",
        "phase_or_privileged_object_inputs": False, "physical_calibration_accepted": False})
    return x, env_features, y, arrays, stats, manifest["split"]


def normalized(value, stats):
    return (value - np.asarray(stats["mean"])) / np.asarray(stats["std"])


class Chunks(Dataset):
    def __init__(self, payload, episodes):
        self.x, self.objects, self.y, self.a, self.stats, _ = payload
        self.ids = np.flatnonzero(np.isin(self.a["episode_index"], episodes))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        tick = self.ids[index]
        end = self.a["ep_start"][tick] + self.a["ep_len"][tick]
        offsets = np.arange(tick, tick + 8)
        ids = np.minimum(offsets, end - 1)
        return {"observation.state": torch.tensor(normalized(self.x[tick], self.stats["state"]), dtype=torch.float32),
            "observation.environment_state": torch.tensor(normalized(self.objects[tick], self.stats["objects"]), dtype=torch.float32),
            "action": torch.tensor(normalized(self.y[ids], self.stats["action"]), dtype=torch.float32),
            "pad": torch.tensor(offsets >= end)}


def train_model(out, payload):
    torch.manual_seed(0)
    config = ACTConfig(input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(4,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        device="cpu", push_to_hub=False, chunk_size=8, n_action_steps=1,
        dim_model=128, n_heads=4, dim_feedforward=512, n_encoder_layers=2, n_decoder_layers=1,
        pretrained_backbone_weights=None, use_vae=False, dropout=0., kl_weight=0.)
    policy = ACTPolicy(config).train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.)
    train_loader = DataLoader(Chunks(payload, payload[-1]["train"]), batch_size=16, shuffle=True)
    val_loader = DataLoader(Chunks(payload, payload[-1]["val"]), batch_size=64)
    iterator = iter(train_loader)
    history = []
    for step in range(1, 2001):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        lr = 3e-4 * min(1., step / 50) * (.01 + .99 * (1 + math.cos(math.pi * step / 2000)) / 2)
        optimizer.param_groups[0]["lr"] = lr
        prediction = policy.model(batch)[0]
        valid = (~batch["pad"]).unsqueeze(-1)
        loss = ((prediction - batch["action"]).abs() * valid).sum() / (valid.sum() * 4)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        optimizer.step()
        if step % 250 == 0:
            error, count = np.zeros(4), 0
            policy.eval()
            with torch.inference_mode():
                for b in val_loader:
                    raw = policy.predict_action_chunk(b)[:, 0] - b["action"][:, 0]
                    error += raw.abs().sum(0).numpy() * np.asarray(payload[4]["action"]["std"])
                    count += len(raw)
            history.append({"step": step, "loss": float(loss), "validation_mae_xyz_m_intent": (error / count).tolist()})
            save(out / "training_history.json", history)
            print(json.dumps(history[-1]), flush=True)
            policy.train()
    policy.eval()
    policy.save_pretrained(out / "checkpoint")
    save(out / "training_plan.json", {"updates": 2000, "maximum_training_anchors": 32000,
        "seed": 0, "batch_size": 16, "state_only_ACT": True, "no_image_encoder": True,
        "source": "SO101 only", "panda_training_examples": 0, "history": history})
    return policy


class PointAdapter:
    def __init__(self, env, robot, objects, *, joint_margin=0., orientation_reference=None):
        self.env, self.robot, self.objects = env, robot, objects
        self.offset = SO_OFFSET if robot == "so101" else PANDA_OFFSET
        self.joint_margin = float(joint_margin)
        self.orientation_reference = orientation_reference
        config = load_yaml(SO_SCENE) if robot == "so101" else env.config
        self.ik = NativeKinematics(env.model, config) if robot == "so101" else PandaIK(env.model, config, task_origin=env.offset)
        self.calls = 0

    def apply(self, desired_point, opened):
        env = self.env
        current = point(env.ee_pose(), self.offset)
        delta = desired_point - current
        requested = current + delta / max(1., np.linalg.norm(delta) / .024)
        self.calls += 1
        if self.robot == "panda":
            T = np.eye(4)
            T[:3, :3] = POINTING_DOWN
            T[:3, 3] = requested - POINTING_DOWN @ self.offset
            width = env.max_opening if opened else 0.
            solved = self.ik.solve(T, env.arm_qpos(), finger_opening=max(width, env.gripper_opening()))
            command = np.r_[solved.qpos if solved.accepted else env.arm_qpos(), width]
            accepted, reason = solved.accepted, solved.reason
        else:
            q = env.sim_state()
            grip = float(np.clip(env.mapping.gripper_real_to_sim_rad(90. if opened else 0.), *env.limits[5]))
            reference = (grasp_orientation(self.objects[:2], 15.) if self.orientation_reference is None
                         else self.orientation_reference)

            def residual(arm):
                T = self.ik.forward(np.r_[arm, grip])
                rotation = Rotation.from_matrix(reference @ T[:3, :3].T).as_rotvec()
                return np.r_[point(T, self.offset) - requested, .002 * rotation, .0001 * (arm - q[:5])]

            lower = env.limits[:5, 0] + self.joint_margin
            upper = env.limits[:5, 1] - self.joint_margin
            fitted = least_squares(residual, np.clip(q[:5], lower, upper),
                bounds=(lower, upper), max_nfev=35, ftol=1e-7, xtol=1e-7, gtol=1e-7)
            target = np.r_[fitted.x, grip]
            error = np.linalg.norm(point(self.ik.forward(target), self.offset) - requested)
            self.ik.data.qpos[:] = env.data.qpos
            accepted, reason, _ = check_joint_request(self.ik, q, env.data.ctrl[env.actuator_ids], target,
                env.max_velocity / env.control_hz, samples=17)
            if error > .003:
                accepted, reason = False, "point_residual"
            command = target if accepted else q
        obs, _, _, truncated, info = env.step(command)
        return obs, truncated, info, accepted, reason, requested


def select_so_orientation(env, objects):
    """Query feasible jaw orientation once; no geometric action is executed."""
    from real2sim.policy.visual_waypoint import VisualWaypointController

    query = VisualWaypointController(env, robot="so101")
    table = env.table_top - env.offset[2]
    query.ball = np.r_[objects[:2], table + env.task["ball"]["radius"]]
    query.bowl = np.r_[objects[2:], table + env.bowl["height"]]
    accepted = query._plan_grasp()
    record = {"accepted": accepted, "ik_setup_queries": query.ik_calls,
        "geometric_actions_executed": 0, "purpose": "jaw orientation prior only; all XYZ and intent remain ACT outputs",
        "failure": query.failure}
    if not accepted:
        return None, record
    record["rotation"] = query.grasp_rotation.tolist()
    return query.grasp_rotation.copy(), record


def evaluate(policy, stats, out, robot, start_point, *, joint_margin=0., parked_perception=False,
             position_offset=0, selected_so_orientation=False):
    directory = out / robot
    directory.mkdir()
    config = load_yaml(SO_SCENE if robot == "so101" else PANDA_SCENE)
    if robot == "panda":
        config["control_hz"] = 12.5
        config["data_config"] = "configs/v2/data.waypoint_20261026.yaml"
    cfg_path = directory / "sim.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
    env_class = So101PickPlaceEnv if robot == "so101" else PandaPickPlaceEnv
    rows = []
    with env_class(cfg_path, render_images=False) as env:
        mapped_start = None
        if robot == "panda":
            env.model.body_pos[env.model.body("link0").id] += [-.15, 0., 0.]
            ik = PandaIK(env.model, env.config, task_origin=env.offset)
            target = np.eye(4)
            target[:3, :3] = POINTING_DOWN
            target[:3, 3] = start_point - POINTING_DOWN @ PANDA_OFFSET
            solved = ik.solve(target, env.initial_qpos, finger_opening=env.max_opening)
            if not solved.accepted:
                raise RuntimeError(f"Panda mapped initial point rejected: {solved.reason}")
            mapped_start = solved.qpos
        for seed in SEEDS:
            setup = None
            if robot == "panda" and parked_perception:
                _, parked_info = env.reset(seed=seed)
                objects = acquire(env)
                _, mapped_info = env.reset(seed=seed, scenario={"joint_positions": mapped_start})
                setup = {"protocol": "RGB at configured parked arm, then reset same seeded layout to declared mapped start",
                    "object_layout_unchanged_scoring_only": {
                        key: np.allclose(parked_info["privileged_metrics"][key], mapped_info["privileged_metrics"][key])
                        for key in ("ball_xyz_task", "bowl_xyz_task")}}
                if not all(setup["object_layout_unchanged_scoring_only"].values()):
                    raise RuntimeError("parked perception reset changed the seeded layout")
            else:
                env.reset(seed=seed, scenario={"joint_positions": mapped_start} if mapped_start is not None else None)
                objects = acquire(env)
            tracker = TrialTracker(success_hold_ticks=math.ceil(12.5), approach_distance_m=.03,
                table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
            orientation, orientation_setup = None, None
            if robot == "so101" and objects is not None and selected_so_orientation:
                orientation, orientation_setup = select_so_orientation(env, objects)
            trial = directory / f"seed_{seed}"
            trial.mkdir()
            writer = CaptionedVideo(trial / "rollout.mp4", 12.5, (360, 640))
            last_point, last_intent, age = None, 1., 0.
            reasons, abort = {}, None
            adapter = PointAdapter(env, robot, objects, joint_margin=joint_margin,
                orientation_reference=orientation) if objects is not None else None
            with (trial / "ticks.jsonl").open("w") as stream:
                try:
                    for tick in range(env.max_episode_steps):
                        if objects is None:
                            abort = "Initial RGB ball/bowl detection unavailable"
                            break
                        if orientation_setup is not None and not orientation_setup["accepted"]:
                            abort = "Robot jaw-orientation setup found no feasible branch"
                            break
                        p = point(env.ee_pose(), adapter.offset)
                        velocity = np.zeros(3) if last_point is None else (p - last_point) * 12.5
                        state = np.r_[p, velocity, last_intent, min(age, 10.)]
                        batch = {"observation.state": torch.tensor(normalized(state, stats["state"]), dtype=torch.float32)[None],
                            "observation.environment_state": torch.tensor(normalized(objects, stats["objects"]), dtype=torch.float32)[None]}
                        with torch.inference_mode():
                            chunk = policy.predict_action_chunk(batch)[0].numpy()
                        raw = chunk[position_offset].copy()
                        raw[3] = chunk[0, 3]
                        action = raw * np.asarray(stats["action"]["std"]) + np.asarray(stats["action"]["mean"])
                        intent = float(action[3] >= .5)
                        obs, truncated, info, accepted, reason, requested = adapter.apply(action[:3], bool(intent))
                        reasons[reason] = reasons.get(reason, 0) + 1
                        pm = info["privileged_metrics"]
                        done = tracker.update(pm, ik_rejected=not accepted,
                            saturation=float(np.mean(info["control"]["limit_clipped"])))
                        stream.write(json.dumps({"tick": tick, "state": state, "rgb_objects": objects,
                            "action": action, "bounded_point": requested, "intent": intent,
                            "position_forecast_offset": position_offset, "intent_forecast_offset": 0,
                            "accepted": accepted, "reason": reason, "control": info["control"],
                            "scoring_only": pm}, default=serial, allow_nan=False) + "\n")
                        rgb = env.render("iphone")
                        writer.write(rgb[:, :, ::-1].copy(), f"FROZEN SO-TRAINED SEMANTIC ACT ({robot})")
                        last_point = p.copy()
                        age = age + 1 / 12.5 if intent == last_intent else 0.
                        last_intent = intent
                        if done or truncated:
                            break
                except (ValueError, RuntimeError) as exc:
                    abort = f"{type(exc).__name__}: {exc}"
                finally:
                    writer.close()
            row = tracker.row(experiment_id="SEMANTIC_ACT_SO_ONLY_20261027", controller_source="FROZEN_SEMANTIC_ACT",
                robot=robot, seed=seed, checkpoint_sha256=digest(out / "checkpoint/model.safetensors"),
                checkpoint_step=2000, mean_latency_ms=None, video_path=str(trial / "rollout.mp4"), abort=abort)
            rows.append({**row, "adapter_reasons": reasons, "rgb_objects": objects,
                         "perception_setup": setup, "joint_command_interior_margin_rad": joint_margin,
                         "position_forecast_offset": position_offset, "intent_forecast_offset": 0,
                         "orientation_setup": orientation_setup})
            save(directory / "report.json", {"summary": summarize(rows), "trials": rows,
                "panda_base_dx_m": -.15 if robot == "panda" else None,
                "mapped_panda_initial_grasp_point": start_point if robot == "panda" else None})
            print(json.dumps({"robot": robot, "seed": seed, "success": row["pick_place_success"],
                "lifted": row["lifted"], "phase": row["phase_failed"], "reasons": reasons, "abort": abort}), flush=True)
    return {"summary": summarize(rows), "trials": rows}


def main(out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    report = {"schema": "semantic_act_transfer_20261027_v1", "status": "preparing",
        "training_robot": "so101", "panda_training_examples": 0,
        "physical_calibration_accepted": False, "real_hf_weights_used": False,
        "privileged_object_or_phase_inputs": False,
        "source_sha256_before": {str(p): digest(p) for p in
            (Path(__file__), Path("src/real2sim/artifacts.py"), Path("src/real2sim/serialization.py"),
             SOURCE / "arrays.npz", SOURCE / "manifest.json", SO_SCENE, PANDA_SCENE)}}
    save(out / "report.json", report)
    payload = prepare(out)
    report["status"] = "training"
    save(out / "report.json", report)
    policy = train_model(out, payload)
    weights = digest(out / "checkpoint/model.safetensors")
    report.update(status="evaluating", checkpoint_sha256=weights)
    save(out / "report.json", report)
    report["so101"] = evaluate(policy, payload[4], out, "so101", payload[0][0, :3])
    report["panda"] = evaluate(policy, payload[4], out, "panda", payload[0][0, :3])
    report.update(status="complete", checkpoint_unchanged=weights == digest(out / "checkpoint/model.safetensors"),
        source_sha256_after={p: digest(p) for p in report["source_sha256_before"]})
    save(out / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/v2/semantic_act_transfer_20261027"))
    main(parser.parse_args().out)
