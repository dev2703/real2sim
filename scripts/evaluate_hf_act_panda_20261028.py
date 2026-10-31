"""Run the actual native HF ACT on live Panda images and virtual SO-101 state.

Robot-only FK/IK maps SO-101 joint commands to Panda grasp points. A wrist camera
sits outside Panda's hand and looks past its fingertips using fixed robot-local
geometry. Both actual Panda views enter the saved native HF processor once.
This is a provisional embodiment adapter, not physical calibration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
import yaml

from real2sim.artifacts import sha256_file
from real2sim.config import load_yaml
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo
from real2sim.geometry.so101_fk import JointMapping
from real2sim.panda.adapter import PandaIK
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.panda.workspace import register_measured_task_workspace
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.serialization import json_default, write_json
from real2sim.sim.env import real_from_sim

if __package__:
    from scripts import diag_v2_learned_panda_retarget_20261027 as retarget
    from scripts import evaluate_hf_act_20261027 as native
else:
    import diag_v2_learned_panda_retarget_20261027 as retarget
    import evaluate_hf_act_20261027 as native


def save(path, payload):
    write_json(path, payload, atomic=True)


def save_rgb(path, image):
    if not cv2.imwrite(str(path), image[:, :, ::-1]):
        raise OSError(f"Could not write image {path}")


def prepare_source(out, joint_offsets_deg, *, source_scene=None, initial_gripper_percent=None,
                   native_contract=None):
    directory = out / "source"
    directory.mkdir()
    scene = native.prepare_scene(directory,
        source=source_scene or Path("configs/v2/sim_so101.waypoint_20261026.yaml"),
        native_contract=native_contract)
    cfg, task = load_yaml(scene), load_yaml(directory / "task.yaml")
    task["robot_so101"]["real_to_sim"]["offset_deg"] = [float(value) for value in joint_offsets_deg]
    # Retain the native physical start under an explicitly changed registration;
    # the corresponding raw degree/percent state is recomputed, never clipped.
    mapping = JointMapping.from_task_config(task)
    if initial_gripper_percent is not None:
        cfg["initial_qpos"][5] = float(mapping.gripper_real_to_sim_rad(initial_gripper_percent))
    cfg["initial_qpos_real_deg"] = real_from_sim(mapping, np.asarray(cfg["initial_qpos"])).tolist()
    (directory / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
    scene.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return scene, task


def panda_start(env, source, base_dx):
    env.model.body_pos[env.model.body("link0").id] += [base_dx, 0., 0.]
    mujoco.mj_forward(env.model, env.data)
    ik = PandaIK(env.model, env.config, task_origin=env.offset)
    target = retarget.target_from_source(source, source.source.initial_qpos)
    source_percent = float(source.mapping.gripper_sim_to_real_percent(source.source.initial_qpos[5]))
    width = retarget.panda_opening_from_percent(source_percent, env.max_opening)
    solved = ik.solve(target, env.initial_qpos, finger_opening=width)
    if not solved.accepted:
        raise RuntimeError(f"Mapped Panda start rejected: {solved.reason}")
    return solved.qpos.copy(), width


def camera_mount_at_start(env, source, mode="external_hand"):
    """Express the source initial wrist optics in Panda's live hand coordinates."""
    source.source.reset(seed=0)
    wrist_key = source.source.task["additional_cameras"][0]["key"]
    camera_name = source.source.camera_bindings[wrist_key]
    camera = source.source.model.camera(camera_name).id
    source_camera_R = source.source.data.cam_xmat[camera].reshape(3, 3)
    source_camera_p_task = source.source.data.cam_xpos[camera] - source.source.offset
    hand = env.model.body("hand").id
    hand_R = env.data.xmat[hand].reshape(3, 3)
    hand_p_world = env.data.xpos[hand]
    camera_p_world = source_camera_p_task + env.offset
    relative_R = hand_R.T @ source_camera_R
    relative_p = hand_R.T @ (camera_p_world - hand_p_world)
    if mode == "external_hand":
        # The initial optical-pose match can sit inside Panda's larger hand.
        # Mount outside its side and look past the fingertips, using hand-local
        # robot geometry only, independent of the ball/bowl or episode outcome.
        relative_p = np.array([.08, 0., 0.])
        forward = np.array([0., 0., .20]) - relative_p
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [1., 0., 0.])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative_R = np.column_stack([right, up, -forward])
    elif mode != "source_optical_match":
        raise ValueError("Unknown Panda wrist camera mount")
    descriptor = copy.deepcopy(source.source.task["additional_cameras"][0])
    descriptor["name"] = "top_hf_panda_nominal"
    descriptor["mounts"]["panda"] = {"body": "hand", "position_body_m": relative_p.tolist(),
        "xyaxes_body": relative_R[:, :2].T.ravel().tolist()}
    descriptor["provenance"]["panda_mount"] = mode
    descriptor["provenance"]["panda_mount_scope"] = "Robot-only fixed hand-relative camera; never aimed at privileged object coordinates"
    return descriptor, {"source_camera_position_task_m": source_camera_p_task,
        "source_camera_rotation_task": source_camera_R, "panda_hand_relative_position_m": relative_p,
        "panda_hand_relative_rotation": relative_R, "mode": mode}


def prepare_panda(out, source, task, base_dx, *, wrist_mount="external_hand", panda_scene=None):
    if panda_scene is None:
        panda_scene = ("configs/v2/measured_registration_20261028/sim_panda_positive_all_scalars.yaml"
            if task.get("operator_measurements") else "configs/v2/sim_panda.waypoint_20261026.yaml")
    cfg = register_measured_task_workspace(load_yaml(panda_scene), task)
    cfg["control_hz"] = 25.
    cfg["render"] = copy.deepcopy(source.source.config["render"])
    data = load_yaml("configs/v2/data.waypoint_20261026.yaml")
    data.update(policy_hz=25., camera_keys=["observation.images.iphone"], excluded_cameras=[])
    panda_task = copy.deepcopy(task)
    panda_task["additional_cameras"] = []
    paths = {name: out / name for name in ("panda_task.yaml", "panda_data.yaml", "sim_panda.yaml")}
    cfg["task_config"], cfg["data_config"] = str(paths["panda_task.yaml"].resolve()), str(paths["panda_data.yaml"].resolve())
    for name, payload in (("panda_task.yaml", panda_task), ("panda_data.yaml", data), ("sim_panda.yaml", cfg)):
        paths[name].write_text(yaml.safe_dump(payload, sort_keys=False))
    with PandaPickPlaceEnv(paths["sim_panda.yaml"], render_images=False) as env:
        start, width = panda_start(env, source, base_dx)
        env.reset(seed=0, scenario={"joint_positions": start, "gripper_opening": width})
        camera, provenance = camera_mount_at_start(env, source, wrist_mount)
    panda_task["additional_cameras"] = [camera]
    data["camera_keys"] = list(source.source.camera_keys)
    paths["panda_task.yaml"].write_text(yaml.safe_dump(panda_task, sort_keys=False))
    paths["panda_data.yaml"].write_text(yaml.safe_dump(data, sort_keys=False))
    return paths["sim_panda.yaml"], provenance


def run(snapshot, out, seeds, *, joint_offsets_deg=(0.,) * 5, execute_steps=None,
        base_dx=-.15, seat_mm=8., video=False, source_scene=None,
        initial_gripper_percent=None, wrist_mount="external_hand", gripper_readback="endpoint",
        head_dir=None, projection_prior="command", panda_scene=None):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    if head_dir is None:
        checkpoint = NativeLeRobotACT(snapshot)
    else:
        from real2sim.policy.local_hf_head import LocalHeadACT

        checkpoint = LocalHeadACT(snapshot, head_dir)
    checkpoint.policy.eval().requires_grad_(False)
    execute_steps = checkpoint.config.n_action_steps if execute_steps is None else execute_steps
    if not 1 <= execute_steps <= checkpoint.config.chunk_size:
        raise ValueError("Execution prefix must fit native saved chunk")
    source_scene, task = prepare_source(out, joint_offsets_deg, source_scene=source_scene,
        initial_gripper_percent=initial_gripper_percent, native_contract=checkpoint.scene_contract)
    source = retarget.VirtualSource(source_scene, panda_offset=np.array([-seat_mm / 1000., 0., 0.]),
        projection_prior=projection_prior)
    try:
        panda_scene, camera_setup = prepare_panda(out, source, task, base_dx,
            wrist_mount=wrist_mount, panda_scene=panda_scene)
        files = [Path(__file__), Path(retarget.__file__), Path(native.__file__),
            Path("src/real2sim/policy/native_lerobot_act.py"),
            Path("src/real2sim/panda/adapter.py"), Path("src/real2sim/panda/env.py"),
            Path("src/real2sim/panda/workspace.py"),
            Path("src/real2sim/sim/cameras.py"), Path("src/real2sim/geometry/so101_fk.py"),
            Path("src/real2sim/serialization.py"), source_scene, panda_scene,
            out / "panda_task.yaml", out / "panda_data.yaml", out / "source/task.yaml", out / "source/data.yaml"]
        files += list(Path(snapshot).glob("*.json")) + list(Path(snapshot).glob("*.safetensors"))
        if head_dir is not None:
            files += [Path(head_dir) / "head.json", Path(head_dir) / "head.pt",
                Path("src/real2sim/policy/local_hf_head.py")]
        hashes = {str(path.resolve()): sha256_file(path) for path in files}
        report = {"schema": "native_hf_act_panda_20261028_v1", "status": "running",
            **checkpoint.checkpoint_identity,
            "weights_sha256": checkpoint.weights_sha256, "execution_steps": execute_steps,
            "policy_hz": 25., "horizon_s": 30., "camera_order": checkpoint.camera_keys,
            "native_scene_contract": checkpoint.scene_contract,
            "policy_call_latency_scope": "source state projection, camera rendering, native preprocessing, ACT and postprocessing",
            "actual_panda_images": True, "saved_native_processors_used_once": True,
            "semantic_act_used": False, "source_actions": "Actual frozen pretrained HF ACT, absolute SO-101 joint degrees plus gripper percent",
            "source_state_projection": "Live Panda grasp-point position to virtual SO-101 position-only IK; explicitly declared measured-jaw readback projection",
            "virtual_source_projection_prior": projection_prior,
            "virtual_source_unobservable_coordinates": "Source orientation is not executed on Panda. Command prior preserves the most recent causal SO-101 command in the position inverse null space; these coordinates are inferred history, not measured Panda joints.",
            "gripper_readback_projection": gripper_readback,
            "initial_gripper_percent": initial_gripper_percent,
            "panda_action_projection": "Source command FK grasp point to Panda fixed-top-down IK; source0closed/90open to Panda0/80mm",
            "joint_offsets_deg": joint_offsets_deg, "panda_base_dx_task_m": base_dx,
            "grasp_point_offsets_m": {"so101": retarget.SOURCE_OFFSET, "panda": source.panda_offset},
            "wrist_camera": camera_setup, "lighting": "Unchanged nominal model lighting; no variation or randomization",
            "no_scripted_object_goals_or_task_corrections": True,
            "physical_calibration_accepted": False, "weights_frozen": True,
            "local_head": getattr(checkpoint, "head_metadata", None),
            "local_head_sha256": getattr(checkpoint, "local_head_sha256", None),
            "source_sha256_before": hashes, "seeds": seeds, "trials": []}
        (out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
        save(out / "report.json", report)
        with PandaPickPlaceEnv(panda_scene, render_images=False) as env:
            if env.camera_keys != checkpoint.camera_keys:
                raise ValueError("Actual Panda cameras do not match checkpoint order")
            start, initial_width = panda_start(env, source, base_dx)
            ik = PandaIK(env.model, env.config, task_origin=env.offset)
            report["mapped_initial_joints"] = start
            for seed in seeds:
                directory = out / f"seed_{seed:03d}"
                directory.mkdir()
                source.q = source.source.initial_qpos.copy()
                source.last_target = source.q.copy()
                _, initial_info = env.reset(seed=seed, scenario={"joint_positions": start, "gripper_opening": initial_width})
                checkpoint.reset()
                tracker = TrialTracker(success_hold_ticks=math.ceil(
                    env.task["episode"]["success_hold_s"] * env.control_hz), approach_distance_m=.03,
                    table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
                queue, latencies, reasons, abort, errors = [], [], {}, None, []
                video_shape = tuple(checkpoint.scene_contract["image_shape_chw"][1:])
                writer = CaptionedVideo(directory / "rollout.mp4", env.control_hz, video_shape) if video else None
                wall_start = time.perf_counter()
                with (directory / "ticks.jsonl").open("w") as handle:
                    try:
                        for tick in range(env.max_episode_steps):
                            infer_ms, state, image_hashes, error = 0., None, None, None
                            if not queue:
                                t0 = time.perf_counter()
                                state, error = source.source_state(env)
                                if gripper_readback == "endpoint":
                                    state[5] = float(np.clip(env.gripper_opening() / env.max_opening, 0., 1.) * 90.)
                                    source.q[5] = source.mapping.gripper_real_to_sim_rad(float(state[5]))
                                elif gripper_readback != "cad":
                                    raise ValueError("Unknown gripper readback projection")
                                errors.append(error)
                                images = {key: env.render(env.camera_bindings[key]) for key in checkpoint.camera_keys}
                                actions = checkpoint.predict_action_chunk(state, images)
                                infer_ms = (time.perf_counter() - t0) * 1000
                                latencies.append(infer_ms)
                                if tick == 0:
                                    for key, image in images.items():
                                        save_rgb(directory / (key.rsplit(".", 1)[-1] + "_initial.png"), image)
                                image_hashes = {key: hashlib.sha256(image.tobytes()).hexdigest() for key, image in images.items()}
                                queue = list(actions[:execute_steps])
                            action = queue.pop(0)
                            raw_source, bounded_source = source.source_action(action)
                            desired = retarget.target_from_source(source, bounded_source)
                            observed = env.ee_pose()
                            delta = desired[:3, 3] - observed[:3, 3]
                            if np.linalg.norm(delta) > .024:
                                desired[:3, 3] = observed[:3, 3] + delta * .024 / np.linalg.norm(delta)
                            percent = float(source.mapping.gripper_sim_to_real_percent(bounded_source[5]))
                            width = retarget.panda_opening_from_percent(percent, env.max_opening)
                            solved = ik.solve(desired, env.arm_qpos(), finger_opening=max(width, env.gripper_opening()))
                            command = np.r_[solved.qpos if solved.accepted else env.arm_qpos(), width]
                            _, _, _, truncated, info = env.step(command)
                            reasons[solved.reason] = reasons.get(solved.reason, 0) + 1
                            done = tracker.update(info["privileged_metrics"], ik_rejected=not solved.accepted,
                                saturation=float(np.mean(info["control"]["limit_clipped"])))
                            record = {"tick": tick, "inference_ms": infer_ms, "native_input_degree_percent": state,
                                "actual_panda_image_sha256": image_hashes, "raw_hf_action_degree_percent": action,
                                "virtual_source_projection_error_m": error,
                                "source_requested_native_rad": raw_source, "source_bounded_native_rad": bounded_source,
                                "panda_requested_pose_task": desired, "panda_ik_accepted": solved.accepted,
                                "panda_ik_reason": solved.reason, "panda_ik_position_error_m": solved.position_error_m,
                                "panda_control": info["control"], "panda_readback": env.sim_state(),
                                "scoring_only": info["privileged_metrics"]}
                            handle.write(json.dumps(record, default=json_default, allow_nan=False) + "\n")
                            if writer:
                                writer.write(env.render("iphone")[:, :, ::-1].copy(), "ACTUAL HF ACT; live Panda wrist+iPhone; virtual SO101")
                            if done or truncated:
                                break
                    except (ValueError, RuntimeError) as exc:
                        abort = f"{type(exc).__name__}: {exc}"
                    finally:
                        if writer:
                            writer.close()
                row = tracker.row(experiment_id="ACTUAL_HF_ACT_PANDA_20261028", controller_source="FROZEN_NATIVE_HF_ACT_LIVE_PANDA",
                    robot="panda", seed=seed, checkpoint_sha256=checkpoint.weights_sha256, checkpoint_step=None,
                    mean_latency_ms=float(np.mean(latencies)) if latencies else None,
                    video_path=str(directory / "rollout.mp4") if video else "", abort=abort)
                row.update(panda_ik_reasons=reasons, policy_calls=len(latencies),
                    local_head_sha256=getattr(checkpoint, "local_head_sha256", None),
                    max_virtual_projection_error_m=max(errors) if errors else None,
                    held_ever=tracker.held_ever, wall_s=time.perf_counter() - wall_start,
                    initial_scoring_only=initial_info["privileged_metrics"], trace=str(directory / "ticks.jsonl"))
                report["trials"].append(row)
                report["summary"] = summarize(report["trials"])
                save(out / "report.json", report)
                print(json.dumps({"seed": seed, "placed": row["pick_place_success"], "lifted": row["lifted"],
                    "phase": row["phase_failed"], "closest_m": row["min_ee_ball_distance_m"],
                    "ik": reasons, "abort": abort, "wall_s": row["wall_s"]}), flush=True)
        after = {path: sha256_file(path) for path in hashes}
        report.update(status="complete", source_sha256_after=after,
            runtime_source_changes=[path for path in hashes if hashes[path] != after[path]],
            checkpoint_unchanged=sha256_file(Path(snapshot) / "model.safetensors") == checkpoint.weights_sha256)
        save(out / "report.json", report)
        return report
    finally:
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=native.SNAPSHOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[300, 301, 302, 303, 304])
    parser.add_argument("--joint-offsets-deg", type=float, nargs=5, default=[0.] * 5)
    parser.add_argument("--execute-steps", type=int)
    parser.add_argument("--base-dx", type=float, default=-.15)
    parser.add_argument("--seat-mm", type=float, default=8.)
    parser.add_argument("--source-scene", type=Path)
    parser.add_argument("--panda-scene", type=Path)
    parser.add_argument("--initial-gripper-percent", type=float)
    parser.add_argument("--wrist-mount", choices=("external_hand", "source_optical_match"), default="external_hand")
    parser.add_argument("--gripper-readback", choices=("endpoint", "cad"), default="endpoint")
    parser.add_argument("--projection-prior", choices=("command", "previous"), default="command",
                        help="Causal source command history or legacy previous-observation null-space prior")
    parser.add_argument("--head-dir", type=Path)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    run(args.snapshot, args.out, args.seeds, joint_offsets_deg=args.joint_offsets_deg,
        execute_steps=args.execute_steps, base_dx=args.base_dx, seat_mm=args.seat_mm, video=args.video,
        source_scene=args.source_scene, initial_gripper_percent=args.initial_gripper_percent,
        wrist_mount=args.wrist_mount, gripper_readback=args.gripper_readback, head_dir=args.head_dir,
        projection_prior=args.projection_prior, panda_scene=args.panda_scene)
