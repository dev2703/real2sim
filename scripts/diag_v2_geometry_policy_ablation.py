"""Full learned rollout in an explicitly unaccepted geometry-only scene hypothesis.

This standalone diagnostic leaves the production calibration gate unchanged. The saved
checkpoint is first validated against its original frozen task. Only listed scene fields may
then change; joint mappings, FK frames, action semantics, rates, and physics remain fixed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from real2sim.config import load_yaml, project_path
from real2sim.evaluation.metrics import TrialTracker
from real2sim.evaluation.run import build_learned_controller, check_contract_timing_and_image
from real2sim.policy.adapters import (
    CartesianActionAdapter,
    ChunkExecutor,
    ContractError,
    ObservationAdapter,
)
from real2sim.policy.paired import file_sha256
from real2sim.sim.contact_solver import contact_solver_settings
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics

SCENE_FIELDS = {
    "geometry_status", "physical_calibration_accepted", "diagnostic_scope",
    "robot_so101.base_z_offset", "table.mat.center_task_xy", "table.mat.yaw_deg",
    "ball.start_region.center_task_xy", "ball.start_region.radius",
    "ball.start_region.provenance", "bowl.center_task_xy", "bowl.center_provenance",
    "ball.support.type", "ball.support.height", "ball.support.size_xy",
    "ball.support.center_task_xy", "ball.support.yaw_deg",
    "camera_iphone.key", "camera_iphone.projection_model", "camera_iphone.width",
    "camera_iphone.height", "camera_iphone.intrinsics_px.fx", "camera_iphone.intrinsics_px.fy",
    "camera_iphone.intrinsics_px.cx", "camera_iphone.intrinsics_px.cy",
    "camera_iphone.distortion_coefficients", "camera_iphone.approx_position_task",
    "camera_iphone.xyaxes_task", "camera_iphone.look_at_task",
    "camera_iphone.extrinsics_provenance",
}
NOTE = ("UNACCEPTED image-based geometry sensitivity hypothesis; not a physical measurement "
        "or accepted camera/joint calibration.")


def leaves(value, prefix=""):
    if isinstance(value, dict):
        return {key: item for name, child in value.items()
                for key, item in leaves(child, f"{prefix}.{name}" if prefix else name).items()}
    return {prefix: value}


def validate_task_ablation(original, candidate):
    """Reject any unlisted change, including physics and policy feature meanings."""
    before, after = leaves(original), leaves(candidate)
    missing = object()
    changed = sorted(key for key in before.keys() | after.keys()
                     if before.get(key, missing) != after.get(key, missing))
    forbidden = set(changed) - SCENE_FIELDS
    if forbidden:
        raise ValueError(f"unlisted scene changes: {sorted(forbidden)}")
    if candidate.get("physical_calibration_accepted") is not False:
        raise ValueError("diagnostic scene must remain physically unaccepted")
    if candidate["frames"] != original["frames"]:
        raise ValueError("policy FK frames must remain identical")
    return changed


def make_task(original, camera, scene):
    task = copy.deepcopy(original)
    task["geometry_status"] = "unaccepted_geometry_only_learned_policy_ablation"
    task["physical_calibration_accepted"] = False
    task["diagnostic_scope"] = NOTE
    task["robot_so101"]["base_z_offset"] = .0024
    task["camera_iphone"].update(camera["candidates"]["early"]["camera"])
    task["camera_iphone"]["extrinsics_provenance"] = NOTE
    task["table"]["mat"]["center_task_xy"] = camera[
        "shared_geometry_metadata_only"]["candidate_mat_center_in_native_base_m"][:2]
    task["table"]["mat"]["yaw_deg"] = 3.861
    task["ball"]["start_region"].update(
        center_task_xy=scene["ball_xy"], radius=0., provenance=NOTE)
    task["ball"]["support"] = copy.deepcopy(scene["support"])
    task["bowl"]["center_task_xy"] = scene["bowl_xy"]
    task["bowl"]["center_provenance"] = NOTE
    validate_task_ablation(original, task)
    return task


def validate_scene_interface(reference, candidate, contract):
    """Inherit checked image/timing semantics without disguising the task hash mismatch."""
    check_contract_timing_and_image(contract, reference)
    validate_task_ablation(reference.task, candidate.task)
    if candidate.camera_keys != reference.camera_keys or candidate.data_config != reference.data_config:
        raise ContractError("scene ablation changed the policy image/data interface")
    if candidate.control_hz != reference.control_hz or candidate.mapping != reference.mapping:
        raise ContractError("scene ablation changed policy timing or joint units")
    before, after = leaves(reference.config), leaves(candidate.config)
    missing = object()
    changed = {key for key in before.keys() | after.keys()
               if before.get(key, missing) != after.get(key, missing)}
    if changed - {"task_config", "ik.workspace_min"}:
        raise ContractError(f"scene ablation changed other simulator settings: {sorted(changed)}")
    if list(candidate.config["ik"]["workspace_min"]) != [
            *reference.config["ik"]["workspace_min"][:2], -.0024]:
        raise ContractError("only the physical table floor may change in IK workspace")
    if contract["camera_keys"] != ["observation.images.iphone"]:
        raise ContractError("wrist-camera mount remains unidentified; use iphone-only checkpoint")
    if contract.get("controller_semantics") != "accepted_target":
        raise ContractError("diagnostic supports the original accepted-target delta contract only")
    try:
        check_contract_timing_and_image(contract, candidate)
    except ContractError as exc:
        if str(exc) != "checkpoint calibration differs from simulator task registration":
            raise
    else:
        raise ContractError("scene hypothesis must retain an explicit changed calibration hash")


def fingerprints(paths):
    return {str(path): file_sha256(path) for path in sorted(set(paths))}


def run(args):
    if args.out.exists() or args.artifacts.exists():
        raise FileExistsError("choose fresh diagnostic result and artifact paths")
    torch.set_num_threads(2)
    original_config = load_yaml(args.reference_config)
    original_task_path = project_path(original_config["task_config"])
    camera = json.loads(args.camera.read_text())
    scenes = json.loads(args.scenes.read_text())
    scene = next(case for case in scenes["cases"] if case["case_id"] == args.case)
    args.artifacts.mkdir(parents=True)
    task_path = args.artifacts / "task.yaml"
    task = make_task(load_yaml(original_task_path), camera, scene)
    task_path.write_text(yaml.safe_dump(task, sort_keys=False))
    sim = copy.deepcopy(original_config)
    sim["task_config"] = str(task_path.resolve())
    sim["ik"]["workspace_min"][2] = -.0024
    sim_path = args.artifacts / "sim.yaml"
    sim_path.write_text(yaml.safe_dump(sim, sort_keys=False))
    frozen = json.loads(args.inputs.read_text())["files"]
    paths = [Path(path) for path in frozen]
    paths += list(Path("src").rglob("*.py"))
    paths += [Path(__file__), args.reference_config, args.camera, args.scenes, args.eval_config,
              original_task_path, project_path(sim["data_config"]), task_path, sim_path]
    paths += [args.checkpoint / name for name in (
        "model.safetensors", "config.json", "stats.json", "real2sim_contract.json", "run_info.json")]
    before = fingerprints(paths)
    if any(before[path] != sha for path, sha in frozen.items()):
        raise ValueError("frozen training inputs changed before diagnostic")
    eval_cfg = load_yaml(args.eval_config)
    t0 = time.perf_counter()
    with closing(So101PickPlaceEnv(args.reference_config, render_images=False)) as reference:
        original_ctrl, contract, meta = build_learned_controller(
            reference, args.checkpoint, eval_cfg, "cpu")
        reference.reset(seed=0)
        reference_state = reference.canonical_proprio()
        with closing(So101PickPlaceEnv(sim_path)) as env:
            validate_scene_interface(reference, env, contract)
            obs, initial = env.reset(seed=0)
            np.testing.assert_array_equal(reference_state, env.canonical_proprio())
            # Check the whole native FK interface, not merely one selected start state.
            old_fk, new_fk = (NativeKinematics(e.model, e.config) for e in (reference, env))
            rng = np.random.default_rng(72)
            max_fk_delta = 0.
            for q in rng.uniform(env.limits[:, 0], env.limits[:, 1], size=(16, 6)):
                max_fk_delta = max(max_fk_delta, float(np.abs(old_fk.forward(q) - new_fk.forward(q)).max()))
            if max_fk_delta > 1e-12:
                raise ContractError("scene changed policy FK; a virtual-FK bridge is required")
            cart = eval_cfg["cartesian"]
            action = CartesianActionAdapter(env, new_fk, max_step_m=cart["max_step_m"],
                                            max_step_rad=cart["max_step_rad"],
                                            target_reference="accepted_target")
            ctrl = ChunkExecutor(original_ctrl.policy, original_ctrl.normalizer,
                                 ObservationAdapter(contract, original_ctrl.normalizer), action)
            ctrl.reset()
            tracker = TrialTracker(math.ceil(task["episode"]["success_hold_s"] * env.control_hz),
                                   eval_cfg["phase"]["approach_distance_m"], env.table_top,
                                   task["ball"]["radius"])
            history, abort = [], None
            cv2.imwrite(str(args.artifacts / "initial_iphone.png"),
                        cv2.cvtColor(env.render("iphone"), cv2.COLOR_RGB2BGR))
            try:
                for tick in range(env.max_episode_steps):
                    result = ctrl.step(obs, env)
                    obs = result.obs
                    metrics = result.info["privileged_metrics"]
                    done = tracker.update(metrics, saturation=result.saturation,
                                          ik_rejected=result.ik_rejected)
                    history.append({**tracker.history[-1], "action": result.detail["action"],
                                    "ik_rejected": result.ik_rejected, "controller": result.detail,
                                    "control": result.info["control"],
                                    "robot_table_penetration_m": metrics["robot_table_penetration_m"],
                                    "robot_support_penetration_m": metrics.get("robot_support_penetration_m", 0.),
                                    "lifted": metrics["lifted"],
                                    "canonical_proprio": env.canonical_proprio().tolist()})
                    if (tick + 1) % 125 == 0:
                        print(f"geometry ablation: {tick + 1}/{env.max_episode_steps} ticks; "
                              f"distance={metrics['ee_ball_distance_m']:.4f}m; "
                              f"IK rejections={tracker.ik_rejections}", flush=True)
                    if done or result.truncated:
                        break
            except (RuntimeError, ValueError) as exc:
                abort = str(exc)
            cv2.imwrite(str(args.artifacts / "final_iphone.png"),
                        cv2.cvtColor(env.render("iphone"), cv2.COLOR_RGB2BGR))
            steady = ctrl.latencies_ms[1:] or ctrl.latencies_ms
            row = tracker.row(experiment_id="UNACCEPTED_GEOMETRY_ONLY_LEARNED_ABLATION",
                              controller_source="LEARNED_ACT", robot="so101", seed=0,
                              checkpoint_sha256=meta["checkpoint_sha256"],
                              checkpoint_step=meta["checkpoint_step"],
                              mean_latency_ms=float(np.mean(steady)) if steady else None,
                              video_path="", abort=abort)
            report = {
                "kind": "UNACCEPTED_GEOMETRY_ONLY_FULL_LEARNED_ROLLOUT", "outcome": row,
                "physical_calibration_accepted": False, "checkpoint_contract_modified": False,
                "production_calibration_gate_unchanged": True,
                "limitations": ("One fixed scene hypothesis, not a calibrated benchmark or success-rate estimate. "
                                "Scene inferred from early episode0 frame200; selected collision-free recorded "
                                "reset is episode9 frame100, so this is not a reconstruction of either capture. "
                                "Physical joint zeros remain unchanged and unverified. GT used only for scoring."),
                "checkpoint": meta, "original_contract": contract,
                "declared_task_hash_mismatch": {"checkpoint": contract["calibration_sha256"],
                                                "scene": file_sha256(task_path)},
                "changed_task_fields": validate_task_ablation(reference.task, task),
                "interface_checks": {"initial_canonical_state_max_difference": 0.,
                                     "random_native_fk_max_difference": max_fk_delta,
                                     "native_fk_samples": 16, "joint_mapping_unchanged": True},
                "scene_case": args.case, "timeout_s": task["episode"]["timeout_s"],
                "simulated_seconds": tracker.ticks / env.control_hz,
                "control_hz": env.control_hz, "policy_calls": ctrl.n_policy_calls,
                "contact_solver": contact_solver_settings(env.model),
                "initial_metrics": initial["privileged_metrics"],
                "per_tick": history, "wall_s": time.perf_counter() - t0,
                "artifacts": str(args.artifacts),
            }
    after = fingerprints(paths)
    report["source_hashes"] = before
    report["runtime_file_changes"] = [path for path in before if before[path] != after[path]]
    report["frozen_training_inputs_unchanged"] = all(after[path] == sha for path, sha in frozen.items())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(row), flush=True)
    if report["runtime_file_changes"]:
        raise RuntimeError("diagnostic source changed during rollout; see result provenance")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", type=Path,
                        default=Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/v2/learning_screen_20261022/runs/L1_resnet_iphone/attempt-001/checkpoints/004000"))
    parser.add_argument("--eval-config", type=Path, default=Path("configs/v2/eval_so101.yaml"))
    parser.add_argument("--camera", type=Path, default=Path("results/v2/camera_render_candidate_20261023.json"))
    parser.add_argument("--scenes", type=Path, default=Path("results/v2/ball_support_scene_candidates_20261023.json"))
    parser.add_argument("--case", default="image_support_10mm")
    parser.add_argument("--inputs", type=Path, default=Path("outputs/v2/learning_screen_20261022/INPUTS.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/v2/geometry_policy_ablation_20261023"))
    parser.add_argument("--out", type=Path, default=Path("results/v2/geometry_policy_ablation_20261023.json"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
