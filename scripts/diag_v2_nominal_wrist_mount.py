"""Bind existing nominal CAD wrist optics to observation.images.top, then verify/render.

Writes isolated configs; never edits an active task, policy contract or source recording.
Choose a reference sim config with the intended data clock/start. Opting in changes the
task hash, so training/inference must use the resulting task and a regenerated contract.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

from real2sim.config import load_yaml, project_path
from real2sim.data.raw import sha256_file
from real2sim.sim.cameras import nominal_so101_wrist_camera
from real2sim.sim.env import So101PickPlaceEnv


def prepare_configs(sim_config: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"choose a new diagnostic directory: {out}")
    sim = load_yaml(sim_config)
    task_path, data_path = (project_path(sim[k]) for k in ("task_config", "data_config"))
    task, data = load_yaml(task_path), load_yaml(data_path)
    if task.get("physical_calibration_accepted") is True:
        raise ValueError("nominal optics must not replace an accepted physical camera")
    if task.get("additional_cameras"):
        raise ValueError("reference task already declares additional cameras; inspect its mounts first")
    descriptor = nominal_so101_wrist_camera(project_path(sim["model"]))
    task["additional_cameras"] = [descriptor]
    task["physical_calibration_accepted"] = False
    task["geometry_status"] = "provisional_geometry_and_nominal_cad_wrist_optics_unverified"
    data["camera_keys"] = [task["camera_iphone"]["key"], descriptor["key"]]
    data["excluded_cameras"] = []
    data["contract_version"] = data.get("contract_version", "so101_v2_12p5") + "_nominal_cad_wrist_v1"
    data["physical_calibration_accepted"] = False
    data["geometry_status"] = task["geometry_status"]
    task_target, data_target = out / "task.yaml", out / "data.yaml"
    data["task_config"] = str(task_target.resolve())
    # Keep simulator-only dynamics and initial state exactly as supplied.
    sim = copy.deepcopy(sim)
    sim.update(task_config=str(task_target.resolve()), data_config=str(data_target.resolve()))
    out.mkdir(parents=True)
    for name, value in (("task.yaml", task), ("data.yaml", data), ("sim.yaml", sim)):
        (out / name).write_text(yaml.safe_dump(value, sort_keys=False))
    (out / "camera.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    return {"reference_sim": str(sim_config), "reference_sim_sha256": sha256_file(sim_config),
            "reference_task": str(task_path), "reference_task_sha256": sha256_file(task_path),
            "reference_data": str(data_path), "reference_data_sha256": sha256_file(data_path),
            "descriptor": descriptor}


def verify(sim_config: Path, out: Path, *, render: bool = True) -> dict:
    report = prepare_configs(sim_config, out)
    with So101PickPlaceEnv(sim_config, render_images=False) as reference:
        with So101PickPlaceEnv(out / "sim.yaml", render_images=render) as env:
            obs, _ = env.reset(seed=0)
            native, bound = (env.model.camera(n).id for n in ("wrist_cam", "top_nominal_cad"))
            # Kinematic samples are not executed physical rollouts or calibration observations.
            samples = np.random.default_rng(24).uniform(env.limits[:, 0], env.limits[:, 1], (5, 6))
            worst_position, worst_axes = 0., 0.
            for q in samples:
                env.data.qpos[env.qadr] = q
                mujoco.mj_kinematics(env.model, env.data)
                mujoco.mj_camlight(env.model, env.data)
                worst_position = max(worst_position, float(np.abs(env.data.cam_xpos[native] - env.data.cam_xpos[bound]).max()))
                worst_axes = max(worst_axes, float(np.abs(env.data.cam_xmat[native] - env.data.cam_xmat[bound]).max()))
            np.testing.assert_allclose(worst_position, 0, atol=1e-12)
            np.testing.assert_allclose(worst_axes, 0, atol=1e-12)
            np.testing.assert_array_equal(env.limits, reference.limits)
            for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity", "exclude_signature", "actuator_ctrlrange", "jnt_range"):
                np.testing.assert_array_equal(getattr(env.model, field), getattr(reference.model, field))
            env.reset(seed=0)
            if render:
                for name in ("iphone", "top_nominal_cad"):
                    cv2.imwrite(str(out / f"{name}.png"), cv2.cvtColor(env.render(name), cv2.COLOR_RGB2BGR))
            report.update(
                kind="NOMINAL_CAD_WRIST_BINDING_DIAGNOSTIC_NOT_CALIBRATION",
                physical_calibration_accepted=False, kinematic_pose_samples=len(samples),
                native_vs_bound_max_position_difference_m=worst_position,
                native_vs_bound_max_axes_difference=worst_axes,
                collision_geometry_and_exclusions_unchanged=True, joint_and_actuator_limits_unchanged=True,
                camera_keys=env.camera_keys,
                observation_shapes={key: list(obs[key].shape) for key in env.camera_keys} if render else {},
                training_inference_requirement="Use generated task consistently in preparation, checkpoint contract and simulator; existing checkpoint task hashes are not bypassed.",
            )
    report["generated_files_sha256"] = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}
    report["code_sha256"] = {str(p): sha256_file(p) for p in (Path(__file__), Path("src/real2sim/sim/cameras.py"))}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()
    result = verify(args.sim_config, args.out, render=not args.no_render)
    print(json.dumps({key: result[key] for key in (
        "camera_keys", "observation_shapes", "native_vs_bound_max_position_difference_m",
        "native_vs_bound_max_axes_difference", "physical_calibration_accepted")}, indent=2))
