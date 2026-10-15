"""Isolated scene from the training-landmark M2 fit; no primary calibration changes."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from real2sim.geometry.so101_fk import JointMapping
from real2sim.policy.paired import file_sha256
from real2sim.sim.camera_fit import backproject_to_plane
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real


def main():
    out = Path("results/metrics/joint_offset_candidate_20261014")
    out.mkdir(parents=True, exist_ok=False)
    source = Path("results/audit/camera_landmark_refit_float64_20261014/report.json")
    fit = json.loads(source.read_text())["models"]["M2_refit_camera_and_joint_offsets"]
    old = Path("results/metrics/geometry_3988577/configs")
    task = copy.deepcopy(yaml.safe_load((old / "task.yaml").read_text()))
    cam = fit["camera"]
    task["camera_iphone"].update({k: cam[k] for k in ["fovy_deg", "xyaxes_task", "look_at_task"]})
    task["camera_iphone"].update(approx_position_task=cam["position_task"],
        intrinsics_px={"fx": cam["focal_px"], "fy": cam["focal_px"], "cx": 640., "cy": 360.},
        extrinsics_provenance="DIAGNOSTIC training-landmark M2 fit, not adopted calibration")
    offsets = [fit["joint_zero_offsets_deg"][j] for j in
               ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]] + [0.]
    task["robot_so101"]["real_to_sim"].update(offset_deg=offsets,
        provenance="DIAGNOSTIC training-landmark M2 fit; no hardware writes")
    rv = np.array(cam["rvec"])
    R = cv2.Rodrigues(rv)[0]
    tv = -R @ cam["position_task"]
    K = np.array([[cam["focal_px"], 0, 640], [0, cam["focal_px"], 360], [0, 0, 1]])
    ztab = -task["robot_so101"]["base_z_offset"]
    split = json.loads(Path(".cache/derived_joint/manifest.json").read_text())["split"]
    detections = json.loads(Path("results/audit/paired_approach_3988577/report.json").read_text())["correspondences"]
    starts = [r for r in detections if r["kind"] == "start" and r["episode"] in split["train"]]
    task["ball"]["start_region"]["center_task_xy"] = backproject_to_plane(
        [r["pixel"] for r in starts], rv, tv, K, ztab + task["ball"]["radius"])[:, :2].mean(0).tolist()
    task["bowl"]["center_task_xy"] = backproject_to_plane(
        [[840, 475]], rv, tv, K, ztab + task["bowl"]["outer_height"])[0, :2].tolist()
    task["table"]["mat"].update(center_task_xy=fit["mat_center_task_xy"], yaw_deg=fit["mat_yaw_deg"])
    cfg = out / "configs"
    cfg.mkdir()
    (cfg / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
    with So101PickPlaceEnv(old / "sim_so101.yaml", render_images=False) as env:
        limits = env.limits.copy()
    for name in ["sim_so101.yaml", "data.yaml", "eval_so101.yaml"]:
        data = yaml.safe_load((old / name).read_text())
        data["task_config"] = str((cfg / "task.yaml").resolve())
        if "data_config" in data:
            data["data_config"] = str((cfg / "data.yaml").resolve())
        if name == "sim_so101.yaml":
            data["initial_qpos"] = np.clip(sim_from_real(JointMapping.from_task_config(task),
                data["initial_qpos_real_deg"]), *limits.T).tolist()
        (cfg / name).write_text(yaml.safe_dump(data, sort_keys=False))
    report = {"kind": "TRAIN_LANDMARK_M2_CONTACT_DIAGNOSTIC_NOT_PRIMARY", "fit_sha256": file_sha256(source),
        "joint_offsets_deg": offsets, "start_episodes": [r["episode"] for r in starts],
        "note": "No checkpoint/labels/splits/limits/actuator changes. Camera, joint offsets and object projection co-vary per one externally fitted candidate.",
        "config_hashes": {p.name: file_sha256(p) for p in cfg.glob("*.yaml")}}
    (out / "protocol.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
