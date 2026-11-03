"""Derive an operator-axis yaw hypothesis from training grasp motions only.

No rollout outcome, validation episode, joint offset, lighting change or distance
scaling enters this derivation. The existing provisional FK and grasp offset are
explicit assumptions; this diagnostic is not independent physical registration.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from real2sim.artifacts import sha256_file
from real2sim.config import load_yaml
from real2sim.data.raw import RawDataset
from real2sim.geometry.so101_fk import JointMapping, So101FK
from real2sim.serialization import write_json

if __package__:
    from scripts import evaluate_hf_act_20261027 as native
else:
    import evaluate_hf_act_20261027 as native


def derive(snapshot, dataset, source_scene, out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    source_scene = Path(source_scene)
    config = load_yaml(source_scene)
    source_task = Path(config["task_config"])
    task = load_yaml(source_task)
    experiment = json.loads((snapshot / "experiment.json").read_text())
    raw = RawDataset(dataset)
    fk = So101FK(mapping=JointMapping.from_task_config(task))
    grasp_offset = np.array([-.010, 0., .030])
    points, episodes = [], []
    for episode in experiment["train_episodes"]:
        rows = raw.episode_rows(episode)
        state = raw.stack(rows, "observation.state")
        action = raw.stack(rows, "action")
        crossing = np.flatnonzero((action[:-1, 5] > 45.) & (action[1:, 5] <= 45.)) + 1
        if not len(crossing):
            episodes.append({"episode": episode, "usable": False, "reason": "no downward 45% crossing"})
            continue
        frame = int(crossing[0])
        pose = fk.fk_real(state[frame])
        point = pose[:3, 3] + pose[:3, :3] @ grasp_offset
        points.append(point)
        episodes.append({"episode": episode, "usable": True, "frame": frame,
                         "state_degree_percent": state[frame].tolist(),
                         "grasp_point_native_base_m": point.tolist()})
    if not points:
        raise ValueError("No training grasp transition is available")
    points = np.asarray(points)
    median = np.median(points, axis=0)
    operator_ball = np.asarray(task["ball"]["start_region"]["center_task_xy"], float)
    yaw = float(np.arctan2(median[1], median[0]) - np.arctan2(operator_ball[1], operator_ball[0]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
    transformed = copy.deepcopy(task)

    def rotate_xy(value):
        return (rotation[:2, :2] @ np.asarray(value, float)).tolist()

    transformed["ball"]["start_region"]["center_task_xy"] = rotate_xy(operator_ball)
    transformed["bowl"]["center_task_xy"] = rotate_xy(task["bowl"]["center_task_xy"])
    mat = transformed["table"]["mat"]
    mat["center_task_xy"] = rotate_xy(mat["center_task_xy"])
    mat["yaw_deg"] = float(mat.get("yaw_deg", 0.)) + float(np.rad2deg(yaw))
    if support := transformed["ball"].get("support"):
        support["center_task_xy"] = rotate_xy(support["center_task_xy"])
        support["yaw_deg"] = float(support.get("yaw_deg", 0.)) + float(np.rad2deg(yaw))
    camera = transformed["camera_iphone"]
    for key in ("approx_position_task", "look_at_task"):
        camera[key] = (rotation @ np.asarray(camera[key], float)).tolist()
    if "xyaxes_task" in camera:
        camera["xyaxes_task"] = (np.asarray(camera["xyaxes_task"]).reshape(2, 3) @ rotation.T).ravel().tolist()
    hypothesis = {
        "schema": "training_fk_operator_frame_yaw_hypothesis_v1",
        "not_independent_physical_calibration": True,
        "training_episodes": experiment["train_episodes"],
        "validation_episodes_not_used": experiment["val_episodes"],
        "transition": "First downward action-gripper 45% crossing in each training episode",
        "grasp_offset_tool_m": grasp_offset.tolist(),
        "estimator": "componentwise median training transition FK point; align XY bearing only",
        "training_grasp_point_median_native_m": median.tolist(),
        "training_grasp_point_std_native_m": points.std(0).tolist(),
        "operator_to_native_yaw_degrees": float(np.rad2deg(yaw)),
        "operator_to_native_rotation": rotation.tolist(),
        "source_ball_operator_xy_m": operator_ball.tolist(),
        "candidate_ball_native_xy_m": transformed["ball"]["start_region"]["center_task_xy"],
        "source_base_ball_distance_m": float(np.linalg.norm(operator_ball)),
        "training_median_base_grasp_distance_m": float(np.linalg.norm(median[:2])),
        "distances_scaled": False, "joint_offsets_changed": False, "lighting_changed": False,
        "assumptions": "Provisional current joint mapping/FK; existing CAD grasp offset; first45% close approximates grasp; fixed base origin and sign-handed operator axes. Camera optics/rotation remain the source scene's provisional hypotheses, expressed in the rotated frame. Existing scenery is native-coordinate context, unchanged.",
        "source_sha256": {str(path): sha256_file(path) for path in [
            Path(__file__), Path("physical_measurements.md"), source_scene, source_task,
            dataset / "meta/info.json", snapshot / "experiment.json",
            snapshot / "model.safetensors", Path("src/real2sim/geometry/so101_fk.py")]},
        "episodes": episodes,
    }
    transformed.update(geometry_status="training_fk_operator_axis_yaw_hypothesis",
                       physical_calibration_accepted=False, operator_frame_hypothesis=hypothesis)
    task_path, sim_path = out / "task.yaml", out / "sim_so101.yaml"
    config["task_config"] = str(task_path.resolve())
    task_path.write_text(yaml.safe_dump(transformed, sort_keys=False))
    sim_path.write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(out / "hypothesis.json", hypothesis, atomic=True)
    return sim_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=native.SNAPSHOT)
    parser.add_argument("--dataset", type=Path,
                        default=Path("~/.cache/huggingface/lerobot/local/so101_act_pickplace_v2").expanduser())
    parser.add_argument("--source-scene", type=Path,
                        default=Path("configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    args = parser.parse_args()
    scene = derive(args.snapshot, args.dataset, args.source_scene, args.out)
    if args.seeds:
        return native.run(args.snapshot, args.out / "raw_policy", args.seeds,
                          video=False, source_scene=scene)


if __name__ == "__main__":
    main()
