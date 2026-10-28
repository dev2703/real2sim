"""Paired latest-ACT image sensitivity before/after a training-only frame hypothesis.

Every comparison holds the real joint vector and frozen model fixed. Recorded
images are a common reference; nominal and rotated scene images are separate
conditions. Evaluation episodes never fit the frame hypothesis. This diagnoses
visual sensitivity, not physical calibration or closed-loop task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import torch

from real2sim.artifacts import sha256_file
from real2sim.data.raw import RawDataset
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.serialization import write_json
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

if __package__:
    from scripts import diagnose_hf_operator_frame as frame_fit
    from scripts import evaluate_hf_act_20261027 as native
else:
    import diagnose_hf_operator_frame as frame_fit
    import evaluate_hf_act_20261027 as native

SOURCE = Path("configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml")
DATASET = Path("~/.cache/huggingface/lerobot/local/so101_act_pickplace_v2").expanduser()


def prediction_metrics(prediction, reference, recorded_prediction, prefix):
    mae = np.abs(prediction[:prefix] - reference[:prefix]).mean(0)
    change = np.abs(prediction[:prefix] - recorded_prediction[:prefix]).mean(0)
    return {"five_arm_joint_mae_degrees": float(mae[:5].mean()),
            "gripper_mae_percent": float(mae[5]),
            "joint_mae_degree_percent": mae.tolist(),
            "image_swap_action_change_degree_percent": change.tolist()}


def shared_start(envs, states):
    for index, state in enumerate(states[:-60]):
        accepted = True
        for env in envs.values():
            q = sim_from_real(env.mapping, state)
            if np.any(q < env.limits[:, 0]) or np.any(q > env.limits[:, 1]):
                accepted = False
                break
            try:
                env.reset(seed=0, scenario={"joint_positions": q})
            except ValueError:
                accepted = False
                break
        if accepted:
            return index
    return None


def run(out, *, snapshot=native.SNAPSHOT, dataset=DATASET, source_scene=SOURCE):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    candidate = frame_fit.derive(snapshot, dataset, source_scene, out / "frame_hypothesis")
    scenes = {}
    for name, source in (("nominal", source_scene), ("training_yaw", candidate)):
        directory = out / name
        directory.mkdir()
        scenes[name] = native.prepare_scene(directory, source)
    model = NativeLeRobotACT(snapshot)
    model.policy.eval().requires_grad_(False)
    experiment = json.loads((snapshot / "experiment.json").read_text())
    raw = RawDataset(dataset)
    watched = [Path(__file__), Path(frame_fit.__file__), Path(native.__file__),
        Path("src/real2sim/policy/native_lerobot_act.py"), source_scene,
        dataset / "meta/info.json", *snapshot.glob("*.json"), *snapshot.glob("*.safetensors")]
    watched += [p for directory in scenes.values() for p in directory.parent.glob("*.yaml")]
    before = {str(p): sha256_file(p) for p in watched}
    report = {"schema": "frozen_latest_alignment_paired_image_diagnostic_v1", "status": "running",
        "weights_sha256": model.weights_sha256, "lighting_changed": False,
        "calibration_adopted": False, "task_success_evaluated": False,
        "candidate_fit_episodes": experiment["train_episodes"],
        "evaluation_episodes": experiment["val_episodes"],
        "selection": "First pose valid in both scenes, then +0,+25,+50source frames. Joint/scene-invalid candidates retained as exclusions; no success or error score selects anchors.",
        "scope": "Same real states across recorded/nominal/training-yaw image conditions. Object layout, camera pose and appearance differ jointly; not isolated camera calibration.",
        "source_sha256_before": before, "cases": [], "excluded": []}
    (out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "report.json", report, atomic=True)
    with ExitStack() as stack:
        envs = {name: stack.enter_context(So101PickPlaceEnv(path, render_images=False))
                for name, path in scenes.items()}
        for episode in experiment["val_episodes"]:
            rows = raw.episode_rows(episode)
            states, actions = (raw.stack(rows, key) for key in ("observation.state", "action"))
            start = shared_start(envs, states)
            if start is None:
                report["excluded"].append({"episode": episode, "reason": "No shared valid source pose"})
                continue
            frames = [start + offset for offset in (0, 25, 50)]
            decoded = {key: {} for key in model.camera_keys}
            for key in model.camera_keys:
                for index, _pts, image in raw.iter_frames(key, episode):
                    if index in frames:
                        decoded[key][index] = cv2.resize(image.to_ndarray(format="rgb24"),
                            (640, 360), interpolation=cv2.INTER_LINEAR)
                    if index >= frames[-1]:
                        break
            for frame in frames:
                if any(frame not in decoded[key] for key in model.camera_keys):
                    report["excluded"].append({"episode": episode, "frame": frame, "reason": "Missing source RGB"})
                    continue
                images = {"recorded": {key: decoded[key][frame] for key in model.camera_keys}}
                try:
                    for name, env in envs.items():
                        q = sim_from_real(env.mapping, states[frame])
                        env.reset(seed=0, scenario={"joint_positions": q})
                        if not np.allclose(env.real_state(), states[frame], atol=1e-5, rtol=0):
                            raise AssertionError("Paired real joint input changed")
                        images[name] = {key: env.render(env.camera_bindings[key]) for key in model.camera_keys}
                except ValueError as exc:
                    report["excluded"].append({"episode": episode, "frame": frame, "reason": str(exc)})
                    continue
                for scene_name in ("nominal", "training_yaw"):
                    images[scene_name + "_wrist_only"] = {
                        key: images[scene_name][key] if key.endswith(".top") else images["recorded"][key]
                        for key in model.camera_keys}
                    images[scene_name + "_global_only"] = {
                        key: images[scene_name][key] if key.endswith(".iphone") else images["recorded"][key]
                        for key in model.camera_keys}
                predictions = {}
                for name, condition in images.items():
                    model.reset()
                    predictions[name] = model.predict_action_chunk(states[frame], condition)
                model.reset()
                repeated = model.predict_action_chunk(states[frame], images["recorded"])
                if not np.array_equal(repeated, predictions["recorded"]):
                    raise AssertionError("Native ACT identical-input repeat changed")
                prefix = min(model.config.n_action_steps, len(actions) - frame)
                record = {"episode": episode, "frame": frame,
                    "raw_joint_input_degree_percent": states[frame].tolist(),
                    "conditions": {name: prediction_metrics(prediction, actions[frame:frame + prefix],
                        predictions["recorded"], prefix) for name, prediction in predictions.items()},
                    "input_image_sha256": {name: {key: hashlib.sha256(rgb.tobytes()).hexdigest()
                        for key, rgb in condition.items()} for name, condition in images.items()}}
                report["cases"].append(record)
                panel = np.vstack([np.hstack([images[name][key] for name in ("recorded", "nominal", "training_yaw")]) for key in model.camera_keys])
                cv2.imwrite(str(out / f"episode_{episode:03d}_frame_{frame:04d}_real_nominal_yaw.jpg"), panel[:, :, ::-1])
                write_json(out / "report.json", report, atomic=True)
            print(json.dumps({"episode": episode, "shared_start": start, "cases_so_far": len(report["cases"])}), flush=True)
    condition_names = ("recorded", "nominal", "training_yaw", "nominal_wrist_only",
                       "nominal_global_only", "training_yaw_wrist_only", "training_yaw_global_only")
    report["summary"] = {name: {
        "cases": len(report["cases"]),
        "mean_five_arm_joint_prefix_mae_degrees": float(np.mean([row["conditions"][name]["five_arm_joint_mae_degrees"] for row in report["cases"]])),
        "mean_gripper_prefix_mae_percent": float(np.mean([row["conditions"][name]["gripper_mae_percent"] for row in report["cases"]])),
        "mean_image_swap_joint_change_degree_percent": np.mean([row["conditions"][name]["image_swap_action_change_degree_percent"] for row in report["cases"]], axis=0).tolist(),
    } for name in condition_names}
    after = {p: sha256_file(Path(p)) for p in before}
    report.update(status="complete", source_sha256_after=after,
        runtime_source_changes=[p for p in before if before[p] != after[p]],
        checkpoint_unchanged=sha256_file(snapshot / "model.safetensors") == model.weights_sha256)
    write_json(out / "report.json", report, atomic=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=native.SNAPSHOT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--source-scene", type=Path, default=SOURCE)
    args = parser.parse_args()
    run(args.out, snapshot=args.snapshot, dataset=args.dataset, source_scene=args.source_scene)
