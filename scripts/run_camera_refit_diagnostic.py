"""Frozen-camera diagnostic: separate rendering changes from reprojected object geometry."""

import copy
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

from real2sim.data.paired_approach import camera_parameters, pose_render
from real2sim.evaluation import oracle, run
from real2sim.evaluation.response_adapter import ApproachResponseAdapter
from real2sim.policy.paired import file_sha256
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.follower_response import FollowerResponse

OUT = Path("results/metrics/camera_refit_verified_20261014")
OLD = Path("results/metrics/geometry_3988577/configs")
REFIT = Path("results/audit/camera_landmark_refit_float64_20261014/report.json")
PAIRED = Path("outputs/act_joint_paired_approach/checkpoints/000500")
JOINT = Path("outputs/act_joint_so101/checkpoints/020000")
RESPONSE = Path("results/audit/follower_response_approach_20261013/model.json")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def snapshot(name, task):
    dest = OUT / name / "configs"
    dest.mkdir(parents=True)
    (dest / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
    for name in ("sim_so101.yaml", "data.yaml", "eval_so101.yaml"):
        cfg = yaml.safe_load((OLD / name).read_text())
        cfg["task_config"] = str((dest / "task.yaml").resolve())
        if "data_config" in cfg:
            cfg["data_config"] = str((dest / "data.yaml").resolve())
        (dest / name).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dest


def prepare():
    fit = json.loads(REFIT.read_text())["models"]["M1_refit_camera"]
    camera = fit["camera"]
    base = yaml.safe_load((OLD / "task.yaml").read_text())
    visual = copy.deepcopy(base)
    visual["camera_iphone"].update({k: camera[k] for k in
                                    ("fovy_deg", "xyaxes_task", "look_at_task")})
    visual["camera_iphone"].update(
        approx_position_task=camera["position_task"],
        intrinsics_px={"fx": camera["focal_px"], "fy": camera["focal_px"], "cx": 640., "cy": 360.},
        extrinsics_provenance=f"DIAGNOSTIC M1 camera from {REFIT}; not adopted in primary config")
    configs = {"camera_only": snapshot("camera_only", visual)}
    rv = np.array(camera["rvec"])
    R = cv2.Rodrigues(rv)[0]
    pos = np.array(camera["position_task"])
    K = np.array([[camera["focal_px"], 0, 640], [0, camera["focal_px"], 360], [0, 0, 1]])

    def backproject(px, z):
        ray = R.T @ np.linalg.solve(K, np.r_[px, 1.])
        return pos + (z - pos[2]) / ray[2] * ray

    split = json.loads(Path(".cache/derived_joint/manifest.json").read_text())["split"]
    prior = json.loads(Path("results/audit/paired_approach_3988577/report.json").read_text())
    starts = [c for c in prior["correspondences"] if c["kind"] == "start" and c["episode"] in split["train"]]
    ztab = -base["robot_so101"]["base_z_offset"]
    xy = np.array([backproject(c["pixel"], ztab + base["ball"]["radius"])[:2] for c in starts])
    full = copy.deepcopy(visual)
    full["ball"]["start_region"]["center_task_xy"] = xy.mean(0).tolist()
    full["ball"]["start_region"]["provenance"] = "DIAGNOSTIC mean of existing TRAIN-only first-frame detections"
    full["bowl"]["center_task_xy"] = backproject([840, 475], ztab + base["bowl"]["outer_height"])[:2].tolist()
    full["bowl"]["center_provenance"] = "DIAGNOSTIC reused hand-read rim pixel (840,475), at rim height"
    full["table"]["mat"].update(center_task_xy=fit["mat_center_task_xy"], yaw_deg=fit["mat_yaw_deg"])
    configs["camera_objects"] = snapshot("camera_objects", full)
    protocol = {
        "kind": "DIAGNOSTIC_CAMERA_REFIT_FROZEN_POLICY_NOT_PRIMARY_BENCHMARK",
        "camera_model": "M1_refit_camera; no joint offset changes",
        "start_detection_episodes": [c["episode"] for c in starts],
        "start_xy_per_episode": xy.tolist(), "start_xy_mean": xy.mean(0).tolist(),
        "bowl_xy": full["bowl"]["center_task_xy"],
        "caveats": ["Validation annotations were adaptively reviewed; this is exploratory.",
                    "Mat residual remains 56 px; geometry is not established ground truth.",
                    "camera_only changes only camera; camera_objects also reprojects objects and mat.",
                    "Frozen paired weights were trained on the OLD renderer.",
                    "No policy consumes object pose; all physical geometry changes are labelled diagnostics."],
        "source_hashes": {str(p): file_sha256(p) for p in
                          [REFIT, PAIRED / "model.safetensors", JOINT / "model.safetensors", RESPONSE]},
    }
    with So101PickPlaceEnv(configs["camera_objects"] / "sim_so101.yaml", render_images=False) as env:
        env.reset(seed=0)
        r2, t2, k2, _, _ = camera_parameters(env)
        points = np.array([[.3, .1, .05], [.1, -.1, .2], [.35, .2, .03]])
        a = cv2.projectPoints(points, rv, -R @ pos, K, None)[0]
        b = cv2.projectPoints(points, r2, t2, k2, None)[0]
        error = float(np.max(abs(a - b)))
        assert error < 1e-3, error
        protocol["camera_roundtrip_max_px"] = error
        arrays = np.load(".cache/derived_joint/arrays.npz")
        images = np.load(".cache/derived_joint/images_iphone.npy", mmap_mode="r")
        tiles = []
        with mujoco.Renderer(env.model, height=144, width=256) as renderer:
            for ep, frame in [(12, 480), (2, 624), (18, 519), (24, 828)]:
                idx = np.flatnonzero((arrays["episode_index"] == ep) & (arrays["frame_index"] == frame))[0]
                pose_render(env, arrays["state"][idx], full["ball"]["start_region"]["center_task_xy"])
                renderer.update_scene(env.data, camera="iphone")
                sim = np.pad(renderer.render(), ((24, 24), (0, 0), (0, 0)), constant_values=114)
                tile = np.concatenate([images[idx], sim], axis=1).copy()
                cv2.putText(tile, f"ep{ep} f{frame}: real | refit (ball static)", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
                tiles.append(tile)
        cv2.imwrite(str(OUT / "real_refit_montage.jpg"), cv2.cvtColor(np.concatenate(tiles), cv2.COLOR_RGB2BGR))
    protocol["config_hashes"] = {str(p): file_sha256(p) for cfg in configs.values() for p in cfg.glob("*.yaml")}
    dump(OUT / "protocol.json", protocol)
    return configs, protocol


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    state = {"status": "preparing", "completed": []}
    dump(OUT / "status.json", state)
    try:
        configs, protocol = prepare()
        jobs = [("camera_only", "paired"), ("camera_objects", "hold"),
                ("camera_objects", "oracle"), ("camera_objects", "joint"),
                ("camera_objects", "paired"), ("camera_objects", "paired_response")]
        for scene, controller in jobs:
            state.update(status="running", phase=f"{scene}/{controller}")
            dump(OUT / "status.json", state)
            cfg = configs[scene]
            common = dict(config=cfg / "sim_so101.yaml", eval_config=cfg / "eval_so101.yaml",
                          n_trials=10, seed_start=0, out=OUT / scene / controller)
            original = run.build_learned_controller
            adapters = []
            if controller == "paired_response":
                model = FollowerResponse(**json.loads(RESPONSE.read_text())["model"])

                def build(env, *args, _original=original, _model=model, _adapters=adapters, **kwargs):
                    ctrl, contract, meta = _original(env, *args, **kwargs)
                    adapter = ApproachResponseAdapter(env, _model, active_ticks=78)
                    ctrl.action_adapter = adapter
                    _adapters.append(adapter)
                    return ctrl, contract, meta

                run.build_learned_controller = build
            try:
                if controller == "oracle":
                    result = oracle.run_trials(**common, video_dir=None)
                else:
                    result = run.run_trials(**common, robot="so101", video=False, device="cpu",
                        checkpoint=None if controller == "hold" else JOINT if controller == "joint" else PAIRED,
                        controller="hold" if controller == "hold" else "learned",
                        experiment_id=f"DIAGNOSTIC_REFIT_{scene}_{controller}")
            finally:
                run.build_learned_controller = original
            for i, adapter in enumerate(adapters):
                dump(common["out"] / f"response_traces_{i}.json", adapter.episodes)
            state["completed"].append({"scene": scene, "controller": controller,
                "summary": result["summary"],
                "mean_closest_m": float(np.mean([t["min_ee_ball_distance_m"] for t in result["trials"]]))})
            dump(OUT / "status.json", state)
        for path, digest in {**protocol["source_hashes"], **protocol["config_hashes"]}.items():
            assert file_sha256(Path(path)) == digest, path
        state.update(status="completed", hashes_verified=True)
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        dump(OUT / "status.json", state)


if __name__ == "__main__":
    main()
