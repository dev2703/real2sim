"""Frozen-checkpoint DIAGNOSTIC: ball visibility, executed prefix and data mismatch.

No training, benchmark changes or physical motion. Simulator poses are diagnostic
setup only. Each run requires a fresh output directory and snapshots its inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from real2sim.policy.act import load_checkpoint
from real2sim.policy.adapters import normalize_image_chw
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def longest_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return int(max(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1), default=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    checkpoint = Path("outputs/act_joint_so101/checkpoints/020000")
    policy, norm, contract = load_checkpoint(checkpoint, device="cpu")
    key = contract["camera_keys"][0]
    episodes = [0, 5, 6, 7, 1, 3]
    ds = DerivedTickDataset(".cache/derived_joint", episodes, policy.config.chunk_size, norm)
    inputs = ["configs/sim_so101.yaml", "configs/task.yaml", "configs/data.yaml",
              "src/real2sim/sim/model.py", "src/real2sim/sim/env.py",
              "results/audit/camera_fit/backprojection.json", str(Path(__file__)),
              ".cache/derived_joint/manifest.json"]
    hashes = {}
    for path in inputs:
        content = Path(path).read_bytes()
        hashes[path] = hashlib.sha256(content).hexdigest()
        (args.output / Path(path).name).write_bytes(content)
    cfg = yaml.safe_load((args.output / "sim_so101.yaml").read_text())
    cfg["task_config"] = str((args.output / "task.yaml").resolve())
    cfg["data_config"] = str((args.output / "data.yaml").resolve())
    frozen_config = args.output / "snapshot_sim.yaml"
    frozen_config.write_text(yaml.safe_dump(cfg))
    bp = json.loads((args.output / "backprojection.json").read_text())["estimates"]["start"]
    ball_xy = dict(zip([e for e, _ in bp["episodes_frames"]], bp["points_xy"], strict=True))
    report = {"diagnostic_only": True, "checkpoint_sha256": sha(checkpoint / "model.safetensors"),
              "input_sha256": hashes, "n_action_steps": policy.config.n_action_steps,
              "chunk_size": policy.config.chunk_size, "episodes": [],
              "method": "Same recorded start state for image interventions. True absence moves "
              "ball to world [100,100,100] without stepping; segmentation verifies visibility. "
              "First two actions execute before replanning; full chunks retained for inspection. "
              "No success inferred. States and images mixed across phases are counterfactuals."}
    env = So101PickPlaceEnv(frozen_config, render_images=True)

    def predict(image, state):
        with torch.inference_mode():
            return norm.unaction(policy.predict_action_chunk({key: image[None],
                                                               "observation.state": state[None]}))[0].numpy()

    def render_case():
        rgb = env.observation()[key]
        env.renderer.enable_segmentation_rendering()
        try:
            seg = env.render()
            visible = int(np.count_nonzero((seg[:, :, 0] == env.ball_geom)
                                          & (seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)))
        finally:
            env.renderer.disable_segmentation_rendering()
        return torch.from_numpy(normalize_image_chw(norm, rgb)), visible

    try:
        for e in episodes:
            ticks = np.flatnonzero(ds.arrays["episode_index"] == e)
            first, last = int(ticks[0]), int(ticks[-1])
            item = ds[int(np.flatnonzero(ds.indices == first)[0])]
            end = ds[int(np.flatnonzero(ds.indices == last)[0])]
            q = np.clip(sim_from_real(env.mapping, ds.arrays["state"][first]),
                        env.limits[:, 0], env.limits[:, 1])
            scenario = {"joint_positions": q}
            if e in ball_xy:
                scenario["ball_xy"] = ball_xy[e]
            env.reset(seed=0, scenario=scenario)
            sim, present_pixels = render_case()
            env.reset(seed=0, scenario={**scenario, "ball_xy": [0.55, 0.42]})
            moved, moved_pixels = render_case()
            env.data.qpos[env._ball_q:env._ball_q + 3] = [100, 100, 100]
            mujoco.mj_forward(env.model, env.data)
            absent, absent_pixels = render_case()
            assert absent_pixels == 0
            env.reset(seed=0, scenario=scenario)
            for _ in range(5):
                env.step(q)
            settled_state = torch.from_numpy(norm.state(env.real_state()).astype(np.float32))
            start_state = item["observation.state"]
            cases = {"real_start": (item[key], start_state),
                     "sim_ball": (sim, start_state), "sim_ball_corner": (moved, start_state),
                     "sim_ball_absent": (absent, start_state),
                     "real_end_start_state": (end[key], start_state),
                     "real_start_end_state": (item[key], end["observation.state"]),
                     "real_end_end_state": (end[key], end["observation.state"]),
                     "real_start_settled_state": (item[key], settled_state)}
            chunks = {name: predict(im, st) for name, (im, st) in cases.items()}
            parity = normalize_image_chw(norm, ds.images[key][first].transpose(2, 0, 1)
                                         .astype(np.float32) / 255)
            row = {"episode": e, "split": "train" if e in ds.manifest["split"]["train"] else "val",
                   "ball_visible_native_pixels": {"present": present_pixels, "corner": moved_pixels,
                                                  "absent": absent_pixels},
                   "preprocessing_max_abs_difference": float(np.max(abs(parity - item[key].numpy()))),
                   "chunks_raw": {name: c.tolist() for name, c in chunks.items()},
                   "first_two_normalized_mae_from_real_start": {
                       name: float(np.mean(abs((c[:2] - chunks["real_start"][:2]) / norm.action_std)))
                       for name, c in chunks.items()}}
            report["episodes"].append(row)
            np.savez_compressed(args.output / f"ep{e}_inputs.npz", **{
                name: im.numpy() for name, (im, _) in cases.items()})
            print(f"episode {e}: visible pixels present/corner/absent "
                  f"{present_pixels}/{moved_pixels}/{absent_pixels}", flush=True)
    finally:
        env.close()

    data_rows = []
    for split in ("train", "val"):
        all_gaps, total, suspicious, stationary = [], 0, 0, 0
        for e in ds.manifest["split"][split]:
            ix = np.flatnonzero(ds.arrays["episode_index"] == e)
            q, u = ds.arrays["state"][ix], ds.arrays["action"][ix]
            dq = np.diff(q, axis=0)
            flag = (abs(dq[:, 1]) < .2) & (abs(u[1:, 1] - q[1:, 1]) > 10)
            total += len(dq)
            suspicious += int(flag.sum())
            stationary += int((np.max(abs(dq[:, :5]), axis=1) < .25).sum())
            all_gaps.append(abs(u - q))
            data_rows.append({"episode": e, "split": split, "duration_s": len(ix) / 10,
                              "stationary_shoulder_with_gap_fraction": float(flag.mean()),
                              "longest_stationary_shoulder_with_gap_s": longest_run(flag) / 10})
        report[f"data_{split}"] = {
            "stationary_arm_fraction": stationary / total,
            "stationary_shoulder_with_gap_fraction": suspicious / total,
            "absolute_leader_follower_gap_p95": np.quantile(np.concatenate(all_gaps), .95, axis=0).tolist()}
    report["data_episodes"] = data_rows
    report["data_thresholds"] = "10 Hz: arm stationary if all five arm increments <0.25 deg; " \
        "shoulder mismatch if increment <0.2 deg and |leader-follower| >10 deg. " \
        "These are descriptive flags, not proof of mechanical stall or task phase."
    report["inputs_changed_during_run"] = [path for path in inputs if sha(path) != hashes[path]]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
