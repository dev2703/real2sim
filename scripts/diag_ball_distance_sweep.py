"""Frozen ACT and recorded-motion radial ball-offset diagnostic, never a benchmark.

Only reset ball xy changes: +1/+2 cm radially away from the robot task origin.
Recorded replay uses reviewed per-episode bowl setup and unchanged real targets.
No training, attachments, physics changes or privileged ACT observations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from real2sim.config import load_yaml
from real2sim.data.bowl_placement import validate_placements
from real2sim.data.paired_approach import camera_parameters
from real2sim.evaluation.metrics import TrialTracker
from real2sim.evaluation.run import build_learned_controller
from real2sim.policy.adapters import ApplyResult
from real2sim.policy.paired import file_sha256
from real2sim.sim.camera_fit import backproject_to_plane
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

BASE = Path("results/metrics/paired_refit_pilot_20261014")
OFFSETS = (0.0, 0.01, 0.02)


def farther(xy, distance):
    xy = np.asarray(xy, dtype=np.float64)
    radius = np.linalg.norm(xy)
    if xy.shape != (2,) or not np.isfinite(xy).all() or radius < 1e-8:
        raise ValueError("finite nonzero task-frame ball xy required")
    result = xy * (1 + distance / radius)
    np.testing.assert_allclose(np.linalg.norm(result) - radius, distance, atol=1e-12)
    return result.tolist()


def trial(env, ecfg, scenario, seed, step, reset, metadata):
    obs, _ = env.reset(seed=seed, scenario=scenario)
    reset()
    contacts = [c for c in env._contact_pairs()
                if "ball_geom" in (c["geom1"], c["geom2"])]
    tracker = TrialTracker(
        success_hold_ticks=math.ceil(env.task["episode"]["success_hold_s"] * env.control_hz),
        approach_distance_m=ecfg["phase"]["approach_distance_m"],
        table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"],
    )
    both_jaw_ticks, abort = 0, None
    try:
        for tick in range(metadata.get("max_ticks", env.max_episode_steps)):
            result = step(tick, obs)
            obs = result.obs
            done = tracker.update(result.info["privileged_metrics"], saturation=result.saturation,
                                  ik_rejected=result.ik_rejected)
            touched = env._ball_contacts()
            both_jaw_ticks += bool(touched & env.gripper_geoms["gripper"] and
                                   touched & env.gripper_geoms["moving_jaw_so101_v1"])
            if done or result.truncated:
                break
    except (RuntimeError, ValueError) as exc:
        abort = f"{type(exc).__name__}: {exc}"
    return {**metadata, "seed": seed, "scenario": scenario, "success": tracker.success,
            "held": tracker.held_ever, "lifted": tracker.lifted,
            "phase_failed": "aborted" if abort else tracker.phase_failed(), "abort": abort,
            "min_distance_m": tracker.min_ee_ball_distance_m,
            "both_jaw_ticks_10hz": both_jaw_ticks,
            "initial_ball_contacts": contacts,
            "invalid_initial_overlap": any(c["distance_m"] < -.002 for c in contacts),
            "history": tracker.history}


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    config, eval_config = BASE / "configs/sim_so101.yaml", BASE / "configs/eval_so101.yaml"
    checkpoint = Path("outputs/act_joint_refit_stationary_20261014/checkpoints/000250")
    arrays_path = Path(".cache/derived_joint/arrays.npz")
    bowls_path = Path("results/audit/bowl_placements_20261014/placements.json")
    frozen = {str(p): file_sha256(p) for p in
              [*sorted((BASE / "configs").glob("*.yaml")), checkpoint / "model.safetensors",
               arrays_path, bowls_path]}
    report = {"kind": "DIAGNOSTIC_RADIAL_BALL_SHIFT_NOT_PRIMARY_BENCHMARK",
              "status": "running", "frozen_sha256": frozen, "offsets_m": OFFSETS,
              "note": "Only ball reset moves outward. No new fitting or ACT privileged inputs. "
                      "Follower-next replay is dynamic target tracking, not directly imposed poses. "
                      "10 Hz jaw sampling can miss short contacts. Replay horizon is at most 30 s.",
              "trials": []}
    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def add(row):
        report["trials"].append(row)
        save()
        print(json.dumps({k: row[k] for k in ("mode", "offset_m", "seed", "success", "held", "lifted",
                                             "min_distance_m", "invalid_initial_overlap", "abort")}), flush=True)

    save()
    ecfg = load_yaml(eval_config)
    try:
        with np.load(arrays_path) as arrays, So101PickPlaceEnv(config, render_images=False) as env:
            env.reset(seed=0)
            rv, tv, K, _, _ = camera_parameters(env)
            prep = json.loads(Path(".cache/paired_approach_3988577/preparation.json").read_text())
            placements = json.loads(bowls_path.read_text())
            validate_placements(placements, json.loads(Path(".cache/derived_joint/manifest.json").read_text()))
            bowls = {r["episode"]: r["pixel_policy"] for r in placements["episodes"]}
            def xy_from_pixel(pixel, height):
                native = (np.asarray(pixel) - [0, 24]) / .2
                return backproject_to_plane([native], rv, tv, K, env.table_top - env.offset[2] + height)[0, :2]

            for ep in [e for e in prep["episodes"] if e["split"] == "val"]:
                ep_id = ep["episode"]
                ball = xy_from_pixel(ep["first_pixel_policy"], env.task["ball"]["radius"])
                bowl = xy_from_pixel(bowls[ep_id], env.task["bowl"]["outer_height"])
                indices = np.flatnonzero(arrays["episode_index"] == ep_id)[:301]
                start = np.clip(sim_from_real(env.mapping, arrays["state"][indices[0]]), *env.limits.T)
                for mode in ("leader_command", "follower_next_state"):
                    targets = arrays["action"][indices[:-1]] if mode == "leader_command" else arrays["state"][indices[1:]]
                    def step(tick, obs, targets=targets):
                        obs, _, _, truncated, info = env.step_real_joint_targets(targets[tick])
                        return ApplyResult(obs, info, truncated, float(np.mean(info["control"]["limit_clipped"])))
                    for offset in OFFSETS:
                        scenario = {"ball_xy": farther(ball, offset), "bowl_xy": bowl.tolist(),
                                    "joint_positions": start.tolist()}
                        add(trial(env, ecfg, scenario, ep_id, step, lambda: None,
                                  {"mode": mode, "episode": ep_id, "offset_m": offset,
                                   "max_ticks": len(targets)}))

        with So101PickPlaceEnv(config, render_images=True) as env:
            ctrl, _, meta = build_learned_controller(env, checkpoint, ecfg, "cpu")
            report["checkpoint"] = meta
            for seed in range(10):
                _, info = env.reset(seed=seed)
                ball = info["privileged_metrics"]["ball_xyz_task"][:2]
                for offset in OFFSETS:
                    add(trial(env, ecfg, {"ball_xy": farther(ball, offset)}, seed,
                              lambda tick, obs: ctrl.step(obs, env), ctrl.reset,
                              {"mode": "frozen_ACT_joint", "offset_m": offset}))
        report["summary"] = []
        for mode in ("leader_command", "follower_next_state", "frozen_ACT_joint"):
            for offset in OFFSETS:
                rows = [r for r in report["trials"] if r["mode"] == mode and r["offset_m"] == offset]
                report["summary"].append({"mode": mode, "offset_m": offset, "n": len(rows),
                    **{key: sum(bool(r[key]) for r in rows) for key in
                       ("success", "held", "lifted", "abort", "invalid_initial_overlap")},
                    "mean_min_distance_m": float(np.mean([r["min_distance_m"] for r in rows])),
                    "two_jaw_trials_10hz": sum(r["both_jaw_ticks_10hz"] > 0 for r in rows)})
        assert all(file_sha256(Path(p)) == checksum for p, checksum in frozen.items())
        report.update(status="completed", frozen_hashes_verified=True)
        save()
        print(json.dumps(report["summary"], indent=2), flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
