"""Record a small, separate synthetic oracle dataset; never train or modify real data.

Each NPZ contains T+1 RGB observations/states and T requested actions. Action t
maps observation t to observation t+1 through the unchanged environment controller.
Privileged oracle/scoring information is confined to the separate episode audit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from real2sim.artifacts import sha256_file as sha256
from real2sim.config import load_yaml
from real2sim.evaluation.metrics import TrialTracker
from real2sim.evaluation.oracle import (
    CAPTION,
    CONTROLLER_SOURCE,
    OracleGeometryController,
    settle_start_pose,
)
from real2sim.evaluation.video import CaptionedVideo, policy_view_to_bgr
from real2sim.policy.adapters import IPHONE_KEY, JOINT_STATE_NAMES
from real2sim.sim.env import So101PickPlaceEnv, real_from_sim, sim_from_real


def rgb(obs: dict) -> np.ndarray:
    return np.rint(obs[IPHONE_KEY].transpose(1, 2, 0) * 255).astype(np.uint8)


def collect(config: Path, eval_config: Path, out: Path, count: int, seed_start: int) -> dict:
    if count < 1:
        raise ValueError("count must be positive")
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier collection.
    ecfg = load_yaml(eval_config)
    manifest = {
        "status": "running", "source": "SYNTHETIC_ORACLE_DEMONSTRATIONS",
        "controller_source": CONTROLLER_SOURCE, "training_started": False,
        "training_authorized": False, "benchmark_result": False,
        "split": "unassigned_separate_from_real_dataset",
        "state_names": JOINT_STATE_NAMES, "action_names": JOINT_STATE_NAMES,
        "units": "five arm degrees, gripper percent; provisional simulator mapping",
        "alignment": "images/state/timestamp have T+1 rows; action[t] drives state[t] to state[t+1]",
        "action_semantics": "requested joint targets BEFORE environment rate limiting",
        "privileged_inputs": "oracle teacher only; excluded from NPZ observation fields",
        "episodes": [],
    }
    def save_manifest():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    env = None
    try:
        env = So101PickPlaceEnv(config, render_images=True)
        manifest["hz"] = env.control_hz
        manifest["resolved"] = {"sim": env.config, "task": env.task,
                                "data": env.data_config, "eval": ecfg}
        manifest["config_sha256"] = {str(p.resolve()): sha256(p) for p in (config, eval_config)}
        for seed in range(seed_start, seed_start + count):
            start, settle = settle_start_pose(env, seed)
            obs, initial = env.reset(seed=seed, scenario={"joint_positions": start})
            ctrl = OracleGeometryController(env)
            tracker = TrialTracker(
                success_hold_ticks=math.ceil(env.task["episode"]["success_hold_s"] * env.control_hz),
                approach_distance_m=float(ecfg["phase"]["approach_distance_m"]),
                table_top_z=env.table_top, ball_radius_m=float(env.task["ball"]["radius"]),
            )
            images, states, native_states, actions, ticks = [rgb(obs)], [env.real_state()], [env.sim_state()], [], []
            video = CaptionedVideo(out / f"seed_{seed}.mp4", env.control_hz, (384, 512))
            video.write(policy_view_to_bgr(obs[IPHONE_KEY]), CAPTION)
            try:
                for _ in range(env.max_episode_steps):
                    result = ctrl.step(obs, env)
                    requested = np.asarray(result.info["control"]["requested_radians"])
                    action = real_from_sim(env.mapping, requested)
                    np.testing.assert_allclose(sim_from_real(env.mapping, action), requested, atol=1e-12, rtol=0)
                    actions.append(action)
                    obs = result.obs
                    images.append(rgb(obs))
                    states.append(env.real_state())
                    native_states.append(env.sim_state())
                    ticks.append({**result.detail, "control": result.info["control"]})
                    done = tracker.update(result.info["privileged_metrics"], saturation=result.saturation,
                                          ik_rejected=result.ik_rejected)
                    video.write(policy_view_to_bgr(obs[IPHONE_KEY]), f"{CAPTION} {result.detail['phase']}")
                    if done or result.truncated:
                        break
            finally:
                video.close()
            path = out / f"seed_{seed}.npz"
            np.savez_compressed(path, images_iphone=np.stack(images), state=np.stack(states),
                                action=np.stack(actions), timestamp=np.arange(len(states)) / env.control_hz)
            # Reload the actual file and execute stored actions, without oracle/GT control.
            # This catches shifted action rows and accidental recording of measured states
            # or post-rate-limit targets as the requested command.
            max_error = 0.0
            with np.load(path, allow_pickle=False) as recorded:
                assert recorded["images_iphone"].shape == (len(actions) + 1, 192, 256, 3)
                assert recorded["images_iphone"].dtype == np.uint8
                assert np.isfinite(recorded["state"]).all() and np.isfinite(recorded["action"]).all()
                env.reset(seed=seed, scenario={"joint_positions": start})
                for i, action in enumerate(recorded["action"]):
                    env.step_real_joint_targets(action)
                    error = float(np.max(np.abs(env.sim_state() - native_states[i + 1])))
                    max_error = max(max_error, error)
            if max_error > 1e-8:
                raise ValueError(f"Recorded command replay mismatch: {max_error} rad")
            audit = {"seed": seed, "source": CONTROLLER_SOURCE, "settle": settle,
                     "initial_privileged": initial["privileged_metrics"],
                     "history": tracker.history, "commands": ticks, "phase_log": ctrl.phase_log,
                     "oracle_params": vars(ctrl.p)}
            audit_path = out / f"seed_{seed}_audit.json"
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")
            row = {"seed": seed, "file": path.name, "sha256": sha256(path),
                   "audit_file": audit_path.name, "audit_sha256": sha256(audit_path),
                   "ticks": len(actions), "success": tracker.success, "held": tracker.held_ever,
                   "lifted": tracker.lifted, "phase_failed": tracker.phase_failed(),
                   "replay_max_joint_error_rad": max_error}
            manifest["episodes"].append(row)
            save_manifest()
            print(json.dumps(row), flush=True)
        manifest["status"] = "completed"
        manifest["successes"] = sum(e["success"] for e in manifest["episodes"])
        save_manifest()
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_manifest()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1000)
    args = parser.parse_args()
    collect(args.config, args.eval_config, args.out, args.count, args.seed_start)
