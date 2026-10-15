"""Privileged recorded-motion replay under the candidate camera; not learned ACT."""

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.data.paired_approach import camera_parameters
from real2sim.evaluation.metrics import TrialTracker
from real2sim.policy.paired import file_sha256
from real2sim.sim.camera_fit import backproject_to_plane
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real


def main(out, selected_episodes=None, bowl_pixel=None, config=None):
    if out.exists():
        raise FileExistsError(out)
    cfg = config or Path("results/metrics/camera_refit_verified_20261014/camera_objects/configs/sim_so101.yaml")
    arrays = np.load(".cache/derived_joint/arrays.npz")
    prep = json.loads(Path(".cache/paired_approach_3988577/preparation.json").read_text())
    episodes = [e for e in prep["episodes"] if e["split"] == "val"]
    if selected_episodes is not None:
        if not set(selected_episodes) <= {e["episode"] for e in episodes}:
            raise ValueError("replay selects held-out episodes only")
        episodes = [e for e in episodes if e["episode"] in selected_episodes]
    if bowl_pixel is not None and len(episodes) != 1:
        raise ValueError("a hand-read bowl pixel requires exactly one selected episode")
    report = {"kind": "PRIVILEGED_RECORDED_MOTION_REPLAY_NOT_ACT", "config_sha256": file_sha256(cfg),
              "horizon": "up to first 30 seconds; same 10 Hz labels and original physics",
              "setup": "initial ball estimated independently per held-out episode from its first image",
              "caveats": ["Follower-next states are privileged diagnostic controls, never policy inputs or new labels.",
                          "Ball is dynamic after initialization; no forced attachment or subsequent pose correction.",
                          "No claim that an episode's grasp/release completes within 30 seconds.",
                          "This is a contact feasibility test, not a benchmark success rate."], "trials": []}
    with So101PickPlaceEnv(cfg, render_images=False) as env:
        env.reset(seed=0)
        rv, tv, K, _, _ = camera_parameters(env)
        bowl_xy = None if bowl_pixel is None else backproject_to_plane(
            [bowl_pixel], rv, tv, K, env.table_top - env.offset[2] + env.task["bowl"]["outer_height"])[0, :2]
        report["bowl_setup"] = {"rim_pixel": bowl_pixel,
                                "bowl_xy": None if bowl_xy is None else bowl_xy.tolist(),
                                "note": "Shared snapshot bowl unless an explicit per-episode image annotation is supplied"}
        for ep in episodes:
            pixel = (np.asarray(ep["first_pixel_policy"]) - [0, 24]) / .2
            xy = backproject_to_plane([pixel], rv, tv, K,
                                     env.table_top - env.offset[2] + env.task["ball"]["radius"])[0, :2]
            indices = np.flatnonzero(arrays["episode_index"] == ep["episode"])[:301]
            for mode in ("leader_command", "follower_next_state"):
                scenario = {"ball_xy": xy,
                    "joint_positions": np.clip(sim_from_real(env.mapping, arrays["state"][indices[0]]), *env.limits.T)}
                if bowl_xy is not None:
                    scenario["bowl_xy"] = bowl_xy
                env.reset(seed=0, scenario=scenario)
                initial_contacts = [c for c in env._contact_pairs()
                                    if "ball_geom" in (c["geom1"], c["geom2"])]
                tracker = TrialTracker(success_hold_ticks=int(env.task["episode"]["success_hold_s"] * env.control_hz),
                    approach_distance_m=.03, table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
                errors = []
                for t, nxt in zip(indices[:-1], indices[1:], strict=True):
                    target = arrays["action"][t] if mode == "leader_command" else arrays["state"][nxt]
                    _, _, _, truncated, info = env.step_real_joint_targets(target)
                    done = tracker.update(info["privileged_metrics"])
                    errors.append((env.real_state() - arrays["state"][nxt]).tolist())
                    if done or truncated:
                        break
                row = {"episode": ep["episode"], "mode": mode, "ball_xy": xy.tolist(),
                       "initial_ball_contacts": initial_contacts,
                       "invalid_initial_ball_overlap": any(c["distance_m"] < -.002 for c in initial_contacts),
                       "held_ever": tracker.held_ever, "lifted": tracker.lifted, "success": tracker.success,
                       "min_distance_m": tracker.min_ee_ball_distance_m, "phase_failed": tracker.phase_failed(),
                       "ticks": tracker.ticks, "joint_mae": np.mean(np.abs(errors), axis=0).tolist(),
                       "history": tracker.history}
                report["trials"].append(row)
    report["summary"] = {}
    for mode in ("leader_command", "follower_next_state"):
        rows = [r for r in report["trials"] if r["mode"] == mode]
        report["summary"][mode] = {"n": len(rows), "held": sum(r["held_ever"] for r in rows),
            "lifted": sum(r["lifted"] for r in rows), "successes": sum(r["success"] for r in rows),
            "invalid_initial_ball_overlap": sum(r["invalid_initial_ball_overlap"] for r in rows),
            "mean_closest_m": float(np.mean([r["min_distance_m"] for r in rows]))}
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("results/audit/camera_landmark_refit_float64_20261014/recorded_replay.json"))
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bowl-pixel", type=float, nargs=2,
                        help="Native-image rim centre for one selected episode; privileged setup only")
    args = parser.parse_args()
    main(args.out, args.episodes, args.bowl_pixel, args.config)
