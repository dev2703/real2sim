"""SCRIPTED REPLAY: compare recorded leader commands with observed follower motion.

Held-out stationary-ball approach prefixes only; no policy inference, fitting,
hardware access, label changes, or benchmark success claims.
"""

import json
from pathlib import Path

import numpy as np

from real2sim.geometry.so101_fk import So101FK
from real2sim.policy.paired import file_sha256
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

OUT = Path("results/audit/paired_command_tracking_3988577")
CONFIG = Path("results/metrics/geometry_3988577/configs/sim_so101.yaml")


def metrics(error):
    return {"mae": np.mean(abs(error), axis=0).tolist(),
            "p95_abs": np.quantile(abs(error), .95, axis=0).tolist()}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    arrays = np.load(".cache/derived_joint/arrays.npz")
    ticks = np.load(".cache/paired_approach_3988577/val/ticks.npy")
    preparation = json.loads(Path(".cache/paired_approach_3988577/preparation.json").read_text())
    episodes = [r for r in preparation["episodes"] if r["split"] == "val"]
    env = So101PickPlaceEnv(CONFIG, render_images=False)
    fk = So101FK(mapping=env.mapping)
    report = {"kind": "SCRIPTED_REPLAY_COMMAND_VS_FOLLOWER", "config_sha256": file_sha256(CONFIG),
              "arrays_sha256": file_sha256(Path(".cache/derived_joint/arrays.npz")),
              "units": ["deg"] * 5 + ["percent"], "episodes": [],
              "note": "No learned inference or success claim. Fixed provisional mapping/dynamics. "
                      "Compare post-step simulated state to recorded next tick; same-tick gap also saved. "
                      "Follower-next-state replay is a privileged tracking diagnostic, never training labels. "
                      "FK distances use the model, not measured physical tool poses. "
                      "Ball starts at image-estimated episode position; only pre-contact prefixes replayed."}
    pooled = {mode: [] for mode in ("leader_command", "follower_next_state")}
    try:
        for ep in episodes:
            selected = ticks[arrays["episode_index"][ticks] == ep["episode"]]
            selected = selected[arrays["episode_index"][selected + 1] == ep["episode"]]
            state = arrays["state"][selected]
            successor = arrays["state"][selected + 1]
            action = arrays["action"][selected]
            moving = np.max(abs(state[:, :5] - state[0, :5]), axis=1) > 5
            reference_xyz = fk.fk_real(successor)[:, :3, 3]
            q0 = sim_from_real(env.mapping, state[0])
            row = {"episode": ep["episode"], "ticks": len(selected), "moving_ticks": int(moving.sum()),
                   "leader_minus_follower_same_tick": metrics(action - state),
                   "leader_minus_follower_next_tick": metrics(action - successor), "modes": {}}
            for mode, commands in (("leader_command", action), ("follower_next_state", successor)):
                env.reset(seed=0, scenario={"joint_positions": np.clip(q0, *env.limits.T),
                                           "ball_xy": ep["ball_xy_estimated"]})
                actual, xyz, clips, rates = [], [], [], []
                for command in commands:
                    _, _, _, truncated, info = env.step_real_joint_targets(command)
                    if truncated:
                        raise RuntimeError("approach prefix exceeded fixed episode cap")
                    actual.append(env.real_state())
                    xyz.append(env.ee_pose()[:3, 3])
                    clips.append(info["control"]["limit_clipped"])
                    rates.append(info["control"]["rate_limited"])
                actual, xyz = np.asarray(actual), np.asarray(xyz)
                error = actual - successor
                distance = np.linalg.norm(xyz - reference_xyz, axis=1)
                row["modes"][mode] = {"sim_minus_follower_next": metrics(error),
                    "moving_sim_minus_follower_next": metrics(error[moving]),
                    "fk_distance_mean_m": float(distance.mean()),
                    "fk_distance_p95_m": float(np.quantile(distance, .95)),
                    "limit_clip_fraction": np.mean(clips, axis=0).tolist(),
                    "rate_limit_fraction": np.mean(rates, axis=0).tolist()}
                pooled[mode].append((error, distance, moving))
                np.savez_compressed(OUT / f"ep{ep['episode']}_{mode}.npz", ticks=selected,
                    command=commands, recorded_state=state, recorded_next=successor,
                    simulated_next=actual, fk_distance_m=distance, moving=moving,
                    limit_clipped=clips, rate_limited=rates)
            report["episodes"].append(row)
        report["pooled"] = {}
        for mode, values in pooled.items():
            error = np.concatenate([v[0] for v in values])
            distance = np.concatenate([v[1] for v in values])
            moving = np.concatenate([v[2] for v in values])
            report["pooled"][mode] = {"ticks": len(error), "sim_minus_follower_next": metrics(error),
                "moving_sim_minus_follower_next": metrics(error[moving]),
                "fk_distance_mean_m": float(distance.mean()),
                "fk_distance_p95_m": float(np.quantile(distance, .95))}
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["pooled"], indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
