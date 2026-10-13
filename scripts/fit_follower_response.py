"""Fit on training recordings, then score frozen response on held-out replay.

This is system identification, not ACT training or a calibrated contact model.
All outputs are isolated, and existing outputs are never overwritten.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.data.raw import RawDataset
from real2sim.geometry.so101_fk import So101FK
from real2sim.policy.paired import file_sha256
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.follower_response import FollowerResponse, fit_joint

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONFIG = Path("results/metrics/geometry_3988577/configs/sim_so101.yaml")


def candidate_score(episodes, joint, gain, inertia, delay, lower, upper):
    """Episode-balanced free-running error; no measured-state feedback after reset."""
    scores = []
    for state, action in episodes:
        q, velocity, error = float(state[0, joint]), 0.0, 0.0
        for t in range(len(state) - 1):
            goal = action[max(0, t - delay), joint]
            step = min(5., max(-5., gain * min(5., max(-5., goal - q)) + inertia * velocity))
            next_q = min(upper, max(lower, q + step))
            velocity, q = next_q - q, next_q
            error += abs(q - state[t + 1, joint])
        scores.append(error / (len(state) - 1))
    return float(np.mean(scores))


def metrics(error):
    return {"mae_per_joint": np.mean(abs(error), axis=0).tolist(),
            "p95_abs_per_joint": np.quantile(abs(error), .95, axis=0).tolist()}


def main(out, fit_scope="full"):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(".cache/derived_joint/manifest.json").read_text())
    train, val = manifest["split"]["train"], manifest["split"]["val"]
    if set(train) & set(val):
        raise ValueError("train/validation overlap")
    # Fixed inner split of training episodes; primary held-out set never selects parameters.
    tune = train[3::4]
    fitting = [e for e in train if e not in tune]
    calpath = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/aligned_follower.json"
    cal = json.loads(calpath.read_text())
    span = [(cal[j]["range_max"] - cal[j]["range_min"]) * 180 / 4095 for j in JOINTS[:5]]
    lower, upper = [-x for x in span] + [0.], span + [100.]
    ds = RawDataset(Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v1")
    if ds.fps != 30:
        raise ValueError("this diagnostic requires the frozen 30 Hz recording clock")

    def read(episodes):
        result = {}
        for e in episodes:
            rows = ds.episode_rows(e)
            result[e] = (ds.stack(rows, "observation.state"), ds.stack(rows, "action"))
        return result

    training = read(train)
    extra_provenance = {}
    if fit_scope == "approach":
        # Restrict using previously cached training-only stationary-ball prefixes.
        # No held-out range, outcome or trace is used to choose hyperparameters.
        derived = np.load(".cache/derived_joint/arrays.npz")
        eligible = np.load(".cache/paired_approach_3988577/train/ticks.npy")
        pair_root = Path(".cache/paired_approach_3988577/train")
        pair_manifest = json.loads((pair_root / "manifest.json").read_text())
        if pair_manifest["split_name"] != "train" or pair_manifest["derived_manifest"] != manifest:
            raise ValueError("approach cache provenance does not match training data")
        if file_sha256(pair_root / "ticks.npy") != pair_manifest["sha256"]["ticks.npy"]:
            raise ValueError("approach tick cache checksum mismatch")
        if not set(derived["episode_index"][eligible]) <= set(train):
            raise ValueError("held-out ticks in approach training cache")
        extra_provenance = {str(pair_root / name): file_sha256(pair_root / name)
                            for name in ("manifest.json", "ticks.npy")}
        for ep in train:
            chosen = eligible[derived["episode_index"][eligible] == ep]
            if not len(chosen):
                raise ValueError(f"no training approach prefix for episode {ep}")
            stop = int(derived["frame_index"][chosen[-1]]) + 1
            s, a = training[ep]
            training[ep] = s[:stop], a[:stop]
    search, selected, first_order = [], [], []
    for j, name in enumerate(JOINTS):
        choices = []
        for momentum in (False, True):
            for delay in (0, 2, 4, 6):
                gain, inertia = fit_joint([training[e] for e in fitting], j, delay, momentum=momentum)
                score = candidate_score([training[e] for e in tune], j, gain, inertia, delay, lower[j], upper[j])
                choices.append(dict(momentum=momentum, delay=delay, gain=gain, inertia=inertia,
                                    inner_episode_mean_free_run_mae=score))
        best = min(choices, key=lambda c: c["inner_episode_mean_free_run_mae"])
        gain, inertia = fit_joint(list(training.values()), j, best["delay"], momentum=best["momentum"])
        selected.append(dict(gain=gain, inertia=inertia, delay=best["delay"]))
        first_order.append(fit_joint(list(training.values()), j, 0, momentum=False)[0])
        search.append({"joint": name, "candidates": choices, "selected": best, "refit_all_train": selected[-1]})
        print(f"fit {name}: gain={gain:.4f} inertia={inertia:.4f} delay={best['delay']}", flush=True)
    model = FollowerResponse(**{k: [r[k] for r in selected] for k in ("gain", "inertia", "delay")},
                             lower=lower, upper=upper)
    first = FollowerResponse(first_order, [0.] * 6, [0] * 6, lower, upper)
    model_file = out / "model.json"
    model_file.write_text(json.dumps({"kind": "OFFLINE_FOLLOWER_RESPONSE", "model": model.to_dict(),
        "fit_scope": fit_scope, "fit_episodes": train,
        "fit_raw_rows_per_episode": {str(e): len(s) for e, (s, _) in training.items()},
        "inner_fit_episodes": fitting, "inner_selection_episodes": tune,
        "source": manifest["source"], "primary_split": manifest["split"], "search": search,
        "first_order_baseline": first.to_dict(),
        "provenance": {**{str(p): file_sha256(p) for p in (calpath, CONFIG, Path('.cache/derived_joint/manifest.json'))},
                       **extra_provenance},
        "bounds_note": "Numeric local calibration endpoints, not verified physical MJCF joint limits.",
        "caps_note": "5 dataset units from documented recording safety cap; velocity capped at 5 units/raw tick.",
        "limitations": "No object/contact/force input. Nominal timestamp clock, no wall-clock motor timestamps. "
                       "Not an ACT adapter; no physical calibration or task-success claim."}, indent=2) + "\n")
    frozen_sha = file_sha256(model_file)
    print("Model frozen; now loading held-out episodes", flush=True)
    heldout = read(val)
    arrays = np.load(".cache/derived_joint/arrays.npz")
    ticks = np.load(".cache/paired_approach_3988577/val/ticks.npy")
    prep = json.loads(Path(".cache/paired_approach_3988577/preparation.json").read_text())
    ball_xy = {r["episode"]: r["ball_xy_estimated"] for r in prep["episodes"] if r["split"] == "val"}
    env = So101PickPlaceEnv(CONFIG, render_images=False)
    fk = So101FK(mapping=env.mapping)
    full_errors = {k: [] for k in ("initial_hold", "instant_target", "first_order", "selected", "selected_10hz_hold")}
    prefix_errors = {k: [] for k in ("response_raw30", "response_held10", "mujoco_response_held10")}
    report = {"kind": "HELD_OUT_FOLLOWER_RESPONSE_REPLAY", "model_sha256": frozen_sha, "fit_scope": fit_scope,
              "joint_order": JOINTS, "episodes": [],
              "note": "Free running from recorded initial state only; no later observed-state feedback. "
                      "Raw30 sees all recorded commands; held10 repeats every third command three times. "
                      "MuJoCo replays predicted held10 positions with unchanged physics/limits. "
                      "This is a scripted model diagnostic, never ACT inference or success."}
    try:
        for ep, (s, a) in heldout.items():
            held_actions = a[(np.arange(len(a)) // 3) * 3]
            pred = model.rollout(s[0], a[:-1])
            pred10 = model.rollout(s[0], held_actions[:-1])
            pred_first = first.rollout(s[0], a[:-1])
            series = {"initial_hold": np.repeat(s[:1], len(s) - 1, axis=0),
                      "instant_target": np.clip(a[:-1], lower, upper), "first_order": pred_first[1:],
                      "selected": pred[1:], "selected_10hz_hold": pred10[1:]}
            ep_result = {"episode": ep, "full_episode": {}}
            for mode, trajectory in series.items():
                error = trajectory - s[1:]
                full_errors[mode].append(error)
                ep_result["full_episode"][mode] = metrics(error)
            selected_ticks = ticks[arrays["episode_index"][ticks] == ep]
            frames = arrays["frame_index"][selected_ticks].astype(int)
            successor = s[frames + 3]
            ref_xyz = fk.fk_real(successor)[:, :3, 3]
            q0 = np.clip(sim_from_real(env.mapping, s[0]), *env.limits.T)
            env.reset(seed=0, scenario={"joint_positions": q0, "ball_xy": ball_xy[ep]})
            sim = []
            for frame in frames:
                _, _, _, truncated, _ = env.step_real_joint_targets(pred10[frame + 3])
                if truncated:
                    raise RuntimeError("prefix exceeded fixed episode cap")
                sim.append(env.real_state())
            sim = np.asarray(sim)
            ep_result["approach_prefix"] = {}
            for mode, trajectory in (("response_raw30", pred[frames + 3]),
                                     ("response_held10", pred10[frames + 3]),
                                     ("mujoco_response_held10", sim)):
                error = trajectory - successor
                distance = np.linalg.norm(fk.fk_real(trajectory)[:, :3, 3] - ref_xyz, axis=1)
                prefix_errors[mode].append((error, distance))
                ep_result["approach_prefix"][mode] = {**metrics(error), "fk_mean_m": float(distance.mean()),
                                                       "fk_p95_m": float(np.quantile(distance, .95))}
            np.savez_compressed(out / f"ep{ep}_traces.npz", recorded=s, action=a, predicted_raw30=pred,
                predicted_held10=pred10, predicted_first_order=pred_first,
                prefix_frames=frames, prefix_simulated=sim)
            report["episodes"].append(ep_result)
            print(f"held-out ep{ep} complete", flush=True)
        report["full_episode_pooled"] = {k: metrics(np.concatenate(v)) for k, v in full_errors.items()}
        report["approach_pooled"] = {}
        for mode, values in prefix_errors.items():
            error = np.concatenate([v[0] for v in values])
            distance = np.concatenate([v[1] for v in values])
            report["approach_pooled"][mode] = {**metrics(error), "ticks": len(error),
                "fk_mean_m": float(distance.mean()), "fk_p95_m": float(np.quantile(distance, .95))}
        if file_sha256(model_file) != frozen_sha:
            raise RuntimeError("model changed during held-out evaluation")
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["approach_pooled"], indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/audit/follower_response_20261013"))
    parser.add_argument("--fit-scope", choices=["full", "approach"], default="full")
    args = parser.parse_args()
    main(args.out, args.fit_scope)
