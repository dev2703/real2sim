"""Read-only recording audit and isolated calibration-range sensitivity config.

No physical hardware access, calibration writes, label edits or model fitting.
Local calibration files are evidence, not proof of recording-time calibration.
"""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from real2sim.data.raw import RawDataset
from real2sim.policy.paired import file_sha256
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

OUT = Path("results/audit/control_contract_20261013")
BASE = Path("results/metrics/geometry_3988577/configs/sim_so101.yaml")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def normalization_evidence(ds, split, calibrations):
    """Test numeric encoder lattices, without claiming physical zero/sign calibration."""
    result = {}
    for field, calibration in calibrations.items():
        values = np.concatenate([ds.stack(ds.episode_rows(ep), field) for ep in split])
        columns = []
        for j, name in enumerate(JOINTS):
            lo, hi = calibration[name]["range_min"], calibration[name]["range_max"]
            if j < 5:
                degree_ticks = values[:, j] * 4095 / 360 + (lo + hi) / 2
                percent_ticks = (values[:, j] + 100) / 200 * (hi - lo) + lo
                columns.append({"joint": name,
                    "degrees_tick_residual_p95": float(np.quantile(abs(degree_ticks - np.rint(degree_ticks)), .95)),
                    "range_m100_100_tick_residual_p95": float(np.quantile(abs(percent_ticks - np.rint(percent_ticks)), .95))})
            else:
                ticks = values[:, j] / 100 * (hi - lo) + lo
                columns.append({"joint": name,
                    "percent_tick_residual_p95": float(np.quantile(abs(ticks - np.rint(ticks)), .95))})
        result[field] = columns
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    source = Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v1"
    calroot = Path.home() / ".cache/huggingface/lerobot/calibration"
    calpath = calroot / "robots/so_follower/aligned_follower.json"
    leaderpath = calroot / "teleoperators/so_leader/aligned_leader.json"
    cal = json.loads(calpath.read_text())
    follower_ranges = np.array([[-(cal[j]["range_max"] - cal[j]["range_min"]) * 180 / 4095,
                                (cal[j]["range_max"] - cal[j]["range_min"]) * 180 / 4095]
                               for j in JOINTS[:5]] + [[0, 100]])
    ds = RawDataset(source)
    manifest = json.loads(Path(".cache/derived_joint/manifest.json").read_text())
    env = So101PickPlaceEnv(BASE, render_images=False)
    native_limits = env.limits.copy()
    sim_degree_limits = np.rad2deg(native_limits[:5])
    report = {"kind": "CONTROL_CONTRACT_AUDIT", "source": manifest["source"],
              "split": manifest["split"], "joint_order": JOINTS, "raw_fps": ds.fps,
              "native_arm_limits_deg": sim_degree_limits.tolist(),
              "local_follower_calibration_ranges_deg_or_percent": follower_ranges.tolist(),
              "provenance": {str(p): file_sha256(p) for p in (BASE, calpath, leaderpath)},
              "limitations": ["Local calibration files may differ from recording-time files.",
                  "Calibration spans establish numeric normalization endpoints, not physical MJCF zero alignment.",
                  "Position-error lag minima are descriptive and confounded by stalls and target holds.",
                  "No schema, labels, offset/sign, primary geometry or hardware settings are changed."],
              "splits": {}}
    report["normalization_encoder_lattice"] = normalization_evidence(
        ds, manifest["split"]["train"] + manifest["split"]["val"],
        {"observation.state": cal, "action": json.loads(leaderpath.read_text())})
    all_raw = {}
    for split in ("train", "val"):
        eps = []
        by_lag = [[] for _ in range(16)]
        moving_by_lag = [[] for _ in range(16)]
        gaps, deltas, states, actions = [], [], [], []
        for ep in manifest["split"][split]:
            rows = ds.episode_rows(ep)
            s, a = ds.stack(rows, "observation.state"), ds.stack(rows, "action")
            ts = rows["timestamp"].to_numpy(float)
            all_raw[ep] = (s, a)
            states.append(s)
            actions.append(a)
            gaps.append(a - s)
            deltas.append(np.diff(s, axis=0))
            # Identical reference ticks at every lag, no cross-episode alignment.
            ref = np.arange(1, len(s) - 15)
            moving = (abs(s[ref] - s[ref - 1]) > .2) | (abs(a[ref] - a[ref - 1]) > .2)
            for lag in range(16):
                err = abs(a[ref] - s[ref + lag])
                by_lag[lag].append(err)
                moving_by_lag[lag].append(np.where(moving, err, np.nan))
            eps.append({"episode": ep, "rows": len(s),
                        "timestamp_delta_median_s": float(np.median(np.diff(ts))),
                        "timestamp_delta_max_s": float(np.max(np.diff(ts))),
                        "shoulder_gap_p95_deg": float(np.quantile(abs(a[:, 1] - s[:, 1]), .95))})
        s, a, gap, delta = map(np.concatenate, (states, actions, gaps, deltas))
        lag_mae = np.array([np.concatenate(v).mean(axis=0) for v in by_lag])
        moving_mae = np.array([np.nanmean(np.concatenate(v), axis=0) for v in moving_by_lag])
        report["splits"][split] = {"episodes": eps, "rows": len(s),
            "state_min": s.min(axis=0).tolist(), "state_max": s.max(axis=0).tolist(),
            "action_min": a.min(axis=0).tolist(), "action_max": a.max(axis=0).tolist(),
            "state_outside_native_arm_fraction": np.mean((s[:, :5] < sim_degree_limits[:, 0]) |
                                                          (s[:, :5] > sim_degree_limits[:, 1]), axis=0).tolist(),
            "state_outside_local_calibration_fraction": np.mean((s < follower_ranges[:, 0] - .1) |
                                                                 (s > follower_ranges[:, 1] + .1), axis=0).tolist(),
            "same_tick_action_state_mae": abs(gap).mean(axis=0).tolist(),
            "same_tick_action_state_p95": np.quantile(abs(gap), .95, axis=0).tolist(),
            "action_state_gap_over_5_fraction": np.mean(abs(gap) > 5.1, axis=0).tolist(),
            "observed_step_over_5_fraction": np.mean(abs(delta) > 5.1, axis=0).tolist(),
            "lag_mae": lag_mae.tolist(), "moving_lag_mae": moving_mae.tolist(),
            "best_lag_frames_descriptive": np.argmin(lag_mae, axis=0).tolist(),
            "best_moving_lag_frames_descriptive": np.argmin(moving_mae, axis=0).tolist()}
    # Exact derivation audit: verify 30->10 Hz subsampling has no state/action row shift.
    derived = np.load(".cache/derived_joint/arrays.npz")
    errors = []
    for ep, (s, a) in all_raw.items():
        ix = np.flatnonzero(derived["episode_index"] == ep)
        frames = derived["frame_index"][ix].astype(int)
        errors.append(max(float(np.max(abs(derived["state"][ix] - s[frames]))),
                          float(np.max(abs(derived["action"][ix] - a[frames])))))
    report["derived_raw_max_abs_difference"] = max(errors)
    best = report["splits"]["train"]["best_moving_lag_frames_descriptive"]
    report["train_selected_lags_frames"] = best
    report["val_mae_at_train_selected_lag"] = [report["splits"]["val"]["moving_lag_mae"][lag][j]
                                               for j, lag in enumerate(best)]
    # One-factor sensitivity: only shoulder lower stop/control bound, using local
    # follower calibration endpoint. No offset fit and no held-out tuning.
    config = yaml.safe_load(BASE.read_text())
    modelpath = Path(config["model"])
    tree = ET.parse(modelpath)
    lower = float(np.deg2rad(follower_ranges[1, 0]))
    for xpath in (".//joint[@name='shoulder_lift']", "./actuator/position[@name='shoulder_lift']"):
        node = tree.find(xpath)
        attr = "range" if node.tag == "joint" else "ctrlrange"
        bounds = [float(x) for x in node.get(attr).split()]
        bounds[0] = lower
        node.set(attr, " ".join(map(str, bounds)))
    modeldest = OUT / "so101_shoulder_range_diagnostic.xml"
    tree.write(modeldest)
    (OUT / "assets").symlink_to(os.path.relpath((modelpath.parent / "assets").resolve(), OUT.resolve()),
                                target_is_directory=True)
    config["model"] = str(modeldest.resolve())
    limits = native_limits.copy()
    limits[1, 0] = lower
    config["initial_qpos"] = np.clip(sim_from_real(env.mapping, config["initial_qpos_real_deg"]),
                                      limits[:, 0], limits[:, 1]).tolist()
    config["diagnostic_provenance"] = {"kind": "LOCAL_CALIBRATION_SHOULDER_RANGE_SENSITIVITY",
        "shoulder_lower_deg": float(follower_ranges[1, 0]),
        "note": "Not adopted as calibrated geometry; only lower stop and matching control bound changed. "
                "Initial native pose now admits the same requested dataset-unit start."}
    (OUT / "sim_shoulder_range.yaml").write_text(yaml.safe_dump(config))
    report["range_sensitivity_config"] = str(OUT / "sim_shoulder_range.yaml")
    report["range_sensitivity_model_sha256"] = file_sha256(modeldest)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    env.close()
    print(json.dumps({"parity": max(errors), "train_lags": best,
                      "range_config": report["range_sensitivity_config"]}))


if __name__ == "__main__":
    main()
