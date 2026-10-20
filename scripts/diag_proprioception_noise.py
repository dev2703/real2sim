"""Audit raw sensor noise and nominal timing without modifying observations or labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from real2sim.data.raw import RawDataset, sha256_file
from real2sim.policy.observation_variants import ObservationVariant, transform_observations


def summarize_joint_trace(q: np.ndarray, fps: float, window: int = 15) -> dict:
    """Descriptive arm-only metrics; retrospective spike tests do not become filters."""
    q = np.asarray(q, np.float64)
    if q.ndim != 2 or q.shape[1] != 5 or len(q) <= window:
        raise ValueError("need more than one window of five arm joints")
    if not np.isfinite(q).all() or fps <= 0:
        raise ValueError("finite joints and positive fps required")
    step = np.diff(q, axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(q, window, axis=0)
    # A low-motion proxy, not ground-truth stationary. Never selects by leader error.
    stationary = (np.ptp(windows, axis=2) <= .5).all(axis=1)
    if stationary.any():
        selected = windows[stationary]
        centered = selected - selected.mean(axis=2, keepdims=True)
        t = np.arange(window, dtype=float) - (window - 1) / 2
        slopes = (centered @ t) / np.dot(t, t)
        residual = centered - slopes[:, :, None] * t
        station_std = np.sqrt(np.mean(residual**2, axis=2))
        station_summary = {"median_detrended_std_deg": np.median(station_std, axis=0).tolist(),
                           "p95_detrended_std_deg": np.percentile(station_std, 95, axis=0).tolist(),
                           "rms_residual_deg": np.sqrt(np.mean(residual**2, axis=(0, 2))).tolist()}
    else:
        station_summary = None
    isolated_spike = (abs(q[1:-1] - .5 * (q[:-2] + q[2:])) > 1.0) & (abs(q[2:] - q[:-2]) < .2)
    lattice = step / (360 / 4095)
    frequencies, power = welch(q, fs=fps, nperseg=min(256, len(q)), axis=0, detrend="linear")
    _, increment_power = welch(step, fs=fps, nperseg=min(256, len(step)), axis=0, detrend="constant")
    # Frequency grids differ only for tiny episodes; use its own grid nevertheless.
    inc_frequency = np.fft.rfftfreq(min(256, len(step)), d=1 / fps)
    def fraction_above_five(freq, p):
        return np.divide(p[freq > 5].sum(axis=0), p.sum(axis=0),
                         out=np.zeros(5), where=p.sum(axis=0) > 0).tolist()
    return {"samples": len(q), "max_abs_step_deg": abs(step).max(axis=0).tolist(),
            "p99_abs_step_deg": np.percentile(abs(step), 99, axis=0).tolist(),
            "p95_abs_velocity_deg_s": np.percentile(abs(step) * fps, 95, axis=0).tolist(),
            "p95_abs_acceleration_deg_s2": np.percentile(abs(np.diff(step, axis=0)) * fps**2, 95, axis=0).tolist(),
            "p95_encoder_increment_lattice_residual_ticks": np.percentile(abs(lattice - np.round(lattice)), 95, axis=0).tolist(),
            "isolated_return_spikes_over_1deg": isolated_spike.sum(axis=0).tolist(),
            "power_fraction_above_5hz_position": fraction_above_five(frequencies, power),
            "power_fraction_above_5hz_increments": fraction_above_five(inc_frequency, increment_power),
            "low_motion_windows": int(stationary.sum()), "all_windows": len(stationary),
            "low_motion_noise": station_summary}


def audit(source: Path, variants: Path, out: Path, raw_root: Path | None = None) -> dict:
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((source / "manifest.json").read_text())
    preparation = json.loads((variants / "preparation.json").read_text())
    ds = RawDataset(raw_root or Path(manifest["source"]["root"]))
    train = preparation["split"]["train"]
    report = {"source_manifest_sha256": sha256_file(source / "manifest.json"),
              "variant_preparation_sha256": sha256_file(variants / "preparation.json"),
              "selection_episodes": train, "source_fps": ds.fps,
              "arm_names": ds.state_names[:5], "episodes": {},
              "definitions": {
                  "low_motion": "all five arm peak-to-peak ranges <=0.5 degree in trailing 15 raw samples; proxy, not proven stationary",
                  "stationary_noise": "linear-detrended residual in overlapping low-motion windows, includes quantization and real micro-motion",
                  "spikes": "retrospective return-spike test: >1 degree from neighbor average, neighbors <0.2 degree apart; diagnostic only",
                  "spectral_power": "Welch power above 5Hz; legitimate rapid motion may contribute, so this is not an identified noise fraction",
                  "timing": "nominal row timestamps, not synchronized camera/motor hardware acquisition times",
                  "selection": "all noise summaries and EMA descriptions use inner-training episodes only; no filters fitted on inner/historical validation"}}
    for ep in train:
        rows = ds.episode_rows(ep)
        q = ds.stack(rows, "observation.state")[:, :5]
        summary = summarize_joint_trace(q, ds.fps)
        dt = np.diff(rows["timestamp"].to_numpy(dtype=float))
        summary["nominal_timing"] = {"min_dt": float(dt.min()), "max_dt": float(dt.max()),
                                      "nonpositive_dt_count": int((dt <= 0).sum()),
                                      "nonuniform_dt_count": int((abs(dt - 1 / ds.fps) > 1e-3).sum())}
        report["episodes"][str(ep)] = summary
    entries = list(report["episodes"].values())
    stationary_entries = [e["low_motion_noise"] for e in entries if e["low_motion_noise"] is not None]
    report["summary"] = {
        "raw_samples": sum(e["samples"] for e in entries),
        "low_motion_windows": sum(e["low_motion_windows"] for e in entries),
        "low_motion_window_fraction": sum(e["low_motion_windows"] for e in entries) / sum(e["all_windows"] for e in entries),
        "max_abs_step_deg": np.max([e["max_abs_step_deg"] for e in entries], axis=0).tolist(),
        "isolated_return_spikes_over_1deg": np.sum([e["isolated_return_spikes_over_1deg"] for e in entries], axis=0).tolist(),
        "episode_median_stationary_rms_deg": np.median([e["rms_residual_deg"] for e in stationary_entries], axis=0).tolist() if stationary_entries else None,
        "episode_median_highfreq_position_power_fraction": np.median([e["power_fraction_above_5hz_position"] for e in entries], axis=0).tolist(),
        "episode_median_highfreq_increment_power_fraction": np.median([e["power_fraction_above_5hz_increments"] for e in entries], axis=0).tolist(),
        "nonuniform_dt_count": sum(e["nominal_timing"]["nonuniform_dt_count"] for e in entries)}
    with np.load(source / "arrays.npz") as loaded:
        arrays = {k: loaded[k] for k in loaded.files}
    mask = np.isin(arrays["episode_index"], train)
    raw_velocity = arrays["state"][mask, 9:12]
    report["velocity_ema"] = {}
    for alpha in (.5, .75):
        filtered = transform_observations(arrays, ObservationVariant("ema_velocity", ema_alpha=alpha))[mask, 9:12]
        report["velocity_ema"][str(alpha)] = {
            "raw_rms_m_s": np.sqrt(np.mean(raw_velocity**2, axis=0)).tolist(),
            "filtered_rms_m_s": np.sqrt(np.mean(filtered**2, axis=0)).tolist(),
            "rms_observation_change_m_s": np.sqrt(np.mean((filtered - raw_velocity)**2, axis=0)).tolist(),
            "theoretical_low_frequency_delay_s": (1 - alpha) / alpha / manifest["policy_hz"],
            "note": "observation velocities only; pose, rotations, gripper and actions unchanged"}
    camera_file = Path("results/audit/camera_quality.json")
    if camera_file.exists():
        camera = json.loads(camera_file.read_text())
        report["existing_camera_audit"] = {"path": str(camera_file), "sha256": sha256_file(camera_file),
                                           "note": "existing image-motion correlation; not a new landmark/hardware timing calibration",
                                           "keys": list(camera)}
        # Persist only selected episodes where schema contains episode-level rows.
        streams = camera.get("cameras", camera.get("streams", camera))
        if isinstance(streams, dict):
            selected = {}
            for key, stream in streams.items():
                if isinstance(stream, dict) and isinstance(stream.get("episodes"), dict):
                    selected[key] = {str(ep): {k: v for k, v in stream["episodes"].get(str(ep), {}).items()
                                              if "lag" in k or "timing" in k} for ep in train}
            report["existing_camera_audit"]["inner_train_lags"] = selected
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = report["summary"]
    text = ["# Proprioception noise audit", "",
            f"Inner-training recordings only: {len(train)} episodes, {summary['raw_samples']} raw rows.",
            f"Low-motion windows: {summary['low_motion_windows']} ({100 * summary['low_motion_window_fraction']:.1f}%). These windows overlap and are proxies for stationarity.",
            f"Maximum arm steps (degrees): {np.round(summary['max_abs_step_deg'], 4).tolist()}.",
            f"Isolated return spikes >1 degree: {summary['isolated_return_spikes_over_1deg']}.",
            f"Episode-median detrended low-motion RMS (degrees): {np.round(summary['episode_median_stationary_rms_deg'], 5).tolist()}.",
            f"Nonuniform nominal timestamp increments: {summary['nonuniform_dt_count']}.", "",
            "These metrics do not establish calibration or actual motor/camera acquisition timing. High-frequency power includes quantization and legitimate motion.",
            "EMA new-sample weights 0.5 / 0.75 introduce theoretical low-frequency delays of 100 / 33 ms at 10 Hz. Their utility must be judged by matched active/phase prediction and physical tracking, not by variance reduction alone.",
            "No raw sample, Cartesian target, gripper event or safety threshold was altered; the experiment variants change policy observation features only.", ""]
    (out / "review.md").write_text("\n".join(text))
    print(json.dumps(report["summary"], indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".cache/derived_cartesian"))
    parser.add_argument("--variants", type=Path, default=Path(".cache/experiments_20261016"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.source, args.variants, args.out, args.raw_root)
