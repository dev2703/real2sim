"""Canonical dataset audit (plan §3.1, §3.2, M0 step 3–5).

    uv run python -m real2sim.data.audit --repo-id bottowork/so101_act_pickplace_v1 \
        --revision <sha> --local-root <dir> --out results/audit

Produces, from the actual files (never from assumptions):

* ``dataset_report.json`` — provenance, features, per-episode table, timing audit,
  action-vs-state audit, file manifest, split manifest summary.
* ``camera_quality.json`` — per image key and per episode: decode rate, black/blank/frozen/
  duplicate counts, sharpness, ball visibility, timestamp coverage, image/joint motion lag,
  explicit USABLE / PARTIALLY_USABLE / UNUSABLE verdicts with reasons and thresholds.
* ``camera_montage.html`` + ``montage/`` — first/middle/last decoded frames per episode per
  camera and the lowest-sharpness frames.
* ``split.json`` — deterministic whole-episode split with SHA-256.
* ``episodes.csv`` — the per-episode table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from real2sim.data.montage import save_thumb, write_montage_html
from real2sim.data.quality import (
    ANALYSIS_SIZE,
    THRESHOLDS,
    FrameMetrics,
    label_stream,
    motion_lag,
    summarize_episode,
)
from real2sim.data.raw import RawDataset, resolve_dataset_root
from real2sim.data.split import make_episode_split

ARM_JOINT_SLICE = slice(0, 5)  # five arm joints; gripper excluded from motion-lag estimate


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _short(key: str) -> str:
    return key.replace("observation.images.", "")


# ----------------------------------------------------------------------------- tables
def audit_episode_table(ds: RawDataset) -> list[dict]:
    rows = []
    for _, r in ds.episodes.iterrows():
        ep = int(r["episode_index"])
        n = int(r["length"])
        row = {
            "episode_index": ep,
            "length": n,
            "duration_s": round(n / ds.fps, 3),
            "task": (list(r["tasks"])[0] if "tasks" in r and r["tasks"] is not None else None),
        }
        for key in ds.image_keys:
            f, t = float(r[f"videos/{key}/from_timestamp"]), float(r[f"videos/{key}/to_timestamp"])
            expected_from_table = int(round((t - f) * ds.fps))
            row[f"{_short(key)}_file"] = int(r[f"videos/{key}/file_index"])
            row[f"{_short(key)}_from_ts"] = round(f, 6)
            row[f"{_short(key)}_to_ts"] = round(t, 6)
            row[f"{_short(key)}_table_frame_deficit"] = expected_from_table - n
        rows.append(row)
    return rows


def audit_timing(ds: RawDataset) -> dict:
    out: dict = {"episodes": {}, "non_uniform_dt_episodes": [], "frame_index_anomalies": []}
    nominal = 1.0 / ds.fps
    for ep in ds.episode_indices:
        g = ds.episode_rows(ep)
        ts = g["timestamp"].to_numpy(dtype=np.float64)
        fi = g["frame_index"].to_numpy()
        d = np.diff(ts) if len(ts) > 1 else np.array([nominal])
        uniform = bool(np.allclose(d, nominal, atol=1e-3))
        contiguous = bool(fi[0] == 0 and np.all(np.diff(fi) == 1))
        out["episodes"][str(ep)] = {
            "rows": int(len(g)),
            "t_first": float(ts[0]),
            "t_last": float(ts[-1]),
            "dt_min": float(d.min()),
            "dt_max": float(d.max()),
            "uniform": uniform,
            "frame_index_contiguous": contiguous,
            "rows_match_length": bool(len(g) == ds.episode_length(ep)),
        }
        if not uniform:
            out["non_uniform_dt_episodes"].append(ep)
        if not contiguous:
            out["frame_index_anomalies"].append(ep)
    return out


def audit_action_state(ds: RawDataset, max_lag: int = 45) -> dict:
    names = ds.state_names
    A = ds.stack(ds.data, "action")
    S = ds.stack(ds.data, "observation.state")
    d = np.abs(A - S)
    per_joint = {}
    eps = [(ds.stack(g, "action"), ds.stack(g, "observation.state")) for g in (ds.episode_rows(e) for e in ds.episode_indices)]
    # within-episode per-frame step (concatenated diff would count episode boundaries)
    within_step = np.concatenate([np.abs(np.diff(s, axis=0)) for _, s in eps if len(s) > 1], axis=0)
    for j, n in enumerate(names):
        best = (0, float("inf"))
        for k in range(0, max_lag + 1):
            errs = [np.abs(a[: len(a) - k, j] - s[k:, j]).mean() for a, s in eps if len(a) > k + 1]
            m = float(np.mean(errs)) if errs else float("inf")
            if m < best[1]:
                best = (k, m)
        per_joint[n] = {
            "state_min": float(S[:, j].min()),
            "state_max": float(S[:, j].max()),
            "action_min": float(A[:, j].min()),
            "action_max": float(A[:, j].max()),
            "state_mean": float(S[:, j].mean()),
            "action_mean": float(A[:, j].mean()),
            "mean_offset_action_minus_state": float((A[:, j] - S[:, j]).mean()),
            "abs_diff_median": float(np.median(d[:, j])),
            "abs_diff_p95": float(np.percentile(d[:, j], 95)),
            "abs_diff_max": float(d[:, j].max()),
            "best_lag_frames": int(best[0]),
            "residual_at_best_lag": float(best[1]),
            "state_max_step_per_frame_within_episode": float(within_step[:, j].max()),
            "state_p99_step_per_frame_within_episode": float(np.percentile(within_step[:, j], 99)),
            "action_at_global_min_fraction": float(np.isclose(A[:, j], A[:, j].min(), atol=0.01).mean()),
            "action_at_global_max_fraction": float(np.isclose(A[:, j], A[:, j].max(), atol=0.01).mean()),
        }
    return {
        "joint_names": names,
        "units_assumed": "degrees for joints, percent for gripper (VERIFY with recording config)",
        "n_rows": int(len(A)),
        "per_joint": per_joint,
        "note": (
            "action = leader goal positions; state = follower measured positions. "
            "Lag/offset are inferred from the data, not asserted."
        ),
    }


# ----------------------------------------------------------------------------- cameras
def audit_camera(
    ds: RawDataset,
    key: str,
    episodes: list[int],
    stride: int,
    out_dir: Path,
    max_worst: int = 10,
) -> tuple[dict, dict[int, dict[str, str]], list[dict]]:
    thumbs: dict[int, dict[str, str]] = {}
    worst: list[tuple[float, int, int, np.ndarray]] = []
    per_ep_raw: dict[int, dict] = {}
    montage_dir = out_dir / "montage" / _short(key)
    w, h = ANALYSIS_SIZE

    for ep in episodes:
        n_expected = ds.episode_length(ep)
        seg = ds.video_segment(key, ep)
        table_deficit = int(round((seg.to_ts - seg.from_ts) * ds.fps)) - n_expected
        m = FrameMetrics()
        failures = 0
        error = None
        want = {0: "first", (n_expected // 2) // stride * stride: "mid", ((n_expected - 1) // stride) * stride: "last"}
        saved: dict[str, str] = {}
        last_rgb: np.ndarray | None = None
        last_pos = -1
        t0 = time.time()
        if not seg.exists:
            error = f"video file missing: {seg.path}"
        else:
            try:
                for pos, pts, frame in ds.iter_frames(key, ep, stride=stride):
                    try:
                        rgb = frame.reformat(width=w, height=h, format="rgb24").to_ndarray()
                    except Exception as e:  # noqa: BLE001 - count and continue
                        failures += 1
                        if error is None:
                            error = f"frame {pos}: {e}"
                        continue
                    m.add(rgb, pts)
                    last_rgb, last_pos = rgb, pos
                    if pos in want and want[pos] not in saved:
                        name = want[pos]
                        p = montage_dir / f"ep{ep:03d}_{name}.jpg"
                        save_thumb(rgb, p, f"ep{ep} f{pos} {name}")
                        saved[name] = str(p.relative_to(out_dir))
                    sharp = m.sharpness[-1]
                    if len(worst) < max_worst:
                        worst.append((sharp, ep, pos, rgb))
                    else:
                        imax = max(range(len(worst)), key=lambda i: worst[i][0])
                        if sharp < worst[imax][0]:
                            worst[imax] = (sharp, ep, pos, rgb)
            except Exception as e:  # noqa: BLE001 - whole-episode decode failure
                failures += 1
                error = f"decode error: {e}"
        if "last" not in saved and last_rgb is not None:
            p = montage_dir / f"ep{ep:03d}_last.jpg"
            save_thumb(last_rgb, p, f"ep{ep} f{last_pos} last(decoded)")
            saved["last"] = str(p.relative_to(out_dir))
        thumbs[ep] = saved

        # image/joint motion lag (decoded-frame units * stride)
        lag, corr = None, float("nan")
        if len(m) > 0:
            g = ds.episode_rows(ep)
            q = ds.stack(g, "observation.state")[:, ARM_JOINT_SLICE]
            speed = np.r_[np.nan, np.abs(np.diff(q, axis=0)).sum(axis=1)][::stride][: len(m)]
            max_lag = max(2, min(30, len(m) // 8))
            lag, corr = motion_lag(np.asarray(m.diff_prev), speed, max_lag=max_lag)
            if lag is not None:
                lag *= stride
        per_ep_raw[ep] = {
            "metrics": m,
            "n_expected": n_expected,
            "failures": failures,
            "table_deficit": table_deficit,
            "decoded_minus_expected": (len(m) * stride - n_expected) if stride == 1 else None,
            "lag": lag,
            "corr": corr,
            "error": error,
            "seconds": round(time.time() - t0, 2),
        }
        print(
            f"  {key} ep{ep:03d}: decoded {len(m)}/{n_expected} (stride {stride}) "
            f"deficit {table_deficit:+d} lag {lag} fails {failures} [{per_ep_raw[ep]['seconds']}s]",
            file=sys.stderr,
        )

    lags = [v["lag"] for v in per_ep_raw.values() if v["lag"] is not None]
    lag_median = float(np.median(lags)) if lags else None
    episodes_out: dict[int, dict] = {}
    for ep, v in per_ep_raw.items():
        s = summarize_episode(
            v["metrics"],
            n_expected=v["n_expected"] // stride + (1 if v["n_expected"] % stride else 0),
            decode_failures=v["failures"],
            video_frame_deficit=v["table_deficit"],
            timing_lag_frames=v["lag"],
            timing_lag_median=lag_median,
        )
        s["frames_expected_raw"] = v["n_expected"]
        s["decoded_minus_expected"] = v["decoded_minus_expected"]
        s["timing_lag_corr"] = None if np.isnan(v["corr"]) else round(float(v["corr"]), 3)
        if v["error"]:
            s["error"] = v["error"]
            if "missing" in v["error"] or len(v["metrics"]) == 0:
                s["label"] = "UNUSABLE"
                s["reasons"] = [v["error"]] + s.get("reasons", [])
        episodes_out[ep] = s

    label, reasons = label_stream(episodes_out)
    frames_expected = sum(v["n_expected"] for v in per_ep_raw.values())
    frames_decoded = sum(len(v["metrics"]) * stride for v in per_ep_raw.values())
    stream = {
        "label": label,
        "reasons": reasons,
        "image_info": ds.image_info(key),
        "n_episodes": len(episodes_out),
        "frames_expected": frames_expected,
        "frames_decoded": min(frames_decoded, frames_expected) if stride > 1 else frames_decoded,
        "decode_rate": round(min(frames_decoded, frames_expected) / max(frames_expected, 1), 5),
        "video_frame_deficit_total": int(sum(v["table_deficit"] for v in per_ep_raw.values())),
        "timing_lag_median_frames": lag_median,
        "sharpness_median_over_episodes": float(np.median([e["sharpness_median"] for e in episodes_out.values() if "sharpness_median" in e])) if any("sharpness_median" in e for e in episodes_out.values()) else None,
        "ball_visible_fraction_mean": float(np.mean([e["ball_visible_fraction"] for e in episodes_out.values() if "ball_visible_fraction" in e])) if any("ball_visible_fraction" in e for e in episodes_out.values()) else None,
        "usable_episodes": [e for e, s in episodes_out.items() if s["label"] == "USABLE"],
        "partial_episodes": [e for e, s in episodes_out.items() if s["label"] == "PARTIALLY_USABLE"],
        "unusable_episodes": [e for e, s in episodes_out.items() if s["label"] == "UNUSABLE"],
        "episodes": {str(e): s for e, s in episodes_out.items()},
    }
    worst_out = []
    for sharp, ep, pos, rgb in sorted(worst, key=lambda x: x[0]):
        p = montage_dir / f"worst_ep{ep:03d}_f{pos:05d}.jpg"
        save_thumb(rgb, p, f"ep{ep} f{pos} sharp {sharp:.1f}")
        worst_out.append({"path": str(p.relative_to(out_dir)), "episode": ep, "frame": pos, "sharpness": float(sharp)})
    return stream, thumbs, worst_out


def select_cameras(streams: dict[str, dict]) -> dict:
    usable = [k for k, s in streams.items() if s["label"] == "USABLE"]
    partial = [k for k, s in streams.items() if s["label"] == "PARTIALLY_USABLE"]
    if len(usable) == 1:
        return {"selected_camera_keys": usable, "rule": "single_usable_stream (plan §3.2): use it; no approval needed"}
    if len(usable) >= 2:
        return {"selected_camera_keys": [], "rule": "multiple_usable_streams: single-vs-multi camera requires user approval (plan §3.2)", "candidates": usable}
    if partial:
        return {"selected_camera_keys": [], "rule": "only partially usable streams: exclusion/handling requires user decision (plan §3.2)", "candidates": partial}
    return {"selected_camera_keys": [], "rule": "no usable stream: vision-policy training blocked (plan §3.2)"}


# ----------------------------------------------------------------------------- driver
def run_audit(
    root: Path,
    out: Path,
    repo_id: str | None,
    revision: str | None,
    stride: int = 1,
    episodes: list[int] | None = None,
    n_val: int = 10,
    seed: int = 0,
    skip_video: bool = False,
    hash_videos: bool = False,
    max_worst: int = 10,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds = RawDataset(root, repo_id=repo_id, revision=revision)
    eps = episodes or ds.episode_indices
    print(f"dataset {repo_id}@{revision} at {root}: {len(ds.episode_indices)} episodes, keys {ds.image_keys}", file=sys.stderr)

    ep_table = audit_episode_table(ds)
    pd.DataFrame(ep_table).to_csv(out / "episodes.csv", index=False)
    timing = audit_timing(ds)
    state_action = audit_action_state(ds)

    streams: dict[str, dict] = {}
    thumbs: dict[str, dict[int, dict[str, str]]] = {}
    worst: dict[str, list[dict]] = {}
    if not skip_video:
        for key in ds.image_keys:
            print(f"auditing {key} ...", file=sys.stderr)
            streams[key], thumbs[key], worst[key] = audit_camera(ds, key, eps, stride, out, max_worst=max_worst)
    selection = select_cameras(streams) if streams else {"selected_camera_keys": [], "rule": "video audit skipped"}

    camera_quality = {
        "generated_at": _now(),
        "source": {"repo_id": repo_id, "revision": revision, "root": str(root)},
        "stride": stride,
        "episodes_audited": eps,
        "analysis_size": list(ANALYSIS_SIZE),
        "thresholds": THRESHOLDS,
        "labels": {k: s["label"] for k, s in streams.items()},
        **selection,
        "streams": streams,
    }
    (out / "camera_quality.json").write_text(json.dumps(camera_quality, indent=2, default=_json_default) + "\n")
    if streams:
        write_montage_html(out / "camera_montage.html", f"{repo_id or root.name} @ {revision or 'local'}", streams, thumbs, worst)

    n_val_eff = min(n_val, max(len(ds.episode_indices) - 1, 0))
    split = make_episode_split(ds.episode_indices, n_val=n_val_eff, seed=seed, repo_id=repo_id, revision=revision)
    split.to_json(out / "split.json")

    license_str = _read_license(root)
    report = {
        "generated_at": _now(),
        "tool": "real2sim.data.audit",
        "source": {
            "repo_id": repo_id,
            "revision": revision,
            "root": str(root),
            "codebase_version": ds.codebase_version,
            "robot_type": ds.robot_type,
            "fps": ds.fps,
            "license": license_str,
        },
        "totals": {
            "episodes": len(ds.episode_indices),
            "frames": int(ds.info.get("total_frames", len(ds.data))),
            "rows": int(len(ds.data)),
            "duration_s": round(len(ds.data) / ds.fps, 1),
            "tasks": ds.task_strings,
        },
        "features": {k: {kk: vv for kk, vv in v.items() if kk != "info"} for k, v in ds.features.items()},
        "image_keys": ds.image_keys,
        "image_info": {k: ds.image_info(k) for k in ds.image_keys},
        "episodes": ep_table,
        "timing": timing,
        "state_action": state_action,
        "camera_quality_labels": camera_quality["labels"],
        "camera_selection": selection,
        "split": {"seed": split.seed, "n_train": split.n_train, "n_val": split.n_val, "sha256": split.sha256, "val": split.val},
        "file_manifest": ds.file_manifest(hash_videos=hash_videos),
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    print(f"wrote {out}/dataset_report.json, camera_quality.json, camera_montage.html, split.json", file=sys.stderr)
    return report


def _read_license(root: Path) -> str | None:
    readme = root / "README.md"
    if not readme.is_file():
        return None
    text = readme.read_text(errors="ignore")
    for line in text.splitlines()[:40]:
        if line.strip().startswith("license:"):
            return line.split(":", 1)[1].strip()
    return None


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m real2sim.data.audit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=None, help="Hugging Face dataset repo id (provenance; download only if no --local-root)")
    p.add_argument("--revision", default=None, help="immutable dataset commit SHA to pin")
    p.add_argument("--local-root", default=None, help="local LeRobot v3.0 dataset directory (preferred)")
    p.add_argument("--out", required=True, help="output directory, e.g. results/audit")
    p.add_argument("--stride", type=int, default=1, help="decode every Nth frame (1 = full audit)")
    p.add_argument("--episodes", default=None, help="comma-separated episode indices to audit (default: all)")
    p.add_argument("--n-val", type=int, default=10, help="validation episodes in the split (D-006)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-video", action="store_true", help="metadata/action audit only")
    p.add_argument("--hash-videos", action="store_true", help="also SHA-256 the video files (slow)")
    p.add_argument("--max-worst", type=int, default=10)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = resolve_dataset_root(args.repo_id, args.revision, args.local_root)
    episodes = [int(e) for e in args.episodes.split(",")] if args.episodes else None
    run_audit(
        root=root,
        out=Path(args.out),
        repo_id=args.repo_id,
        revision=args.revision,
        stride=max(1, args.stride),
        episodes=episodes,
        n_val=args.n_val,
        seed=args.seed,
        skip_video=args.skip_video,
        hash_videos=args.hash_videos,
        max_worst=args.max_worst,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
