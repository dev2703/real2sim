"""Per-frame image quality metrics and the USABLE / PARTIALLY_USABLE / UNUSABLE rules.

Everything here is pure numpy/OpenCV on small arrays so it can be unit-tested without video.
Thresholds are explicit constants and are written into the audit JSON next to the verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

THRESHOLDS = {
    "black_mean_max": 10.0,  # grey mean below this => BLACK
    "blank_std_max": 5.0,  # grey std below this => BLANK (featureless)
    "frozen_diff_max": 0.5,  # mean |frame - prev| below this counts toward a frozen run
    "frozen_run_min": 15,  # consecutive low-diff frames to call a FROZEN run (0.5 s at 30 fps)
    "duplicate_diff_max": 0.05,  # near-identical consecutive frames
    "low_sharpness_median": 5.0,  # Laplacian variance (320x180 grey) below this => no detail
    "ball_min_area_px": 20,  # yellow blob area at 320x180 to count the ball as visible
    "episode_unusable_decode_rate": 0.90,
    "episode_partial_decode_rate": 0.995,
    "episode_unusable_dark_fraction": 0.50,
    "episode_partial_dark_fraction": 0.02,
    "episode_unusable_frozen_fraction": 0.80,
    "episode_partial_frozen_fraction": 0.30,
    "episode_partial_ball_visible_fraction": 0.20,
    "timing_lag_outlier_frames": 3,
    "stream_unusable_episode_fraction": 0.50,
}

ANALYSIS_SIZE = (320, 180)  # (width, height) used for all per-frame metrics


def ball_area_px(rgb_small: np.ndarray) -> int:
    """Largest tennis-ball-yellow connected component area (pixels) in a small RGB frame."""
    from real2sim.perception.ball_hsv import BALL_HSV

    hsv = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2HSV)
    window = BALL_HSV["audit"]
    mask = cv2.inRange(hsv, window.lower, window.upper)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


@dataclass
class FrameMetrics:
    """Accumulates per-frame statistics for one episode of one camera."""

    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    sharpness: list[float] = field(default_factory=list)
    diff_prev: list[float] = field(default_factory=list)
    ball_area: list[int] = field(default_factory=list)
    pts: list[float] = field(default_factory=list)
    _prev: np.ndarray | None = None

    def add(self, rgb_small: np.ndarray, pts: float) -> None:
        grey = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)
        g32 = grey.astype(np.float32)
        self.mean.append(float(g32.mean()))
        self.std.append(float(g32.std()))
        self.sharpness.append(float(cv2.Laplacian(grey, cv2.CV_64F).var()))
        self.diff_prev.append(
            float(np.abs(g32 - self._prev).mean()) if self._prev is not None else float("nan")
        )
        self.ball_area.append(ball_area_px(rgb_small))
        self.pts.append(float(pts))
        self._prev = g32

    def __len__(self) -> int:
        return len(self.mean)


def frozen_runs(diff_prev: np.ndarray, diff_max: float, run_min: int) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs of runs with ``diff_prev < diff_max`` of length >= run_min."""
    low = np.nan_to_num(diff_prev, nan=np.inf) < diff_max
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(low):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= run_min:
                runs.append((start, i))
            start = None
    if start is not None and len(low) - start >= run_min:
        runs.append((start, len(low)))
    return runs


def summarize_episode(
    m: FrameMetrics,
    n_expected: int,
    decode_failures: int = 0,
    video_frame_deficit: int = 0,
    timing_lag_frames: float | None = None,
    timing_lag_median: float | None = None,
    thresholds: dict | None = None,
) -> dict:
    """Roll per-frame metrics into an episode verdict with explicit reasons."""
    t = {**THRESHOLDS, **(thresholds or {})}
    n = len(m)
    out: dict = {
        "frames_expected": int(n_expected),
        "frames_decoded": int(n),
        "decode_failures": int(decode_failures),
        "video_frame_deficit": int(video_frame_deficit),
    }
    if n == 0:
        out.update({"label": "UNUSABLE", "reasons": ["no frames decoded"]})
        return out
    mean = np.asarray(m.mean)
    std = np.asarray(m.std)
    sharp = np.asarray(m.sharpness)
    diff = np.asarray(m.diff_prev, dtype=np.float64)
    ball = np.asarray(m.ball_area)

    black = mean < t["black_mean_max"]
    blank = std < t["blank_std_max"]
    dup = np.nan_to_num(diff, nan=np.inf) < t["duplicate_diff_max"]
    runs = frozen_runs(diff, t["frozen_diff_max"], int(t["frozen_run_min"]))
    frozen_frames = int(sum(e - s for s, e in runs))
    decode_rate = n / max(n_expected, 1)
    dark_fraction = float(np.mean(black | blank))
    frozen_fraction = frozen_frames / n
    ball_visible_fraction = float(np.mean(ball >= t["ball_min_area_px"]))
    median_sharp = float(np.median(sharp))

    out.update(
        {
            "decode_rate": round(decode_rate, 5),
            "black_frames": int(black.sum()),
            "blank_frames": int(blank.sum()),
            "duplicate_frames": int(dup.sum()),
            "frozen_runs": [[int(s), int(e)] for s, e in runs],
            "frozen_frames": frozen_frames,
            "frozen_fraction": round(frozen_fraction, 4),
            "dark_fraction": round(dark_fraction, 4),
            "sharpness_median": round(median_sharp, 2),
            "sharpness_p10": round(float(np.percentile(sharp, 10)), 2),
            "sharpness_min": round(float(sharp.min()), 2),
            "grey_mean_median": round(float(np.median(mean)), 2),
            "grey_std_median": round(float(np.median(std)), 2),
            "ball_visible_fraction": round(ball_visible_fraction, 4),
            "pts_first": round(float(m.pts[0]), 5),
            "pts_last": round(float(m.pts[-1]), 5),
            "timing_lag_frames": timing_lag_frames,
        }
    )

    reasons: list[str] = []
    label = "USABLE"

    def unusable(msg: str) -> None:
        nonlocal label
        label = "UNUSABLE"
        reasons.append(msg)

    def partial(msg: str) -> None:
        nonlocal label
        if label != "UNUSABLE":
            label = "PARTIALLY_USABLE"
        reasons.append(msg)

    if decode_rate < t["episode_unusable_decode_rate"]:
        unusable(f"decode rate {decode_rate:.3f} < {t['episode_unusable_decode_rate']}")
    elif decode_rate < t["episode_partial_decode_rate"] or decode_failures > 0:
        partial(f"decode rate {decode_rate:.4f} / {decode_failures} failures")
    if dark_fraction > t["episode_unusable_dark_fraction"]:
        unusable(f"black/blank fraction {dark_fraction:.2f}")
    elif dark_fraction > t["episode_partial_dark_fraction"]:
        partial(f"black/blank fraction {dark_fraction:.3f}")
    if median_sharp < t["low_sharpness_median"]:
        unusable(f"median sharpness {median_sharp:.2f} < {t['low_sharpness_median']} (no scene detail)")
    if frozen_fraction > t["episode_unusable_frozen_fraction"]:
        unusable(f"frozen fraction {frozen_fraction:.2f}")
    elif frozen_fraction > t["episode_partial_frozen_fraction"]:
        partial(f"frozen fraction {frozen_fraction:.2f}")
    if video_frame_deficit != 0:
        partial(f"video has {video_frame_deficit:+d} frames vs data rows (possible misalignment)")
    if ball_visible_fraction < t["episode_partial_ball_visible_fraction"]:
        partial(f"ball visible in only {ball_visible_fraction:.2f} of frames")
    if (
        timing_lag_frames is not None
        and timing_lag_median is not None
        and abs(timing_lag_frames - timing_lag_median) > t["timing_lag_outlier_frames"]
    ):
        partial(
            f"image/joint motion lag {timing_lag_frames:+.0f} frames vs stream median "
            f"{timing_lag_median:+.0f}"
        )
    out["label"] = label
    out["reasons"] = reasons
    return out


def label_stream(episode_summaries: dict[int, dict], thresholds: dict | None = None) -> tuple[str, list[str]]:
    """Combine episode verdicts into a stream verdict."""
    t = {**THRESHOLDS, **(thresholds or {})}
    labels = [s["label"] for s in episode_summaries.values()]
    if not labels:
        return "UNUSABLE", ["no episodes audited"]
    n = len(labels)
    n_unusable = labels.count("UNUSABLE")
    n_partial = labels.count("PARTIALLY_USABLE")
    reasons = [f"{n - n_unusable - n_partial}/{n} USABLE, {n_partial} PARTIAL, {n_unusable} UNUSABLE"]
    if n_unusable / n > t["stream_unusable_episode_fraction"]:
        return "UNUSABLE", reasons
    if n_unusable or n_partial:
        return "PARTIALLY_USABLE", reasons
    return "USABLE", reasons


def motion_lag(image_motion: np.ndarray, joint_speed: np.ndarray, max_lag: int = 30) -> tuple[int | None, float]:
    """Lag (frames) maximising correlation between image motion energy and joint speed.

    Positive lag means the image motion trails the joint motion (camera later than state).
    Returns ``(lag, correlation)``; ``(None, nan)`` when a series is too short or constant.
    """
    a = np.asarray(image_motion, dtype=np.float64)
    b = np.asarray(joint_speed, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 4 * max_lag or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return None, float("nan")
    a = (a - np.nanmean(a)) / np.nanstd(a)
    b = (b - np.nanmean(b)) / np.nanstd(b)
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    best_lag, best_c = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = float(np.mean(a[lag:] * b[: n - lag]))
        else:
            c = float(np.mean(a[: n + lag] * b[-lag:]))
        if c > best_c:
            best_lag, best_c = lag, c
    return best_lag, best_c
