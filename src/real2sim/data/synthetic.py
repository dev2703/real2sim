"""Generate a tiny LeRobot v3.0-shaped dataset for unit tests and CI.

Nothing produced here is ever used for training. Episodes contain a moving bright square and a
yellow "ball" blob so the audit's motion-lag and ball-visibility checks have signal; optional
cameras are black or frozen to exercise the failure paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _pick_codec() -> str:
    for name in ("libx264", "mpeg4"):
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:  # noqa: BLE001 - codec probing
            continue
    raise RuntimeError("no usable video encoder in PyAV")


def _render(kind: str, i: int, n: int, w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "black":
        return np.zeros((h, w, 3), np.uint8)
    frame = np.full((h, w, 3), 120, np.uint8)
    # static texture so the frame is not BLANK
    frame[:, ::8] = 90
    frame[::8, :] = 150
    if kind == "frozen":
        return frame
    # moving bright square proportional to progress, plus a yellow ball blob
    x = int((w - 40) * i / max(n - 1, 1))
    frame[h // 2 - 20 : h // 2 + 20, x : x + 40] = 240
    cx, cy = w // 4, int(h * 0.75)
    yy, xx = np.ogrid[:h, :w]
    ball = (xx - cx) ** 2 + (yy - cy) ** 2 <= 10**2
    frame[ball] = (220, 230, 40)  # tennis-ball yellow in RGB
    noise = rng.integers(-3, 4, size=frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_synthetic_dataset(
    root: Path,
    cameras: dict[str, str] | None = None,
    episode_lengths: tuple[int, ...] = (24, 30),
    fps: int = 30,
    width: int = 128,
    height: int = 96,
    seed: int = 0,
) -> Path:
    """Write a dataset under ``root`` and return it.

    ``cameras`` maps image key -> kind in {"moving", "black", "frozen"}.
    """
    cameras = cameras or {"observation.images.cam_good": "moving"}
    root = Path(root)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    codec = _pick_codec()

    rows = []
    ep_rows = []
    global_index = 0
    for ep, n in enumerate(episode_lengths):
        t = np.arange(n) / fps
        # smooth joint motion in degrees; gripper closes mid-episode
        q = np.stack(
            [30 * np.sin(2 * np.pi * t / 2.0 + k) for k in range(5)] + [np.where(t > t[-1] / 2, 2.0, 60.0)],
            axis=1,
        ).astype(np.float32)
        a = (q + rng.normal(0, 0.5, q.shape)).astype(np.float32)
        for i in range(n):
            rows.append(
                {
                    "action": a[i],
                    "observation.state": q[i],
                    "timestamp": np.float32(t[i]),
                    "frame_index": i,
                    "episode_index": ep,
                    "index": global_index,
                    "task_index": 0,
                }
            )
            global_index += 1
        ep_rows.append({"episode_index": ep, "length": n})
    df = pd.DataFrame(rows)
    df.to_parquet(root / "data" / "chunk-000" / "file-000.parquet", index=False)

    starts = np.cumsum([0] + list(episode_lengths[:-1]))
    ep_table = []
    for ep, n in enumerate(episode_lengths):
        row = {
            "episode_index": ep,
            "tasks": ["synthetic pick place"],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": int(starts[ep]),
            "dataset_to_index": int(starts[ep] + n),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for key in cameras:
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = 0
            row[f"videos/{key}/from_timestamp"] = float(starts[ep] / fps)
            row[f"videos/{key}/to_timestamp"] = float((starts[ep] + n) / fps)
        ep_table.append(row)
    pd.DataFrame(ep_table).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["synthetic pick place"], name="task")).to_parquet(
        root / "meta" / "tasks.parquet"
    )

    features: dict = {
        "action": {"dtype": "float32", "shape": [6], "names": JOINTS},
        "observation.state": {"dtype": "float32", "shape": [6], "names": JOINTS},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key, kind in cameras.items():
        features[key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.height": height, "video.width": width, "video.codec": codec, "video.fps": fps},
        }
        vdir = root / "videos" / key / "chunk-000"
        vdir.mkdir(parents=True, exist_ok=True)
        with av.open(str(vdir / "file-000.mp4"), mode="w") as container:
            stream = container.add_stream(codec, rate=fps)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            if codec == "libx264":
                stream.options = {"g": "2", "crf": "18", "preset": "ultrafast"}
            for n in episode_lengths:
                for i in range(n):
                    arr = _render(kind, i, n, width, height, rng)
                    frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "features": features,
        "total_episodes": len(episode_lengths),
        "total_frames": int(sum(episode_lengths)),
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "robot_type": "so_follower",
        "splits": {"train": f"0:{len(episode_lengths)}"},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return root
