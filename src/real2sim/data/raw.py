"""Read-only access to a LeRobot v3.0 dataset directory.

The raw recording is never rewritten. This module exposes metadata, the episode table, the
frame-level parquet rows and timestamp-bounded video decoding for one episode and one image key.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pandas as pd

from real2sim.config import sha256_file as sha256_file  # re-export for existing call sites

SUPPORTED_CODEBASE_VERSIONS = ("v3.0",)


@dataclass(frozen=True)
class VideoSegment:
    """Location of one episode inside one concatenated video file."""

    key: str
    episode_index: int
    path: Path
    from_ts: float
    to_ts: float

    @property
    def exists(self) -> bool:
        return self.path.is_file()


def _read_parquet_dir(directory: Path) -> pd.DataFrame:
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {directory}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


class RawDataset:
    """Thin reader over a LeRobot v3.0 dataset root.

    Parameters
    ----------
    root:
        Directory containing ``meta/``, ``data/`` and ``videos/``.
    repo_id, revision:
        Provenance only; recorded into reports. ``revision`` should be the immutable commit SHA.
    """

    def __init__(self, root: str | Path, repo_id: str | None = None, revision: str | None = None):
        self.root = Path(root)
        self.repo_id = repo_id
        self.revision = revision
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"missing {info_path}")
        self.info: dict = json.loads(info_path.read_text())
        version = self.info.get("codebase_version")
        if version not in SUPPORTED_CODEBASE_VERSIONS:
            raise ValueError(
                f"unsupported LeRobot codebase_version {version!r}; "
                f"supported: {SUPPORTED_CODEBASE_VERSIONS}"
            )
        self.fps = float(self.info["fps"])
        self.features: dict = self.info["features"]
        self.episodes = (
            _read_parquet_dir(self.root / "meta" / "episodes")
            .sort_values("episode_index")
            .reset_index(drop=True)
        )
        tasks_path = self.root / "meta" / "tasks.parquet"
        self.tasks = pd.read_parquet(tasks_path) if tasks_path.is_file() else None
        stats_path = self.root / "meta" / "stats.json"
        self.stats = json.loads(stats_path.read_text()) if stats_path.is_file() else None
        self._data: pd.DataFrame | None = None

    # ------------------------------------------------------------------ metadata
    @property
    def codebase_version(self) -> str:
        return str(self.info["codebase_version"])

    @property
    def robot_type(self) -> str | None:
        return self.info.get("robot_type")

    @property
    def image_keys(self) -> list[str]:
        return [k for k, v in self.features.items() if v.get("dtype") in ("video", "image")]

    @property
    def state_names(self) -> list[str]:
        return list(self.features["observation.state"]["names"])

    @property
    def action_names(self) -> list[str]:
        return list(self.features["action"]["names"])

    @property
    def episode_indices(self) -> list[int]:
        return [int(e) for e in self.episodes["episode_index"].tolist()]

    @property
    def task_strings(self) -> list[str]:
        if self.tasks is None:
            return []
        if "task" in self.tasks.columns:
            return [str(t) for t in self.tasks["task"].tolist()]
        return [str(t) for t in self.tasks.index.tolist()]

    def image_info(self, key: str) -> dict:
        feat = self.features[key]
        shape = feat.get("shape") or []
        out = {
            "key": key,
            "dtype": feat.get("dtype"),
            "height": int(shape[0]) if len(shape) >= 2 else None,
            "width": int(shape[1]) if len(shape) >= 2 else None,
            "channels": int(shape[2]) if len(shape) >= 3 else None,
        }
        out.update({k: v for k, v in (feat.get("info") or {}).items() if k.startswith("video.")})
        return out

    # ------------------------------------------------------------------ frame table
    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            df = _read_parquet_dir(self.root / "data")
            if "index" in df.columns:
                df = df.sort_values("index")
            self._data = df.reset_index(drop=True)
        return self._data

    def episode_rows(self, episode_index: int) -> pd.DataFrame:
        df = self.data
        return df[df["episode_index"] == episode_index].reset_index(drop=True)

    def episode_length(self, episode_index: int) -> int:
        row = self.episodes[self.episodes["episode_index"] == episode_index]
        if len(row) != 1:
            raise KeyError(f"episode {episode_index} not in episode table")
        return int(row["length"].iloc[0])

    @staticmethod
    def stack(df: pd.DataFrame, column: str) -> np.ndarray:
        return np.stack(df[column].to_numpy()).astype(np.float64)

    # ------------------------------------------------------------------ video access
    def video_segment(self, key: str, episode_index: int) -> VideoSegment:
        row = self.episodes[self.episodes["episode_index"] == episode_index]
        if len(row) != 1:
            raise KeyError(f"episode {episode_index} not in episode table")
        row = row.iloc[0]
        chunk = int(row[f"videos/{key}/chunk_index"])
        file_index = int(row[f"videos/{key}/file_index"])
        rel = self.info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file_index)
        return VideoSegment(
            key=key,
            episode_index=int(episode_index),
            path=self.root / rel,
            from_ts=float(row[f"videos/{key}/from_timestamp"]),
            to_ts=float(row[f"videos/{key}/to_timestamp"]),
        )

    def iter_frames(
        self, key: str, episode_index: int, stride: int = 1, tol: float = 1e-4
    ) -> Iterator[tuple[int, float, av.VideoFrame]]:
        """Yield ``(frame_position, pts_seconds, av.VideoFrame)`` for one episode.

        ``frame_position`` counts decoded frames inside ``[from_ts, to_ts)`` starting at 0.
        Frames are decoded in presentation order from the nearest preceding keyframe.
        """
        seg = self.video_segment(key, episode_index)
        if not seg.exists:
            raise FileNotFoundError(seg.path)
        with av.open(str(seg.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = float(stream.time_base)
            container.seek(int(seg.from_ts / tb), stream=stream, backward=True, any_frame=False)
            pos = 0
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = frame.pts * tb
                if t < seg.from_ts - tol:
                    continue
                if t >= seg.to_ts - tol:
                    break
                if pos % stride == 0:
                    yield pos, t, frame
                pos += 1

    def count_frames(self, key: str, episode_index: int, tol: float = 1e-4) -> int:
        """Number of video frames inside the episode's ``[from_ts, to_ts)`` window (demux only)."""
        seg = self.video_segment(key, episode_index)
        if not seg.exists:
            raise FileNotFoundError(seg.path)
        n = 0
        with av.open(str(seg.path)) as container:
            stream = container.streams.video[0]
            tb = float(stream.time_base)
            container.seek(int(seg.from_ts / tb), stream=stream, backward=True, any_frame=False)
            for packet in container.demux(stream):
                if packet.pts is None:
                    continue
                t = packet.pts * tb
                if seg.from_ts - tol <= t < seg.to_ts - tol:
                    n += 1
                elif t >= seg.to_ts - tol and packet.dts is not None and packet.dts * tb >= seg.to_ts - tol:
                    break
        return n

    # ------------------------------------------------------------------ provenance
    def file_manifest(self, hash_videos: bool = False, hash_limit_bytes: int = 64 << 20) -> list[dict]:
        rows = []
        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or ".cache" in p.parts:
                continue
            size = p.stat().st_size
            is_video = p.suffix.lower() in (".mp4", ".mkv", ".webm")
            do_hash = (hash_videos or not is_video) and size <= hash_limit_bytes
            rows.append(
                {
                    "path": str(p.relative_to(self.root)),
                    "bytes": size,
                    "sha256": sha256_file(p) if do_hash else None,
                }
            )
        return rows


def resolve_dataset_root(
    repo_id: str | None, revision: str | None, local_root: str | Path | None
) -> Path:
    """Return a local directory for the dataset, downloading only when no local root is given."""
    if local_root is not None:
        root = Path(local_root)
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"{root} does not look like a LeRobot dataset (no meta/info.json)")
        return root
    if repo_id is None:
        raise ValueError("either --local-root or --repo-id is required")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision))
