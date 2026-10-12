"""Deterministic whole-episode train/validation split with a hashed manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    n_train: int
    n_val: int
    train: list[int]
    val: list[int]
    sha256: str
    source_repo_id: str | None = None
    source_revision: str | None = None

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> SplitManifest:
        return cls(**json.loads(Path(path).read_text()))


def split_hash(train: list[int], val: list[int], seed: int) -> str:
    payload = json.dumps({"seed": seed, "train": sorted(train), "val": sorted(val)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_episode_split(
    episode_ids: list[int],
    n_val: int,
    seed: int = 0,
    repo_id: str | None = None,
    revision: str | None = None,
) -> SplitManifest:
    ids = sorted(int(e) for e in episode_ids)
    if n_val < 0 or n_val >= len(ids):
        raise ValueError(f"n_val={n_val} must be in [0, {len(ids) - 1}]")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    val = sorted(ids[i] for i in perm[:n_val])
    train = sorted(ids[i] for i in perm[n_val:])
    assert not set(train) & set(val)
    return SplitManifest(
        seed=seed,
        n_train=len(train),
        n_val=len(val),
        train=train,
        val=val,
        sha256=split_hash(train, val, seed),
        source_repo_id=repo_id,
        source_revision=revision,
    )
