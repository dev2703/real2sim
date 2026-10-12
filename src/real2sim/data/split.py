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
    excluded: list[int] | None = None

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> SplitManifest:
        return cls(**json.loads(Path(path).read_text()))


def split_hash(train: list[int], val: list[int], seed: int, excluded: list[int] | None = None) -> str:
    body = {"seed": seed, "train": sorted(train), "val": sorted(val)}
    if excluded:
        body["excluded"] = sorted(excluded)
    payload = json.dumps(body, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_episode_split(
    episode_ids: list[int],
    n_val: int,
    seed: int = 0,
    repo_id: str | None = None,
    revision: str | None = None,
    exclude: list[int] | None = None,
) -> SplitManifest:
    """Seeded whole-episode split.

    The permutation is drawn over *all* ``episode_ids`` and ``exclude`` is removed afterwards, so
    excluding an episode never reshuffles the others (D-010). ``n_val`` counts validation episodes
    before exclusion.
    """
    ids = sorted(int(e) for e in episode_ids)
    if n_val < 0 or n_val >= len(ids):
        raise ValueError(f"n_val={n_val} must be in [0, {len(ids) - 1}]")
    excluded = sorted({int(e) for e in (exclude or [])})
    unknown = set(excluded) - set(ids)
    if unknown:
        raise ValueError(f"excluded episodes not in dataset: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    val = sorted(ids[i] for i in perm[:n_val] if ids[i] not in excluded)
    train = sorted(ids[i] for i in perm[n_val:] if ids[i] not in excluded)
    assert not set(train) & set(val)
    if not train:
        raise ValueError("exclusions leave no training episodes")
    return SplitManifest(
        seed=seed,
        n_train=len(train),
        n_val=len(val),
        train=train,
        val=val,
        sha256=split_hash(train, val, seed, excluded),
        source_repo_id=repo_id,
        source_revision=revision,
        excluded=excluded or None,
    )
