"""Expert-correction dataset for the recovery specialist (plan §4 M6 steps 4-5).

Table columns (CSV or parquet), one row per policy tick of a correction episode:

* ``correction_episode_id`` (int/str), ``tick`` (int)
* ``controller_source``, ``intervention_start``, ``intervention_end`` (ticks, inclusive)
* ``expert_<a>`` for ``a`` in :data:`ACTION_NAMES` — the expert's 7-D action (target ``y``)
* the 29 feature columns named exactly as :data:`FEATURE_NAMES` (input ``x``); the
  ``base_proposed_<a>`` columns inside them are the frozen ACT's proposed action at that tick
* ``reason`` (detector reason), ``success`` (episode outcome), ``provenance``

Rows are used only while the expert is in control (``intervention_start <= tick <=
intervention_end``). ``provenance`` must be one of ``ALLOWED_PROVENANCE``; any other value is a
hard error so fabricated or unlabelled corrections can never train the specialist (AGENTS.md
rule 4). The train/held-out split is by correction episode id, deterministic given ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from real2sim.data.labels import ACTION_NAMES
from real2sim.recovery.detector import FEATURE_NAMES

ALLOWED_PROVENANCE = ("scripted_oracle_sim", "human_teleop_sim", "real_intervention")
EXPERT_COLUMNS = [f"expert_{a}" for a in ACTION_NAMES]
META_COLUMNS = [
    "correction_episode_id",
    "tick",
    "controller_source",
    "intervention_start",
    "intervention_end",
    "reason",
    "success",
    "provenance",
]
ALL_COLUMNS = META_COLUMNS + EXPERT_COLUMNS + FEATURE_NAMES


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported correction file {path}; use .csv or .parquet")


@dataclass
class Split:
    train_ids: list[Any]
    heldout_ids: list[Any]


def split_episode_ids(ids: list[Any], heldout_fraction: float, seed: int) -> Split:
    uniq = sorted(set(ids), key=str)
    if len(uniq) < 2:
        raise ValueError("need at least two correction episodes to hold one out")
    rng = np.random.default_rng(seed)
    order = [uniq[i] for i in rng.permutation(len(uniq))]
    n_held = max(1, int(round(heldout_fraction * len(uniq))))
    n_held = min(n_held, len(uniq) - 1)
    return Split(train_ids=order[n_held:], heldout_ids=order[:n_held])


class CorrectionDataset:
    """(x[29], y[7]) pairs from a correction table, split by correction episode id."""

    def __init__(
        self,
        frame: pd.DataFrame,
        allowed_provenance: tuple[str, ...] = ALLOWED_PROVENANCE,
    ) -> None:
        missing = [c for c in ALL_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"correction table missing columns: {missing}")
        bad = frame.loc[~frame["provenance"].isin(allowed_provenance), "provenance"]
        if len(bad):
            raise ValueError(
                f"refusing {len(bad)} row(s) with provenance {sorted(set(map(str, bad)))}; "
                f"allowed: {allowed_provenance}"
            )
        active = (frame["tick"] >= frame["intervention_start"]) & (frame["tick"] <= frame["intervention_end"])
        self.frame = frame.loc[active].reset_index(drop=True)
        self.n_dropped_outside_intervention = int((~active).sum())
        if len(self.frame) == 0:
            raise ValueError("no rows inside an intervention window")
        self.x = self.frame[FEATURE_NAMES].to_numpy(np.float32)
        self.y = self.frame[EXPERT_COLUMNS].to_numpy(np.float32)
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.y)):
            raise ValueError("non-finite feature or expert-action values")
        self.episode_ids = self.frame["correction_episode_id"].tolist()

    @classmethod
    def from_file(cls, path: str | Path, allowed_provenance=ALLOWED_PROVENANCE) -> CorrectionDataset:
        return cls(read_table(path), allowed_provenance)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    def split(self, heldout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, Split]:
        """Boolean masks (train, heldout) over rows plus the id split."""
        s = split_episode_ids(self.episode_ids, heldout_fraction, seed)
        held = set(map(str, s.heldout_ids))
        is_held = np.array([str(e) in held for e in self.episode_ids])
        return ~is_held, is_held, s
