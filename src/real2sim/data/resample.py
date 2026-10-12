"""Episode-preserving resampling to the policy clock and action-chunk indexing (D-007).

The recording clock is uniform (audited), so 10 Hz from 30 Hz is exact row selection: every
``stride``-th row of each episode, starting at row 0. Chunks never cross episode boundaries;
positions past the episode end are padded with the last valid action and flagged in
``action_is_pad`` so the ACT loss ignores them.
"""

from __future__ import annotations

import numpy as np


def policy_stride(source_fps: float, policy_hz: float) -> int:
    ratio = source_fps / policy_hz
    stride = int(round(ratio))
    if stride < 1 or abs(ratio - stride) > 1e-6:
        raise ValueError(f"source fps {source_fps} is not an integer multiple of policy rate {policy_hz}")
    return stride


def episode_tick_rows(n_rows: int, stride: int) -> np.ndarray:
    """Row indices (within one episode) sampled at the policy clock."""
    if n_rows <= 0:
        raise ValueError("episode has no rows")
    return np.arange(0, n_rows, stride, dtype=np.int64)


def chunk_indices(ep_start: np.ndarray, ep_len: np.ndarray, sample: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """For global tick ``sample`` return ``(indices[chunk], is_pad[chunk])`` into the tick arrays.

    ``ep_start``/``ep_len`` describe each tick's episode (arrays indexed by global tick).
    """
    s0 = int(ep_start[sample])
    n = int(ep_len[sample])
    k = sample - s0
    local = k + np.arange(chunk)
    is_pad = local >= n
    idx = s0 + np.minimum(local, n - 1)
    return idx.astype(np.int64), is_pad
