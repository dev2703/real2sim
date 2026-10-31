"""Bounded ACT ensembling with explicit control-tick alignment.

Unlike queue concatenation, overlapping predictions are combined only when they
refer to the same absolute action tick. A checkpoint trained on a prefix must
declare that prefix: its unsupervised tail never enters the ensemble. This module
does not change an action representation, perform IK or compensate motor lag.
"""

from __future__ import annotations

from collections import deque
from numbers import Integral

import numpy as np


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class BoundedTemporalEnsembler:
    """Average predictions for one action tick across overlapping chunks.

    ``chunk[0]`` addresses ``start_tick``; ``chunk[i]`` addresses
    ``start_tick + i``. Call :meth:`add_chunk` with the observation's control tick,
    then :meth:`action_at` with the current execution tick. These may differ when
    inference was delayed. Expired predictions are discarded rather than shifted
    onto a later tick. Ticks are nonnegative integers within an episode.

    Weights follow LeRobot ACT's ``exp(-coefficient * i)``, where ``i=0`` is
    the OLDEST contributing prediction. Positive coefficients favour older
    predictions, negative coefficients newer predictions, and zero is uniform.
    ``i`` is the rank among contributing chunks, including when inference skips
    ticks. Numerical evaluation uses an equivalent stable softmax.

    Inputs are finite, unnormalised action arrays of shape ``(chunk_size, D)``.
    The caller remains responsible for ensuring those actions share a frame and
    meaning before averaging, particularly for rotational representations.
    """

    def __init__(self, chunk_size: int, coefficient: float,
                 supervised_prefix_steps: int | None = None):
        self.chunk_size = _integer(chunk_size, "chunk_size", minimum=1)
        if isinstance(coefficient, (bool, str)):
            raise ValueError("coefficient must be a finite number")
        try:
            self.coefficient = float(coefficient)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("coefficient must be a finite number") from exc
        if not np.isfinite(self.coefficient):
            raise ValueError("coefficient must be a finite number")
        self.supervised_prefix_steps = supervised_prefix_steps
        if supervised_prefix_steps is None:
            self.horizon = self.chunk_size
        else:
            self.horizon = _integer(supervised_prefix_steps, "supervised_prefix_steps", minimum=1)
            if self.horizon > self.chunk_size:
                raise ValueError("supervised_prefix_steps exceeds chunk_size")
        self.reset()

    def reset(self) -> None:
        """Remove all episode history, timing and action-dimension state."""
        self._chunks: deque[tuple[int, np.ndarray]] = deque()
        self._last_start_tick: int | None = None
        self._last_action_tick: int | None = None
        self._action_dim: int | None = None
        self.last_selection: dict | None = None

    @property
    def pending_chunks(self) -> int:
        return len(self._chunks)

    def add_chunk(self, chunk: np.ndarray, *, start_tick: int) -> None:
        """Retain a validated copy of the supervised prefix only.

        Chunk start ticks must increase strictly, so duplicate and out-of-order
        inference responses cannot accidentally gain extra ensemble weight.
        Validation errors leave the existing history unchanged.
        """
        tick = _integer(start_tick, "start_tick")
        if self._last_start_tick is not None and tick <= self._last_start_tick:
            raise ValueError("chunk start_tick must increase strictly")
        try:
            actions = np.asarray(chunk, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("chunk must contain finite numeric predictions") from exc
        if actions.ndim != 2 or actions.shape[0] != self.chunk_size or actions.shape[1] < 1:
            raise ValueError(f"chunk must have shape ({self.chunk_size}, action_dim)")
        if not np.isfinite(actions).all():
            raise ValueError("chunk must contain finite numeric predictions")
        if self._action_dim is not None and actions.shape[1] != self._action_dim:
            raise ValueError("chunk action dimension changed within the episode")
        earliest_tick = max(tick, 0 if self._last_action_tick is None else self._last_action_tick + 1)
        if tick + self.horizon <= earliest_tick:
            raise ValueError("chunk has no unexpired supervised predictions")
        retained = actions[:self.horizon].copy()
        while self._chunks and self._chunks[0][0] + self.horizon <= earliest_tick:
            self._chunks.popleft()
        self._chunks.append((tick, retained))
        self._last_start_tick = tick
        self._action_dim = actions.shape[1]

    def action_at(self, *, tick: int) -> np.ndarray:
        """Consume one action at its exact tick; refuse missing/stale coverage.

        Skipping execution ticks is allowed; consuming a tick twice or moving
        backwards requires an episode reset. No previous action is reused when
        every supervised prediction has expired.
        """
        action_tick = _integer(tick, "tick")
        if self._last_action_tick is not None and action_tick <= self._last_action_tick:
            raise ValueError("action tick must increase strictly")
        if self._last_start_tick is not None and action_tick < self._last_start_tick:
            raise ValueError("action tick precedes the latest observation tick")
        contributions = [(start, actions[action_tick - start])
                         for start, actions in self._chunks
                         if 0 <= action_tick - start < self.horizon]
        if not contributions:
            raise ValueError("no supervised prediction covers this action tick")
        ranks = np.arange(len(contributions), dtype=np.float64)
        # All exponent arguments are nonpositive, including extreme finite coefficients.
        with np.errstate(over="ignore"):
            logits = (-self.coefficient * ranks if self.coefficient >= 0
                      else self.coefficient * (len(contributions) - 1 - ranks))
        weights = np.exp(logits)
        weights /= weights.sum()
        values = np.stack([value for _, value in contributions])
        # Scaling prevents a finite convex combination overflowing on large inputs.
        scales = np.maximum(np.abs(values).max(axis=0), np.finfo(np.float64).tiny)
        scaled = values / scales
        combined = np.clip(weights @ scaled, scaled.min(axis=0), scaled.max(axis=0)) * scales
        while self._chunks and self._chunks[0][0] + self.horizon <= action_tick + 1:
            self._chunks.popleft()
        self._last_action_tick = action_tick
        self.last_selection = {
            "action_tick": action_tick,
            "start_ticks": [start for start, _ in contributions],
            "offsets": [action_tick - start for start, _ in contributions],
            "weights": weights.tolist(),
            "supervised_horizon": self.horizon,
        }
        return combined
