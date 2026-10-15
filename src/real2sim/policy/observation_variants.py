"""Causal observation ablations shared by derived datasets and online inference.

Action targets and raw observations are never edited. Joint inputs are measured
SO-101 positions (five degrees plus gripper percent), not leader commands. History
contains observations only, oldest first, with age in seconds and a validity flag.
Missing slots repeat the first available observation but have age=valid=0; they
cannot be mistaken for observations from a preceding episode.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from real2sim.data.labels import PROPRIO_NAMES

KINDS = ("canonical", "no_velocity", "ema_velocity", "joints", "canonical_joints")
ARM_NAMES = [f"joint_{name}_deg" for name in
             ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]


@dataclass(frozen=True)
class ObservationVariant:
    kind: str = "canonical"
    ema_alpha: float = 0.5
    history: int = 1
    policy_hz: float = 10.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown observation kind {self.kind!r}; expected one of {KINDS}")
        if not np.isfinite(self.ema_alpha) or not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if isinstance(self.history, bool) or not isinstance(self.history, (int, np.integer)) or self.history < 1:
            raise ValueError("history must be a positive integer")
        if not np.isfinite(self.policy_hz) or self.policy_hz <= 0:
            raise ValueError("policy_hz must be finite and positive")

    @classmethod
    def from_dict(cls, config: dict | None) -> ObservationVariant:
        return cls(**(config or {}))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def requires_joints(self) -> bool:
        return self.kind in ("joints", "canonical_joints")


def base_state_names(config: ObservationVariant) -> list[str]:
    if config.kind == "no_velocity":
        return [*PROPRIO_NAMES[:9], PROPRIO_NAMES[12]]
    if config.kind == "joints":
        return [*ARM_NAMES, "gripper_obs"]
    if config.kind == "canonical_joints":
        return [*PROPRIO_NAMES, *ARM_NAMES]
    return list(PROPRIO_NAMES)


def state_names(config: ObservationVariant) -> list[str]:
    names = base_state_names(config)
    if config.history == 1:
        return names
    return [f"history_{lag}_{name}"
            for lag in reversed(range(config.history))
            for name in [*names, "age_s", "valid"]]


class StreamingObservationTransform:
    """One transform instance per rollout; call ``reset`` on every environment reset."""

    def __init__(self, config: ObservationVariant | dict | None = None):
        self.config = config if isinstance(config, ObservationVariant) else ObservationVariant.from_dict(config)
        self.reset()

    def reset(self) -> None:
        self._velocity: np.ndarray | None = None
        self._last_timestamp: float | None = None
        self._history: deque = deque(maxlen=self.config.history)

    def update(self, canonical_state: np.ndarray, joint_state: np.ndarray | None = None,
               timestamp: float | None = None) -> np.ndarray:
        canonical = np.asarray(canonical_state, dtype=np.float32)
        if canonical.shape != (13,) or not np.isfinite(canonical).all():
            raise ValueError("canonical_state must be a finite 13-vector")
        cfg = self.config
        if timestamp is None:
            timestamp = 0.0 if self._last_timestamp is None else self._last_timestamp + 1 / cfg.policy_hz
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or (self._last_timestamp is not None and timestamp <= self._last_timestamp):
            raise ValueError("timestamps must be finite and strictly increasing within an episode")
        joints = None
        if cfg.requires_joints:
            joints = np.asarray(joint_state, dtype=np.float32)
            if joints.shape != (6,) or not np.isfinite(joints).all():
                raise ValueError("measured SO-101 joint_state must be a finite 6-vector")
        if cfg.kind == "no_velocity":
            feature = np.r_[canonical[:9], canonical[12]]
        elif cfg.kind == "ema_velocity":
            feature = canonical.copy()
            if self._velocity is None:
                self._velocity = np.zeros(3, dtype=np.float32)
            else:
                self._velocity = cfg.ema_alpha * canonical[9:12] + (1 - cfg.ema_alpha) * self._velocity
            feature[9:12] = self._velocity
        elif cfg.kind == "joints":
            feature = np.r_[joints[:5], joints[5] / 100]
        elif cfg.kind == "canonical_joints":
            feature = np.r_[canonical, joints[:5]]
        else:
            feature = canonical.copy()
        self._last_timestamp = timestamp
        self._history.append((timestamp, feature.copy()))
        if cfg.history == 1:
            return feature.astype(np.float32)
        first = self._history[0][1]
        slots = [np.r_[first, 0.0, 0.0] for _ in range(cfg.history - len(self._history))]
        slots.extend(np.r_[old, timestamp - when, 1.0] for when, old in self._history)
        return np.concatenate(slots).astype(np.float32)


def transform_observations(arrays: dict, config: ObservationVariant) -> np.ndarray:
    """Transform all ordered episode rows without reading labels or future states."""
    canonical = np.asarray(arrays["state"])
    episodes = np.asarray(arrays["episode_index"])
    if canonical.ndim != 2 or canonical.shape[1] != 13 or len(canonical) != len(episodes):
        raise ValueError("expected canonical state[N,13] and matching episode_index[N]")
    if not len(canonical):
        raise ValueError("no observations")
    joints = arrays.get("joint_state")
    if config.requires_joints and (joints is None or np.asarray(joints).shape != (len(canonical), 6)):
        raise ValueError("joint input variants require measured joint_state[N,6]")
    timestamps = arrays.get("timestamp")
    if timestamps is not None and np.asarray(timestamps).shape != (len(canonical),):
        raise ValueError("timestamp length differs from states")
    transform = StreamingObservationTransform(config)
    out = np.empty((len(canonical), len(state_names(config))), dtype=np.float32)
    previous, seen = None, set()
    for i, ep in enumerate(episodes):
        if ep != previous:
            if ep in seen:
                raise ValueError("episode rows must form contiguous blocks")
            seen.add(ep)
            previous = ep
            transform.reset()
        out[i] = transform.update(canonical[i], None if joints is None else joints[i],
                                  None if timestamps is None else timestamps[i])
    return out


def fit_state_statistics(features: np.ndarray, episode_ids: np.ndarray,
                         training_episode_ids: list[int], config: ObservationVariant) -> dict:
    """Fit only training observations; history slots share current-state statistics.

    Ages and validity flags retain their seconds / binary units. Repeated padding
    never contributes additional samples to fitting sensor normalization.
    """
    features = np.asarray(features, dtype=np.float64)
    episode_ids = np.asarray(episode_ids)
    if features.ndim != 2 or features.shape != (len(episode_ids), len(state_names(config))):
        raise ValueError("feature dimensions differ from observation variant")
    if not np.isfinite(features).all():
        raise ValueError("non-finite observation features")
    if not training_episode_ids or not set(training_episode_ids) <= set(episode_ids.tolist()):
        raise ValueError("training episode IDs must be nonempty and present in the arrays")
    mask = np.isin(episode_ids, training_episode_ids)
    selected = features[mask]
    n_base = len(base_state_names(config))
    current = selected if config.history == 1 else selected[:, -n_base - 2:-2]
    mean, std = current.mean(axis=0), np.maximum(current.std(axis=0), 1e-3)
    if config.history > 1:
        mean = np.tile(np.r_[mean, 0.0, 0.0], config.history)
        std = np.tile(np.r_[std, 1.0, 1.0], config.history)
    return {"mean": mean.tolist(), "std": std.tolist(), "min": selected.min(axis=0).tolist(),
            "max": selected.max(axis=0).tolist(), "count": int(mask.sum())}
