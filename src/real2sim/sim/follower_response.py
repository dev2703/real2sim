"""Offline identified follower response; not an ACT action adapter or physics model.

At the recording clock, velocity = inertia * previous velocity + gain * clipped
target error. Both velocity and state are bounded. Delay is causal and measured
in recording ticks. This model has no contact/force inputs and cannot predict a
grasp or establish physical calibration.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class FollowerResponse:
    gain: list[float]
    inertia: list[float]
    delay: list[int]
    lower: list[float]
    upper: list[float]
    hz: float = 30.0
    error_cap: float = 5.0
    step_cap: float = 5.0

    def __post_init__(self):
        n = len(self.gain)
        if not n or any(len(v) != n for v in (self.inertia, self.delay, self.lower, self.upper)):
            raise ValueError("all per-joint parameters must have the same nonzero size")
        if not np.all(np.isfinite([self.gain, self.inertia, self.lower, self.upper])):
            raise ValueError("non-finite model parameters")
        if np.any((np.asarray(self.gain) < 0) | (np.asarray(self.gain) > 1)):
            raise ValueError("gain must be in [0,1]")
        if np.any((np.asarray(self.inertia) < 0) | (np.asarray(self.inertia) > .98)):
            raise ValueError("inertia must be in [0,.98]")
        if any(int(d) != d or d < 0 for d in self.delay):
            raise ValueError("delay must be a nonnegative integer")
        if np.any(np.asarray(self.lower) >= np.asarray(self.upper)):
            raise ValueError("invalid position bounds")
        if not all(np.isfinite(v) and v > 0 for v in (self.hz, self.error_cap, self.step_cap)):
            raise ValueError("clock and caps must be finite and positive")

    def to_dict(self):
        return asdict(self)

    def advance(self, state, velocity, delayed_action):
        state, velocity, delayed_action = [np.asarray(v, float) for v in (state, velocity, delayed_action)]
        if any(v.shape != (len(self.gain),) or not np.isfinite(v).all()
               for v in (state, velocity, delayed_action)):
            raise ValueError("state, velocity and action must be finite joint vectors")
        error = np.clip(delayed_action - state, -self.error_cap, self.error_cap)
        step = np.clip(np.asarray(self.gain) * error + np.asarray(self.inertia) * velocity,
                       -self.step_cap, self.step_cap)
        next_state = np.clip(state + step, self.lower, self.upper)
        return next_state, next_state - state

    def rollout(self, initial, actions):
        actions = np.asarray(actions, float)
        if actions.ndim != 2 or actions.shape[1] != len(self.gain) or not np.isfinite(actions).all():
            raise ValueError("actions must be a finite time-by-joint array")
        state = np.asarray(initial, float)
        if state.shape != (len(self.gain),) or not np.isfinite(state).all():
            raise ValueError("initial must be a finite joint vector")
        if np.any(state < self.lower) or np.any(state > self.upper):
            raise ValueError("initial lies outside model position bounds")
        velocity = np.zeros_like(state)
        result = [state.copy()]
        for t in range(len(actions)):
            delayed = actions[np.maximum(0, t - np.asarray(self.delay)), np.arange(len(state))]
            state, velocity = self.advance(state, velocity, delayed)
            result.append(state.copy())
        return np.asarray(result)


def fit_joint(episodes, joint, delay, *, momentum, error_cap=5.0):
    """Bounded least squares on supplied training episodes; no episode-boundary mixing."""
    features, targets = [], []
    for state, action in episodes:
        t = np.arange(max(1, delay), len(state) - 1)
        error = np.clip(action[t - delay, joint] - state[t, joint], -error_cap, error_cap)
        columns = [error]
        if momentum:
            columns.append(state[t, joint] - state[t - 1, joint])
        features.append(np.column_stack(columns))
        targets.append(state[t + 1, joint] - state[t, joint])
    if not features or not sum(len(v) for v in targets):
        raise ValueError("no training transitions")
    fit = lsq_linear(np.concatenate(features), np.concatenate(targets),
                     bounds=(np.zeros(2 if momentum else 1), [1, .98] if momentum else [1]))
    if not fit.success:
        raise RuntimeError(f"response fit failed: {fit.message}")
    return float(fit.x[0]), float(fit.x[1]) if momentum else 0.0
