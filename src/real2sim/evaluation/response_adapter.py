"""Experimental causal follower-response adapter for a fixed early time window.

Reads only measured joint state and executed policy commands. No object poses,
contact flags, phase labels or future chunk actions are used to gate or predict.
Not enabled by the production evaluation entry point.
"""

from collections import deque

import numpy as np

from real2sim.policy.adapters import JointActionAdapter
from real2sim.sim.follower_response import FollowerResponse


class ApproachResponseAdapter(JointActionAdapter):
    def __init__(self, env, model: FollowerResponse, *, active_ticks: int):
        super().__init__(env)
        ratio = model.hz / env.control_hz
        if not ratio.is_integer() or ratio < 1:
            raise ValueError("response clock must be an integer multiple of policy clock")
        if len(model.gain) != 6 or int(active_ticks) != active_ticks or active_ticks < 1:
            raise ValueError("requires six joints and a positive integer horizon")
        self.model, self.subticks = model, int(ratio)
        self.active_ticks = int(active_ticks)
        self.episodes = []
        self._ready = False

    def reset(self):
        self.tick = 0
        self.previous_state = None
        self.actions = deque(maxlen=max(self.model.delay) + 1)
        self.trace = []
        self.episodes.append(self.trace)
        self._ready = True

    def apply(self, action):
        if not self._ready:
            raise RuntimeError("reset response adapter before use")
        action = np.asarray(action, float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("joint action must be six finite values")
        measured = np.asarray(self.env.real_state(), float)
        if measured.shape != (6,) or not np.isfinite(measured).all():
            raise ValueError("measured state must be six finite values")
        active = self.tick < self.active_ticks
        target = action.copy()
        if active:
            # Approximate previous raw-tick velocity using the last observed 100 ms
            # change; predictions are restarted from actual measured state each tick.
            velocity = (np.zeros(6) if self.previous_state is None else
                        np.clip((measured - self.previous_state) / self.subticks,
                                -self.model.step_cap, self.model.step_cap))
            target = np.clip(measured, self.model.lower, self.model.upper)
            for _ in range(self.subticks):
                self.actions.append(action.copy())
                delayed = np.array([self.actions[max(0, len(self.actions) - 1 - d)][j]
                                    for j, d in enumerate(self.model.delay)])
                target, velocity = self.model.advance(target, velocity, delayed)
        self.previous_state = measured.copy()
        result = super().apply(target)
        self.trace.append({"tick": self.tick, "response_active": active,
                           "observed_state": measured.tolist(), "policy_target": action.tolist(),
                           "applied_target": target.tolist(),
                           "limit_clipped": result.detail["limit_clipped"],
                           "rate_limited": result.detail["rate_limited"]})
        self.tick += 1
        return result
