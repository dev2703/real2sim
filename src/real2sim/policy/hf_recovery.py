"""Causal approach-stall detection for explicitly assisted frozen ACT runs."""

from collections import deque

import numpy as np


class ApproachMonitor:
    """Use RGB target distance and elapsed control time, never task truth."""

    def __init__(self, hz, *, patience_s=2., max_attempt_s=5., progress_m=.003):
        if not np.isfinite([hz, patience_s, max_attempt_s, progress_m]).all() or min(
                hz, patience_s, max_attempt_s, progress_m) <= 0:
            raise ValueError("Approach monitor parameters must be positive and finite")
        self.window = max(1, round(hz * patience_s))
        self.deadline = max(self.window, round(hz * max_attempt_s))
        self.progress_m = progress_m
        self.distances = deque(maxlen=self.window + 1)

    def update(self, tick, distance_m):
        if distance_m is not None:
            if not np.isfinite(distance_m) or distance_m < 0:
                raise ValueError("RGB approach distance must be finite and nonnegative")
            self.distances.append(float(distance_m))
            if (len(self.distances) == self.window + 1 and distance_m > .04
                    and self.distances[0] - min(self.distances) < self.progress_m):
                return "RGB approach made less than 3mm progress over the patience interval"
        if tick >= self.deadline:
            return "Bounded HF approach attempt elapsed; request RGB recovery"
        return None
