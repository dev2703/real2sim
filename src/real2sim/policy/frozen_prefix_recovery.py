"""Observable, latched handover from a frozen policy to learned recovery.

RGB references and measured robot pose decide only when to hand over. They
never generate an action here. Both control stages retain their own action
sources, native guards and explicit assistance accounting.
"""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np

from real2sim.geometry.so3 import is_rotation


class FrozenPrefixRecoveryController:
    """Run frozen ACT until an injected observable monitor requests recovery.

    Monitor/reference factories create fresh episode state on reset. Drivers
    implement reset()/step(obs, env); learned recovery also implements close().
    The wrapper owns no environment or frozen model resources. Once handover
    occurs, it remains in learned recovery for the rest of that episode.
    """

    def __init__(self, recovery, frozen, monitor_factory, references_factory, grasp_offset):
        if not callable(monitor_factory) or not callable(references_factory):
            raise ValueError("Frozen-prefix monitor and RGB references require episode factories")
        offset = np.asarray(grasp_offset, float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Frozen-prefix grasp offset must contain three finite coordinates")
        self.recovery, self.frozen = recovery, frozen
        self.env = recovery.env
        self.monitor_factory, self.references_factory = monitor_factory, references_factory
        self.grasp_offset = offset.copy()
        self._closed = False
        self.reset()

    def __getattr__(self, name):
        # Existing evaluation code can still inspect learned-head/guard fields.
        return getattr(self.recovery, name)

    @property
    def trigger(self):
        return deepcopy(self._trigger)

    @property
    def prefix_policy_calls(self):
        return len(self.frozen.inference_ms)

    def reset(self):
        self.tick = self.prefix_ticks = self.recovery_ticks = 0
        self._trigger = None
        self.monitor = self.monitor_factory()
        self.references = self.references_factory()
        self.frozen.reset()
        self.recovery.reset()

    def _distance(self, env):
        ball = self.references.ball
        if ball is None:
            return None
        target = np.asarray(ball, float)
        pose = np.asarray(env.ee_pose(), float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Frozen-prefix RGB ball reference must be finite with three coordinates")
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0., 0., 0., 1.]) or not is_rotation(pose[:3, :3])):
            raise ValueError("Frozen-prefix measured TCP requires a finite proper pose")
        grasp = pose[:3, 3] + pose[:3, :3] @ self.grasp_offset
        return float(np.linalg.norm(grasp - target))

    def step(self, obs, env):
        if env is not self.env:
            raise ValueError("Frozen-prefix controller is bound to a different environment")
        started = perf_counter()
        if self._trigger is None:
            # The tracker owns its refresh cadence and may retain an aged RGB
            # reference during occlusion. Missing references give no distance.
            self.references.update(self.tick)
            distance = self._distance(env)
            reason = self.monitor.update(self.tick, distance)
            if reason:
                self._trigger = {"tick": self.tick, "reason": str(reason), "distance_m": distance}
                # Discard all recovery history and queued ACT features captured
                # before the handover. Its first observation is the actual
                # post-prefix robot/RGB state, not a stale episode-start state.
                self.recovery.reset()
                inherit = getattr(self.recovery, "inherit_rgb_references", None)
                if inherit is not None:
                    inherit(self.references, self.tick)
        prefix_active = self._trigger is None
        result = (self.frozen if prefix_active else self.recovery).step(obs, env)
        self.prefix_ticks += int(prefix_active)
        self.recovery_ticks += int(not prefix_active)
        result.detail.setdefault("inference_ms", 0.)
        result.detail.setdefault("head_inference_ms", 0.)
        result.detail.update(prefix_tick=self.tick, prefix_trigger=self.trigger,
            prefix_active=prefix_active, learned_recovery_active=not prefix_active,
            prefix_ticks=self.prefix_ticks, recovery_ticks=self.recovery_ticks,
            prefix_policy_calls=self.prefix_policy_calls, no_runtime_geometric_teacher=True,
            controller_latency_ms=(perf_counter() - started) * 1000.)
        self.tick += 1
        return result

    def close(self):
        if not self._closed:
            self._closed = True
            self.recovery.close()
