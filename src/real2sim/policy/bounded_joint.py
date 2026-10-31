"""Executable joint commands with explicit gripper actuator saturation.

Regressed gripper percentages can cross the physical closed/open endpoints by
small amounts. Saturate that actuator before checking the complete command so
an overclosed gripper prediction does not freeze a valid arm movement. Arm
targets, collision/workspace gates, velocity bounds and all training labels are
unchanged. Raw requests and saturation remain visible in every execution trace.
"""

from __future__ import annotations

import numpy as np

from real2sim.policy.adapters import ApplyResult, JointActionAdapter
from real2sim.policy.follower_joint import check_joint_request
from real2sim.sim.env import sim_from_real
from real2sim.sim.kinematics import NativeKinematics


def bound_gripper_command(mapping, limits, action):
    action = np.asarray(action, float)
    limits = np.asarray(limits, float)
    if action.shape != (6,) or not np.isfinite(action).all():
        raise ValueError("six finite degree/percent joint targets required")
    if limits.shape != (6, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("six finite ordered native joint limits required")
    requested = sim_from_real(mapping, action)
    mapped = np.sort(mapping.gripper_real_to_sim_rad(np.array([0., 100.])))
    lower, upper = max(mapped[0], limits[5, 0]), min(mapped[1], limits[5, 1])
    if lower > upper:
        raise ValueError("gripper percentage interval and native limits do not intersect")
    bounded = requested.copy()
    bounded[5] = np.clip(requested[5], lower, upper)
    return requested, bounded


class BoundedGripperJointActionAdapter(JointActionAdapter):
    """Keep arm hard gates; independently saturate the gripper actuator."""

    def __init__(self, env, *, samples=17):
        super().__init__(env)
        self.ik = NativeKinematics(env.model, env.config)
        self.samples = samples

    def apply(self, action):
        requested, bounded = bound_gripper_command(self.env.mapping, self.env.limits, action)
        current = self.env.sim_state()
        start = self.env.data.ctrl[self.env.actuator_ids].copy()
        self.ik.data.qpos[:] = self.env.data.qpos
        accepted, reason, applied = check_joint_request(self.ik, current, start, bounded,
            self.env.max_velocity / self.env.control_hz, samples=self.samples)
        obs, _, _, truncated, info = self.env.step(bounded if accepted else current)
        saturated = bool(requested[5] != bounded[5])
        saturation = np.asarray(info["control"]["limit_clipped"], bool).copy()
        saturation[5] |= saturated
        return ApplyResult(obs, info, truncated, float(np.mean(saturation)),
            ik_rejected=not accepted,
            detail={"requested_native_rad": requested.tolist(),
                "bounded_native_target_rad": bounded.tolist(),
                "gripper_saturated": saturated,
                "gripper_saturation_rad": float(bounded[5] - requested[5]),
                "checked_applied_native_rad": applied.tolist(), "guard_reason": reason,
                "arm_target_clipping": False, "native_limits_changed": False,
                "task_corrections": 0, "corrective_ik_calls": 0})
