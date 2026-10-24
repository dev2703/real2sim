"""Explicit joint-target inference with existing native/workspace/collision gates.

Static interpolation checks are conservative command-path probes, not a guarantee
about swept mesh collision or physical tracking. Native servo dynamics and numerical
abort rules remain authoritative and unchanged.
"""

from __future__ import annotations

import numpy as np

from real2sim.data.follower_joint_targets import VARIANTS, VERSION
from real2sim.policy.adapters import ApplyResult, ContractError
from real2sim.sim.env import sim_from_real


def validate_semantics(contract: dict) -> int:
    value = contract.get("joint_target_semantics", {})
    name = value.get("variant")
    if (value.get("version") != VERSION or name not in VARIANTS
            or value.get("inference") != "guarded_absolute_native_joint_target_v1"
            or value.get("arm_units") != "degrees"
            or value.get("gripper_target") != "current_recorded_leader_command_percent"
            or value.get("lead_policy_ticks") != (1 if str(name).startswith("follower") else 0)):
        raise ContractError("unsupported explicit joint_target_semantics")
    count = value.get("collision_path_samples")
    if type(count) is not int or not 3 <= count <= 101:
        raise ContractError("joint-target collision samples must be an integer in [3,101]")
    return count


def check_joint_request(kinematics, current, command_start, requested, max_delta,
                        samples: int = 17) -> tuple[bool, str, np.ndarray]:
    """Check the requested endpoint and both rate-bounded command-path proxies."""
    current, command_start, requested, max_delta = map(
        lambda value: np.asarray(value, float), (current, command_start, requested, max_delta))
    if (any(v.shape != (6,) or not np.isfinite(v).all() for v in
            (current, command_start, requested, max_delta)) or np.any(max_delta <= 0)):
        raise ValueError("six finite joints and positive per-tick rate bounds required")
    if type(samples) is not int or samples < 3:
        raise ValueError("at least three static path samples required")
    limits = kinematics.limits
    if np.any(current < limits[:, 0]) or np.any(current > limits[:, 1]):
        return False, "current_native_limits", current.copy()
    if np.any(command_start < limits[:, 0]) or np.any(command_start > limits[:, 1]):
        return False, "command_native_limits", current.copy()
    if np.any(requested < limits[:, 0]) or np.any(requested > limits[:, 1]):
        return False, "target_native_limits", current.copy()
    applied = command_start + np.clip(requested - command_start, -max_delta, max_delta)

    def probe(q):
        pose = kinematics.forward(q)
        if (np.any(pose[:3, 3] < kinematics.config["workspace_min"])
                or np.any(pose[:3, 3] > kinematics.config["workspace_max"])):
            return "outside_workspace"
        return "collision" if kinematics._collision() else None

    reason = probe(requested)
    if reason:
        return False, "target_" + reason, current.copy()
    for start in (current, command_start):
        for fraction in np.linspace(0, 1, samples):
            reason = probe(start + fraction * (applied - start))
            if reason:
                return False, "path_" + reason, current.copy()
    return True, "accepted", applied


class GuardedJointActionAdapter:
    """Saved degree/% absolute targets; rejected commands hold the observed joints."""

    dim = 6

    def __init__(self, env, kinematics, contract):
        self.env, self.ik = env, kinematics
        self.samples = validate_semantics(contract)
        self.ik_rejections = 0

    def reset(self):
        self.ik_rejections = 0

    def apply(self, action):
        a = np.asarray(action, float)
        if a.shape != (6,) or not np.isfinite(a).all():
            raise ValueError("six finite degree/percent joint targets required")
        desired = sim_from_real(self.env.mapping, a)
        current = self.env.sim_state()
        start = self.env.data.ctrl[self.env.actuator_ids].copy()
        # Preserve independent probes while copying only state used by collision geometry.
        self.ik.data.qpos[:] = self.env.data.qpos
        accepted, reason, applied = check_joint_request(
            self.ik, current, start, desired, self.env.max_velocity / self.env.control_hz, self.samples)
        if not accepted:
            self.ik_rejections += 1
        obs, _, _, truncated, info = self.env.step(desired if accepted else current)
        return ApplyResult(obs, info, truncated, float(np.mean(info["control"]["limit_clipped"])),
                           ik_rejected=not accepted,
                           detail={"joint_target_semantics": VERSION, "ik_reason": reason,
                                   "requested_native_target_rad": desired.tolist(),
                                   "checked_rate_bounded_target_rad": applied.tolist(),
                                   "collision_path_samples": self.samples,
                                   "collision_scope": "static sampled joint-path proxies; dynamics unchanged",
                                   "rejection_behavior": "hold observed arm and gripper",
                                   "rate_limited": info["control"]["rate_limited"]})
