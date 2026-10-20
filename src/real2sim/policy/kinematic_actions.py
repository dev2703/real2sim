"""Observable-joint retargeting ablations; no object poses enter these adapters.

This is a bounded one-step Jacobian/FK experiment, not a collision planner or an
implementation of LEGATO's hierarchical optimizer / EmbodiSteer's denoising guidance.
The original IK acceptance, actuator limits and rejection handling remain authoritative.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from real2sim.geometry.so3 import apply_delta, delta_action


def bounded_joint_step(jacobian, delta, q, limits, max_step, *, rotation_weight=.1, damping=.001):
    """Regularized task error in metres; rotation_weight converts radians to metres."""
    j, d, q, lim, cap = map(lambda x: np.asarray(x, dtype=float),
                           (jacobian, delta, q, limits, max_step))
    n = len(q)
    if j.shape != (6, n) or d.shape != (6,) or lim.shape != (n, 2) or cap.shape != (n,):
        raise ValueError("invalid bounded kinematic projection shapes")
    if not all(np.isfinite(x).all() for x in (j, d, q, lim, cap)):
        raise ValueError("kinematic projection inputs must be finite")
    if (np.any(cap <= 0) or np.any(lim[:, 0] >= lim[:, 1])
            or not np.isfinite(rotation_weight) or rotation_weight <= 0
            or not np.isfinite(damping) or damping <= 0):
        raise ValueError("positive projection weights, damping, limits and motion caps required")
    if np.any(q < lim[:, 0]) or np.any(q > lim[:, 1]):
        raise ValueError("projection requires an in-range measured state; never silently clip it")
    weights = np.array([1., 1., 1., rotation_weight, rotation_weight, rotation_weight])
    matrix = np.vstack([weights[:, None] * j, damping * np.eye(n)])
    target = np.r_[weights * d, np.zeros(n)]
    lower, upper = np.maximum(lim[:, 0] - q, -cap), np.minimum(lim[:, 1] - q, cap)
    result = lsq_linear(matrix, target, bounds=(lower, upper), tol=1e-10)
    if not result.success:
        raise RuntimeError("bounded kinematic projection did not converge")
    return result.x


class KinematicDeltaAdapter:
    """Optional tool-axis conversion and/or joint-bounded task target projection."""

    dim = 7
    needs_obs = True

    def __init__(self, adapter, *, tool_frame=False, projection=None):
        self.wrapped = adapter
        self.env = adapter.env
        self.controller = getattr(adapter, "panda", adapter)
        self.tool_frame = tool_frame
        self.projection = projection
        if projection is not None and set(projection) - {"rotation_weight", "damping"}:
            raise ValueError("unknown kinematic projection settings")
        if projection is not None and any(not np.isfinite(v) or v <= 0 for v in projection.values()):
            raise ValueError("kinematic projection settings must be finite and positive")

    def reset(self):
        self.wrapped.reset()

    @property
    def stats(self):
        return getattr(self.wrapped, "stats", None)

    @property
    def ik_rejections(self):
        return self.wrapped.ik_rejections

    def apply(self, action, obs=None):
        a = np.asarray(action, dtype=float).copy()
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError("expected seven finite delta action values")
        current = self.env.ee_pose()
        # Labels use the observed tool axes, even when integrating onto an accepted target.
        if self.tool_frame:
            a[:3] = current[:3, :3] @ a[:3]
            a[3:6] = current[:3, :3] @ a[3:6]
        detail = {"delta_axes": "observed_tool" if self.tool_frame else "task"}
        if self.projection is not None:
            c = self.controller
            reference = getattr(c, "_accepted_target", None)
            reference = current if reference is None or c.target_reference == "current_pose" else reference
            desired = apply_delta(reference, a[:6])
            ik = c.ik
            is_panda = hasattr(self.wrapped, "panda")
            q = self.env.arm_qpos() if is_panda else self.env.sim_state()
            count = 7 if is_panda else 5  # gripper joints never solve arm motion
            try:
                ik.forward(q)
                jp, jr = np.zeros((3, ik.model.nv)), np.zeros((3, ik.model.nv))
                mujoco.mj_jacSite(ik.model, ik.data, jp, jr, ik.site)
                j = np.vstack([jp, jr])[:, ik.vadr[:count]]
                dq = bounded_joint_step(
                    j, delta_action(current, desired), q[:count], ik.limits[:count],
                    self.env.max_velocity[:count] / self.env.control_hz, **self.projection,
                )
                target_q = q.copy()
                target_q[:count] += dq
                feasible = ik.forward(target_q)
                a[:6] = delta_action(reference, feasible)
                residual = delta_action(feasible, desired)
                detail.update(projection_applied=True, joint_step_rad=dq.tolist(),
                              projection_position_change_m=float(np.linalg.norm(residual[:3])),
                              projection_rotation_change_rad=float(np.linalg.norm(residual[3:])))
            except (ValueError, RuntimeError) as error:
                # Retain the original request and let existing IK reject/hold it.
                detail.update(projection_applied=False, projection_failure=str(error))
        result = self.wrapped.apply(a, obs) if getattr(self.wrapped, "needs_obs", False) else self.wrapped.apply(a)
        result.detail["kinematic_delta"] = detail
        return result
