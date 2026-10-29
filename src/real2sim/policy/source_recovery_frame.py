"""Optional source-schema recovery on a measured Panda virtual SO-101 state.

This adapter supplies a source-trained head with source TCP coordinates and
maps its source grasp-point displacement back to the native Panda interface.
The source orientation is a causal command prior, never a Panda measurement.
Native IK, collision, rate and source-representability checks remain external.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from real2sim.geometry.so3 import apply_delta, delta_action
from real2sim.panda.model import POINTING_DOWN
from real2sim.policy.common_recovery import clip_common


class _SourceObservation:
    """Only the measured/proxy robot reads needed by RecoveryObservation."""

    def __init__(self, pose, q, mapping):
        self._pose, self._q, self.mapping = pose.copy(), q.copy(), mapping

    def ee_pose(self):
        return self._pose.copy()

    def sim_state(self):
        return self._q.copy()


class SourceRecoveryFrame:
    """Snapshot source observations and map source actions onto measured Panda.

    Call update() before an ACT query or proposal on every physical control
    tick, even when ACT consumes a cached chunk. Encode its observation_env
    with RecoveryObservation(robot="so101"). This explicitly selects the
    head's source schema; it does not assert a native SO-101 embodiment.

    hf_to_common() then records the current ACT proposal as the causal source
    rotation prior, while both it and map_action() use this tick's unchanged
    source snapshot. Keep selected-action history in the source schema and
    audit mapped native actions separately.
    """

    schema = "source_so101_recovery_on_measured_panda_v1"

    def __init__(self, env, source, *, source_offset):
        self.env, self.source = env, source
        self.source_offset = np.asarray(source_offset, float).copy()
        if self.source_offset.shape != (3,) or not np.isfinite(self.source_offset).all():
            raise ValueError("Source recovery requires a finite 3D source grasp offset")
        self.reset()

    def reset(self):
        self.observation_env = None
        self.source_pose = self.native_pose = None
        self.last_projection = self.last_mapping = None
        self._projection_residual = None

    def update(self):
        """Refresh the source representation using robot-only measured feedback."""
        # A rejected update must not leave a usable snapshot from an older tick.
        self.reset()
        pose = np.asarray(self.env.ee_pose(), float).copy()
        opening = float(self.env.gripper_opening())
        semantic = float(self.env.gripper_semantic())
        if not np.isfinite([opening, semantic]).all():
            raise ValueError("Source recovery requires finite measured Panda jaw feedback")
        measured = SimpleNamespace(ee_pose=lambda: pose.copy(),
            gripper_opening=lambda: opening, max_opening=self.env.max_opening)
        self.source.source_state(measured)
        source_pose = self.source.fk.fk_sim(self.source.q[:5])
        q = self.source.q.copy()
        # Source-state CAD gap inversion and semantic intent are distinct. The
        # recovery head's jaw channel always retains measured native semantics.
        q[5] = self.source.mapping.gripper_real_to_sim_rad(np.clip(semantic, 0., 1.) * 90.)
        self.source_pose, self.native_pose = source_pose.copy(), pose
        self.observation_env = _SourceObservation(source_pose, q, self.source.mapping)
        self.last_projection = deepcopy(self.source.last_projection)
        native_point = pose[:3, 3] + pose[:3, :3] @ self.source.panda_offset
        self._projection_residual = self._source_point(source_pose) - native_point
        return self.observation_env

    def _require_snapshot(self):
        if self.observation_env is None:
            raise RuntimeError("Update the source recovery frame before encoding or mapping actions")

    def _source_point(self, pose):
        return pose[:3, 3] + pose[:3, :3] @ self.source_offset

    def hf_to_common(self, raw):
        """ACT's current source-FK proposal in the source head's action schema."""
        self._require_snapshot()
        raw = np.asarray(raw, float)
        _, bounded = self.source.source_action(raw)
        target = self.source.fk.fk_sim(bounded[:5])
        return clip_common(np.r_[delta_action(self.source_pose, target), raw[5] / 90.])

    def map_action(self, action):
        """Preserve source grasp-point change, anchored on measured native pose.

        Anchoring avoids turning the tolerated source projection residual into
        a geometric corrective motion. Source rotation deltas contribute their
        actual grasp-point displacement instead of being discarded by Panda's
        fixed task orientation. The unchanged native common-action cap applies
        afterward, before the external native executor's feasibility checks.
        """
        self._require_snapshot()
        action = clip_common(action)
        source_target = apply_delta(self.source_pose, action[:6])
        displacement = self._source_point(source_target) - self._source_point(self.source_pose)
        native_point = (self.native_pose[:3, 3]
                        + self.native_pose[:3, :3] @ self.source.panda_offset)
        target = np.eye(4)
        target[:3, :3] = POINTING_DOWN
        target[:3, 3] = native_point + displacement - POINTING_DOWN @ self.source.panda_offset
        requested = np.r_[delta_action(self.native_pose, target), action[6]]
        mapped = clip_common(requested)
        self.last_mapping = {"source_common_action": action.tolist(),
            "source_grasp_displacement_task_m": displacement.tolist(),
            "unbounded_native_common_action": requested.tolist(),
            "native_common_action": mapped.tolist(),
            "native_action_clipped": not np.allclose(requested, mapped, rtol=0., atol=1e-12)}
        return mapped

    def diagnostics(self):
        self._require_snapshot()
        return {"schema": self.schema, "encoded_robot_schema": "so101",
            "native_robot": "panda", "source_orientation_is_measured": False,
            "action_mapping": "source_grasp_displacement_anchored_on_measured_panda",
            "projection": deepcopy(self.last_projection),
            "projection_residual_task_m": self._projection_residual.tolist(),
            "projection_residual_norm_m": float(np.linalg.norm(self._projection_residual)),
            "source_tcp_pose": self.source_pose.tolist(),
            "measured_native_tcp_pose": self.native_pose.tolist(),
            "mapping": deepcopy(self.last_mapping)}
