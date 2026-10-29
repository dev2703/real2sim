"""Actual virtual-source FK/projection and optional source recovery agree."""

import numpy as np
import pytest

from real2sim.geometry.so3 import apply_delta
from real2sim.panda.model import POINTING_DOWN
from real2sim.policy.common_recovery import semantic_gripper
from real2sim.policy.source_recovery_frame import SourceRecoveryFrame
from scripts.diag_v2_learned_panda_retarget_20261027 import (
    SOURCE_OFFSET,
    VirtualSource,
    target_from_source,
)

pytestmark = pytest.mark.integration


class ObservedPanda:
    max_opening = .08

    def __init__(self, pose):
        self.pose = pose

    def ee_pose(self):
        return self.pose.copy()

    def gripper_opening(self):
        return .02

    def gripper_semantic(self):
        return .25


def test_source_rotation_is_retained_as_native_grasp_point_motion():
    source = VirtualSource("configs/v2/sim_so101.waypoint_20261026.yaml")
    try:
        command = source.q.copy()
        command[1:5] += [.05, -.03, -.1, .1]
        source.last_target = command.copy()
        native = ObservedPanda(target_from_source(source, command))
        adapter = SourceRecoveryFrame(native, source, source_offset=SOURCE_OFFSET)
        proxy = adapter.update()
        assert semantic_gripper(proxy, "so101") == pytest.approx(.25)
        assert adapter.diagnostics()["projection"]["accepted"]
        assert adapter.diagnostics()["projection"]["position_tolerance_m"] == .003
        np.testing.assert_allclose(proxy.ee_pose(), source.fk.fk_sim(command[:5]), atol=1e-9)
        np.testing.assert_allclose(adapter.map_action(np.zeros(7)), np.zeros(7), atol=1e-12)
        action = np.array([.002, -.001, .001, .08, -.04, .02, .3])
        source_target = apply_delta(proxy.ee_pose(), action[:6])
        source_grasp = source_target[:3, 3] + source_target[:3, :3] @ SOURCE_OFFSET
        mapped = adapter.map_action(action)
        native_target = apply_delta(native.pose, mapped[:6])
        native_grasp = native_target[:3, 3] + native_target[:3, :3] @ source.panda_offset
        np.testing.assert_allclose(native_grasp, source_grasp, atol=1e-9)
        np.testing.assert_allclose(native_target[:3, :3], POINTING_DOWN, atol=1e-12)
        assert adapter.diagnostics()["source_orientation_is_measured"] is False
    finally:
        source.close()
