"""Position-only transfer preserves commanded source coordinates causally."""

import numpy as np
import pytest

from scripts.diag_v2_learned_panda_retarget_20261027 import VirtualSource, target_from_source

pytestmark = pytest.mark.integration


class ObservedPanda:
    max_opening = .08

    def __init__(self, pose):
        self.pose = pose

    def ee_pose(self):
        return self.pose.copy()

    def gripper_opening(self):
        return self.max_opening


def test_achieved_command_round_trip_preserves_source_wrist_coordinates():
    source = VirtualSource("configs/v2/sim_so101.waypoint_20261026.yaml")
    try:
        command = source.q.copy()
        command[3:5] += [-.5, .4]
        source.last_target = command.copy()
        observed = ObservedPanda(target_from_source(source, command))
        state, error = source.source_state(observed)
        assert error < 1e-10
        np.testing.assert_allclose(source.q, command, atol=1e-9)
        np.testing.assert_allclose(state[:5],
            (np.rad2deg(command[:5]) - source.mapping.offset_deg) / source.mapping.sign,
            atol=1e-5)
    finally:
        source.close()


def test_measured_tracking_error_corrects_virtual_command_position():
    source = VirtualSource("configs/v2/sim_so101.waypoint_20261026.yaml")
    try:
        command = source.q.copy()
        command[1:4] += [.1, -.05, -.2]
        source.last_target = command.copy()
        measured = command.copy()
        measured[0] += .03
        observed = ObservedPanda(target_from_source(source, measured))
        equivalent = source.ball_point(measured)
        state, error = source.source_state(observed)
        assert np.isfinite(state).all()
        assert error < 1e-5
        np.testing.assert_allclose(source.ball_point(source.q), equivalent, atol=1e-5)
        assert np.linalg.norm(source.ball_point(command) - equivalent) > .001
        assert not np.allclose(source.q[:5], command[:5])
    finally:
        source.close()


def test_projection_rejects_unknown_null_space_prior():
    with pytest.raises(ValueError, match="projection prior"):
        VirtualSource("unused", projection_prior="future")
