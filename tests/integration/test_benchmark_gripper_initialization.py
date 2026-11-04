"""Matched starts can retain the configured source jaw without changing arm pose."""

from contextlib import ExitStack

import numpy as np
import pytest

from real2sim.config import load_yaml
from scripts.benchmark_frozen_act import MEASURED_SCENE, SharedEpisode

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("robot", ["so101", "panda"])
def test_configured_jaw_is_preserved_and_default_override_is_explicit(tmp_path, robot):
    configured_scene = load_yaml(MEASURED_SCENE)
    configured_q = np.asarray(configured_scene["initial_qpos"], float)
    configured_percent = configured_scene["initial_qpos_real_deg"][5]
    assert configured_percent == pytest.approx(42.52717208862305)
    with ExitStack() as resources:
        default = SharedEpisode(robot, tmp_path / "default")
        resources.callback(default.close)
        retained = SharedEpisode(robot, tmp_path / "configured", initial_gripper_percent=None)
        resources.callback(retained.close)
        default.reset(seed=14000)
        retained.reset(seed=14000)

        np.testing.assert_array_equal(retained.source.source.initial_qpos, configured_q)
        assert retained.source.source.real_state()[5] == pytest.approx(configured_percent, abs=1e-5)
        assert default.source.source.mapping.gripper_sim_to_real_percent(
            default.source.source.initial_qpos[5]) == pytest.approx(1.2)
        np.testing.assert_array_equal(default.source.source.initial_qpos[:5], configured_q[:5])
        np.testing.assert_array_equal(default.source.source.initial_qpos[:5],
                                      retained.source.source.initial_qpos[:5])

        if robot == "so101":
            np.testing.assert_array_equal(default.env.sim_state()[:5], retained.env.sim_state()[:5])
        else:
            np.testing.assert_allclose(default.env.arm_qpos(), retained.env.arm_qpos(), atol=1e-12)
            expected_width = configured_percent / 90. * retained.env.max_opening
            assert retained.env.gripper_opening() == pytest.approx(expected_width, abs=1e-12)
            assert default.env.gripper_opening() == pytest.approx(1.2 / 90. * default.env.max_opening,
                                                                abs=1e-12)
        np.testing.assert_allclose(default.env.ee_pose(), retained.env.ee_pose(), atol=1e-12)
