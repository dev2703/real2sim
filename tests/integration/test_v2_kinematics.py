"""Real MuJoCo Jacobians and native controller paths, with no rendered images."""

import numpy as np
import pytest

from real2sim.panda.adapter import PandaCartesianActionAdapter
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.policy.adapters import CartesianActionAdapter, PandaActionAdapter
from real2sim.policy.kinematic_actions import KinematicDeltaAdapter
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics


@pytest.mark.integration
@pytest.mark.parametrize("robot", ["so101", "panda"])
def test_projection_uses_native_joints_and_retains_original_controller(robot):
    cls = PandaPickPlaceEnv if robot == "panda" else So101PickPlaceEnv
    with cls(render_images=False) as env:
        obs, _ = env.reset(seed=0)
        if robot == "panda":
            native = PandaActionAdapter(PandaCartesianActionAdapter(env, target_reference="accepted_target"))
        else:
            native = CartesianActionAdapter(env, NativeKinematics(env.model, env.config),
                                            target_reference="accepted_target")
        adapter = KinematicDeltaAdapter(native, tool_frame=True,
                                        projection={"rotation_weight": .1, "damping": .001})
        adapter.reset()
        result = adapter.apply(np.array([.001, 0, 0, 0, 0, 0, .5]), obs)
        detail = result.detail["kinematic_delta"]
        assert detail["projection_applied"], detail
        dq = np.asarray(detail["joint_step_rad"])
        assert len(dq) == (7 if robot == "panda" else 5)
        assert np.all(np.abs(dq) <= env.max_velocity[:len(dq)] / env.control_hz + 1e-9)
        assert np.isfinite(detail["projection_position_change_m"])
        assert "ik_reason" in result.detail
        adapter.reset()
        controller = getattr(native, "panda", native)
        assert controller._accepted_target is None
