"""Position-only IK probes retain full-forward geometry and acceptance decisions."""

import mujoco
import numpy as np
import pytest

from real2sim.panda.adapter import PandaIK
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics

pytestmark = pytest.mark.integration


@pytest.fixture(params=("so101", "panda"))
def setup(request):
    env_class = So101PickPlaceEnv if request.param == "so101" else PandaPickPlaceEnv
    with env_class(render_images=False) as env:
        env.reset(seed=7)
        solver = (NativeKinematics(env.model, env.config) if request.param == "so101"
                  else PandaIK(env.model, env.config, task_origin=env.offset))
        solver.data.qpos[:] = env.data.qpos
        yield env, solver


def test_position_stage_matches_full_forward_poses_jacobians_and_contacts(setup):
    env, solver = setup
    reference = mujoco.MjData(env.model)
    rng = np.random.default_rng(31)
    contact_count = 0
    for q in rng.uniform(solver.limits[:, 0], solver.limits[:, 1], size=(32, len(solver.qadr))):
        solver.forward(q)
        reference.qpos[:] = solver.data.qpos
        mujoco.mj_forward(env.model, reference)
        for name in ("site_xpos", "site_xmat", "cam_xpos", "cam_xmat"):
            np.testing.assert_array_equal(getattr(solver.data, name), getattr(reference, name))
        actual_jac = np.zeros((6, env.model.nv))
        reference_jac = np.zeros_like(actual_jac)
        mujoco.mj_jacSite(env.model, solver.data, actual_jac[:3], actual_jac[3:], solver.site)
        mujoco.mj_jacSite(env.model, reference, reference_jac[:3], reference_jac[3:], solver.site)
        np.testing.assert_array_equal(actual_jac, reference_jac)
        assert solver.data.ncon == reference.ncon
        ncon = reference.ncon
        contact_count += ncon
        for name in ("geom1", "geom2", "dist", "pos", "frame"):
            np.testing.assert_array_equal(getattr(solver.data.contact, name)[:ncon],
                                          getattr(reference.contact, name)[:ncon])
    assert contact_count > 0  # Include actual collision probes, not only free-space FK.


def test_ik_solutions_and_rejections_match_full_forward(setup, monkeypatch):
    env, solver = setup
    seed = (np.array([0., -.8, .8, .8, 0., .3]) if len(solver.qadr) == 6
            else env.initial_qpos.copy())
    target = solver.forward(seed)
    nearby = target.copy()
    nearby[0, 3] += .005
    outside = target.copy()
    outside[0, 3] = 5.
    outcomes = []
    for requested in (target, nearby, outside):
        actual = solver.solve(requested, seed)
        with monkeypatch.context() as context:
            context.setattr(mujoco, "mj_fwdPosition", mujoco.mj_forward)
            reference = solver.solve(requested, seed)
        assert actual.accepted == reference.accepted
        assert actual.reason == reference.reason
        assert actual.iterations == reference.iterations
        assert actual.position_error_m == reference.position_error_m
        assert actual.orientation_error_rad == reference.orientation_error_rad
        np.testing.assert_array_equal(actual.qpos, reference.qpos)
        outcomes.append(actual.accepted)
    assert any(outcomes) and not all(outcomes)
