"""Submillimetre Cartesian commands must survive optional IK polishing."""
import copy

import numpy as np
import pytest

from real2sim.panda.adapter import PandaIK
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics

pytestmark = pytest.mark.integration


@pytest.fixture(params=["so101", "panda"])
def case(request):
    cls = So101PickPlaceEnv if request.param == "so101" else PandaPickPlaceEnv
    with cls(render_images=False) as env:
        env.reset(seed=0)
        cfg = copy.deepcopy(env.config)
        cfg["ik"].update(position_convergence_m=1e-5, orientation_convergence_rad=1e-4)
        if request.param == "so101":
            seed = np.array([0., -.8, .8, .8, 0., .3])
            def build(c):
                return NativeKinematics(env.model, c)
            n = 5
        else:
            seed = env.initial_qpos.copy()
            def build(c):
                return PandaIK(env.model, c, task_origin=env.offset)
            n = 7
        yield env, cfg, build, seed, n


def test_small_reachable_command_moves_arm_and_reduces_residual(case):
    env, cfg, build, seed, n = case
    kin = build(cfg)
    target_q = seed.copy()
    target_q[0] += .0008
    target = kin.forward(target_q)
    initial_error = np.linalg.norm(target[:3, 3] - kin.forward(seed)[:3, 3])
    assert 1e-5 < initial_error < .0003
    before = env.data.qpos.copy()
    result = kin.solve(target, seed)
    assert result.accepted
    assert result.iterations > 0
    assert np.max(np.abs(result.qpos[:n] - seed[:n])) > 1e-5
    assert result.position_error_m < initial_error / 5
    np.testing.assert_array_equal(env.data.qpos, before)
    # Frozen configs still stop at the same coarse tolerance as previous benchmarks.
    legacy = build(env.config).solve(target, seed)
    assert legacy.accepted and legacy.iterations == 0
    np.testing.assert_array_equal(legacy.qpos, seed)


def test_polishing_preserves_workspace_and_collision_rejection(case, monkeypatch):
    _, cfg, build, seed, _ = case
    kin = build(cfg)
    target = kin.forward(seed)
    target[0, 3] = 5
    rejected = kin.solve(target, seed)
    assert not rejected.accepted and rejected.reason == "outside_workspace"
    np.testing.assert_array_equal(rejected.qpos, seed)
    target = kin.forward(seed)
    attr = "_collision" if isinstance(kin, NativeKinematics) else "_penetrates"
    monkeypatch.setattr(kin, attr, lambda: True)
    rejected = kin.solve(target, seed)
    assert not rejected.accepted and rejected.reason == "collision"
    np.testing.assert_array_equal(rejected.qpos, seed)


@pytest.mark.parametrize("field", ["position_convergence_m", "orientation_convergence_rad"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 1.0])
def test_invalid_convergence_refused(case, field, value):
    _, cfg, build, _, _ = case
    cfg["ik"][field] = value
    with pytest.raises(ValueError, match="convergence"):
        build(cfg)


def test_iteration_budget_does_not_apply_unaccepted_partial_target(case):
    _, cfg, build, seed, _ = case
    cfg["ik"]["max_iterations"] = 0
    kin = build(cfg)
    target_q = seed.copy()
    target_q[0] += .1
    target = kin.forward(target_q)
    result = kin.solve(target, seed)
    assert not result.accepted
    assert result.iterations == 0
    assert result.position_error_m > cfg["ik"]["position_tolerance_m"]
    np.testing.assert_array_equal(result.qpos, seed)
