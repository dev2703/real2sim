"""Real learned-target regressions for the optional numerical IK fallback."""

import copy

import numpy as np
import pytest

from real2sim.geometry.so3 import delta_action
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics

pytestmark = pytest.mark.integration

# Recorded learned rollout seed 1, tick 187. The native simulator's soft wrist
# stop overshot slightly; the adapter clips that measured seed before solving.
RESIDUAL_SEED = np.array([-.4794674664265274, .9167133166090302, -1.281067510569535,
                          1.6580864276076321, -.15955410599246211, .8909650930547597])
RESIDUAL_TARGET = np.array([
    [.21196979562583204, -.26492128327914966, .9406835383953891, .3179099410915888],
    [.07864649321470409, .964058706120222, .2537824664133262, .14270345058682338],
    [-.974106531570018, .020187243980544476, .2251864568154522, .06777048703237087],
    [0., 0., 0., 1.],
])
GRIP = .8882052680015565

# Seed 0, tick 278: the policy requested a pose with gripper/table penetration.
COLLISION_SEED = np.array([.11549601995275965, .01393079238004808, .6179149408455855,
                           .9565245354209064, .2062053687655447, -.17285606645014298])
COLLISION_TARGET = np.array([
    [-.020763056794554223, -.03052317345694377, .9993183833767211, .21907075867889952],
    [-.024360589961289175, .9992525508547289, .03001501735889715, -.021805676422472827],
    [-.9994875972863714, -.023720781868106375, -.021491099978401755, -.007911510868373717],
    [0., 0., 0., 1.],
])


@pytest.fixture
def setup():
    with So101PickPlaceEnv(render_images=False) as env:
        config = copy.deepcopy(env.config)
        config["ik"].update(position_convergence_m=1e-5, orientation_convergence_rad=1e-4)
        yield env, config


def test_position_priority_recovers_feasible_recorded_target_without_changing_gates(setup):
    env, config = setup
    seed = np.clip(RESIDUAL_SEED, env.limits[:, 0], env.limits[:, 1])
    baseline = NativeKinematics(env.model, config).solve(RESIDUAL_TARGET, seed, gripper=GRIP)
    assert baseline.reason == "residual_rejected"
    assert baseline.position_error_m > config["ik"]["position_tolerance_m"]
    config["ik"]["position_priority_fallback"] = True
    unchanged = copy.deepcopy(config)
    live_before = env.data.qpos.copy()
    kin = NativeKinematics(env.model, config)
    result = kin.solve(RESIDUAL_TARGET, seed, gripper=GRIP)
    assert result.accepted
    actual_error = delta_action(kin.forward(result.qpos), RESIDUAL_TARGET)
    assert np.linalg.norm(actual_error[:3]) <= config["ik"]["position_tolerance_m"]
    assert np.linalg.norm(actual_error[3:]) <= config["ik"]["orientation_tolerance_rad"]
    assert not kin._collision()
    assert np.all(result.qpos >= kin.limits[:, 0])
    assert np.all(result.qpos <= kin.limits[:, 1])
    assert config == unchanged
    np.testing.assert_array_equal(env.data.qpos, live_before)


def test_position_priority_preserves_real_collision_and_returns_original_seed(setup):
    env, config = setup
    baseline = NativeKinematics(env.model, config).solve(
        COLLISION_TARGET, COLLISION_SEED, gripper=-.17201138297915458)
    config["ik"]["position_priority_fallback"] = True
    fallback = NativeKinematics(env.model, config).solve(
        COLLISION_TARGET, COLLISION_SEED, gripper=-.17201138297915458)
    assert baseline.reason == fallback.reason == "collision"
    assert not fallback.accepted
    np.testing.assert_array_equal(fallback.qpos, COLLISION_SEED)
    assert fallback.position_error_m == baseline.position_error_m
    assert fallback.orientation_error_rad == baseline.orientation_error_rad


def test_impossible_fallback_preserves_original_failure_and_seed(setup):
    env, config = setup
    seed = np.clip(RESIDUAL_SEED, env.limits[:, 0], env.limits[:, 1])
    target = RESIDUAL_TARGET.copy()
    target[:3, 3] = [.5, .4, .55]  # inside the box, outside native arm reach
    baseline = NativeKinematics(env.model, config).solve(target, seed, gripper=GRIP)
    config["ik"]["position_priority_fallback"] = True
    kin = NativeKinematics(env.model, config)
    fallback = kin.solve(target, seed, gripper=GRIP)
    assert not baseline.accepted and not fallback.accepted
    assert baseline.reason == fallback.reason == "residual_rejected"
    assert baseline.position_error_m == fallback.position_error_m
    assert baseline.orientation_error_rad == fallback.orientation_error_rad
    np.testing.assert_array_equal(fallback.qpos, seed)
    candidate_pose = kin.forward(kin.data.qpos[kin.qadr].copy())
    candidate_error = delta_action(candidate_pose, target)
    assert np.linalg.norm(candidate_error[:3]) == pytest.approx(fallback.position_error_m)
    assert np.linalg.norm(candidate_error[3:]) == pytest.approx(fallback.orientation_error_rad)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_fallback_flag_rejects_non_boolean_values(setup, value):
    env, config = setup
    config["ik"]["position_priority_fallback"] = value
    with pytest.raises(ValueError, match="must be a boolean"):
        NativeKinematics(env.model, config)
