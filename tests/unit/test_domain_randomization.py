"""Episode randomization is declared, repeatable and independent of episode order."""

import json

import numpy as np
import pytest

from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.sim.domain_randomization import DomainRandomizationConfig, DomainRandomizer
from real2sim.sim.env import So101PickPlaceEnv


@pytest.fixture(params=[So101PickPlaceEnv, PandaPickPlaceEnv], ids=["so101", "panda"])
def env(request):
    environment = request.param(render_images=False)
    try:
        yield environment
    finally:
        environment.close()


def model_parameters(env):
    return {name: np.asarray(getattr(env.model, name)).copy() for name in DomainRandomizer._MODEL_ARRAYS}


def test_sampling_is_pure_and_can_round_trip_through_json(env):
    randomizer = DomainRandomizer(env)
    before = model_parameters(env)
    first = randomizer.sample(201)
    assert first == randomizer.sample(201)
    assert first == json.loads(json.dumps(first, allow_nan=False))
    assert first["sha256"] != randomizer.sample(202)["sha256"]
    assert first["profile_sha256"] == randomizer.profile_sha256
    for name, array in before.items():
        np.testing.assert_array_equal(getattr(env.model, name), array)
    assert len(randomizer.baseline_sha256) == 64


def test_apply_restores_baseline_before_each_episode_and_does_not_accumulate(env):
    randomizer = DomainRandomizer(env)
    original = model_parameters(env)
    first = randomizer.apply(42)
    first_model = model_parameters(env)
    # Reset moves the bowl's model body. A later application still returns to
    # the immutable original geometry before drawing its next layout.
    env.reset(seed=42, scenario=first["scenario"])
    randomizer.apply(43)
    repeated = randomizer.apply(42)
    assert repeated == first
    for name, array in first_model.items():
        np.testing.assert_array_equal(getattr(env.model, name), array)
    randomizer.restore()
    for name, array in original.items():
        np.testing.assert_array_equal(getattr(env.model, name), array)


def test_physics_variation_scales_ball_inertia_and_contact_friction_without_changing_robot(env):
    model = env.model
    before = model_parameters(env)
    protected = {name: np.asarray(getattr(model, name)).copy() for name in (
        "jnt_range", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "geom_size")}
    task_radius, task_mass = env.task["ball"]["radius"], env.task["ball"]["mass"]
    descriptor = DomainRandomizer(env).apply(303)
    parameters = descriptor["parameters"]
    ball_body = model.body("ball").id
    others = np.arange(model.nbody) != ball_body
    scale = parameters["ball_mass_scale"]
    assert model.body_mass[ball_body] == pytest.approx(before["body_mass"][ball_body] * scale)
    np.testing.assert_allclose(model.body_inertia[ball_body], before["body_inertia"][ball_body] * scale)
    np.testing.assert_array_equal(model.body_mass[others], before["body_mass"][others])
    np.testing.assert_array_equal(model.body_inertia[others], before["body_inertia"][others])
    collision = (model.geom_contype | model.geom_conaffinity) != 0
    np.testing.assert_allclose(model.geom_friction[collision, 0],
                               before["geom_friction"][collision, 0] * parameters["friction_scale"])
    np.testing.assert_array_equal(model.geom_friction[:, 1:], before["geom_friction"][:, 1:])
    np.testing.assert_array_equal(model.geom_friction[~collision], before["geom_friction"][~collision])
    for name, array in protected.items():
        np.testing.assert_array_equal(getattr(model, name), array)
    assert env.task["ball"]["radius"] == task_radius
    assert env.task["ball"]["mass"] == task_mass  # declared reference measurement remains intact


def test_camera_appearance_and_layout_draws_respect_declared_bounds(env):
    randomizer = DomainRandomizer(env)
    before = model_parameters(env)
    cfg = randomizer.config
    descriptor = randomizer.apply(800)
    params, scenario = descriptor["parameters"], descriptor["scenario"]
    assert np.max(np.abs(env.model.cam_pos - before["cam_pos"])) <= cfg.camera_position_jitter_m
    np.testing.assert_allclose(np.linalg.norm(env.model.cam_quat, axis=1), 1., atol=1e-12)
    assert not np.array_equal(env.model.cam_quat, before["cam_quat"])
    assert np.max(np.linalg.norm(params["camera_rotation_vector_rad"], axis=1)) <= np.deg2rad(cfg.camera_rotation_jitter_deg)
    ratios = env.model.cam_fovy / before["cam_fovy"]
    assert np.all((ratios >= cfg.camera_fovy_scale[0]) & (ratios <= cfg.camera_fovy_scale[1]))
    np.testing.assert_array_equal(env.model.geom_rgba[:, 3], before["geom_rgba"][:, 3])
    np.testing.assert_array_equal(env.model.mat_rgba[:, 3], before["mat_rgba"][:, 3])
    assert np.all((env.model.geom_rgba >= 0) & (env.model.geom_rgba <= 1))
    assert np.linalg.norm(np.asarray(scenario["ball_xy"]) - env.task["ball"]["start_region"]["center_task_xy"]) <= cfg.ball_position_jitter_m
    assert np.linalg.norm(np.asarray(scenario["bowl_xy"]) - env.task["bowl"]["center_task_xy"]) <= cfg.bowl_position_jitter_m
    _, info = env.reset(seed=800, scenario=scenario)
    np.testing.assert_allclose(info["privileged_metrics"]["ball_xyz_task"][:2], scenario["ball_xy"])
    np.testing.assert_allclose(info["privileged_metrics"]["bowl_xyz_task"][:2], scenario["bowl_xy"])
    assert np.isfinite(env.data.qpos).all()


def test_shared_seeds_have_identical_layout_and_physics_draws_across_arms():
    so101, panda = So101PickPlaceEnv(render_images=False), PandaPickPlaceEnv(render_images=False)
    try:
        # Match the task geometry explicitly; arm model sizes/camera counts differ.
        panda.task["ball"]["start_region"]["center_task_xy"] = list(so101.task["ball"]["start_region"]["center_task_xy"])
        panda.task["bowl"]["center_task_xy"] = list(so101.task["bowl"]["center_task_xy"])
        source, target = DomainRandomizer(so101), DomainRandomizer(panda)
        first, second = source.sample(500), target.sample(500)
        assert first["scenario"] == second["scenario"]
        for name in ("friction_scale", "ball_mass_scale", "light_scale"):
            assert first["parameters"][name] == second["parameters"][name]
    finally:
        so101.close()
        panda.close()


@pytest.mark.parametrize("field,value", [
    ("unknown", 1), ("ball_position_jitter_m", -1), ("camera_position_jitter_m", True),
    ("bowl_position_jitter_m", float("nan")), ("camera_rotation_jitter_deg", 181),
    ("color_offset", float("inf")), ("friction_scale", [0, 1]),
    ("ball_mass_scale", [2, 1]), ("light_scale", [.5]), ("camera_fovy_scale", "bad"),
])
def test_invalid_profiles_are_rejected(field, value):
    with pytest.raises(ValueError):
        DomainRandomizationConfig.from_dict({field: value})


@pytest.mark.parametrize("seed", [-1, 1.5, True, "42"])
def test_invalid_episode_seeds_are_rejected(env, seed):
    with pytest.raises(ValueError, match="nonnegative integer"):
        DomainRandomizer(env).apply(seed)


def test_config_json_and_explicit_identity_profile(env, tmp_path):
    payload = {"ball_position_jitter_m": 0., "bowl_position_jitter_m": 0.,
               "camera_position_jitter_m": 0., "camera_rotation_jitter_deg": 0.,
               "camera_fovy_scale": [1., 1.], "light_scale": [1., 1.],
               "color_gain": [1., 1.], "color_offset": 0.,
               "friction_scale": [1., 1.], "ball_mass_scale": [1., 1.]}
    path = tmp_path / "randomization.json"
    path.write_text(json.dumps(payload))
    config = DomainRandomizationConfig.from_json(path)
    assert config.to_dict() == payload
    randomizer = DomainRandomizer(env, path)
    before = model_parameters(env)
    randomizer.apply(66)
    for name, array in before.items():
        np.testing.assert_allclose(getattr(env.model, name), array, atol=1e-14)
