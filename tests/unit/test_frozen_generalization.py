"""The frozen evaluation changes declared nuisance factors, not policy inputs."""

from copy import deepcopy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from scripts import evaluate_v2_frozen_generalization_20261027 as generalization


def tiny_environment():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <visual><headlight ambient=".2 .3 .4" diffuse=".5 .6 .7" specular=".1 .2 .3"/></visual>
      <worldbody>
        <light ambient=".1 .2 .3" diffuse=".4 .5 .6" specular=".2 .3 .4"/>
        <geom name="table" type="plane" size="1 1 .1" friction=".6 .005 .0001"/>
        <geom name="decoration" type="sphere" size=".01" contype="0" conaffinity="0"
              friction=".8 .007 .0002"/>
        <body name="ball" pos="0 0 .04">
          <freejoint/>
          <geom name="ball_geom" type="sphere" size=".03" mass=".058"
                friction=".9 .006 .0003"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    return SimpleNamespace(model=model, ball_geom=model.geom("ball_geom").id,
                           ball_body=model.body("ball").id)


def test_dim_lighting_changes_built_lights_without_changing_contacts_or_dynamics():
    env = tiny_environment()
    model = env.model
    friction = model.geom_friction.copy()
    inertia = model.body_inertia.copy()
    mass = model.body_mass.copy()
    before = {name: getattr(model, "light_" + name).copy()
              for name in ("ambient", "diffuse", "specular")}
    heads = {name: getattr(model.vis.headlight, name).copy() for name in before}
    generalization.vary_model(env, generalization.CONDITIONS["lighting_dim"])
    for name in before:
        np.testing.assert_allclose(getattr(model, "light_" + name), .6 * before[name])
        np.testing.assert_allclose(getattr(model.vis.headlight, name), .6 * heads[name])
    np.testing.assert_array_equal(model.geom_friction, friction)
    np.testing.assert_array_equal(model.body_inertia, inertia)
    np.testing.assert_array_equal(model.body_mass, mass)


def test_friction_changes_effective_collision_materials_and_preserves_other_axes():
    env = tiny_environment()
    model = env.model
    before = model.geom_friction.copy()
    light = model.light_diffuse.copy()
    solref = model.geom_solref.copy()
    enabled = (model.geom_contype | model.geom_conaffinity) != 0
    generalization.vary_model(env, generalization.CONDITIONS["friction_low"])
    np.testing.assert_allclose(model.geom_friction[enabled, 0], before[enabled, 0] * 2 / 3)
    np.testing.assert_array_equal(model.geom_friction[~enabled], before[~enabled])
    np.testing.assert_array_equal(model.geom_friction[:, 1:], before[:, 1:])
    np.testing.assert_array_equal(model.geom_solref, solref)
    np.testing.assert_array_equal(model.light_diffuse, light)
    # The max mixing rule must change the actual ball/table contact coefficient.
    data = mujoco.MjData(model)
    data.qpos[2] = .02
    mujoco.mj_forward(model, data)
    ball_contacts = [contact for contact in data.contact[:data.ncon]
                     if env.ball_geom in (contact.geom1, contact.geom2)]
    assert ball_contacts
    assert all(np.isclose(contact.friction[0], .6) for contact in ball_contacts)


@pytest.mark.parametrize("fail", [False, True])
def test_frontend_cannot_see_varied_radius_and_true_geometry_is_restored(monkeypatch, fail):
    env = SimpleNamespace(task={"ball": {"radius": .034}})
    original = deepcopy(env.task)

    def acquire(candidate):
        assert candidate.task["ball"]["radius"] == .032
        if fail:
            raise ValueError("detection failed")
        return np.array([.3, .2, .31, -.03])

    monkeypatch.setattr(generalization.experiment, "acquire", acquire)
    if fail:
        with pytest.raises(ValueError, match="detection failed"):
            generalization.acquire_frozen_perception(env, .032)
    else:
        result = generalization.acquire_frozen_perception(env, .032)
        assert result.shape == (4,)
    assert env.task == original


@pytest.mark.integration
@pytest.mark.parametrize("robot", ["so101", "panda"])
def test_size_variation_rebuilds_physical_radius_and_inertia_with_constant_mass(tmp_path, robot):
    env_class = (generalization.experiment.So101PickPlaceEnv if robot == "so101"
                 else generalization.experiment.PandaPickPlaceEnv)
    snapshots = {}
    for condition in ("nominal", "ball_small"):
        _, config = generalization.configured_scene(
            tmp_path, robot, condition, generalization.SOURCE)
        with env_class(config, render_images=False) as env:
            snapshots[condition] = {
                "radius": env.model.geom_size[env.ball_geom, 0],
                "task_radius": env.task["ball"]["radius"],
                "mass": env.model.body_mass[env.ball_body],
                "inertia": env.model.body_inertia[env.ball_body].copy(),
                "friction": env.model.geom_friction.copy(),
            }
    nominal, small = snapshots["nominal"], snapshots["ball_small"]
    assert small["radius"] == small["task_radius"]
    assert small["radius"] == pytest.approx(nominal["radius"] * .95)
    assert small["mass"] == nominal["mass"]
    np.testing.assert_allclose(small["inertia"], nominal["inertia"] * .95 ** 2)
    np.testing.assert_array_equal(small["friction"], nominal["friction"])


def test_frozen_dependency_mutation_is_rejected_before_evaluation(tmp_path):
    dependency = tmp_path / "weights"
    dependency.write_bytes(b"frozen")
    protocol = {"source_sha256_before": {
        str(dependency): generalization.sha256_file(dependency)}}
    generalization.verify_sources(protocol)
    dependency.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Frozen dependencies changed"):
        generalization.verify_sources(protocol)


def test_predeclared_layouts_are_reproducible_paired_and_vary_both_objects(tmp_path, monkeypatch):
    dependency = tmp_path / "dependency"
    dependency.write_text("fixed")
    source = tmp_path / "source"
    (source / "checkpoint").mkdir(parents=True)
    (source / "checkpoint/model.safetensors").write_bytes(b"frozen")
    monkeypatch.setattr(generalization, "dependency_paths", lambda _: [dependency])
    first = generalization.prepare(tmp_path / "first", source)
    second = generalization.prepare(tmp_path / "second", source)
    assert first["layouts"] == second["layouts"]
    assert len(first["layouts"]) == 20
    assert len({tuple(layout["ball_xy"]) for layout in first["layouts"]}) == 20
    assert len({tuple(layout["bowl_xy"]) for layout in first["layouts"]}) == 20
    assert set(first["robots"]) == {"so101", "panda"}
    assert all(set(layout) == {"seed", "ball_xy", "bowl_xy"} for layout in first["layouts"])
    # One shared schedule serves every robot and factor, rather than independent RNG draws.
    assert len(first["conditions"]) == 7
    assert first["frozen_settings"]["training_updates"] == 0
    assert first["frozen_settings"]["position_forecast_offset"] == 3

