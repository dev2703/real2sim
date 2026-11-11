"""Both scene loaders interpret explicit camera clipping distances in metres."""

import numpy as np
import pytest
import yaml

from real2sim.config import load_yaml
from real2sim.panda import model as panda
from real2sim.sim import model as so101


@pytest.fixture(params=[so101, panda], ids=["so101", "panda"])
def scene_builder(request):
    return request.param


def write_scene(tmp_path, builder, distance):
    config = load_yaml(builder.DEFAULT_CONFIG)
    config["render"]["near_clip_m"] = distance
    path = tmp_path / "sim.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_near_clip_metres_uses_compiled_extent_and_preserves_scene(tmp_path, scene_builder):
    original, _, _ = scene_builder.load_scene()
    configured, config, _ = scene_builder.load_scene(write_scene(tmp_path, scene_builder, .005))

    assert config["render"]["near_clip_m"] == .005
    assert configured.stat.extent != pytest.approx(1.)
    assert configured.vis.map.znear * configured.stat.extent == pytest.approx(.005, rel=1e-6)
    assert configured.vis.map.znear != original.vis.map.znear
    assert configured.vis.map.zfar == original.vis.map.zfar
    for field in ("body_pos", "body_mass", "body_inertia", "geom_size", "geom_friction",
                  "jnt_range", "actuator_ctrlrange", "cam_pos", "cam_quat", "cam_fovy"):
        np.testing.assert_array_equal(getattr(configured, field), getattr(original, field))


def test_absent_near_clip_preserves_upstream_setting(scene_builder):
    import mujoco

    model, config, task = scene_builder.load_scene()
    assert "near_clip_m" not in config["render"]
    upstream = mujoco.MjModel.from_xml_string(scene_builder.scene_xml(config, task))
    assert model.vis.map.znear == upstream.vis.map.znear
    assert model.vis.map.zfar == upstream.vis.map.zfar


@pytest.mark.parametrize("distance", [0., -.005, float("nan"), float("inf"), None, "0.005", True])
def test_near_clip_rejects_invalid_distances(tmp_path, scene_builder, distance):
    with pytest.raises(ValueError, match="render.near_clip_m.*finite positive"):
        scene_builder.load_scene(write_scene(tmp_path, scene_builder, distance))


def test_near_clip_cannot_exceed_far_plane(tmp_path, scene_builder):
    original, _, _ = scene_builder.load_scene()
    far_distance = float(original.vis.map.zfar) * original.stat.extent
    with pytest.raises(ValueError, match="render.near_clip_m.*far clipping"):
        scene_builder.load_scene(write_scene(tmp_path, scene_builder, far_distance))


@pytest.mark.parametrize("robot", ["so101", "panda"])
def test_benchmark_preparation_preserves_near_clip_for_both_live_models(tmp_path, robot):
    from scripts.benchmark_frozen_act import SharedEpisode

    config = load_yaml(
        "configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml")
    config["render"]["near_clip_m"] = .005
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(config))
    contract = {"camera_keys": ["observation.images.wrist", "observation.images.iphone"],
                "wrist_key": "observation.images.wrist", "image_shape_chw": [3, 180, 320]}

    shared = SharedEpisode(robot, tmp_path / "episode", source_scene=source,
                           native_contract=contract)
    try:
        shared.reset(seed=12050)
        for env in (shared.source.source, shared.env):
            assert env.config["render"] == {"width": 320, "height": 180, "near_clip_m": .005}
            assert env.model.vis.map.znear * env.model.stat.extent == pytest.approx(.005, rel=1e-6)
    finally:
        shared.close()
