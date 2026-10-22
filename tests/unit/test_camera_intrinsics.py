"""Check source-resolution camera intrinsics against compiled frusta and real pixels."""

import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from real2sim.config import load_yaml
from real2sim.sim.cameras import add_link_cameras, camera_projection_attributes


def camera():
    return {
        "projection_model": "opencv_pinhole", "width": 1280, "height": 720,
        "intrinsics_px": {"fx": 1000., "fy": 800., "cx": 720., "cy": 320.},
        "distortion_coefficients": [0., 0., 0., 0., 0.],
    }


def model_for(camera_config, point=(.1, -.05, -1.)):
    root = ET.Element("mujoco")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="360")
    ET.SubElement(visual, "headlight", ambient="1 1 1", diffuse="0 0 0", specular="0 0 0")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "camera", name="cam", **camera_projection_attributes(camera_config))
    ET.SubElement(world, "geom", type="sphere", size="0.015", pos=" ".join(map(str, point)),
                  rgba="1 0 0 1")
    return mujoco.MjModel.from_xml_string(ET.tostring(root).decode())


def test_compiled_projection_matches_opencv_with_noncentral_anisotropic_intrinsics():
    spec = camera()
    m = model_for(spec)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    view, scene = mujoco.MjvCamera(), mujoco.MjvScene(m, maxgeom=10)
    view.type, view.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, 0
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, view, mujoco.mjtCatBit.mjCAT_ALL, scene)
    frustum = scene.camera[0]
    near = frustum.frustum_near
    point = np.array([.1, -.05, -1.])
    on_plane = point[:2] * near / -point[2]
    u = 1280 * (.5 + (on_plane[0] - frustum.frustum_center) / (2 * frustum.frustum_width))
    v = 720 * (frustum.frustum_top - on_plane[1]) / (frustum.frustum_top - frustum.frustum_bottom)
    expected = [1000 * .1 + 720, 800 * .05 + 320]
    np.testing.assert_allclose([u, v], expected, atol=1e-4)


@pytest.mark.parametrize("size", [(320, 180), (640, 360)])
def test_rendered_sphere_matches_source_intrinsics_at_both_render_sizes(size):
    width, height = size
    m = model_for(camera())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    renderer = mujoco.Renderer(m, width=width, height=height)
    try:
        renderer.update_scene(d, camera="cam")
        rgb = renderer.render()
    finally:
        renderer.close()
    mask = (rgb[..., 0] > 100) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)
    assert mask.sum() > 10
    ys, xs = np.nonzero(mask)
    observed = [xs.mean() + .5, ys.mean() + .5]
    expected = np.array([820., 360.]) * [width / 1280, height / 720]
    np.testing.assert_allclose(observed, expected, atol=1.)


def test_legacy_fovy_is_an_explicit_unchanged_default():
    assert camera_projection_attributes({"fovy_deg": 22.79, "intrinsics_px": {}}) == {"fovy": "22.79"}


@pytest.mark.parametrize("edit", [
    {"distortion_coefficients": [0., .01, 0., 0., 0.]},
    {"distortion_coefficients": None},
    {"projection_model": "unknown"},
    {"width": 1280.5},
    {"intrinsics_px": {"fx": 0., "fy": 800., "cx": 720., "cy": 320.}},
])
def test_invalid_or_unrepresented_projection_fails(edit):
    spec = camera()
    spec.update(edit)
    with pytest.raises(ValueError):
        camera_projection_attributes(spec)


def test_changing_render_aspect_does_not_silently_change_camera_geometry():
    with pytest.raises(ValueError, match="aspect ratio"):
        camera_projection_attributes(camera(), {"width": 640, "height": 480})


def test_link_cameras_use_same_intrinsics_in_both_robot_mounts():
    spec = camera()
    spec.update(key="observation.images.top", name="wrist", mounts={robot: {
        "body": "link", "position_body_m": [0., 0., .1], "xyaxes_body": [1., 0., 0., 0., 1., 0.],
    } for robot in ("so101", "panda")})
    task = {"camera_iphone": {"key": "observation.images.iphone"}, "additional_cameras": [spec]}
    for robot in ("so101", "panda"):
        root = ET.fromstring('<mujoco><worldbody><body name="link"/></worldbody></mujoco>')
        add_link_cameras(root, copy.deepcopy(task), robot, {"width": 640, "height": 360})
        attrs = root.find(".//camera").attrib
        assert attrs["principalpixel"] == "-80 40"
        assert "fovy" not in attrs


@pytest.mark.parametrize("robot", ["so101", "panda"])
def test_complete_robot_scene_preserves_explicit_source_intrinsics(robot):
    if robot == "so101":
        from real2sim.sim.model import scene_xml
    else:
        from real2sim.panda.model import scene_xml
    config = load_yaml(f"configs/sim_{robot}.yaml")
    task = load_yaml(config["task_config"])
    task["camera_iphone"].update(camera())
    model = mujoco.MjModel.from_xml_string(scene_xml(config, task))
    cam = model.camera("iphone")
    np.testing.assert_array_equal(model.cam_resolution[cam.id], [1280, 720])
    np.testing.assert_allclose(model.cam_intrinsic[cam.id], [1000 / 1280, 800 / 720, -80 / 1280, 40 / 720])
