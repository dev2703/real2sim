"""The optional view copies existing CAD optics and preserves collision/limit gates."""

import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from real2sim.geometry.so101_fk import DEFAULT_MJCF
from real2sim.sim.cameras import add_link_cameras, camera_bindings, nominal_so101_wrist_camera


def test_nominal_optics_reproduce_native_transform_across_joint_poses():
    spec = nominal_so101_wrist_camera(DEFAULT_MJCF)
    task = {"camera_iphone": {"key": "observation.images.iphone"}, "additional_cameras": [spec]}
    root = ET.parse(DEFAULT_MJCF).getroot()
    root.find("compiler").set("meshdir", str(DEFAULT_MJCF.parent / "assets"))
    original = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    add_link_cameras(root, copy.deepcopy(task), "so101")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    native, bound = (model.camera(n).id for n in ("wrist_cam", "top_nominal_cad"))
    for q in np.random.default_rng(2).uniform(-.5, .5, size=(4, model.nq)):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.cam_xpos[bound], data.cam_xpos[native], atol=1e-12)
        np.testing.assert_allclose(data.cam_xmat[bound], data.cam_xmat[native], atol=1e-12)
    assert model.cam_fovy[bound] == model.cam_fovy[native]
    for field in ("jnt_range", "actuator_ctrlrange", "geom_size", "geom_pos", "exclude_signature"):
        np.testing.assert_array_equal(getattr(original, field), getattr(model, field))
    assert spec["physical_calibration_accepted"] is False
    assert spec["provenance"]["source_camera"] == "wrist_cam"
    assert camera_bindings(task, "so101")["observation.images.top"] == "top_nominal_cad"
    with pytest.raises(ValueError, match="missing panda mount"):
        camera_bindings(task, "panda")
