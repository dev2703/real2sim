"""Grasp visibility: the wrist near plane must not follow scenery extent.

Floor and wall boxes inflate MuJoCo stat.extent, so the default near clip
(vis.map.znear of 0.01 times that extent) is about 8.5 cm. Jaw tips and the ball
at grasp are closer, and the wrist image never shows the jaws. vis.map.znear is
a fraction of extent shared by wrist_cam and the global iphone camera.
"""

import mujoco
import numpy as np
import pytest

from real2sim.config import REPO_ROOT
from real2sim.sim.cameras import RENDER_NEAR_CLIP_M
from real2sim.sim.model import load_scene

MEASURED = REPO_ROOT / "configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml"


def _near_clip_m(model, camera_name: str) -> float:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = model.camera(camera_name).id
    zclip = np.zeros(2, np.float32)
    mujoco.mjv_cameraFrustum(np.zeros(2, np.float32), np.zeros(2, np.float32), zclip, model, camera)
    return float(zclip[0])


def test_measured_scene_near_clip_is_five_millimetres():
    model, _, _ = load_scene(MEASURED)
    extent = float(model.stat.extent)
    # The measured floor and wall boxes make extent several metres, so the untouched
    # default (0.01 * extent) would sit near 8.5 cm and hide the grasp.
    assert extent > 1.0
    assert 0.01 * extent > 0.05
    assert float(model.vis.map.znear) == pytest.approx(RENDER_NEAR_CLIP_M / extent, rel=1e-5)
    for name in ("wrist_cam", "iphone"):
        assert _near_clip_m(model, name) == pytest.approx(RENDER_NEAR_CLIP_M, abs=1e-5)
