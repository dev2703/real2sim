"""The SO-101 and Panda scenes must share one task frame and one camera (plan E3 validity).

Before 2026-10-13 the Panda scene ignored the D-014 base lift and the fitted camera roll, so the
frozen policy saw a 12 mm proprio shift and a rotated image on the Panda that had nothing to do
with the embodiment. These tests pin the shared registration.
"""

import mujoco
import numpy as np
import pytest

from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.panda.model import panda_base_world
from real2sim.panda.model import task_frame_offset as panda_offset
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.model import base_z_offset
from real2sim.sim.model import task_frame_offset as so101_offset

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def envs():
    so = So101PickPlaceEnv(render_images=False)
    pa = PandaPickPlaceEnv(render_images=False)
    yield so, pa
    so.close()
    pa.close()


def test_task_frame_origin_is_shared(envs):
    so, pa = envs
    np.testing.assert_allclose(panda_offset(pa.task), so101_offset(so.task), atol=1e-12)
    np.testing.assert_allclose(pa.offset, so.offset, atol=1e-12)


def test_panda_base_stands_on_the_table_below_the_origin(envs):
    _, pa = envs
    base = pa.model.body("link0").id
    mujoco.mj_forward(pa.model, pa.data)
    np.testing.assert_allclose(pa.data.xpos[base], panda_base_world(pa.task), atol=1e-12)
    assert pa.data.xpos[base][2] == pytest.approx(pa.offset[2] - base_z_offset(pa.task))


def test_camera_pose_and_axes_are_identical(envs):
    so, pa = envs
    cs, cp = so.model.camera("iphone").id, pa.model.camera("iphone").id
    np.testing.assert_allclose(pa.model.cam_pos[cp], so.model.cam_pos[cs], atol=1e-9)
    np.testing.assert_allclose(pa.model.cam_quat[cp], so.model.cam_quat[cs], atol=1e-9)
    assert pa.model.cam_fovy[cp] == pytest.approx(so.model.cam_fovy[cs])
    fitted = np.asarray(so.task["camera_iphone"]["xyaxes_task"], float)
    mujoco.mj_forward(pa.model, pa.data)
    xmat = pa.data.cam_xmat[cp].reshape(3, 3)
    np.testing.assert_allclose(xmat[:, 0], fitted[:3], atol=1e-4)
    np.testing.assert_allclose(xmat[:, 1], fitted[3:], atol=1e-4)


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_same_seed_places_ball_and_bowl_identically_in_task_frame(envs, seed):
    so, pa = envs
    _, i_so = so.reset(seed=seed)
    _, i_pa = pa.reset(seed=seed)
    m_so, m_pa = i_so["privileged_metrics"], i_pa["privileged_metrics"]
    np.testing.assert_allclose(m_pa["ball_xyz_task"], m_so["ball_xyz_task"], atol=2e-3)
    np.testing.assert_allclose(m_pa["bowl_xyz_task"][:2], m_so["bowl_xyz_task"][:2], atol=1e-9)


def test_ee_pose_is_in_the_task_frame_not_the_panda_base_frame(envs):
    _, pa = envs
    pa.reset(seed=0)
    site = pa.data.site_xpos[pa.site_id]
    np.testing.assert_allclose(pa.ee_pose()[:3, 3], site - pa.offset, atol=1e-12)
    np.testing.assert_allclose(pa.ee_pose()[:3, 3], pa.config["initial_tcp_task_xyz"], atol=3e-3)
