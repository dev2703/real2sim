"""Regressions for the diagnostic refit's joint-offset numerical precision."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.optimize import least_squares

from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.model import scene_xml

spec = importlib.util.spec_from_file_location(
    "landmark_refit", Path(__file__).resolve().parents[2] / "scripts/diag_camera_landmark_refit.py")
refit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refit)


@pytest.fixture
def model():
    with So101PickPlaceEnv(render_images=False) as env:
        env.reset(seed=0)
        rows = []
        for state in ([10, -40, 55, -35, 12, 50], [-15, 20, 30, -60, -15, 20],
                      [25, -60, 70, 25, 30, 70]):
            for joint in [*refit.HORN, "wrist_roll_servo"]:
                rows.append({"episode": 0, "joint": joint, "state": np.array(state, np.float32)})
        yield refit.Model(env, rows, {}, {0}, {})


def test_zero_offset_finite_difference_matches_resolvable_step(model):
    base = model.landmark_points(np.zeros(4), np.zeros(3), np.zeros(3))
    for joint in range(4):
        derivatives = []
        for step in (np.sqrt(np.finfo(float).eps), 1e-3):
            offset = np.zeros(4)
            offset[joint] = step
            derivatives.append((model.landmark_points(offset, np.zeros(3), np.zeros(3)) - base) / step)
        assert np.linalg.norm(derivatives[0]) > 1e-4
        np.testing.assert_allclose(derivatives[0], derivatives[1], rtol=1e-3, atol=1e-7)


def test_recovers_synthetic_offsets_from_zero_initialization(model):
    expected = np.array([2., -3., 4., -5.])
    truth = model.landmark_points(expected, np.zeros(3), np.zeros(3))
    result = least_squares(
        lambda x: (model.landmark_points(x, np.zeros(3), np.zeros(3)) - truth).ravel(),
        np.zeros(4), diff_step=1e-4, gtol=1e-12, xtol=1e-12, ftol=1e-12)
    np.testing.assert_allclose(result.x, expected, atol=1e-5)


def test_rendered_mat_corners_match_fitted_yaw(model):
    import copy

    task = copy.deepcopy(model.env.task)
    task["table"]["mat"]["yaw_deg"] = 17.0
    compiled = mujoco.MjModel.from_xml_string(scene_xml(model.env.config, task))
    data = mujoco.MjData(compiled)
    mujoco.mj_forward(compiled, data)
    mat = compiled.geom("mat_visual")
    sx, sy = mat.size[:2]
    local = np.array([[-sx, sy, 0], [sx, sy, 0], [sx, -sy, 0], [-sx, -sy, 0]])
    corners = local @ data.geom_xmat[mat.id].reshape(3, 3).T + data.geom_xpos[mat.id]
    model.mat = {"size_m_operator": task["table"]["mat"]["size_xy"]}
    expected = model.mat_points(*task["table"]["mat"]["center_task_xy"], np.deg2rad(17.))
    np.testing.assert_allclose(corners[:, :2] - model.env.offset[:2], expected[:, :2], atol=1e-8)
    assert mat.contype[0] == mat.conaffinity[0] == 0
