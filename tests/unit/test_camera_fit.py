"""Camera-fit helpers: OpenCV <-> MuJoCo camera convention, detector, PnP on synthetic data.

The round-trip test renders a real MuJoCo scene through a camera built from the converted
parameters and compares the rendered sphere centroid with ``cv2.projectPoints``; the analytic
test compares the closed-form MuJoCo projection with OpenCV on random poses.
"""

import cv2
import mujoco
import numpy as np
import pytest

from real2sim.sim import camera_fit as cf


def _random_pose(rng):
    rvec = rng.normal(size=3) * 0.8
    tvec = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0.6, 1.2)])
    return rvec, tvec


def test_fovy_focal_roundtrip():
    assert cf.fy_from_fovy_deg(cf.fovy_deg_from_fy(924.0, 720), 720) == pytest.approx(924.0)
    assert cf.fovy_deg_from_fy(924.0, 720) == pytest.approx(42.6, abs=0.05)


def test_opencv_to_mujoco_analytic_projection_matches_cv2():
    rng = np.random.default_rng(0)
    W, H, f = 1280, 720, 924.0
    K = cf.intrinsics_matrix(f, f, W / 2, H / 2)
    for _ in range(20):
        rvec, tvec = _random_pose(rng)
        conv = cf.opencv_to_mujoco(rvec, tvec)
        # points in front of the camera
        R, _ = cv2.Rodrigues(rvec)
        pc = np.column_stack([rng.uniform(-0.2, 0.2, 10), rng.uniform(-0.2, 0.2, 10), rng.uniform(0.4, 1.0, 10)])
        pw = (pc - tvec) @ R
        uv_cv, _ = cv2.projectPoints(pw, rvec, tvec, K, np.zeros(5))
        uv_mj = cf.project_mujoco(pw, conv["pos"], conv["xaxis"], conv["yaxis"], cf.fovy_deg_from_fy(f, H), W, H)
        np.testing.assert_allclose(uv_mj, uv_cv.reshape(-1, 2), atol=1e-6)
        # camera centre and optical axis agree
        np.testing.assert_allclose(conv["pos"], -R.T @ tvec, atol=1e-12)
        np.testing.assert_allclose(conv["forward"], R.T @ [0, 0, 1], atol=1e-12)


def test_opencv_to_mujoco_rendered_sphere_matches_cv2_projection():
    """Build a MuJoCo camera from the converted pose and render a small sphere; its centroid
    must land where cv2.projectPoints puts the sphere centre (within ~1 px)."""
    W, H = 320, 240
    fovy = 45.0
    f = cf.fy_from_fovy_deg(fovy, H)
    K = cf.intrinsics_matrix(f, f, W / 2, H / 2)
    rng = np.random.default_rng(1)
    for trial in range(3):
        # camera somewhere above the origin looking roughly at it, with roll
        rvec, tvec = _random_pose(rng)
        conv = cf.opencv_to_mujoco(rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        p_cam = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0.8])
        p_world = R.T @ (p_cam - tvec)
        xyaxes = " ".join(f"{v:.9g}" for v in [*conv["xaxis"], *conv["yaxis"]])
        xml = f"""
        <mujoco>
          <visual><global offwidth="{W}" offheight="{H}"/><headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
          <worldbody>
            <camera name="cam" pos="{' '.join(f'{v:.9g}' for v in conv['pos'])}" xyaxes="{xyaxes}" fovy="{fovy}"/>
            <geom type="sphere" size="0.01" pos="{' '.join(f'{v:.9g}' for v in p_world)}" rgba="1 0 0 1"/>
          </worldbody>
        </mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=H, width=W)
        try:
            renderer.update_scene(data, camera="cam")
            rgb = renderer.render()
        finally:
            renderer.close()
        mask = (rgb[..., 0] > 100) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)
        assert mask.sum() > 20, f"sphere not rendered (trial {trial})"
        vs, us = np.nonzero(mask)
        centroid = np.array([us.mean() + 0.5, vs.mean() + 0.5])  # pixel-centre convention
        uv_cv, _ = cv2.projectPoints(p_world[None], rvec, tvec, K, np.zeros(5))
        np.testing.assert_allclose(centroid, uv_cv.reshape(2), atol=1.5)


def test_roll_free_axes_match_full_axes_when_no_roll():
    pos, target = np.array([0.5, -0.4, 0.4]), np.array([0.2, 0.0, 0.0])
    x, y = cf.roll_free_axes(pos, target)
    z = np.cross(x, y)
    np.testing.assert_allclose(z, (pos - target) / np.linalg.norm(pos - target), atol=1e-12)
    assert abs(x[2]) < 1e-12  # horizontal x axis = no roll


def test_ball_centroid_px_on_synthetic_disc():
    img = np.full((180, 320, 3), 60, np.uint8)
    cv2.circle(img, (210, 70), 25, (220, 230, 40), -1)  # tennis-ball yellow (RGB)
    cv2.circle(img, (40, 140), 6, (220, 230, 40), -1)  # small distractor below min area
    u, v, area = cf.ball_centroid_px(img, min_area=500)
    assert abs(u - 210) < 1.0 and abs(v - 70) < 1.0
    assert 1800 < area < 2100
    assert cf.ball_centroid_px(np.zeros((180, 320, 3), np.uint8)) is None


def test_solve_pose_recovers_synthetic_camera_with_outliers():
    rng = np.random.default_rng(2)
    W, H, f = 1280, 720, 924.0
    K = cf.intrinsics_matrix(f, f, W / 2, H / 2)
    rvec_true = np.array([1.9, 0.3, -0.4])
    tvec_true = np.array([-0.2, 0.1, 0.7])
    pts = np.column_stack([rng.uniform(0.2, 0.4, 30), rng.uniform(-0.2, 0.1, 30), rng.uniform(0.0, 0.15, 30)])
    uv, _ = cv2.projectPoints(pts, rvec_true, tvec_true, K, np.zeros(5))
    uv = uv.reshape(-1, 2) + rng.normal(scale=1.0, size=(30, 2))
    uv[:4] += 150.0  # gross outliers
    fit = cf.solve_pose(pts, uv, K, image_size=(W, H))
    assert fit["ok"] and fit["n_inliers"] >= 24 and fit["inlier_mask"][:4].sum() == 0
    assert fit["rms_inliers_px"] < 3.0
    np.testing.assert_allclose(fit["tvec"], tvec_true, atol=0.02)
    np.testing.assert_allclose(fit["rvec"], rvec_true, atol=0.02)


def test_hold_runs_respects_min_length():
    state = np.array([1, 30, 30, 30, 30, 1, 30, 30], float)
    action = np.array([1, 1, 1, 1, 1, 1, 1, 1], float)
    assert cf.hold_runs(state, action, min_frames=4) == [(1, 5)]
    assert cf.hold_runs(state, action, min_frames=2) == [(1, 5), (6, 8)]
