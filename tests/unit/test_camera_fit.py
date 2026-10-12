"""Camera-fit helpers: OpenCV <-> MuJoCo camera convention, detector, PnP on synthetic data.

The round-trip test renders a real MuJoCo scene through a camera built from the converted
parameters and compares the rendered sphere centroid with ``cv2.projectPoints``; the analytic
test compares the closed-form MuJoCo projection with OpenCV on random poses.
"""

import json

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


# --------------------------------------------------------------------------- back-projection
def test_backproject_to_plane_inverts_projection():
    """A known 3-D point on a horizontal plane, projected with cv2, back-projects to itself."""
    rng = np.random.default_rng(3)
    W, H, f = 1280, 720, 924.0
    K = cf.intrinsics_matrix(f, f, W / 2, H / 2)
    for _ in range(20):
        rvec, tvec = _random_pose(rng)
        R, _ = cv2.Rodrigues(rvec)
        z0 = rng.uniform(0.0, 0.1)
        # random points on the plane, in front of the camera (drop any behind it)
        pts = np.column_stack([rng.uniform(-0.3, 0.3, 8), rng.uniform(-0.3, 0.3, 8), np.full(8, z0)])
        pts = pts[(R @ pts.T + tvec[:, None])[2] > 0.1]
        if len(pts) < 2:
            continue
        uv, _ = cv2.projectPoints(pts, rvec, tvec, K, np.zeros(5))
        back = cf.backproject_to_plane(uv.reshape(-1, 2), rvec, tvec, K, z0)
        np.testing.assert_allclose(back, pts, atol=1e-6)
    # a downward-looking camera cannot reach a plane above itself in front of it: refused
    rvec_down, tvec_down = np.array([2.3, 0.08, -0.23]), np.array([-0.21, -0.01, 0.62])
    cam_z = cf.opencv_to_mujoco(rvec_down, tvec_down)["pos"][2]
    with pytest.raises(ValueError):
        cf.backproject_to_plane([[W / 2, H / 2]], rvec_down, tvec_down, K, cam_z + 1.0)


def test_backproject_fx_scaling_moves_points_along_ray_only_in_depth():
    """Scaling fx keeps the ray direction's image position but changes its angle, so a point
    shifts in the plane in the direction away from / towards the camera foot."""
    rvec, tvec = np.array([2.3, 0.08, -0.23]), np.array([-0.21, -0.01, 0.62])  # ~fitted pose
    K = cf.intrinsics_matrix(924.0, 924.0, 640.0, 360.0)
    p = cf.backproject_to_plane([[820.0, 530.0]], rvec, tvec, K, 0.0381)[0]
    assert np.isfinite(p).all()
    shifts = []
    for sc in (0.9, 1.1):
        Ks = K.copy()
        Ks[0, 0] *= sc
        Ks[1, 1] *= sc
        shifts.append(cf.backproject_to_plane([[820.0, 530.0]], rvec, tvec, Ks, 0.0381)[0] - p)
    assert np.sign(shifts[0][0]) == -np.sign(shifts[1][0])  # opposite directions
    assert all(0.001 < np.linalg.norm(s[:2]) < 0.05 for s in shifts)


def _synthetic_report(rng, K, rvec, tvec, task, *, bowl_xy, start_xy, noise_px, n=8, area_scale=None):
    """Correspondences of balls at the two rest heights of ``task`` (so a task with a base lift
    puts them on the lowered planes); blob areas are those of a full ball unless ``area_scale``
    gives a per-index factor."""
    z_bowl = cf.ball_rest_height_in_bowl(task)
    z_start = cf.ball_rest_height_on_mat(task)
    radius = float(task["ball"]["radius"])
    cs = []
    for kind, xy, z in (("bowl", bowl_xy, z_bowl), ("start", start_xy, z_start)):
        for i in range(n):
            p = np.array([*xy, z]) + np.array([*rng.normal(scale=0.004, size=2), 0.0])
            uv, _ = cv2.projectPoints(p[None], rvec, tvec, K, np.zeros(5))
            uv = uv.reshape(2) + rng.normal(scale=noise_px, size=2)
            r_px = cf.expected_ball_radius_px(p, rvec, tvec, K, radius)
            scale = 1.0 if area_scale is None else area_scale(kind, i)
            cs.append({"episode": i, "frame": 0, "kind": kind, "point_base": p.tolist(),
                       "pixel": uv.tolist(), "ball_area_px": int(scale * np.pi * r_px ** 2), "inlier": True})
    for i in range(5):
        p = np.array([rng.uniform(0.25, 0.4), rng.uniform(-0.1, 0.1), rng.uniform(0.03, 0.12)])
        uv, _ = cv2.projectPoints(p[None], rvec, tvec, K, np.zeros(5))
        cs.append({"episode": 20 + i, "frame": 7, "kind": "held", "point_base": p.tolist(),
                   "pixel": uv.reshape(2).tolist(), "ball_area_px": 8000, "inlier": True,
                   "state_deg": [0, 0, 0, 0, 0, 30]})
    return {"opencv_pose": {"rvec": rvec.tolist(), "tvec": tvec.tolist()},
            "intrinsics_used": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]},
            "correspondences": cs}


def test_backproject_report_and_task_write(tmp_path):
    from real2sim.config import load_yaml

    task_path = tmp_path / "task.yaml"
    task_path.write_text((cf.DEFAULT_TASK).read_text())
    task = load_yaml(task_path)
    rng = np.random.default_rng(4)
    K = cf.intrinsics_matrix(924.0, 924.0, 640.0, 360.0)
    rvec, tvec = np.array([2.315, 0.085, -0.232]), np.array([-0.209, -0.0086, 0.616])
    bowl_true, start_true = (0.328, -0.163), (0.331, 0.069)
    report = _synthetic_report(rng, K, rvec, tvec, task, bowl_xy=bowl_true, start_xy=start_true, noise_px=3.0)
    out = cf.backproject_report(report, task)
    for kind, truth in (("bowl", bowl_true), ("start", start_true)):
        est = out["estimates"][kind]
        assert est["n"] == 8 and est["accepted"]
        np.testing.assert_allclose(est["mean_xy"], truth, atol=0.006)
        assert all(s < 0.01 for s in est["std_xy_m"])
        assert set(est["focal_sensitivity"]) == {"fx_x0.9", "fx_x1.1"}
    assert out["held_ball_check"]["rms_xy_m_all"][0] < 1e-6  # noise-free held points
    # the writer touches only the four owned keys
    before = task_path.read_text()
    (tmp_path / "report.json").write_text(json.dumps(report))
    res = cf.run_backproject(tmp_path / "report.json", task_path, cf.DEFAULT_DATA, tmp_path / "out",
                             write_config=True, render=False, refine=False)
    assert res["config_written"] == {"bowl": True, "start": True} and res["refined"] is False
    after = task_path.read_text()
    new = load_yaml(task_path)
    np.testing.assert_allclose(new["bowl"]["center_task_xy"], out["estimates"]["bowl"]["mean_xy"], atol=1e-4)
    np.testing.assert_allclose(new["ball"]["start_region"]["center_task_xy"], out["estimates"]["start"]["mean_xy"], atol=1e-4)
    assert new["bowl"]["center_provenance"].startswith("DATA-DERIVED v3: area-consistent HSV centroids of the in-bowl ball")
    assert new["ball"]["start_region"]["provenance"].startswith("DATA-DERIVED v3: area-consistent HSV centroids of the start-marker ball")
    assert "n=8/8" in new["bowl"]["center_provenance"]
    assert new["ball"]["start_region"]["radius"] == task["ball"]["start_region"]["radius"]
    changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines(), strict=True) if a != b]
    assert len(changed) == 3  # start_region line, bowl centre line, bowl provenance line
    assert new["camera_iphone"] == task["camera_iphone"] and new["table"] == task["table"]
    assert (tmp_path / "out" / "backprojection.json").is_file()


def test_backproject_rejects_wide_spread_and_leaves_task_untouched(tmp_path):
    from real2sim.config import load_yaml

    task_path = tmp_path / "task.yaml"
    task_path.write_text((cf.DEFAULT_TASK).read_text())
    task = load_yaml(task_path)
    rng = np.random.default_rng(5)
    K = cf.intrinsics_matrix(924.0, 924.0, 640.0, 360.0)
    rvec, tvec = np.array([2.315, 0.085, -0.232]), np.array([-0.209, -0.0086, 0.616])
    # 60 px pixel noise -> several cm of spread on the plane
    report = _synthetic_report(rng, K, rvec, tvec, task, bowl_xy=(0.33, -0.16), start_xy=(0.33, 0.07), noise_px=60.0)
    before = task_path.read_text()
    (tmp_path / "report.json").write_text(json.dumps(report))
    res = cf.run_backproject(tmp_path / "report.json", task_path, cf.DEFAULT_DATA, tmp_path / "out",
                             write_config=True, render=False, refine=False)
    assert not res["estimates"]["bowl"]["accepted"] and "reason" in res["estimates"]["bowl"]
    assert res["config_written"] == {"bowl": False, "start": False}
    assert task_path.read_text() == before


def test_main_backproject_subcommand_and_fit_default(tmp_path, capsys):
    from real2sim.config import load_yaml

    task_path = tmp_path / "task.yaml"
    task_path.write_text((cf.DEFAULT_TASK).read_text())
    task = load_yaml(task_path)
    rng = np.random.default_rng(6)
    K = cf.intrinsics_matrix(924.0, 924.0, 640.0, 360.0)
    rvec, tvec = np.array([2.315, 0.085, -0.232]), np.array([-0.209, -0.0086, 0.616])
    report = _synthetic_report(rng, K, rvec, tvec, task, bowl_xy=(0.33, -0.16), start_xy=(0.33, 0.07), noise_px=2.0)
    (tmp_path / "report.json").write_text(json.dumps(report))
    rc = cf.main(["backproject", "--report", str(tmp_path / "report.json"), "--task", str(task_path),
                  "--out", str(tmp_path / "out"), "--no-write", "--no-render", "--no-refine"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert '"config_written"' in printed and '"accepted": true' in printed and '"refined": false' in printed
    assert load_yaml(task_path) == task  # --no-write
    # the default (no subcommand) still parses the fit options
    with pytest.raises(SystemExit):
        cf.main(["--help"])


# --------------------------------------------------------------------------- detection consistency
def test_area_gate_excludes_truncated_and_leaked_blobs_without_refinement():
    """Raw centroids whose blob area is inconsistent with a full ball at the back-projected
    depth are excluded (their centroid is biased in a way the spread test cannot detect)."""
    from real2sim.config import load_yaml

    task = load_yaml(cf.DEFAULT_TASK)
    rng = np.random.default_rng(7)
    K = cf.intrinsics_matrix(924.0, 924.0, 640.0, 360.0)
    rvec, tvec = np.array([2.315, 0.085, -0.232]), np.array([-0.209, -0.0086, 0.616])
    bowl_true = (0.328, -0.170)
    bad = {("bowl", 1): 0.65, ("bowl", 5): 1.4}  # truncated / leaked, as in the real frames
    report = _synthetic_report(rng, K, rvec, tvec, task, bowl_xy=bowl_true, start_xy=(0.33, 0.07),
                               noise_px=1.0, area_scale=lambda k, i: bad.get((k, i), 1.0))
    for c in report["correspondences"]:
        if (c["kind"], c["episode"]) in bad:
            c["pixel"][1] -= 80.0  # the biased centroid that goes with a cut/leaked blob
    out = cf.backproject_report(report, task)
    est = out["estimates"]["bowl"]
    assert est["n_detections"] == 8 and est["n"] == 6 and est["accepted"]
    excluded = {d["episode"]: d for d in est["detections"] if not d["used"]}
    assert sorted(excluded) == [1, 5]
    assert all("band" in d["reason"] and not d["area_consistent"] for d in excluded.values())
    # ratios are taken at the biased pixel's own (larger) depth, so they drift up a little
    assert 0.65 <= excluded[1]["area_ratio"] < cf.AREA_RATIO_BAND[0]
    assert excluded[5]["area_ratio"] > cf.AREA_RATIO_BAND[1]
    np.testing.assert_allclose(est["mean_xy"], bowl_true, atol=0.004)
    # with the gate off (all areas consistent) the same biased pixels pull the mean ~1 cm
    for c in report["correspondences"]:
        c["ball_area_px"] = int(np.pi * cf.expected_ball_radius_px(c["point_base"], rvec, tvec, K, 0.0335) ** 2)
    est_all = cf.backproject_report(report, task)["estimates"]["bowl"]
    assert est_all["n"] == 8 and abs(est_all["mean_xy"][1] - bowl_true[1]) > 0.006


def _bowl_frame(centre, r_px, *, size=(720, 1280), occlude_frac=0.35, leak=True):
    """Synthetic iphone frame: dark mat, grey bowl, yellow ball whose lower ``occlude_frac`` is hidden
    behind a pale rim and (optionally) a pale low-saturation patch leaking from the top of the ball."""
    img = np.full((*size, 3), (28, 30, 34), np.uint8)
    cu, cv_ = int(round(centre[0])), int(round(centre[1]))
    cv2.ellipse(img, (cu, cv_ + int(0.3 * r_px)), (int(1.5 * r_px), int(1.0 * r_px)), 0, 0, 360, (150, 150, 142), -1)
    cv2.circle(img, (cu, cv_), int(round(r_px)), (225, 235, 40), -1)
    if leak:  # pale yellow-grey: inside the HSV window (S ~ 80) but low yellowness contrast
        cv2.ellipse(img, (cu - int(0.2 * r_px), cv_ - int(0.9 * r_px)), (int(1.2 * r_px), int(0.5 * r_px)),
                    0, 180, 360, (205, 205, 140), -1)
    # near rim: hides the lower part of the ball
    y_cut = cv_ + int((1 - 2 * occlude_frac) * r_px)
    cv2.ellipse(img, (cu, y_cut + int(0.4 * r_px)), (int(1.6 * r_px), int(0.55 * r_px)), 0, 180, 360, (190, 190, 180), -1)
    cv2.rectangle(img, (cu - int(1.6 * r_px), y_cut), (cu + int(1.6 * r_px), y_cut + int(0.8 * r_px)), (190, 190, 180), -1)
    return img


def test_refine_ball_circle_recovers_centre_of_truncated_and_leaked_ball():
    true_c, r_px = (640.0, 420.0), 75.0
    img = _bowl_frame(true_c, r_px)
    u, v, area = cf.ball_centroid_px(img)
    assert v < true_c[1] - 20  # the HSV centroid is pulled up by the cut lower half and the leak
    # the leak adds about what the occlusion removes, so the blob area alone looks plausible here:
    # the area gate cannot catch this case, the outline fit must
    assert 0.8 * np.pi * r_px ** 2 <= area <= 1.25 * np.pi * r_px ** 2
    fit = cf.refine_ball_circle(img, (u, v), r_px)
    assert fit["ok"], fit
    np.testing.assert_allclose(fit["centre"], true_c, atol=2.0)  # measured <= 1.2 px (rasterised disc)
    assert fit["arc_coverage"] >= 0.4 and fit["disc_fill"] > 0.6 and fit["rms_px"] < 1.5
    # the same fit must not report ok on a frame without a ball (grey rim ring of the same radius)
    ring = np.full((720, 1280, 3), (28, 30, 34), np.uint8)
    cv2.circle(ring, (640, 420), 75, (190, 190, 180), 6)
    assert not cf.refine_ball_circle(ring, (640.0, 420.0), r_px)["ok"]


def test_refine_detections_feeds_backprojection():
    """Synthetic frames rendered at the projected rest positions: the biased raw centroids are
    replaced by the outline fit and the plane estimate recovers the truth."""
    from real2sim.config import load_yaml

    task = load_yaml(cf.DEFAULT_TASK)
    rng = np.random.default_rng(8)
    K = cf.intrinsics_matrix(924.0, 924.0, 640.0, 360.0)
    rvec, tvec = np.array([2.315, 0.085, -0.232]), np.array([-0.209, -0.0086, 0.616])
    bowl_true, start_true = (0.326, -0.172), (0.333, 0.070)
    report = _synthetic_report(rng, K, rvec, tvec, task, bowl_xy=bowl_true, start_xy=start_true, noise_px=0.0,
                               n=4, area_scale=lambda k, i: 0.7)
    truth = {}
    for c in report["correspondences"]:
        if c["kind"] in ("bowl", "start"):
            truth[(c["episode"], c["kind"])] = (list(c["pixel"]),
                                                cf.expected_ball_radius_px(c["point_base"], rvec, tvec, K, 0.0335))
            c["pixel"][1] -= 20.0  # biased centroid of a truncated blob
    frames = {}
    for (ep, kind), (uv, r_px) in truth.items():
        frames[(ep, 0, kind)] = _bowl_frame(uv, r_px, leak=(kind == "bowl"))

    def load_frame(ep, fr):  # bowl and start share (ep, frame) in the synthetic report: pick by call order
        return frames[(ep, fr, kinds.pop(0))]

    kinds = [c["kind"] for c in report["correspondences"] if c["kind"] in ("bowl", "start")]
    refined = cf.refine_detections(report["correspondences"], task, rvec, tvec, K, load_frame)
    assert len(refined) == 8 and all(r["used"] for r in refined)
    for r in refined:  # raw centroids were 20 px off; 2.5 px ~ 1 mm on the plane at this depth
        np.testing.assert_allclose(r["pixel_refined"], truth[(r["episode"], r["kind"])][0], atol=2.5)
        # 0.7 of a full ball, taken at the biased pixel's own depth: a little above 0.7, below the band
        assert 0.65 < r["area_ratio"] < cf.AREA_RATIO_BAND[0] and not r["area_consistent"]
    out = cf.backproject_report(report, task, refined=refined)
    assert out["detection_source"].startswith("known-radius circle fit")
    for kind, tr in (("bowl", bowl_true), ("start", start_true)):
        est = out["estimates"][kind]
        assert est["n"] == 4 and est["accepted"] and all(d["shift_px"] is not None for d in est["detections"])
        np.testing.assert_allclose(est["mean_xy"], tr, atol=0.004)
    prov = cf.centre_provenance("bowl", out["estimates"]["bowl"], refined=True)
    assert prov.startswith("DATA-DERIVED v3: known-radius circle fit of the in-bowl ball") and "n=4/4" in prov


def test_rest_planes_follow_base_z_offset(tmp_path):
    """With D-014's base lift the table top is at z_task = -base_z_offset; the planes (and the
    fit's 3-D points) must move with it, and a fit on the unlifted planes is visibly biased."""
    from real2sim.config import load_yaml

    text = cf.DEFAULT_TASK.read_text()
    assert "base_z_offset" not in text
    lifted = tmp_path / "task.yaml"
    lifted.write_text(text.replace("robot_so101:\n", "robot_so101:\n  base_z_offset: 0.012\n"))
    task0, task1 = load_yaml(cf.DEFAULT_TASK), load_yaml(lifted)
    assert cf.table_top_z_task(task0) == 0.0 and cf.table_top_z_task(task1) == pytest.approx(-0.012)
    assert cf.ball_rest_height_in_bowl(task1) == pytest.approx(cf.ball_rest_height_in_bowl(task0) - 0.012)
    assert cf.ball_rest_height_on_mat(task1) == pytest.approx(0.0335 - 0.012)
    rng = np.random.default_rng(9)
    K = cf.intrinsics_matrix(924.0, 924.0, 640.0, 360.0)
    rvec, tvec = np.array([2.315, 0.085, -0.232]), np.array([-0.209, -0.0086, 0.616])
    bowl_true, start_true = (0.328, -0.170), (0.331, 0.069)
    report = _synthetic_report(rng, K, rvec, tvec, task1, bowl_xy=bowl_true, start_xy=start_true, noise_px=0.0)
    assert report["correspondences"][0]["point_base"][2] == pytest.approx(cf.ball_rest_height_in_bowl(task1))
    good = cf.backproject_report(report, task1)
    assert good["table_top_z_task_m"] == pytest.approx(-0.012)
    np.testing.assert_allclose(good["estimates"]["bowl"]["mean_xy"], bowl_true, atol=0.003)
    np.testing.assert_allclose(good["estimates"]["start"]["mean_xy"], start_true, atol=0.003)
    wrong = cf.backproject_report(report, task0)  # planes 1.2 cm too high
    shift = np.linalg.norm(np.asarray(wrong["estimates"]["bowl"]["mean_xy"]) - bowl_true)
    assert 0.006 < shift < 0.03


def test_sim_config_for_task_builds_scene_from_that_task(tmp_path):
    """render_comparisons must render the scene from the task.yaml it was given, not the repo one."""
    from real2sim.config import load_yaml
    from real2sim.sim.env import So101PickPlaceEnv

    task_path = tmp_path / "task.yaml"
    task_path.write_text(cf.DEFAULT_TASK.read_text())
    cf.write_task_centres(task_path, bowl=([0.30, -0.20], "test"), start=([0.35, 0.09], "test"))
    cfg = cf._sim_config_for_task(task_path, cf.DEFAULT_DATA, tmp_path / "sim.yaml")
    assert load_yaml(cfg)["task_config"] == str(task_path.resolve())
    with So101PickPlaceEnv(cfg, render_images=False) as env:
        assert env.task["bowl"]["center_task_xy"] == [0.30, -0.20]
        assert env.task["ball"]["start_region"]["center_task_xy"] == [0.35, 0.09]
        np.testing.assert_allclose(env.model.body("bowl").pos[:2], [0.30, -0.20])
    assert load_yaml(cf.DEFAULT_TASK)["bowl"]["center_task_xy"] != [0.30, -0.20]
