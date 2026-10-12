"""Fit the simulated ``iphone`` camera pose to the real viewpoint with PnP (plan M2 step 1, §3.4).

Correspondences are *observable* geometry only:

* ball held in the gripper during a sustained hold -> 3-D point = gripper-frame FK of the
  recorded follower state (``So101FK.fk_real``), 2-D point = HSV centroid of the ball;
* ball resting on the start marker in the first frame of an episode -> 3-D point = task.yaml
  ``ball.start_region.center_task_xy`` at ball-radius height;
* ball resting in the bowl in the last frame of an episode -> 3-D point = bowl centre at
  floor-rest height.

The camera pose ``T_cam_base`` (OpenCV optical convention: +z forward, +y down) is solved with
``cv2.solvePnPRansac`` + LM refinement using the provisional intrinsics; the focal length is
refined only if the fixed-focal RMS stays above ``REFINE_FOCAL_ABOVE_PX``. The result is converted
to the MuJoCo camera convention (looks along -z, +y up) and written to ``configs/task.yaml``
``camera_iphone`` only if the RMS reprojection error is below ``ACCEPT_RMS_PX``. Everything,
including a failed fit, is written to ``results/audit/camera_fit/report.json``.

Nothing here touches policy observations; it only sets the render camera.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import cv2
import numpy as np

from real2sim.config import load_yaml
from real2sim.data.raw import RawDataset, VideoSegment, resolve_dataset_root
from real2sim.geometry.so101_fk import JointMapping, So101FK

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASK = ROOT / "configs" / "task.yaml"
DEFAULT_DATA = ROOT / "configs" / "data.yaml"
OUT_DIR = ROOT / "results" / "audit" / "camera_fit"

HSV_LO = (22, 70, 90)  # same tennis-ball yellow window as real2sim.data.quality.ball_area_px
HSV_HI = (50, 255, 255)
MIN_BALL_AREA_PX = 1500  # at 1280x720 a free ball is ~17 000 px; a half-occluded held ball ~5 000
IMAGE_LAG_FRAMES = 2  # the iphone image trails the joint state by ~2 frames (docs/data_contract.md)
HOLD_LEADER_MAX_PCT = 8.0
HOLD_FOLLOWER_RANGE_PCT = (12.0, 60.0)
HOLD_MIN_FRAMES = 30
RANSAC_REPROJ_PX = 30.0
REFINE_FOCAL_ABOVE_PX = 15.0
ACCEPT_RMS_PX = 40.0


# --------------------------------------------------------------------------- camera conventions
def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def fovy_deg_from_fy(fy: float, height: int) -> float:
    return math.degrees(2.0 * math.atan(0.5 * height / fy))


def fy_from_fovy_deg(fovy_deg: float, height: int) -> float:
    return 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)


def opencv_to_mujoco(rvec: np.ndarray, tvec: np.ndarray) -> dict:
    """OpenCV extrinsics (``x_cam = R x_world + t``) -> MuJoCo camera ``pos`` and ``xyaxes``.

    OpenCV looks along +z with +y down; MuJoCo looks along -z with +y up. Both share +x (right),
    so the MuJoCo camera frame is the OpenCV frame rotated by 180 degrees about x.
    """
    R_cw, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))
    t = np.asarray(tvec, float).reshape(3)
    R_wc = R_cw.T  # columns = OpenCV camera axes in world coordinates
    pos = -R_wc @ t
    R_wc_mj = R_wc @ np.diag([1.0, -1.0, -1.0])
    return {"pos": pos, "xaxis": R_wc_mj[:, 0], "yaxis": R_wc_mj[:, 1], "forward": -R_wc_mj[:, 2]}


def project_mujoco(points_world: np.ndarray, pos, xaxis, yaxis, fovy_deg: float,
                    width: int, height: int) -> np.ndarray:
    """Pixel coordinates (u right, v down, origin top-left) of a MuJoCo camera defined by
    ``pos``/``xyaxes``/``fovy``; square pixels, principal point at the image centre."""
    x = np.asarray(xaxis, float) / np.linalg.norm(xaxis)
    y = np.asarray(yaxis, float)
    y = y - x * (x @ y)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)  # camera looks along -z
    rel = np.asarray(points_world, float).reshape(-1, 3) - np.asarray(pos, float)
    xc, yc, zc = rel @ x, rel @ y, rel @ z
    depth = -zc
    f = fy_from_fovy_deg(fovy_deg, height)
    u = width / 2.0 + f * xc / depth
    v = height / 2.0 - f * yc / depth
    return np.stack([u, v], axis=1)


def roll_free_axes(position, target) -> tuple[np.ndarray, np.ndarray]:
    """The ``xyaxes`` that ``real2sim.sim.model`` derives from a position and look-at point."""
    z = np.asarray(position, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    return x, np.cross(z, x)


# --------------------------------------------------------------------------- detection
def ball_centroid_px(rgb: np.ndarray, min_area: int = MIN_BALL_AREA_PX) -> tuple[float, float, int] | None:
    """Centroid ``(u, v)`` and area of the largest tennis-ball-yellow component, or ``None``."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[k, cv2.CC_STAT_AREA])
    if area < min_area:
        return None
    return float(centroids[k][0]), float(centroids[k][1]), area


def decode_frame(seg: VideoSegment, frame_pos: int, fps: float, tol: float = 1e-4) -> np.ndarray:
    """Decode one RGB frame of an episode by seeking (keyframe interval is 2 in this dataset)."""
    target = seg.from_ts + frame_pos / fps
    with av.open(str(seg.path)) as container:
        stream = container.streams.video[0]
        tb = float(stream.time_base)
        container.seek(int(target / tb), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if frame.pts * tb >= target - tol:
                return frame.to_ndarray(format="rgb24")
    raise ValueError(f"frame {frame_pos} of episode {seg.episode_index} not found")


# --------------------------------------------------------------------------- correspondences
@dataclass
class Correspondence:
    episode: int
    frame: int
    kind: str  # held | start | bowl
    point_base: list[float]
    pixel: list[float]
    ball_area_px: int
    state_deg: list[float] | None = None
    reproj_px: list[float] | None = None
    error_px: float | None = None
    inlier: bool | None = None


def hold_runs(state_g: np.ndarray, action_g: np.ndarray, min_frames: int = HOLD_MIN_FRAMES) -> list[tuple[int, int]]:
    lo, hi = HOLD_FOLLOWER_RANGE_PCT
    hold = (action_g < HOLD_LEADER_MAX_PCT) & (state_g > lo) & (state_g < hi)
    runs, i = [], 0
    while i < len(hold):
        if hold[i]:
            j = i
            while j < len(hold) and hold[j]:
                j += 1
            if j - i >= min_frames:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def ball_rest_height_in_bowl(task: dict) -> float:
    """Ball centre height (base frame) when resting on the bowl floor: floor slab + radius."""
    bowl = task["bowl"]
    floor = float(bowl["outer_height"]) - float(bowl["inner_depth"])
    return floor + float(task["ball"]["radius"])


def collect(ds: RawDataset, task: dict, key: str, episodes: list[int], fk: So101FK, *,
            holds_per_episode: int = 3, max_hold_episodes: int = 9, start_episodes: int = 8,
            bowl_episodes: int = 8, min_area: int = MIN_BALL_AREA_PX) -> list[Correspondence]:
    fps = float(ds.info["fps"])
    ball = task["ball"]
    start_xy = [float(v) for v in ball["start_region"]["center_task_xy"]]
    bowl_xy = [float(v) for v in task["bowl"]["center_task_xy"]]
    out: list[Correspondence] = []
    n_hold_eps = 0
    for ep in episodes:
        rows = ds.episode_rows(ep)
        s = ds.stack(rows, "observation.state")
        a = ds.stack(rows, "action")
        runs = hold_runs(s[:, 5], a[:, 5])
        if not runs or n_hold_eps >= max_hold_episodes:
            continue
        n_hold_eps += 1
        seg = ds.video_segment(key, ep)
        i0, i1 = max(runs, key=lambda r: r[1] - r[0])
        # Spread samples through the longest hold; the state row is lagged by IMAGE_LAG_FRAMES.
        for q in np.linspace(0.15, 0.85, holds_per_episode):
            frame = int(i0 + q * (i1 - i0))
            # prefer a slow frame nearby so the 2-frame lag matters less
            lo, hi = max(i0 + 1, frame - 15), min(i1 - 1, frame + 15)
            speed = np.abs(np.diff(s[lo - 1:hi + 1, :5], axis=0)).sum(axis=1)
            frame = lo + int(np.argmin(speed[: hi - lo]))
            state = s[frame - IMAGE_LAG_FRAMES]
            det = ball_centroid_px(decode_frame(seg, frame, fps), min_area)
            if det is None:
                continue
            p = fk.fk_real(state)[:3, 3]
            out.append(Correspondence(ep, frame, "held", p.tolist(), [det[0], det[1]], det[2],
                                      state_deg=np.round(state, 2).tolist()))
    for ep in episodes[:start_episodes]:
        seg = ds.video_segment(key, ep)
        det = ball_centroid_px(decode_frame(seg, 0, fps), min_area)
        if det is not None:
            out.append(Correspondence(ep, 0, "start", [*start_xy, float(ball["radius"])],
                                      [det[0], det[1]], det[2]))
    z_bowl = ball_rest_height_in_bowl(task)
    for ep in episodes[:bowl_episodes]:
        seg = ds.video_segment(key, ep)
        last = ds.episode_length(ep) - 1
        det = ball_centroid_px(decode_frame(seg, last, fps), min_area)
        if det is not None:
            out.append(Correspondence(ep, last, "bowl", [*bowl_xy, z_bowl], [det[0], det[1]], det[2]))
    return out


# --------------------------------------------------------------------------- PnP
def solve_pose(points: np.ndarray, pixels: np.ndarray, K: np.ndarray, *, refine_focal: bool = False,
               image_size: tuple[int, int] = (1280, 720), reproj_px: float = RANSAC_REPROJ_PX) -> dict:
    """RANSAC PnP + LM refinement on the inliers. Returns rvec, tvec, K, per-point errors."""
    pts = np.ascontiguousarray(points, np.float64).reshape(-1, 1, 3)
    pix = np.ascontiguousarray(pixels, np.float64).reshape(-1, 1, 2)
    dist = np.zeros(5)
    cv2.setRNGSeed(0)  # deterministic RANSAC
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(pts, pix, K, dist, reprojectionError=reproj_px,
                                                 iterationsCount=2000, confidence=0.999,
                                                 flags=cv2.SOLVEPNP_SQPNP)
    if not ok or inliers is None or len(inliers) < 6:
        return {"ok": False, "reason": "solvePnPRansac failed or < 6 inliers"}
    inl = inliers.reshape(-1)
    K_out = K.copy()
    if refine_focal:
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
                 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
        _, K_out, _, rvecs, tvecs = cv2.calibrateCamera(
            [pts[inl].reshape(-1, 3).astype(np.float32)], [pix[inl].reshape(-1, 2).astype(np.float32)],
            image_size, K.copy(), dist.copy(), flags=flags)
        rvec, tvec = rvecs[0], tvecs[0]
    rvec, tvec = cv2.solvePnPRefineLM(pts[inl], pix[inl], K_out, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K_out, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - pix.reshape(-1, 2), axis=1)
    mask = np.zeros(len(err), bool)
    mask[inl] = True
    return {
        "ok": True, "rvec": rvec.reshape(3), "tvec": tvec.reshape(3), "K": K_out, "reproj": proj.reshape(-1, 2),
        "errors": err, "inlier_mask": mask, "rms_inliers_px": float(np.sqrt(np.mean(err[mask] ** 2))),
        "rms_all_px": float(np.sqrt(np.mean(err ** 2))), "max_inlier_px": float(err[mask].max()),
        "n_inliers": int(mask.sum()), "n_points": int(len(err)),
    }


def mujoco_camera_params(rvec, tvec, K, height: int, look_depth_point) -> dict:
    conv = opencv_to_mujoco(rvec, tvec)
    pos, fwd = conv["pos"], conv["forward"]
    d = float((np.asarray(look_depth_point, float) - pos) @ fwd)
    look_at = pos + fwd * d
    x_rf, y_rf = roll_free_axes(pos, look_at)
    roll_deg = math.degrees(math.atan2(np.cross(x_rf, conv["xaxis"]) @ (-fwd), x_rf @ conv["xaxis"]))
    pitch_deg = math.degrees(math.asin(-fwd[2]))
    return {
        "position_task": pos.tolist(), "look_at_task": look_at.tolist(),
        "xaxis": conv["xaxis"].tolist(), "yaxis": conv["yaxis"].tolist(),
        "fovy_deg": fovy_deg_from_fy(float(K[1, 1]), height),
        "roll_about_optical_axis_deg": roll_deg, "pitch_down_deg": pitch_deg,
        "distance_to_look_at_m": d,
    }


# --------------------------------------------------------------------------- outputs
def draw_overlay(rgb: np.ndarray, cs: list[Correspondence], title: str) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for c in cs:
        u, v = int(round(c.pixel[0])), int(round(c.pixel[1]))
        cv2.circle(img, (u, v), 12, (0, 255, 0), 2)
        if c.reproj_px is not None:
            ru, rv = int(round(c.reproj_px[0])), int(round(c.reproj_px[1]))
            colour = (0, 0, 255) if c.inlier else (0, 165, 255)
            cv2.drawMarker(img, (ru, rv), colour, cv2.MARKER_CROSS, 24, 2)
            cv2.line(img, (u, v), (ru, rv), (255, 255, 255), 1)
            cv2.putText(img, f"{c.kind} {c.error_px:.0f}px", (u + 14, v - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
    cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def write_task_camera_block(task_path: Path, params: dict, provenance: str, K: np.ndarray) -> None:
    """Replace only the ``camera_iphone`` mapping in task.yaml; other blocks stay byte-identical."""
    text = task_path.read_text()
    m = re.search(r"^camera_iphone:\n(?:[ \t]+.*\n)*", text, re.M)
    if m is None:
        raise ValueError("camera_iphone block not found in task.yaml")
    task = load_yaml(task_path)
    cam = task["camera_iphone"]
    f6 = [f"{v:.4f}" for v in params["position_task"]]
    l6 = [f"{v:.4f}" for v in params["look_at_task"]]
    xa = [f"{v:.5f}" for v in params["xaxis"]]
    ya = [f"{v:.5f}" for v in params["yaxis"]]
    block = (
        "camera_iphone:\n"
        f"  key: {cam['key']}\n"
        f"  width: {cam['width']}\n"
        f"  height: {cam['height']}\n"
        f"  fovy_deg: {params['fovy_deg']:.2f}\n"
        f"  intrinsics_px: {{fx: {K[0, 0]:.1f}, fy: {K[1, 1]:.1f}, cx: {K[0, 2]:.1f}, cy: {K[1, 2]:.1f}}}\n"
        f"  extrinsics_provenance: \"{provenance}\"\n"
        "  # Base/task frame (m). look_at = point on the optical axis nearest the workspace centre; the\n"
        "  # roll-free look-at form drops the roll about the optical axis, kept in xyaxes_task below.\n"
        f"  approx_position_task: [{', '.join(f6)}]\n"
        f"  look_at_task: [{', '.join(l6)}]\n"
        f"  xyaxes_task: [{', '.join(xa + ya)}]   # MuJoCo xyaxes (x right, y up); roll {params['roll_about_optical_axis_deg']:.1f} deg\n"
    )
    task_path.write_text(text[: m.start()] + block + text[m.end():])


def render_preview(task: dict, params: dict, state_deg: np.ndarray, ball_base: np.ndarray,
                   real_rgb: np.ndarray, out_path: Path, *, sim_config: Path | None = None,
                   title: str = "") -> None:
    """Side-by-side: MuJoCo render through the fitted camera (full xyaxes, with roll) at the real
    joint state next to the real frame. Visual check only; nothing is measured here."""
    import mujoco

    from real2sim.sim.env import sim_from_real
    from real2sim.sim.model import DEFAULT_CONFIG, scene_xml, task_frame_offset

    config = load_yaml(sim_config or DEFAULT_CONFIG)
    xml = scene_xml(config, task)
    offset = task_frame_offset(task)
    pos = np.asarray(params["position_task"], float) + offset
    xyaxes = [*params["xaxis"], *params["yaxis"]]
    xml = re.sub(r'<camera name="iphone"[^>]*/>',
                 f'<camera name="iphone" pos="{" ".join(f"{v:.6g}" for v in pos)}" '
                 f'xyaxes="{" ".join(f"{v:.6g}" for v in xyaxes)}" fovy="{params["fovy_deg"]:.4g}" />', xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mapping = JointMapping.from_task_config(task)
    q = sim_from_real(mapping, np.asarray(state_deg, float))
    for name, val in zip(config["joint_names"], q, strict=True):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = val
    ball_adr = model.jnt_qposadr[model.joint("ball_free").id]
    data.qpos[ball_adr:ball_adr + 3] = np.asarray(ball_base, float) + offset
    mujoco.mj_forward(model, data)
    w, h = int(config["render"]["width"]), int(config["render"]["height"])
    renderer = mujoco.Renderer(model, height=h, width=w)
    try:
        renderer.update_scene(data, camera="iphone")
        sim_rgb = renderer.render().copy()
    finally:
        renderer.close()
    real_small = cv2.resize(real_rgb, (w, h), interpolation=cv2.INTER_AREA)
    both = cv2.cvtColor(np.concatenate([sim_rgb, real_small], axis=1), cv2.COLOR_RGB2BGR)
    cv2.putText(both, "sim, fitted camera (joint state from data; ball at FK point)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(both, f"real {title}", (w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), both, [cv2.IMWRITE_JPEG_QUALITY, 85])


def run(task_path: Path = DEFAULT_TASK, data_path: Path = DEFAULT_DATA, out_dir: Path = OUT_DIR, *,
        write_config: bool = True, n_overlays: int = 4) -> dict:
    t0 = time.time()
    task = load_yaml(task_path)
    data_cfg = load_yaml(data_path)
    cam = task["camera_iphone"]
    key = cam["key"]
    W, H = int(cam["width"]), int(cam["height"])
    kp = cam["intrinsics_px"]
    K0 = intrinsics_matrix(float(kp["fx"]), float(kp["fy"]), float(kp["cx"]), float(kp["cy"]))
    root = resolve_dataset_root(data_cfg["source"]["repo_id"], data_cfg["source"]["revision"],
                                Path(data_cfg["source"]["local_root_hint"]).expanduser())
    ds = RawDataset(root)
    excluded = set(data_cfg.get("excluded_episodes", []))
    episodes = [e for e in ds.episode_indices if e not in excluded]
    fk = So101FK(mapping=JointMapping.from_task_config(task))
    cs = collect(ds, task, key, episodes, fk)
    report: dict = {
        "generated_by": "real2sim.sim.camera_fit", "dataset_root": str(root),
        "source_revision": data_cfg["source"]["revision"], "image_size": [W, H],
        "intrinsics_provisional": {"fx": K0[0, 0], "fy": K0[1, 1], "cx": K0[0, 2], "cy": K0[1, 2]},
        "detector": {"hsv_lo": HSV_LO, "hsv_hi": HSV_HI, "min_area_px": MIN_BALL_AREA_PX},
        "image_lag_frames_applied": IMAGE_LAG_FRAMES,
        "point_sources": {
            "held": "gripperframe FK of follower state (task.yaml provisional joint mapping) = ball centre",
            "start": "task.yaml ball.start_region centre at radius height (DATA-DERIVED preliminary)",
            "bowl": "task.yaml bowl centre at floor-rest height (DATA-DERIVED preliminary)",
        },
        "n_points": len(cs), "n_by_kind": {k: sum(c.kind == k for c in cs) for k in ("held", "start", "bowl")},
        "fit_accepted": False, "config_written": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(cs) < 6:
        report["failure"] = f"only {len(cs)} correspondences; need >= 6"
        report["correspondences"] = [asdict(c) for c in cs]
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        return report

    P = np.array([c.point_base for c in cs])
    U = np.array([c.pixel for c in cs])
    fit = solve_pose(P, U, K0, image_size=(W, H))
    report["fit_fixed_focal"] = _fit_summary(fit)
    focal_refined = False
    if fit["ok"] and fit["rms_inliers_px"] > REFINE_FOCAL_ABOVE_PX:
        fit2 = solve_pose(P, U, K0, refine_focal=True, image_size=(W, H))
        report["fit_refined_focal"] = _fit_summary(fit2)
        if fit2["ok"] and fit2["rms_inliers_px"] < fit["rms_inliers_px"]:
            fit, focal_refined = fit2, True
    # Diagnostic only: the pan sign is unverified (docs/calibration.md section 2).
    report["diagnostic_pan_sign_flipped"] = _pan_flip_diagnostic(cs, task, K0, (W, H))
    if not fit["ok"]:
        report["failure"] = fit.get("reason", "PnP failed")
        report["correspondences"] = [asdict(c) for c in cs]
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        return report

    for c, e, r, m in zip(cs, fit["errors"], fit["reproj"], fit["inlier_mask"], strict=True):
        c.error_px, c.reproj_px, c.inlier = float(e), r.tolist(), bool(m)
    params = mujoco_camera_params(fit["rvec"], fit["tvec"], fit["K"], H,
                                  [float(task["table"]["mat"]["center_task_xy"][0]), 0.0, 0.0])
    # Check the OpenCV->MuJoCo conversion on the fitted pose itself (full xyaxes and roll-free form).
    mj_full = project_mujoco(P, params["position_task"], params["xaxis"], params["yaxis"], params["fovy_deg"], W, H)
    x_rf, y_rf = roll_free_axes(params["position_task"], params["look_at_task"])
    mj_rf = project_mujoco(P, params["position_task"], x_rf, y_rf, params["fovy_deg"], W, H)
    report["conversion_check"] = {
        "max_abs_px_cv_vs_mujoco_full_xyaxes": float(np.abs(mj_full - fit["reproj"]).max()),
        "rms_px_roll_free_lookat_vs_detections": float(np.sqrt(np.mean(np.sum((mj_rf - U) ** 2, axis=1)))),
    }
    rms = fit["rms_inliers_px"]
    report.update({
        "focal_refined": focal_refined, "rms_inliers_px": rms, "rms_all_px": fit["rms_all_px"],
        "n_inliers": fit["n_inliers"], "opencv_pose": {"rvec": fit["rvec"].tolist(), "tvec": fit["tvec"].tolist(),
                                                       "convention": "x_cam = R x_base + t; +z forward, +y down"},
        "intrinsics_used": {"fx": float(fit["K"][0, 0]), "fy": float(fit["K"][1, 1]),
                            "cx": float(fit["K"][0, 2]), "cy": float(fit["K"][1, 2])},
        "mujoco_camera": params, "fit_accepted": rms <= ACCEPT_RMS_PX,
        "provisional_camera": {"approx_position_task": cam["approx_position_task"],
                               "look_at_task": cam["look_at_task"], "fovy_deg": cam["fovy_deg"]},
    })
    provenance = (f"DATA-DERIVED PnP, reprojection error {rms:.1f} px RMS over {fit['n_inliers']} inliers"
                  f" of {fit['n_points']} points"
                  + (", focal refined" if focal_refined else ", provisional intrinsics fixed")
                  + "; 3-D points from provisional FK mapping and preliminary start/bowl centres")
    report["provenance"] = provenance
    if report["fit_accepted"] and write_config:
        write_task_camera_block(task_path, params, provenance, fit["K"])
        report["config_written"] = True
    # Overlays: one frame per episode with the most points (held frames), plus a start and a bowl frame.
    fps = float(ds.info["fps"])
    chosen: list[tuple[int, int]] = []
    for kind in ("held", "start", "bowl"):
        for c in sorted((c for c in cs if c.kind == kind), key=lambda c: -(c.error_px or 0))[:2]:
            if (c.episode, c.frame) not in chosen:
                chosen.append((c.episode, c.frame))
    for ep, frame in chosen[: max(n_overlays, 3)]:
        rgb = decode_frame(ds.video_segment(key, ep), frame, fps)
        group = [c for c in cs if c.episode == ep and c.frame == frame]
        img = draw_overlay(rgb, group, f"ep{ep:03d} f{frame} green=detected ball, red=reprojected 3-D point"
                           f" (RMS {rms:.1f}px)")
        cv2.imwrite(str(out_dir / f"overlay_{ep:03d}_{frame:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    # Visual check: sim render through the fitted camera next to the real frame (held + start).
    held = [c for c in cs if c.kind == "held" and c.inlier]
    if held:
        c = min(held, key=lambda c: c.error_px or 0)
        rgb = decode_frame(ds.video_segment(key, c.episode), c.frame, fps)
        try:
            render_preview(task, params, np.array(c.state_deg), np.array(c.point_base), rgb,
                           out_dir / f"sim_vs_real_fit_{c.episode:03d}_{c.frame:05d}.jpg",
                           title=f"ep{c.episode:03d} f{c.frame}")
            report["preview"] = f"sim_vs_real_fit_{c.episode:03d}_{c.frame:05d}.jpg"
        except Exception as exc:  # rendering is optional (needs an offscreen GL context)
            report["preview_error"] = repr(exc)
    report["per_kind"] = {
        k: {"n": len(g), "n_inliers": sum(c.inlier for c in g),
            "rms_px": float(np.sqrt(np.mean([c.error_px ** 2 for c in g]))),
            "mean_signed_du_px": float(np.mean([c.reproj_px[0] - c.pixel[0] for c in g])),
            "mean_signed_dv_px": float(np.mean([c.reproj_px[1] - c.pixel[1] for c in g]))}
        for k in ("held", "start", "bowl") if (g := [c for c in cs if c.kind == k])
    }
    report["notes"] = [
        "Only correspondences with inlier=true (RANSAC, 30 px) enter the LM refinement.",
        "A systematic offset of a kind (mean_signed_du/dv) indicates its 3-D point source is off, not the camera.",
        "The camera depth along the optical axis is weakly constrained: all points lie in ~0.15 x 0.25 x 0.15 m.",
        "Focal refinement is attempted only when the fixed-focal RMS exceeds "
        f"{REFINE_FOCAL_ABOVE_PX} px and is kept only if it lowers the RMS.",
    ]
    report["correspondences"] = [asdict(c) for c in cs]
    report["runtime_s"] = time.time() - t0
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _fit_summary(fit: dict) -> dict:
    if not fit["ok"]:
        return {"ok": False, "reason": fit.get("reason")}
    return {k: fit[k] for k in ("rms_inliers_px", "rms_all_px", "max_inlier_px", "n_inliers", "n_points")} | {
        "ok": True, "fx": float(fit["K"][0, 0])}


def _pan_flip_diagnostic(cs: list[Correspondence], task: dict, K: np.ndarray, image_size) -> dict:
    mapping = JointMapping.from_task_config(task)
    flipped = JointMapping(sign=(-mapping.sign[0], *mapping.sign[1:]), offset_deg=mapping.offset_deg,
                           gripper_sim_range_rad=mapping.gripper_sim_range_rad)
    fk = So101FK(mapping=flipped)
    P = np.array([fk.fk_real(np.array(c.state_deg))[:3, 3] if c.kind == "held" else c.point_base for c in cs])
    U = np.array([c.pixel for c in cs])
    fit = solve_pose(P, U, K, image_size=image_size)
    return {"note": "shoulder_pan sign flipped for the FK-derived points only; diagnostic, not applied"} | _fit_summary(fit)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", type=Path, default=DEFAULT_TASK)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--no-write", action="store_true", help="never modify task.yaml")
    args = ap.parse_args(argv)
    rep = run(args.task, args.data, args.out, write_config=not args.no_write)
    keys = ("n_points", "n_by_kind", "rms_inliers_px", "n_inliers", "focal_refined", "fit_accepted",
            "config_written", "failure", "mujoco_camera", "diagnostic_pan_sign_flipped", "runtime_s")
    print(json.dumps({k: rep[k] for k in keys if k in rep}, indent=2))
    return 0 if rep.get("fit_accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
