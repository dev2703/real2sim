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

The ``backproject`` subcommand inverts the fit: the in-bowl and start-marker ball detections are
back-projected through the fitted camera onto their known rest-height planes to estimate the bowl
centre and the start-region centre (written to task.yaml only if the spread over detections is
below ``BACKPROJECT_ACCEPT_STD_M``), and two sim-vs-real images are rendered for a visual check.
The HSV centroid of a ball resting in the bowl is biased (the near rim and the ball's own shadow
cut the lower half of the blob; the pale bowl interior leaks into it in some frames), so each
start/bowl detection is first refined by fitting a circle of the radius the camera predicts for
the ball to the ball's chromatic outline (``refine_ball_circle``); detections whose outline
fit or blob area is inconsistent with a full ball are excluded, and the raw centroid's
area-consistency ratio is recorded either way.

All rest-height planes are measured from the table top at ``z_task = -robot_so101.base_z_offset``
(``real2sim.sim.model.base_z_offset``; 0 until D-014 is approved), the same convention the scene
assembly uses, so an approved base lift moves the planes and the fit's 3-D points together.

Nothing here touches policy observations; it only sets the render camera and scene geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import cv2
import numpy as np

from real2sim.config import REPO_ROOT, load_yaml
from real2sim.data.raw import RawDataset, VideoSegment, resolve_dataset_root
from real2sim.geometry.so101_fk import JointMapping, So101FK

ROOT = REPO_ROOT
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


def table_top_z_task(task: dict) -> float:
    """Height of the table top in the task (= base) frame: ``-robot_so101.base_z_offset``.

    ``real2sim.sim.model`` places the robot base ``base_z_offset`` above the table top while the
    task frame stays equal to the base frame, so the table (and everything resting on it) sits at
    ``z_task = -base_z_offset``. Zero until D-014 is approved.
    """
    from real2sim.sim.model import base_z_offset

    return 0.0 - base_z_offset(task)  # 0.0 - x avoids printing -0.0 for the default


def ball_rest_height_in_bowl(task: dict) -> float:
    """Ball centre height (task frame) when resting on the bowl floor: table top + floor slab
    + radius. The bowl base sits on the table top (``real2sim.sim.model.scene_xml``)."""
    bowl = task["bowl"]
    floor = float(bowl["outer_height"]) - float(bowl["inner_depth"])
    return table_top_z_task(task) + floor + float(task["ball"]["radius"])


def ball_rest_height_on_mat(task: dict) -> float:
    """Ball centre height (task frame) when resting on the mat: table top + radius."""
    return table_top_z_task(task) + float(task["ball"]["radius"])


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
    z_start = ball_rest_height_on_mat(task)
    for ep in episodes[:start_episodes]:
        seg = ds.video_segment(key, ep)
        det = ball_centroid_px(decode_frame(seg, 0, fps), min_area)
        if det is not None:
            out.append(Correspondence(ep, 0, "start", [*start_xy, z_start], [det[0], det[1]], det[2]))
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
    x_rf, _ = roll_free_axes(pos, look_at)
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
        "table_top_z_task_m": table_top_z_task(task),
        "point_sources": {
            "held": "gripperframe FK of follower state (task.yaml provisional joint mapping) = ball centre",
            "start": "task.yaml ball.start_region centre at table top + radius",
            "bowl": "task.yaml bowl centre at table top + bowl floor + radius",
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


# --------------------------------------------------------------------------- back-projection
BACKPROJECT_ACCEPT_STD_M = 0.02  # write a centre to task.yaml only if its x and y std are below this
FOCAL_SENSITIVITY_SCALES = (0.9, 1.1)  # depth is weakly constrained (focal refinement diverged)
# Area-consistency band for the raw HSV blob: ball_area_px / (pi r_px^2) with r_px the silhouette
# radius the camera predicts at the back-projected depth. Below: truncated (shadow / rim); above:
# the mask leaked into the surroundings. Used as the gate when no outline refinement is possible.
AREA_RATIO_BAND = (0.8, 1.25)
# Known-radius circle fit of the ball outline on the chromatic "yellowness" image.
CIRCLE_CANNY = (40, 120)
CIRCLE_MIN_GRADIENT = 6.0  # yellowness units per px; drops faint texture edges
CIRCLE_INLIER_TOL_PX = 2.5
CIRCLE_MIN_COS = 0.6  # the yellowness gradient at an inlier must point towards the centre
CIRCLE_RANSAC_ITERS = 600
CIRCLE_SEARCH_ANNULUS = (0.4, 1.8)  # edge points kept, in units of r_px around the coarse centroid
CIRCLE_MIN_ARC_COVERAGE = 0.3  # fraction of 36 angular bins with an inlier (a rim-occluded ball ~0.4)
CIRCLE_MIN_DISC_FILL = 0.5  # fraction of the fitted disc that is yellow


def expected_ball_radius_px(point_base, rvec, tvec, K: np.ndarray, radius: float) -> float:
    """Silhouette radius (px) of a ball of ``radius`` centred at ``point_base`` seen by the OpenCV
    camera: ``f * R / sqrt(d^2 - R^2)`` with ``d`` the optical depth (square pixels assumed)."""
    R_cw, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))
    depth = float((R_cw @ np.asarray(point_base, float).reshape(3) + np.asarray(tvec, float).reshape(3))[2])
    if depth <= radius:
        raise ValueError("ball centre is not in front of the camera")
    return float(K[1, 1]) * radius / math.sqrt(depth * depth - radius * radius)


def yellowness(rgb: np.ndarray) -> np.ndarray:
    """``(R + G) / 2 - B`` clipped to uint8: high on the tennis ball (also in its shadow), ~0 on the
    grey bowl, the white seams and the dark mat, so its edges outline the ball and not the rim."""
    f = rgb.astype(np.float32)
    return np.clip((f[..., 0] + f[..., 1]) / 2 - f[..., 2], 0, 255).astype(np.uint8)


def fit_known_radius_circle(points: np.ndarray, normals: np.ndarray, radius_px: float, *,
                            tol_px: float = CIRCLE_INLIER_TOL_PX, min_cos: float = CIRCLE_MIN_COS,
                            iterations: int = CIRCLE_RANSAC_ITERS, seed: int = 0) -> dict | None:
    """RANSAC + Gauss-Newton fit of a circle of fixed ``radius_px`` to oriented edge points.

    ``normals[i]`` is the (unit) image gradient at ``points[i]``; a point is an inlier only when it
    lies within ``tol_px`` of the circle *and* its gradient points towards the centre (cosine above
    ``min_cos``), i.e. the yellow side is inside. Returns ``None`` if no circle has >= 3 inliers.
    """
    pts = np.asarray(points, float).reshape(-1, 2)
    nrm = np.asarray(normals, float).reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return None
    r = float(radius_px)

    def inlier_mask(c):
        d = pts - c
        dist = np.linalg.norm(d, axis=1)
        return (np.abs(dist - r) < tol_px) & ((-d * nrm).sum(axis=1) > min_cos * dist)

    rng = np.random.default_rng(seed)
    best_n, best_c = 0, None
    for _ in range(iterations):
        i, j = rng.choice(n, 2, replace=False)
        p, q = pts[i], pts[j]
        chord = q - p
        d = float(np.linalg.norm(chord))
        if d < 0.5 * r or d >= 2 * r:
            continue
        mid, h = (p + q) / 2, math.sqrt(r * r - (d / 2) ** 2)
        perp = np.array([-chord[1], chord[0]]) / d
        for sgn in (1.0, -1.0):
            c = mid + sgn * h * perp
            k = int(inlier_mask(c).sum())
            if k > best_n:
                best_n, best_c = k, c
    if best_c is None or best_n < 3:
        return None
    c = best_c
    for _ in range(10):  # Gauss-Newton on the centre with the radius fixed
        inl = inlier_mask(c)
        if inl.sum() < 3:
            break
        d = pts[inl] - c
        dist = np.linalg.norm(d, axis=1)
        step = np.linalg.lstsq(-d / dist[:, None], -(dist - r), rcond=None)[0]
        c = c + step
        if np.linalg.norm(step) < 1e-3:
            break
    inl = inlier_mask(c)
    d = pts[inl] - c
    dist = np.linalg.norm(d, axis=1)
    bins = np.unique(((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi) * 36).astype(int) % 36)
    return {"centre": c, "n_inliers": int(inl.sum()), "arc_coverage": len(bins) / 36.0,
            "rms_px": float(np.sqrt(np.mean((dist - r) ** 2))) if inl.any() else float("nan")}


def refine_ball_circle(rgb: np.ndarray, centre_uv, radius_px: float, *, seed: int = 0) -> dict:
    """Refine a coarse ball detection to the centre of a circle of known ``radius_px`` fitted to
    the ball's chromatic outline. Returns ``centre`` (refined, or the input when the fit is not
    ``ok``), the fit statistics, ``disc_fill`` (fraction of the fitted disc that is yellow) and
    ``ok``: enough inliers spread over >= ``CIRCLE_MIN_ARC_COVERAGE`` of the circumference and a
    disc that is at least ``CIRCLE_MIN_DISC_FILL`` yellow (rejects a fit to the bowl rim)."""
    c0 = np.asarray(centre_uv, float).reshape(2)
    yel = cv2.GaussianBlur(yellowness(rgb), (5, 5), 0)
    edges = cv2.Canny(yel, *CIRCLE_CANNY)
    yf = yel.astype(np.float32)
    gx = cv2.Sobel(yf, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(yf, cv2.CV_32F, 0, 1, ksize=3) / 8
    gm = np.hypot(gx, gy)
    ys, xs = np.nonzero((edges > 0) & (gm > CIRCLE_MIN_GRADIENT))
    pts = np.column_stack([xs, ys]).astype(float) + 0.5
    dist = np.linalg.norm(pts - c0, axis=1)
    lo, hi = CIRCLE_SEARCH_ANNULUS
    sel = (dist > lo * radius_px) & (dist < hi * radius_px)
    pts, nrm = pts[sel], np.column_stack([gx[ys, xs], gy[ys, xs]])[sel] / gm[ys, xs][sel, None]
    out = {"centre": c0.tolist(), "radius_px": float(radius_px), "n_edge": int(len(pts)),
           "n_inliers": 0, "arc_coverage": 0.0, "rms_px": float("nan"), "disc_fill": float("nan"), "ok": False}
    fit = fit_known_radius_circle(pts, nrm, radius_px, seed=seed)
    if fit is None:
        out["reason"] = "no circle of the expected radius on the ball outline"
        return out
    c = fit["centre"]
    yy, xx = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
    disc = (xx + 0.5 - c[0]) ** 2 + (yy + 0.5 - c[1]) ** 2 < radius_px ** 2
    fill = float((yel[disc] > 30).mean()) if disc.any() else 0.0
    out.update({"n_inliers": fit["n_inliers"], "arc_coverage": fit["arc_coverage"], "rms_px": fit["rms_px"],
                "disc_fill": fill})
    if fit["arc_coverage"] < CIRCLE_MIN_ARC_COVERAGE:
        out["reason"] = f"inliers cover only {fit['arc_coverage']:.2f} of the circumference"
    elif fill < CIRCLE_MIN_DISC_FILL:
        out["reason"] = f"fitted disc is only {fill:.2f} yellow (rim, not ball)"
    else:
        out["ok"] = True
        out["centre"] = c.tolist()
    return out


def refine_detections(cs: list[dict], task: dict, rvec, tvec, K: np.ndarray, load_frame, *,
                      kinds=("bowl", "start")) -> list[dict]:
    """Per start/bowl correspondence: raw centroid, area-consistency ratio and the known-radius
    circle refinement on the decoded frame (``load_frame(episode, frame) -> rgb``).

    The expected radius comes from the depth of the raw centroid back-projected onto the
    detection's rest plane, then once more from the refined centre (the depth changes by < 1 %).
    """
    radius = float(task["ball"]["radius"])
    planes = {"bowl": ball_rest_height_in_bowl(task), "start": ball_rest_height_on_mat(task)}
    out = []
    for c in cs:
        if c["kind"] not in kinds:
            continue
        uv = np.asarray(c["pixel"], float)
        rgb = load_frame(int(c["episode"]), int(c["frame"]))
        rec = {"episode": c["episode"], "frame": c["frame"], "kind": c["kind"], "pixel_centroid": uv.tolist(),
               "ball_area_px": c["ball_area_px"]}
        p0 = backproject_to_plane([uv], rvec, tvec, K, planes[c["kind"]])[0]
        r0 = expected_ball_radius_px(p0, rvec, tvec, K, radius)
        rec["area_ratio"] = float(c["ball_area_px"]) / (math.pi * r0 * r0)
        rec["area_consistent"] = bool(AREA_RATIO_BAND[0] <= rec["area_ratio"] <= AREA_RATIO_BAND[1])
        fit = refine_ball_circle(rgb, uv, r0)
        if fit["ok"]:
            p1 = backproject_to_plane([fit["centre"]], rvec, tvec, K, planes[c["kind"]])[0]
            fit = refine_ball_circle(rgb, fit["centre"], expected_ball_radius_px(p1, rvec, tvec, K, radius))
        rec["circle_fit"] = {k: v for k, v in fit.items() if k != "centre"}
        rec["pixel_refined"] = fit["centre"] if fit["ok"] else None
        rec["shift_px"] = (np.asarray(fit["centre"]) - uv).tolist() if fit["ok"] else None
        rec["used"] = bool(fit["ok"])
        if not fit["ok"]:
            rec["reason"] = fit.get("reason", "outline fit failed")
        out.append(rec)
    return out


def backproject_to_plane(pixels, rvec, tvec, K: np.ndarray, z_plane: float) -> np.ndarray:
    """Intersect the rays through ``pixels`` (OpenCV camera ``x_cam = R x_base + t``) with the
    horizontal plane ``z = z_plane`` of the base/task frame. Returns ``[n, 3]`` base-frame points.

    Inverse of ``cv2.projectPoints`` with zero distortion for points known to lie on the plane.
    Raises if a ray does not hit the plane in front of the camera.
    """
    R_cw, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))
    t = np.asarray(tvec, float).reshape(3)
    centre = -R_cw.T @ t
    uv = np.asarray(pixels, float).reshape(-1, 2)
    rays_cam = np.column_stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1],
                                np.ones(len(uv))])
    rays = rays_cam @ R_cw  # row i = R_cw.T @ rays_cam[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (float(z_plane) - centre[2]) / rays[:, 2]
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError(f"ray(s) do not reach the plane z={z_plane} in front of the camera")
    return centre + s[:, None] * rays


def plane_estimate(pixels, rvec, tvec, K: np.ndarray, z_plane: float) -> dict:
    """Mean and sample std (ddof=1) of the back-projected points; ``n < 2`` gives NaN std."""
    pts = backproject_to_plane(pixels, rvec, tvec, K, z_plane)
    std = pts[:, :2].std(axis=0, ddof=1) if len(pts) > 1 else np.full(2, np.nan)
    return {"n": int(len(pts)), "z_plane_m": float(z_plane), "mean_xy": pts[:, :2].mean(axis=0).tolist(),
            "std_xy_m": std.tolist(), "max_abs_dev_xy_m": np.abs(pts[:, :2] - pts[:, :2].mean(axis=0)).max(axis=0).tolist(),
            "points_xy": pts[:, :2].tolist()}


def backproject_report(report: dict, task: dict, *, refined: list[dict] | None = None,
                       accept_std_m: float = BACKPROJECT_ACCEPT_STD_M,
                       focal_scales=FOCAL_SENSITIVITY_SCALES) -> dict:
    """Bowl centre and start-region centre from the PnP camera and the ball detections.

    * in-bowl detections (last frames) are back-projected onto ``z = table top + bowl floor +
      ball radius``;
    * start-marker detections (first frames) onto ``z = table top + ball radius`` (ball on the
      mat); the table top is ``table_top_z_task`` (``-base_z_offset``).

    ``refined`` (from :func:`refine_detections`) supplies the outline-fitted centres: only
    detections with ``used=True`` enter the estimate, at their refined pixel. Without it the raw
    centroids are used, gated by the blob-area consistency band ``AREA_RATIO_BAND`` (an
    under- or over-segmented blob has a biased centroid that the spread test cannot see).

    The pose is the fitted ``opencv_pose`` of ``report``; the intrinsics are the ones used by
    the fit. ``focal_scales`` re-run the back-projection with ``fx = fy`` scaled and the pose
    held fixed to show how the weak depth constraint moves the estimates. Held-ball detections
    are back-projected onto their own FK height as a metre-scale check of the camera.
    """
    pose = report["opencv_pose"]
    rvec, tvec = np.asarray(pose["rvec"], float), np.asarray(pose["tvec"], float)
    ki = report["intrinsics_used"]
    K = intrinsics_matrix(float(ki["fx"]), float(ki["fy"]), float(ki["cx"]), float(ki["cy"]))
    cs = report["correspondences"]
    radius = float(task["ball"]["radius"])
    planes = {"bowl": ball_rest_height_in_bowl(task), "start": ball_rest_height_on_mat(task)}
    previous = {"bowl": [float(v) for v in task["bowl"]["center_task_xy"]],
                "start": [float(v) for v in task["ball"]["start_region"]["center_task_xy"]]}
    out: dict = {
        "generated_by": "real2sim.sim.camera_fit backproject",
        "method": ("ray through each detected ball centre from the PnP camera (pose and intrinsics of "
                   "report.json held fixed) intersected with a horizontal plane of known height in the "
                   "task frame; centre = mean over the detections that pass the outline/area checks"),
        "detection_source": ("known-radius circle fit of the ball outline (refine_ball_circle)" if refined
                             else "raw HSV centroid, gated by AREA_RATIO_BAND"),
        "table_top_z_task_m": table_top_z_task(task), "area_ratio_band": list(AREA_RATIO_BAND),
        "camera_position_task": (-cv2.Rodrigues(rvec)[0].T @ tvec).tolist(),
        "intrinsics_used": dict(ki), "accept_std_m": float(accept_std_m), "estimates": {},
    }
    by_key = {(r["episode"], r["frame"], r["kind"]): r for r in (refined or [])}
    for kind in ("bowl", "start"):
        group = [c for c in cs if c["kind"] == kind]
        if not group:
            out["estimates"][kind] = {"n": 0, "accepted": False, "reason": "no detections of this kind"}
            continue
        detections = []
        for c in group:
            uv = np.asarray(c["pixel"], float)
            rec = {"episode": c["episode"], "frame": c["frame"], "pixel_centroid": uv.tolist(),
                   "ball_area_px": c["ball_area_px"]}
            ref = by_key.get((c["episode"], c["frame"], kind))
            if ref is not None:
                rec.update({k: ref[k] for k in ("area_ratio", "area_consistent", "circle_fit", "pixel_refined",
                                                "shift_px", "used") if k in ref})
                if not ref["used"]:
                    rec["reason"] = ref.get("reason", "outline fit failed")
            else:
                p0 = backproject_to_plane([uv], rvec, tvec, K, planes[kind])[0]
                r0 = expected_ball_radius_px(p0, rvec, tvec, K, radius)
                rec["area_ratio"] = float(c["ball_area_px"]) / (math.pi * r0 * r0)
                rec["area_consistent"] = bool(AREA_RATIO_BAND[0] <= rec["area_ratio"] <= AREA_RATIO_BAND[1])
                rec["used"] = rec["area_consistent"]
                if not rec["used"]:
                    rec["reason"] = (f"blob area is {rec['area_ratio']:.2f} of a full ball at this depth "
                                     f"(band {AREA_RATIO_BAND[0]}-{AREA_RATIO_BAND[1]})")
            detections.append(rec)
        used = [d for d in detections if d["used"]]
        pix = [d.get("pixel_refined") or d["pixel_centroid"] for d in used]
        if not pix:
            out["estimates"][kind] = {"n": 0, "n_detections": len(group), "accepted": False,
                                      "reason": "no detection passed the outline/area checks",
                                      "detections": detections, "previous_task_yaml_xy": previous[kind],
                                      "z_plane_m": planes[kind]}
            continue
        est = plane_estimate(pix, rvec, tvec, K, planes[kind])
        est["n_detections"] = len(group)
        est["episodes_frames"] = [[d["episode"], d["frame"]] for d in used]
        est["ball_area_px"] = [d["ball_area_px"] for d in used]
        est["detections"] = detections
        est["previous_task_yaml_xy"] = previous[kind]
        est["shift_from_previous_xy_m"] = (np.asarray(est["mean_xy"]) - np.asarray(previous[kind])).tolist()
        est["focal_sensitivity"] = {}
        for sc in focal_scales:
            Ks = K.copy()
            Ks[0, 0] *= sc
            Ks[1, 1] *= sc
            m = plane_estimate(pix, rvec, tvec, Ks, planes[kind])["mean_xy"]
            est["focal_sensitivity"][f"fx_x{sc:g}"] = {
                "mean_xy": m, "shift_xy_m": (np.asarray(m) - np.asarray(est["mean_xy"])).tolist()}
        std = np.asarray(est["std_xy_m"])
        est["accepted"] = bool(est["n"] >= 2 and np.all(np.isfinite(std)) and np.all(std < accept_std_m))
        if not est["accepted"]:
            est["reason"] = (f"std {std.tolist()} m not below {accept_std_m} m in both axes"
                             if est["n"] >= 2 else "fewer than 2 detections")
        out["estimates"][kind] = est
    held = [c for c in cs if c["kind"] == "held"]
    if held:
        fk = np.array([c["point_base"] for c in held])
        bp = np.vstack([backproject_to_plane([c["pixel"]], rvec, tvec, K, c["point_base"][2]) for c in held])
        res = bp[:, :2] - fk[:, :2]
        inl = np.array([bool(c.get("inlier")) for c in held])
        out["held_ball_check"] = {
            "note": ("held detections back-projected onto their own FK height, minus the FK point: "
                     "metre-scale residual of the camera on the points that constrained it"),
            "n": int(len(held)), "n_inliers": int(inl.sum()),
            "rms_xy_m_all": np.sqrt((res ** 2).mean(axis=0)).tolist(),
            "rms_xy_m_inliers": (np.sqrt((res[inl] ** 2).mean(axis=0)).tolist() if inl.any() else None),
            "mean_signed_xy_m_all": res.mean(axis=0).tolist(),
            "fk_mean_xyz_m": fk.mean(axis=0).tolist(), "fk_std_xyz_m": fk.std(axis=0, ddof=1).tolist(),
        }
    n_inl = {k: sum(bool(c.get("inlier")) for c in cs if c["kind"] == k) for k in ("start", "bowl")}
    out["notes"] = [
        "Simulator ground truth is not involved: inputs are real-image detections and the fitted camera.",
        "The PnP camera was fitted with the start/bowl 3-D points at their *previous* task.yaml values "
        f"({n_inl['start']} start inliers, {n_inl['bowl']} bowl inliers), so the start estimate is weakly "
        "circular; the held-ball FK points dominate the fit.",
        "Depth along the optical axis is weakly constrained (focal refinement diverged); the "
        "focal_sensitivity entries bound the resulting shift of the planar estimates.",
        "The in-bowl ball may rest off the bowl centre; the spread over episodes includes that.",
        "The HSV centroid of a ball in the bowl is biased (near rim and shadow cut the lower half; the "
        "pale interior leaks into the mask in some frames). The estimate therefore uses the centre of a "
        "circle of the camera-predicted radius fitted to the ball outline; area_ratio records how far "
        "each raw blob is from a full ball.",
    ]
    return out


def write_task_centres(task_path: Path, *, bowl: tuple[list[float], str] | None = None,
                       start: tuple[list[float], str] | None = None) -> None:
    """Rewrite only ``bowl.center_task_xy``/``bowl.center_provenance`` and
    ``ball.start_region.center_task_xy``/``provenance`` lines of task.yaml; everything else stays
    byte-identical. Each argument is ``(xy, provenance)`` or ``None`` to leave that block alone."""
    text = task_path.read_text()
    if bowl is not None:
        xy, prov = bowl
        m = re.search(r"^bowl:\n(?:[ \t]+.*\n)*", text, re.M)
        if m is None:
            raise ValueError("bowl block not found in task.yaml")
        block = m.group(0)
        block, n1 = re.subn(r"^(  center_task_xy: )\[[^\]]*\]", lambda mm: f"{mm.group(1)}[{xy[0]:.4f}, {xy[1]:.4f}]",
                            block, count=1, flags=re.M)
        block, n2 = re.subn(r'^(  center_provenance: )"[^"\n]*"', lambda mm: f'{mm.group(1)}"{prov}"',
                            block, count=1, flags=re.M)
        if n1 != 1 or n2 != 1:
            raise ValueError("bowl.center_task_xy / center_provenance lines not found")
        text = text[: m.start()] + block + text[m.end():]
    if start is not None:
        xy, prov = start
        pattern = r'^(  start_region: \{center_task_xy: )\[[^\]]*\](, radius: [^,}]*, provenance: )"[^"\n]*"(\})'
        text, n = re.subn(pattern, lambda mm: f'{mm.group(1)}[{xy[0]:.4f}, {xy[1]:.4f}]{mm.group(2)}"{prov}"{mm.group(3)}',
                          text, count=1, flags=re.M)
        if n != 1:
            raise ValueError("ball.start_region line not found in the expected inline form")
    task_path.write_text(text)


def centre_provenance(kind: str, est: dict, *, refined: bool) -> str:
    what = "in-bowl ball" if kind == "bowl" else "start-marker ball"
    how = "known-radius circle fit of the" if refined else "area-consistent HSV centroids of the"
    sx, sy = (100 * v for v in est["std_xy_m"])
    return (f"DATA-DERIVED v3: {how} {what} outline back-projected through the PnP camera, "
            f"n={est['n']}/{est.get('n_detections', est['n'])}, std {sx:.1f}/{sy:.1f} cm")


def _sim_config_for_task(task_path: Path, data_path: Path, out_path: Path) -> Path:
    """Copy of ``configs/sim_so101.yaml`` whose ``task_config``/``data_config`` point at the given
    files, so ``So101PickPlaceEnv`` builds its scene from the same task.yaml the comparison marks
    are projected from (the default config always reads the repo task.yaml)."""
    import yaml

    from real2sim.sim.model import DEFAULT_CONFIG

    config = load_yaml(DEFAULT_CONFIG)
    config["task_config"] = str(Path(task_path).resolve())
    config["data_config"] = str(Path(data_path).resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("# generated by real2sim.sim.camera_fit for one render; task/data paths overridden\n"
                        + yaml.safe_dump(config, sort_keys=False))
    return out_path


def _side_by_side(sim_rgb: np.ndarray, real_rgb: np.ndarray, left: str, right: str, marks: dict,
                  cam: dict, w: int, h: int, full_w: int, full_h: int) -> np.ndarray:
    """Sim render | real frame, with task-frame points projected through the task.yaml camera
    drawn on both halves (same camera, so the marks land at the same pixels)."""
    real_small = cv2.resize(real_rgb, (w, h), interpolation=cv2.INTER_AREA)
    both = cv2.cvtColor(np.concatenate([sim_rgb, real_small], axis=1), cv2.COLOR_RGB2BGR)
    x, y = cam["xyaxes_task"][:3], cam["xyaxes_task"][3:]
    scale = w / full_w
    for label, (p, colour) in marks.items():
        u, v = project_mujoco(np.asarray(p, float), cam["approx_position_task"], x, y, cam["fovy_deg"],
                              full_w, full_h)[0] * scale
        for dx in (0, w):
            cv2.drawMarker(both, (int(round(u)) + dx, int(round(v))), colour, cv2.MARKER_CROSS, 18, 2)
            cv2.putText(both, label, (int(u) + dx + 10, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    cv2.putText(both, left, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(both, right, (w + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return both


def render_comparisons(task_path: Path, report: dict, data_path: Path, out_dir: Path,
                       compare_out: Path, *, grasp_episode: int = 12, grasp_frame: int = 769) -> dict:
    """Two sim-vs-real images through the task.yaml camera (visual check, nothing measured):

    1. ``compare_out``: ``So101PickPlaceEnv(render_images=True).reset(seed=0)`` next to the first
       real frame of episode 0 (the frame of results/audit/montage/iphone/ep000_first.jpg).
    2. ``out_dir/sim_vs_real_grasp_<ep>_<frame>.jpg``: the recorded follower state of the held
       correspondence at (grasp_episode, grasp_frame) applied through the task.yaml joint mapping
       as ``scenario['joint_positions']`` (clipped to the model limits), ball at the FK gripper
       point. This is a SCRIPTED_REPLAY of one recorded state, never a policy rollout.
    """
    import tempfile

    from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

    task = load_yaml(task_path)
    data_cfg = load_yaml(data_path)
    cam = task["camera_iphone"]
    full_w, full_h = int(cam["width"]), int(cam["height"])
    bowl_xy = [float(v) for v in task["bowl"]["center_task_xy"]]
    start_xy = [float(v) for v in task["ball"]["start_region"]["center_task_xy"]]
    z_mat = ball_rest_height_on_mat(task)
    marks = {"bowl centre (task.yaml)": ([*bowl_xy, ball_rest_height_in_bowl(task)], (0, 200, 255)),
             "start centre (task.yaml)": ([*start_xy, z_mat], (255, 200, 0))}
    root = resolve_dataset_root(data_cfg["source"]["repo_id"], data_cfg["source"]["revision"],
                                Path(data_cfg["source"]["local_root_hint"]).expanduser())
    ds = RawDataset(root)
    fps = float(ds.info["fps"])
    key = cam["key"]
    result: dict = {"sim_task_path": str(Path(task_path).resolve()), "table_top_z_task_m": table_top_z_task(task)}
    with tempfile.TemporaryDirectory() as tmp, So101PickPlaceEnv(
            _sim_config_for_task(task_path, data_path, Path(tmp) / "sim_so101.yaml"), render_images=True) as env:
        for k, want in (("bowl", bowl_xy), ("start", start_xy)):
            got = env.task["bowl" if k == "bowl" else "ball"]
            got = got["center_task_xy"] if k == "bowl" else got["start_region"]["center_task_xy"]
            if not np.allclose(got, want):
                raise RuntimeError(f"sim scene {k} centre {got} differs from {task_path} {want}")
        w, h = int(env.config["render"]["width"]), int(env.config["render"]["height"])
        env.reset(seed=0)
        sim0 = env.render()
        real0 = decode_frame(ds.video_segment(key, 0), 0, fps)
        img = _side_by_side(sim0, real0, "sim: So101PickPlaceEnv.reset(seed=0), fitted camera, NO POLICY",
                            "real: ep000 frame 0 (montage ep000_first.jpg)", marks, cam, w, h, full_w, full_h)
        compare_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(compare_out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result["reset_seed0"] = str(compare_out)
        held = [c for c in report["correspondences"]
                if c["kind"] == "held" and c["episode"] == grasp_episode and c["frame"] == grasp_frame]
        if not held:
            result["grasp_error"] = f"no held correspondence for episode {grasp_episode} frame {grasp_frame}"
            return result
        c = held[0]
        q = np.clip(sim_from_real(env.mapping, np.asarray(c["state_deg"], float)), env.limits[:, 0], env.limits[:, 1])
        fk = np.asarray(c["point_base"], float)
        env.reset(seed=0, scenario={"joint_positions": q, "ball_xy": fk[:2],
                                    "ball_drop_height": max(0.0, float(fk[2]) - z_mat)})
        sim_g = env.render()
        real_g = decode_frame(ds.video_segment(key, grasp_episode), grasp_frame, fps)
        marks_g = dict(marks)
        marks_g["detected ball (real)"] = (backproject_to_plane(
            [c["pixel"]], report["opencv_pose"]["rvec"], report["opencv_pose"]["tvec"],
            intrinsics_matrix(*(float(report["intrinsics_used"][k]) for k in ("fx", "fy", "cx", "cy"))),
            float(fk[2]))[0].tolist(), (0, 255, 0))
        img = _side_by_side(
            sim_g, real_g,
            f"sim: SCRIPTED_REPLAY of real state ep{grasp_episode:03d} f{grasp_frame} via mapping; ball at FK point",
            f"real: ep{grasp_episode:03d} f{grasp_frame}", marks_g, cam, w, h, full_w, full_h)
        grasp_out = out_dir / f"sim_vs_real_grasp_{grasp_episode:03d}_{grasp_frame:05d}.jpg"
        cv2.imwrite(str(grasp_out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result["grasp"] = str(grasp_out)
        result["grasp_state_deg"] = c["state_deg"]
        result["grasp_joint_positions_rad"] = q.tolist()
        result["grasp_ball_fk_task"] = fk.tolist()
    return result


def run_backproject(report_path: Path = OUT_DIR / "report.json", task_path: Path = DEFAULT_TASK,
                    data_path: Path = DEFAULT_DATA, out_dir: Path = OUT_DIR, *, write_config: bool = True,
                    render: bool = True, refine: bool = True, compare_out: Path | None = None) -> dict:
    """``refine`` decodes each start/bowl frame again and replaces the HSV centroid by the
    known-radius outline fit; if the dataset cannot be read the raw centroids are used under the
    area gate and ``refine_error`` records why."""
    report = json.loads(Path(report_path).read_text())
    if "opencv_pose" not in report:
        raise ValueError(f"{report_path} holds no accepted PnP pose")
    task = load_yaml(task_path)
    refined = None
    refine_error = None
    if refine:
        try:
            data_cfg = load_yaml(data_path)
            root = resolve_dataset_root(data_cfg["source"]["repo_id"], data_cfg["source"]["revision"],
                                        Path(data_cfg["source"]["local_root_hint"]).expanduser())
            ds = RawDataset(root)
            fps = float(ds.info["fps"])
            key = task["camera_iphone"]["key"]
            ki = report["intrinsics_used"]
            refined = refine_detections(
                report["correspondences"], task, report["opencv_pose"]["rvec"], report["opencv_pose"]["tvec"],
                intrinsics_matrix(float(ki["fx"]), float(ki["fy"]), float(ki["cx"]), float(ki["cy"])),
                lambda ep, fr: decode_frame(ds.video_segment(key, ep), fr, fps))
        except Exception as exc:  # dataset missing: fall back to the gated raw centroids
            refine_error = repr(exc)
    out = backproject_report(report, task, refined=refined)
    out["report_path"] = str(report_path)
    out["refined"] = refined is not None
    if refine_error is not None:
        out["refine_error"] = refine_error
    out["config_written"] = {}
    writes = {}
    for kind in ("bowl", "start"):
        est = out["estimates"][kind]
        if est.get("accepted") and write_config:
            writes[kind] = ([float(v) for v in est["mean_xy"]],
                            centre_provenance(kind, est, refined=refined is not None))
        out["config_written"][kind] = bool(est.get("accepted") and write_config)
    if writes:
        write_task_centres(task_path, bowl=writes.get("bowl"), start=writes.get("start"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if render:
        try:
            out["renders"] = render_comparisons(task_path, report, data_path, out_dir,
                                               compare_out or ROOT / "results" / "audit" / "sim_vs_real_iphone.jpg")
        except Exception as exc:  # rendering needs the dataset and an offscreen GL context
            out["render_error"] = repr(exc)
    (out_dir / "backprojection.json").write_text(json.dumps(out, indent=2))
    return out


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


def main_backproject(argv) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m real2sim.sim.camera_fit backproject",
        description="Bowl centre and start-region centre by back-projecting the ball detections of "
                    "report.json through the fitted camera onto their rest-height planes.")
    ap.add_argument("--report", type=Path, default=OUT_DIR / "report.json")
    ap.add_argument("--task", type=Path, default=DEFAULT_TASK)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="directory for backprojection.json and renders")
    ap.add_argument("--compare-out", type=Path, default=None,
                    help="side-by-side reset(seed=0) image (default results/audit/sim_vs_real_iphone.jpg)")
    ap.add_argument("--no-write", action="store_true", help="never modify task.yaml")
    ap.add_argument("--no-render", action="store_true", help="skip the sim-vs-real renders")
    ap.add_argument("--no-refine", action="store_true",
                    help="use the raw HSV centroids (area-gated) instead of the outline circle fit")
    args = ap.parse_args(argv)
    rep = run_backproject(args.report, args.task, args.data, args.out, write_config=not args.no_write,
                          render=not args.no_render, refine=not args.no_refine, compare_out=args.compare_out)
    hide = ("points_xy", "episodes_frames", "ball_area_px", "detections")
    shown = {k: {kk: vv for kk, vv in v.items() if kk not in hide} for k, v in rep["estimates"].items()}
    print(json.dumps({"estimates": shown, "held_ball_check": rep.get("held_ball_check"),
                      "refined": rep["refined"], "refine_error": rep.get("refine_error"),
                      "config_written": rep["config_written"], "renders": rep.get("renders"),
                      "render_error": rep.get("render_error")}, indent=2))
    return 0 if all(e.get("accepted") for e in rep["estimates"].values()) else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "backproject":
        return main_backproject(argv[1:])
    if argv and argv[0] == "fit":
        argv = argv[1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog="Subcommands: fit (default), backproject")
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
