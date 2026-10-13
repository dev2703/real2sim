"""DIAGNOSTIC camera fit anchored on operator measurements (2026-10-13): camera 24 in horizontally from the robot's
nearest edge and 21 in above the table; mat 16 x 12 in; ball ~15 in from the table edge. Unknowns: focal length, camera
orientation, camera x, mat centre. Observations: hand-read joint pixels (lift/elbow weighted highest) and mat corners."""

import json
import sys

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares

from real2sim.data.raw import RawDataset
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

IN = 0.0254
W, H = 1280, 720
OBS = {
    (5, 0): {"lift": (305, 235), "elbow": (75, 210), "wrist": (335, 118), "tip": (505, 305)},
    (5, 150): {"lift": (315, 220), "elbow": (60, 180), "wrist": (355, 95), "tip": (560, 200)},
    (5, 240): {"lift": (330, 215), "elbow": (545, 55)},
    (5, 330): {"lift": (330, 215), "elbow": (535, 50), "wrist": (690, 140)},
}
SIG = {"lift": 12.0, "elbow": 15.0, "wrist": 30.0, "tip": 40.0}
SIG_M = 10.0
MAT_PX = np.array([[330, 195], [915, 258], [1000, 690], [340, 697]], float)
BALL_PX = (844, 231)
ds = RawDataset(sys.argv[1])
env = So101PickPlaceEnv("configs/sim_so101.yaml", render_images=False)
task = env.task
bz = task["robot_so101"]["base_z_offset"]
ztab = -bz
r_ball = task["ball"]["radius"]
# robot's nearest edge to a camera on the -y side: min y over base-body geoms' bounding boxes (task frame)
base_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "base")
mujoco.mj_forward(env.model, env.data)
ymin = min(
    float(env.data.geom_xpos[g][1] - env.offset[1] - env.model.geom_rbound[g])
    for g in range(env.model.ngeom)
    if env.model.geom_bodyid[g] == base_body
)
cam_y = -(24 * IN - ymin)
cam_z = 21 * IN - bz  # task frame (base origin 12 mm above the table)
print(
    f"base nearest edge y = {ymin:+.3f} m -> camera fixed at y = {cam_y:.3f}, z = {cam_z:.3f} (task frame)"
)
jid = {
    n: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, n)
    for n in ("shoulder_lift", "elbow_flex", "wrist_flex")
}
site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
pts3, pts2, sig = [], [], []
for (e, f), obs in OBS.items():
    S = ds.stack(ds.episode_rows(e), "observation.state")
    env.reset(
        seed=0,
        scenario={
            "joint_positions": np.clip(
                sim_from_real(env.mapping, S[f]), env.limits[:, 0], env.limits[:, 1]
            )
        },
    )
    anc = {
        "lift": env.data.xanchor[jid["shoulder_lift"]],
        "elbow": env.data.xanchor[jid["elbow_flex"]],
        "wrist": env.data.xanchor[jid["wrist_flex"]],
        "tip": env.data.site_xpos[site],
    }
    for n, px in obs.items():
        pts3.append(anc[n].copy() - env.offset)
        pts2.append(px)
        sig.append(SIG[n])
pts3, pts2, sig = np.array(pts3), np.array(pts2, float), np.array(sig)


def project(p3, f, rvec, tvec):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    px, _ = cv2.projectPoints(p3.reshape(-1, 1, 3), rvec, tvec, K, None)
    return px.reshape(-1, 2)


def corners(cx, cy, sx, sy):
    return np.array(
        [
            [cx - sx / 2, cy + sy / 2, ztab],
            [cx + sx / 2, cy + sy / 2, ztab],
            [cx + sx / 2, cy - sy / 2, ztab],
            [cx - sx / 2, cy - sy / 2, ztab],
        ]
    )


def unpack(x):
    return x[0], x[1:4], np.array([x[4], cam_y, cam_z]), x[5], x[6]


def resid(x, sx, sy):
    f, rvec, pos, mcx, mcy = unpack(x)
    Rm, _ = cv2.Rodrigues(rvec)
    tvec = -Rm @ pos
    return np.concatenate(
        [
            ((project(pts3, f, rvec, tvec) - pts2) / sig[:, None]).ravel(),
            ((project(corners(mcx, mcy, sx, sy), f, rvec, tvec) - MAT_PX) / SIG_M).ravel(),
        ]
    )


def look_rvec(pos, target):
    z = (pos - target) / np.linalg.norm(pos - target)
    x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)  # MuJoCo-style camera axes
    Rcw = np.stack([x, -y, -z])  # OpenCV: x right, y down, z forward
    return cv2.Rodrigues(Rcw)[0].ravel()


results = {}
for sx, sy in ((0.406, 0.305), (0.305, 0.406)):
    best = None
    for f0 in (900, 1300, 1800, 2500):
        for cx0 in (-0.1, 0.1, 0.3):
            x0 = np.r_[
                f0,
                look_rvec(np.array([cx0, cam_y, cam_z]), np.array([0.25, 0.0, ztab])),
                cx0,
                0.25,
                0.0,
            ]
            try:
                r = least_squares(resid, x0, args=(sx, sy), method="lm")
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    f, rvec, pos, mcx, mcy = unpack(best.x)
    Rm, _ = cv2.Rodrigues(rvec)
    tvec = -Rm @ pos
    fwd = Rm.T @ [0, 0, 1.0]
    rj = project(pts3, f, rvec, tvec) - pts2
    rm = project(corners(mcx, mcy, sx, sy), f, rvec, tvec) - MAT_PX
    ray = Rm.T @ (
        np.linalg.inv(np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]))
        @ [BALL_PX[0], BALL_PX[1], 1.0]
    )
    s_ = (ztab + r_ball - pos[2]) / ray[2]
    ball = pos + s_ * ray
    print(
        f"mat {sx:.3f}x{sy:.3f}: f={f:.0f} (fovy {2 * np.degrees(np.arctan(H / 2 / f)):.1f})  cam x={pos[0]:+.3f}  pitch {np.degrees(np.arcsin(-fwd[2])):.1f}  mat centre ({mcx:.3f},{mcy:.3f})  joints rms {np.sqrt((rj**2).sum(1).mean()):.1f} px (lift/elbow {np.sqrt((rj[sig <= 15] ** 2).sum(1).mean()):.1f})  mat rms {np.sqrt((rm**2).sum(1).mean()):.1f} px  cost {best.cost:.1f}  ball xy ({ball[0]:.3f},{ball[1]:.3f})"
    )
    results[(sx, sy)] = (
        best.cost,
        dict(
            f=float(f),
            rvec=rvec.tolist(),
            tvec=tvec.tolist(),
            K=[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]],
            position_task=pos.tolist(),
            xyaxes_task=[*(Rm.T @ [1, 0, 0.0]).tolist(), *(-(Rm.T @ [0, 1, 0.0])).tolist()],
            look_at_task=(pos + fwd * np.linalg.norm(pos - [mcx, mcy, ztab])).tolist(),
            fovy_deg=float(2 * np.degrees(np.arctan(H / 2 / f))),
            mat_size=[sx, sy],
            mat_center=[float(mcx), float(mcy)],
            ball_xy_ep5_f0=ball[:2].tolist(),
            joint_rms_px=float(np.sqrt((rj**2).sum(1).mean())),
            mat_rms_px=float(np.sqrt((rm**2).sum(1).mean())),
        ),
    )
key = min(results, key=lambda k: results[k][0])
json.dump(results[key][1], open("outputs/tmp_appearance/camera_measured.json", "w"), indent=1)
print("saved best:", key)
env.close()
