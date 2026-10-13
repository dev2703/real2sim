"""DIAGNOSTIC camera + mat-size fit from hand-read robot joint pixels (known link lengths are the scale) plus the
mat outline. Joint anchors come from the MuJoCo model at the recorded state through the current mapping."""

import json
import sys

import cv2
import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

from real2sim.data.raw import RawDataset
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

W, H = 1280, 720
# hand-read pixels (full res) from gridded crops: joint name -> (u, v); None = not visible
OBS = {
    (5, 0): {"lift": (305, 235), "elbow": (75, 210), "wrist": (335, 118), "tip": (505, 305)},
    (5, 150): {"lift": (315, 220), "elbow": (60, 180), "wrist": (355, 95), "tip": (560, 200)},
    (5, 240): {"lift": (330, 215), "elbow": (545, 55)},
    (5, 330): {"lift": (330, 215), "elbow": (535, 50), "wrist": (690, 140)},
}
MAT_PX = np.array(
    [[330, 195], [915, 258], [1000, 690], [340, 697]], float
)  # far-left, far-right, near-right, near-left
SIG_J, SIG_M = 15.0, 8.0
ds = RawDataset(sys.argv[1])
env = So101PickPlaceEnv("configs/sim_so101.yaml", render_images=False)
jid = {
    n: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, n)
    for n in ("shoulder_lift", "elbow_flex", "wrist_flex")
}
site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
task = yaml.safe_load(open("configs/task.yaml"))
ztab = -task["robot_so101"]["base_z_offset"]
cxm, cym = task["table"]["mat"]["center_task_xy"]
pts3, pts2 = [], []
for (e, f), obs in OBS.items():
    S = ds.stack(ds.episode_rows(e), "observation.state")
    q = np.clip(sim_from_real(env.mapping, S[f]), env.limits[:, 0], env.limits[:, 1])
    env.reset(seed=0, scenario={"joint_positions": q})
    anchors = {
        "lift": env.data.xanchor[jid["shoulder_lift"]],
        "elbow": env.data.xanchor[jid["elbow_flex"]],
        "wrist": env.data.xanchor[jid["wrist_flex"]],
        "tip": env.data.site_xpos[site],
    }
    for n, px in obs.items():
        pts3.append(anchors[n].copy() - env.offset)
        pts2.append(px)
pts3, pts2 = np.array(pts3), np.array(pts2, float)


def mat_corners(sx, sy):
    return np.array(
        [
            [cxm - sx / 2, cym + sy / 2, ztab],
            [cxm + sx / 2, cym + sy / 2, ztab],
            [cxm + sx / 2, cym - sy / 2, ztab],
            [cxm - sx / 2, cym - sy / 2, ztab],
        ]
    )


def project(p3, f, rvec, tvec):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    px, _ = cv2.projectPoints(p3.reshape(-1, 1, 3), rvec, tvec, K, None)
    return px.reshape(-1, 2)


def resid(x, f_fixed=None):
    f = x[0] if f_fixed is None else f_fixed
    o = 0 if f_fixed is None else -1
    rvec, tvec, sx, sy = x[1 + o : 4 + o], x[4 + o : 7 + o], x[7 + o], x[8 + o]
    return np.concatenate(
        [
            ((project(pts3, f, rvec, tvec) - pts2) / SIG_J).ravel(),
            ((project(mat_corners(sx, sy), f, rvec, tvec) - MAT_PX) / SIG_M).ravel(),
        ]
    )


def init(f):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    ok, rvec, tvec = cv2.solvePnP(
        np.vstack([pts3, mat_corners(0.45, 0.40)]),
        np.vstack([pts2, MAT_PX]),
        K,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    return rvec.ravel(), tvec.ravel()


print("fixed-f sweep (joint rms px, mat rms px, mat size, camera pos task):")
for f in (800, 1000, 1300, 1600, 2000, 2500, 3000):
    rv, tv = init(f)
    r = least_squares(lambda x, focal=f: resid(x, focal), np.r_[rv, tv, 0.45, 0.40], method="lm")
    rj = r.fun[: len(pts2) * 2].reshape(-1, 2) * SIG_J
    rm = r.fun[len(pts2) * 2 :].reshape(-1, 2) * SIG_M
    Rm, _ = cv2.Rodrigues(r.x[0:3])
    pos = -Rm.T @ r.x[3:6]
    print(
        f"  f={f:5d}: joints {np.sqrt((rj**2).sum(1).mean()):5.1f}  mat {np.sqrt((rm**2).sum(1).mean()):5.1f}  mat {r.x[6]:.3f} x {r.x[7]:.3f} m  cam {np.round(pos, 2).tolist()}  cost {r.cost:.2f}"
    )
FIX = (
    [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
)  # optional fixed mat size "sx,sy" (m)
if FIX:

    def resid_fix(x, f_fixed=None):
        f = x[0] if f_fixed is None else f_fixed
        o = 0 if f_fixed is None else -1
        rvec, tvec = x[1 + o : 4 + o], x[4 + o : 7 + o]
        return np.concatenate(
            [
                ((project(pts3, f, rvec, tvec) - pts2) / SIG_J).ravel(),
                ((project(mat_corners(*FIX), f, rvec, tvec) - MAT_PX) / SIG_M).ravel(),
            ]
        )

    print(f"mat FIXED at {FIX[0]:.3f} x {FIX[1]:.3f} m; fixed-f sweep:")
    for f in (1000, 1300, 1600, 2000, 2500, 3000):
        rv, tv = init(f)
        r = least_squares(lambda x, focal=f: resid_fix(x, focal), np.r_[rv, tv], method="lm")
        rj = r.fun[: len(pts2) * 2].reshape(-1, 2) * SIG_J
        rm = r.fun[len(pts2) * 2 :].reshape(-1, 2) * SIG_M
        Rm, _ = cv2.Rodrigues(r.x[0:3])
        pos = -Rm.T @ r.x[3:6]
        fwd = Rm.T @ [0, 0, 1.0]
        print(
            f"  f={f:5d}: joints {np.sqrt((rj**2).sum(1).mean()):5.1f}  mat {np.sqrt((rm**2).sum(1).mean()):5.1f}  cam {np.round(pos, 3).tolist()}  pitch {np.degrees(np.arcsin(-fwd[2])):.1f}  cost {r.cost:.2f}"
        )
    best = None
    for f0 in (1000, 1500, 2200, 3000):
        rv, tv = init(f0)
        r = least_squares(resid_fix, np.r_[f0, rv, tv], method="lm")
        if best is None or r.cost < best.cost:
            best = r
    x = np.r_[best.x, FIX]
    best.x = x
    resid = resid_fix
best = best if FIX else None
for f0 in [] if FIX else (1000, 1500, 2200, 3000):
    rv, tv = init(f0)
    r = least_squares(resid, np.r_[f0, rv, tv, 0.45, 0.40], method="lm")
    if best is None or r.cost < best.cost:
        best = r
x = best.x
f = x[0]
Rm, _ = cv2.Rodrigues(x[1:4])
nj = len(pts2) * 2
best.fun = (
    best.fun
    if not FIX
    else np.concatenate(
        [
            ((project(pts3, f, x[1:4], x[4:7]) - pts2) / SIG_J).ravel(),
            ((project(mat_corners(*FIX), f, x[1:4], x[4:7]) - MAT_PX) / SIG_M).ravel(),
        ]
    )
)
pos = -Rm.T @ x[4:7]
fwd = Rm.T @ [0, 0, 1.0]
xax = Rm.T @ [1, 0, 0.0]
yax = -(Rm.T @ [0, 1, 0.0])
rj = best.fun[: len(pts2) * 2].reshape(-1, 2) * SIG_J
rm = best.fun[len(pts2) * 2 :].reshape(-1, 2) * SIG_M
print(
    f"FREE f: f={f:.0f} fovy {2 * np.degrees(np.arctan(H / 2 / f)):.1f}  joints rms {np.sqrt((rj**2).sum(1).mean()):.1f} px  mat rms {np.sqrt((rm**2).sum(1).mean()):.1f} px  mat {x[7]:.3f} x {x[8]:.3f} m  cam {np.round(pos, 3).tolist()}  pitch {np.degrees(np.arcsin(-fwd[2])):.1f}"
)
print("per-point joint residuals px:", np.round(np.linalg.norm(rj, axis=1), 0).tolist())
json.dump(
    {
        "f": float(f),
        "rvec": x[1:4].tolist(),
        "tvec": x[4:7].tolist(),
        "K": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]],
        "mat_size": [float(x[7]), float(x[8])],
        "position_task": pos.tolist(),
        "xyaxes_task": [*xax.tolist(), *yax.tolist()],
        "look_at_task": (pos + fwd * np.linalg.norm(pos - [cxm, cym, ztab])).tolist(),
        "fovy_deg": float(2 * np.degrees(np.arctan(H / 2 / f))),
        "joint_px": OBS and {f"{k[0]}_{k[1]}": v for k, v in OBS.items()},
        "mat_px": MAT_PX.tolist(),
    },
    open("outputs/tmp_appearance/camera_from_robot.json", "w"),
    indent=1,
)
env.close()
