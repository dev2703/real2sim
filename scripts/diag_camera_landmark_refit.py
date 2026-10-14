"""DIAGNOSTIC camera/robot correspondence refit from independent hand-read joint landmarks (2026-10-14).

Observations: servo-horn boss centres of shoulder_lift / elbow_flex / wrist_flex, the wrist_roll servo
body centre, and the operator-measured mat outline. Train episodes fit; held-out validation episodes
only score. Nuisance parameters: one along-axis offset per horn landmark and a 3-D offset for the
roll-servo landmark in the wrist body frame. Models: M0 current camera (+nuisance only), M1 refit
camera (fovy, rotation, position; operator position prior), M2 M1 + joint zero offsets (pan, lift,
elbow, wrist_flex). Nothing here changes configs; the report is written to a fresh directory.
"""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares

from real2sim.data.paired_approach import camera_parameters, pose_render
from real2sim.sim.env import So101PickPlaceEnv

W, H = 1280.0, 720.0
IN = 0.0254
HORN = ["shoulder_lift", "elbow_flex", "wrist_flex"]
OFFSET_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]


def load_rows(env, landmarks, arrays):
    rows = []
    for r in landmarks["rows"]:
        if r.get("dropped"):
            continue
        idx = np.flatnonzero((arrays["episode_index"] == r["episode"]) & (arrays["frame_index"] == r["frame"]))
        if len(idx) != 1:
            raise ValueError(f"no unique recorded state for episode {r['episode']} frame {r['frame']}")
        rows.append({**r, "state": arrays["state"][idx[0]].copy()})
    return rows


class Model:
    def __init__(self, env, rows, mat, train_eps, cam_prior):
        self.env, self.rows, self.mat, self.cam_prior = env, rows, mat, cam_prior
        self.train = np.array([r["episode"] in train_eps for r in rows])
        self.jid = {n: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in HORN + OFFSET_JOINTS}
        wrist_body = env.model.body("wrist").id
        self.roll_geom = next(g for g in range(env.model.ngeom)
                              if env.model.geom_bodyid[g] == wrist_body and env.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                              and env.model.mesh(env.model.geom_dataid[g]).name == "sts3215_03a_no_horn_v1")
        self.wrist_body = wrist_body
        self.ztab = -env.task["robot_so101"]["base_z_offset"]
        self.ball_xy = env.task["ball"]["start_region"]["center_task_xy"]
        self._cache = {}

    # --- geometry -------------------------------------------------------------------------
    def landmark_points(self, joint_offsets_deg, horn_d, roll_off):
        key = tuple(float(v) for v in joint_offsets_deg)
        if key not in self._cache:
            anchors, axes, roll_pos, roll_rot = [], [], [], []
            for r in self.rows:
                q = r["state"].copy()
                q[:4] += joint_offsets_deg
                pose_render(self.env, q, self.ball_xy)
                d = self.env.data
                anchors.append({n: d.xanchor[self.jid[n]] - self.env.offset for n in HORN})
                axes.append({n: d.xaxis[self.jid[n]].copy() for n in HORN})
                roll_pos.append(d.geom_xpos[self.roll_geom] - self.env.offset)
                roll_rot.append(d.xmat[self.wrist_body].reshape(3, 3).copy())
            self._cache = {key: (anchors, axes, roll_pos, roll_rot)}
        anchors, axes, roll_pos, roll_rot = self._cache[key]
        pts = []
        for i, r in enumerate(self.rows):
            if r["joint"] == "wrist_roll_servo":
                pts.append(roll_pos[i] + roll_rot[i] @ roll_off)
            else:
                k = HORN.index(r["joint"])
                pts.append(anchors[i][r["joint"]] + horn_d[k] * axes[i][r["joint"]])
        return np.asarray(pts)

    def mat_points(self, cx, cy, yaw):
        sx, sy = self.mat["size_m_operator"]
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.array([[-sx / 2, sy / 2], [sx / 2, sy / 2], [sx / 2, -sy / 2], [-sx / 2, -sy / 2]])
        xy = local @ np.array([[c, s], [-s, c]]) + [cx, cy]
        return np.column_stack([xy, np.full(4, self.ztab)])

    @staticmethod
    def project(p3, f, rvec, pos):
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        R = cv2.Rodrigues(np.asarray(rvec, float))[0]
        tvec = -R @ np.asarray(pos, float)
        return cv2.projectPoints(np.asarray(p3, float).reshape(-1, 1, 3), np.asarray(rvec, float), tvec, K, None)[0].reshape(-1, 2)

    # --- parameter packing ------------------------------------------------------------------
    def unpack(self, x, fit_camera, fit_joints, cam0):
        i = 0
        if fit_camera:
            f, rvec, pos = x[0], x[1:4], x[4:7]
            i = 7
        else:
            f, rvec, pos = cam0
        horn_d, roll_off = x[i:i + 3], x[i + 3:i + 6]
        cx, cy, yaw = x[i + 6], x[i + 7], x[i + 8]
        i += 9
        joint_off = x[i:i + 4] if fit_joints else np.zeros(4)
        return f, rvec, pos, horn_d, roll_off, (cx, cy, yaw), joint_off

    def residuals(self, x, fit_camera, fit_joints, cam0, train_only=True):
        f, rvec, pos, horn_d, roll_off, mat_c, joint_off = self.unpack(x, fit_camera, fit_joints, cam0)
        pts = self.landmark_points(joint_off, horn_d, roll_off)
        obs = np.array([r["px"] for r in self.rows], float)
        sig = np.array([r["sigma_px"] for r in self.rows], float)
        res = (self.project(pts, f, rvec, pos) - obs) / sig[:, None]
        if train_only:
            res = res[self.train]
        mat_res = (self.project(self.mat_points(*mat_c), f, rvec, pos) - np.asarray(self.mat["px"], float)) / self.mat["sigma_px"]
        parts = [res.ravel(), mat_res.ravel()]
        if fit_camera:
            parts.append((pos[1:] - self.cam_prior["mean"][1:]) / self.cam_prior["sigma"][1:])  # operator y/z prior, x free
            parts.append(np.r_[0.0])
        if fit_joints:
            parts.append(joint_off / 10.0)  # weak prior: offsets of order 10 degrees are plausible, larger are penalised
        return np.concatenate(parts)

    def score(self, x, fit_camera, fit_joints, cam0):
        f, rvec, pos, horn_d, roll_off, mat_c, joint_off = self.unpack(x, fit_camera, fit_joints, cam0)
        pts = self.landmark_points(joint_off, horn_d, roll_off)
        obs = np.array([r["px"] for r in self.rows], float)
        err = np.linalg.norm(self.project(pts, f, rvec, pos) - obs, axis=1)
        mat_err = np.linalg.norm(self.project(self.mat_points(*mat_c), f, rvec, pos) - np.asarray(self.mat["px"], float), axis=1)
        out = {"train_rms_px": float(np.sqrt(np.mean(err[self.train] ** 2))),
               "val_rms_px": float(np.sqrt(np.mean(err[~self.train] ** 2))),
               "train_median_px": float(np.median(err[self.train])), "val_median_px": float(np.median(err[~self.train])),
               "mat_rms_px": float(np.sqrt(np.mean(mat_err ** 2))), "per_row_error_px": err.tolist()}
        for j in HORN + ["wrist_roll_servo"]:
            m = np.array([r["joint"] == j for r in self.rows])
            out[f"val_rms_px_{j}"] = float(np.sqrt(np.mean(err[m & ~self.train] ** 2)))
            out[f"train_rms_px_{j}"] = float(np.sqrt(np.mean(err[m & self.train] ** 2)))
        R = cv2.Rodrigues(np.asarray(rvec, float))[0]
        fwd = R.T @ [0, 0, 1.0]
        out["camera"] = {"focal_px": float(f), "fovy_deg": float(2 * np.degrees(np.arctan(H / 2 / f))),
                         "position_task": np.asarray(pos, float).tolist(), "rvec": np.asarray(rvec, float).tolist(),
                         "xyaxes_task": [*(R.T @ [1, 0, 0.0]).tolist(), *(-(R.T @ [0, 1, 0.0])).tolist()],
                         "look_at_task": (np.asarray(pos, float) + fwd * 0.8).tolist(),
                         "pitch_down_deg": float(np.degrees(np.arcsin(-fwd[2])))}
        out["horn_axis_offsets_m"] = np.asarray(horn_d, float).tolist()
        out["roll_servo_offset_wrist_frame_m"] = np.asarray(roll_off, float).tolist()
        out["mat_center_task_xy"] = [float(mat_c[0]), float(mat_c[1])]
        out["mat_yaw_deg"] = float(np.degrees(mat_c[2]))
        out["joint_zero_offsets_deg"] = dict(zip(OFFSET_JOINTS, np.asarray(joint_off, float).tolist(), strict=True))
        return out


def main(out):
    if out.exists() and any(p.name != "landmarks.json" for p in out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads((out / "landmarks.json").read_text())
    arrays = np.load(".cache/derived_joint/arrays.npz")
    env = So101PickPlaceEnv("results/metrics/geometry_3988577/configs/sim_so101.yaml", render_images=False)
    try:
        env.reset(seed=0)
        rv, tv, K, cam_pos, _ = camera_parameters(env)
        cam0 = (float(K[0, 0]), np.asarray(rv, float), np.asarray(cam_pos, float))
        rows = load_rows(env, landmarks, arrays)
        train_eps = {0, 8, 12, 16, 20, 28, 36}
        val_eps = sorted({r["episode"] for r in rows} - train_eps)
        task_cam = env.task["camera_iphone"]
        prior = {"mean": np.asarray(task_cam["approx_position_task"], float),
                 "sigma": np.array([1.0, 0.03 * 24 * IN, 0.03 * 21 * IN])}
        model = Model(env, rows, landmarks["mat_corners_px"], train_eps, prior)
        report = {"kind": "DIAGNOSTIC_CAMERA_LANDMARK_REFIT_NOT_ADOPTED",
                  "n_rows": len(rows), "n_train_rows": int(model.train.sum()), "n_val_rows": int((~model.train).sum()),
                  "train_episodes": sorted(train_eps), "val_episodes": val_eps,
                  "camera_prior": {"mean_task": prior["mean"].tolist(), "sigma_m": prior["sigma"].tolist(),
                                   "note": "operator 24 in / 21 in +-3 percent on y and z; x unconstrained"},
                  "models": {}}
        x_nuis0 = np.r_[np.zeros(3), np.zeros(3), env.task["table"]["mat"]["center_task_xy"], 0.0]

        def run(name, fit_camera, fit_joints, starts):
            best = None
            for x0 in starts:
                r = least_squares(model.residuals, x0, args=(fit_camera, fit_joints, cam0), method="trf", loss="soft_l1", f_scale=2.0, max_nfev=4000, diff_step=1e-4)
                if best is None or r.cost < best.cost:
                    best = r
            s = model.score(best.x, fit_camera, fit_joints, cam0)
            s["cost"] = float(best.cost)
            s["n_params"] = int(len(best.x))
            report["models"][name] = s
            print(f"{name}: train {s['train_rms_px']:.1f} px  val {s['val_rms_px']:.1f} px (median {s['val_median_px']:.1f})  mat {s['mat_rms_px']:.1f}  "
                  f"f={s['camera']['focal_px']:.0f} pos={np.round(s['camera']['position_task'],3).tolist()} joints={ {k: round(v,1) for k,v in s['joint_zero_offsets_deg'].items()} }")
            return best.x

        # M0: current camera, only nuisance offsets and mat centre.
        x0 = run("M0_current_camera_nuisance_only", False, False, [x_nuis0])
        # M0 without any nuisance: pure reprojection through the current camera (reference).
        s = model.score(np.r_[np.zeros(6), env.task["table"]["mat"]["center_task_xy"], 0.0], False, False, cam0)
        report["models"]["M0_current_camera_raw"] = {k: v for k, v in s.items() if k != "per_row_error_px"} | {"per_row_error_px": s["per_row_error_px"]}
        print(f"M0 raw: train {s['train_rms_px']:.1f} val {s['val_rms_px']:.1f}")
        # M1: refit camera.
        cam_starts = []
        for f0 in (cam0[0], 1300.0, 2300.0):
            for dx in (0.0, -0.15, 0.15):
                cam_starts.append(np.r_[f0, cam0[1], cam0[2] + [dx, 0, 0], x0])
        x1 = run("M1_refit_camera", True, False, cam_starts)
        # M2: refit camera + joint zero offsets.
        x2 = run("M2_refit_camera_and_joint_offsets", True, True, [np.r_[x1, np.zeros(4)], np.r_[x1, 0, -5, 0, 0], np.r_[x1, 0, 5, 0, 0]])
        # M2b: joint offsets with the current camera fixed (does the camera alone explain it?).
        run("M2b_current_camera_joint_offsets", False, True, [np.r_[x0, np.zeros(4)]])
        report["rows"] = [{k: v for k, v in r.items() if k != "state"} | {"state_deg": r["state"].tolist(), "split": "train" if r["episode"] in train_eps else "val"} for r in rows]
        report["caveats"] = [
            "Hand-read landmarks (~15-30 px); horn-boss along-axis offsets, the roll-servo offset and the mat centre/yaw are nuisance fits, not measurements.",
            "Joint zero offsets are fitted jointly with the camera and are only as trustworthy as the held-out validation improvement.",
            "Image/state timing was mitigated by choosing low-velocity ticks but not corrected.",
            "Validation episodes are the approved held-out split; they were never used in fitting here, but the earlier (superseded) episode-5 landmarks were training-only.",
            "No config, dataset, label or checkpoint was changed by this script.",
        ]
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/audit/camera_landmark_refit_20261014"))
    main(p.parse_args().out)
