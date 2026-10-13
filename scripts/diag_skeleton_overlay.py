"""DIAGNOSTIC: project the simulated joint skeleton (current mapping) into real frames through the camera in task.yaml-equivalent json."""

import json
import sys

import cv2
import mujoco
import numpy as np

from real2sim.data.raw import RawDataset
from real2sim.sim.camera_fit import decode_frame
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

cam = json.load(open(sys.argv[2]))
K = np.array(cam["K"])
rvec = np.array(cam["rvec"])
tvec = np.array(cam["tvec"])
ds = RawDataset(sys.argv[1])
env = So101PickPlaceEnv("configs/sim_so101.yaml", render_images=False)
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
jid = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
tiles = []
for e, frames in ((5, (0, 150, 240, 330)), (0, (0,)), (6, (0,))):
    rows = ds.episode_rows(e)
    S = ds.stack(rows, "observation.state")
    seg = ds.video_segment("observation.images.iphone", e)
    for f in frames:
        img = decode_frame(seg, f, ds.fps)
        env.reset(
            seed=0,
            scenario={
                "joint_positions": np.clip(
                    sim_from_real(env.mapping, S[f]), env.limits[:, 0], env.limits[:, 1]
                )
            },
        )
        sk = np.array(
            [env.data.xanchor[j] - env.offset for j in jid]
            + [env.data.site_xpos[site] - env.offset]
        )
        px, _ = cv2.projectPoints(sk, rvec, tvec, K, None)
        px = px.reshape(-1, 2)
        for i in range(len(px) - 1):
            cv2.line(img, tuple(px[i].astype(int)), tuple(px[i + 1].astype(int)), (255, 0, 0), 4)
        for p, n in zip(px, ["pan", "lift", "elbow", "wrist", "roll", "tip"], strict=False):
            cv2.circle(img, tuple(p.astype(int)), 9, (0, 0, 255), -1)
            cv2.putText(
                img,
                n,
                tuple((p + [10, -10]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        cv2.putText(
            img,
            f"ep{e} f{f} state {np.round(S[f][:4]).astype(int).tolist()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        tiles.append(cv2.resize(img, (960, 540)))
env.close()
grid = np.concatenate([np.concatenate(tiles[i : i + 2], 1) for i in (0, 2, 4)], 0)
cv2.imwrite(sys.argv[3], cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print("ok")
