"""DIAGNOSTIC: first-frame comparison after the camera refit. Ball positions are re-estimated per episode by
back-projecting the detected ball through the refit camera onto the ball-centre plane."""

import json
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from real2sim.data.raw import RawDataset
from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.sim.camera_fit import decode_frame
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

root = sys.argv[1]
eps = [5, 6, 0, 7]
cam = json.load(open("outputs/tmp_appearance/camera_measured.json"))
K = np.array(cam["K"])
Rm, _ = cv2.Rodrigues(np.array(cam["rvec"]))
t = np.array(cam["tvec"])
task = yaml.safe_load(open("configs/task.yaml"))
ztab = -task["robot_so101"]["base_z_offset"]
r = task["ball"]["radius"]


def ball_xy_from_frame(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array([22, 70, 90]), np.array([50, 255, 255]))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    u, v = cen[k]
    ray_c = np.linalg.inv(K) @ np.array([u, v, 1.0])
    ray_w = Rm.T @ ray_c
    o = -Rm.T @ t
    s = (ztab + r - o[2]) / ray_w[2]
    p = o + s * ray_w
    return p[:2].tolist(), (u, v), int(st[k, cv2.CC_STAT_AREA])


policy, norm, contract = load_checkpoint("outputs/act_joint_so101/checkpoints/020000", device="cpu")
key = contract["camera_keys"][0]
dsd = DerivedTickDataset(".cache/derived_joint", eps, policy.config.chunk_size, normalizer=norm)
arr = dsd.arrays
ep = arr["episode_index"]
fr = arr["frame_index"]
joints = arr["joint_state"] if "joint_state" in arr else arr["state"]
mn, sd = torch.as_tensor(norm.img_mean), torch.as_tensor(norm.img_std)
bb = policy.model.backbone


def feat(img01):
    with torch.no_grad():
        out = bb(((img01 - mn) / sd)[None])
        out = list(out.values())[-1] if isinstance(out, dict) else out
        return F.adaptive_avg_pool2d(out, 1).flatten()


def chunk(img01, st):
    with torch.no_grad():
        return norm.unaction(
            policy.predict_action_chunk(
                {"observation.state": st[None], key: ((img01 - mn) / sd)[None]}
            )
        )[0].numpy()


raw = RawDataset(root)
env = So101PickPlaceEnv("configs/sim_so101.yaml", render_images=True)
tiles = []
feats = {}
for e in eps:
    tk = np.flatnonzero(ep == e)
    f0 = tk[np.argmin(fr[tk])]
    it = dsd[int(np.flatnonzero(dsd.indices == f0)[0])]
    real = (it[key] * sd + mn).clamp(0, 1)
    st = it["observation.state"]
    full = decode_frame(raw.video_segment(key, e), 0, raw.fps)
    bxy, (u, v), area = ball_xy_from_frame(full)
    q = np.clip(sim_from_real(env.mapping, joints[f0]), env.limits[:, 0], env.limits[:, 1])
    env.reset(seed=0, scenario={"joint_positions": q, "ball_xy": bxy})
    sim = torch.from_numpy(env.observation()[key]).float()
    feats[e] = (feat(real), feat(sim))
    cr, cs = chunk(real, st), chunk(sim, st)
    print(
        f"ep{e}: ball px ({u:.0f},{v:.0f}) area {area} -> task xy {np.round(bxy, 3).tolist()} | feat dist real-sim {float(torch.norm(feats[e][0] - feats[e][1])):.2f} | wflex real {cr[0, 3]:.0f}->{cr[-1, 3]:.0f}  sim {cs[0, 3]:.0f}->{cs[-1, 3]:.0f} | lift end real {cr[-1, 1]:.0f} sim {cs[-1, 1]:.0f}"
    )
    tiles.append(
        np.concatenate([real.numpy().transpose(1, 2, 0), sim.numpy().transpose(1, 2, 0)], 1)
    )
print(
    "real-real baseline feat dist:",
    round(float(torch.norm(feats[5][0] - feats[6][0])), 2),
    round(float(torch.norm(feats[0][0] - feats[7][0])), 2),
)
cv2.imwrite(
    "outputs/tmp_appearance/after_camera_refit.jpg",
    cv2.cvtColor((np.concatenate(tiles, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
)
env.close()
