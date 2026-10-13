"""DIAGNOSTIC: where do the backbone features of the sim render differ from the real first frame, and does the
difference survive low-pass filtering? Spatial feature-difference heatmaps are written next to the grid results."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

policy, norm, contract = load_checkpoint("outputs/act_joint_so101/checkpoints/020000", device="cpu")
key = contract["camera_keys"][0]
eps = [5, 6]
ds = DerivedTickDataset(".cache/derived_joint", eps, policy.config.chunk_size, normalizer=norm)
arr = ds.arrays
ep = arr["episode_index"]
fr = arr["frame_index"]
joints = arr["joint_state"] if "joint_state" in arr else arr["state"]
m, s = torch.as_tensor(norm.img_mean), torch.as_tensor(norm.img_std)
bb = policy.model.backbone


def fmap(img01):
    with torch.no_grad():
        out = bb(((img01 - m) / s)[None])
        out = list(out.values())[-1] if isinstance(out, dict) else out
        return out[0]  # [C,h,w]


def lowpass(img, k):
    if k == 1:
        return img
    a = img.numpy().transpose(1, 2, 0)
    h, w = a.shape[:2]
    small = cv2.resize(a, (w // k, h // k), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(
        cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    )


env = So101PickPlaceEnv("configs/sim_so101.yaml", render_images=True)
bp = json.load(open("results/audit/camera_fit/backprojection.json"))["estimates"]["start"]
ball_xy = dict(zip([a for a, _ in bp["episodes_frames"]], bp["points_xy"], strict=False))
out = Path("outputs/tmp_appearance")
imgs = {}
for e in eps:
    t = np.flatnonzero(ep == e)
    f = t[np.argmin(fr[t])]
    it = ds[int(np.flatnonzero(ds.indices == f)[0])]
    real = (it[key] * s + m).clamp(0, 1)
    q = np.clip(sim_from_real(env.mapping, joints[f]), env.limits[:, 0], env.limits[:, 1])
    env.reset(seed=0, scenario={"joint_positions": q, "ball_xy": ball_xy[e]})
    imgs[e] = (real, torch.from_numpy(env.observation()[key]).float())
env.close()
print("low-pass factor | feat dist real5-vs-sim5 | real5-vs-real6 (baseline) | ratio")
for k in (1, 2, 4, 8, 16):
    r5, s5 = (lowpass(x, k) for x in imgs[5])
    r6 = lowpass(imgs[6][0], k)
    d_sim = float(
        torch.norm(F.adaptive_avg_pool2d(fmap(r5), 1) - F.adaptive_avg_pool2d(fmap(s5), 1))
    )
    d_real = float(
        torch.norm(F.adaptive_avg_pool2d(fmap(r5), 1) - F.adaptive_avg_pool2d(fmap(r6), 1))
    )
    print(f"{k:15d} | {d_sim:23.2f} | {d_real:25.2f} | {d_sim / d_real:5.2f}")
# spatial heatmap of feature difference (k=1)
real, sim = imgs[5]
fr_, fs_ = fmap(real), fmap(sim)
fr6 = fmap(imgs[6][0])
diff = torch.norm(fr_ - fs_, dim=0)
base = torch.norm(fr_ - fr6, dim=0)
print(
    "feature map",
    tuple(fr_.shape),
    f" per-cell diff sim: mean {diff.mean():.1f} max {diff.max():.1f} | real-real: mean {base.mean():.1f} max {base.max():.1f}",
)
h, w = real.shape[1:]


def heat(d):
    d = (d / max(float(diff.max()), 1e-6)).numpy()
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1] / 255.0


panel = np.concatenate(
    [real.numpy().transpose(1, 2, 0), sim.numpy().transpose(1, 2, 0), heat(diff), heat(base)], 1
)
cv2.imwrite(
    str(out / "feature_heatmap_ep5.jpg"),
    cv2.cvtColor((panel * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
)
# cell grid of the diff (rows x cols) so the location can be read numerically
np.set_printoptions(linewidth=200, precision=0, suppress=True)
print("sim-vs-real per-cell diff grid (top=image top):\n", diff.numpy())
print("real-vs-real grid:\n", base.numpy())
