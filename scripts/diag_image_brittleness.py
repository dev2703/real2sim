"""DIAGNOSTIC: how sensitive is the joint ACT policy to mild perturbations of a REAL first frame,
compared with the sim render? Reports the predicted wrist trajectory (hold ~ +70; rest ~ -40..-80)
and the ResNet backbone feature distance to the unperturbed real frame. Real-vs-real distance
between two episodes' first frames is the natural baseline.
"""

import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

policy, norm, contract = load_checkpoint("outputs/act_joint_so101/checkpoints/020000", device="cpu")
key = contract["camera_keys"][0]
eps = [5, 6, 7, 0]
ds = DerivedTickDataset(".cache/derived_joint", eps, policy.config.chunk_size, normalizer=norm)
arr = ds.arrays
ep = arr["episode_index"]
fr = arr["frame_index"]
joints = arr["joint_state"] if "joint_state" in arr else arr["state"]
m, s = torch.as_tensor(norm.img_mean), torch.as_tensor(norm.img_std)


def first_item(e):
    t = np.flatnonzero(ep == e)
    f = t[np.argmin(fr[t])]
    return ds[int(np.flatnonzero(ds.indices == f)[0])], f


backbone = None
for name in ("backbone",):
    backbone = getattr(getattr(policy, "model", policy), name, None)


def feat(img01):  # [3,H,W] in [0,1]
    x = ((img01 - m) / s)[None]
    with torch.no_grad():
        out = backbone(x)
        if isinstance(out, dict):
            out = list(out.values())[-1]
        return F.adaptive_avg_pool2d(out, 1).flatten()


def wflex(img01, state_n):
    with torch.no_grad():
        c = norm.unaction(
            policy.predict_action_chunk(
                {"observation.state": state_n[None], key: ((img01 - m) / s)[None]}
            )
        )[0].numpy()
    return c[0, 3], c[-1, 3], c[-1, 1]


def perturb(img, kind):
    x = img.clone()
    if kind == "bright+20%":
        return (x * 1.2).clamp(0, 1)
    if kind == "bright-20%":
        return x * 0.8
    if kind == "warm_tint":
        x[0] = (x[0] * 1.1).clamp(0, 1)
        x[2] = x[2] * 0.9
        return x
    if kind == "blur3":
        a = x.numpy().transpose(1, 2, 0)
        return torch.from_numpy(cv2.GaussianBlur(a, (0, 0), 1.5).transpose(2, 0, 1))
    if kind == "shift8px":
        return torch.roll(x, shifts=(0, 8), dims=(1, 2))
    if kind == "shift_down6px":
        return torch.roll(x, shifts=(6, 0), dims=(1, 2))
    if kind == "gray":
        g = x.mean(0, keepdim=True)
        return g.expand_as(x).clone()
    if kind == "noise_sd0.03":
        return (x + 0.03 * torch.randn_like(x)).clamp(0, 1)
    if kind == "contrast0.7":
        return ((x - 0.5) * 0.7 + 0.5).clamp(0, 1)
    if kind == "flip_lr":
        return torch.flip(x, dims=[2])
    raise KeyError(kind)


env = So101PickPlaceEnv("configs/sim_so101.yaml", render_images=True)
bp = json.load(open("results/audit/camera_fit/backprojection.json"))["estimates"]["start"]
ball_xy = dict(zip([a for a, _ in bp["episodes_frames"]], bp["points_xy"], strict=False))
rows = []
torch.manual_seed(0)
for e in eps[:2]:
    item, f = first_item(e)
    real = (item[key] * s + m).clamp(0, 1)
    st = item["observation.state"]
    f0 = feat(real)
    other, _ = first_item(eps[2] if e != eps[2] else eps[3])
    real_other = (other[key] * s + m).clamp(0, 1)
    q = np.clip(sim_from_real(env.mapping, joints[f]), env.limits[:, 0], env.limits[:, 1])
    env.reset(seed=0, scenario={"joint_positions": q, "ball_xy": ball_xy[e]})
    sim = torch.from_numpy(env.observation()[key]).float()
    print(
        f"\n=== episode {e}  (wflex first->last of chunk; lift last; feature L2 distance to real frame)"
    )
    cases = [("real", real), ("real_other_episode", real_other), ("sim_render", sim)] + [
        (k, perturb(real, k))
        for k in (
            "bright+20%",
            "bright-20%",
            "warm_tint",
            "contrast0.7",
            "blur3",
            "noise_sd0.03",
            "shift8px",
            "shift_down6px",
            "gray",
            "flip_lr",
        )
    ]
    for name, img in cases:
        w0, w1, l1 = wflex(img, st)
        d = float(torch.norm(feat(img) - f0))
        print(
            f"  {name:20s} wflex {w0:6.1f} -> {w1:6.1f}   lift_end {l1:7.1f}   feat_dist {d:6.2f}"
        )
        rows.append(
            {
                "episode": e,
                "case": name,
                "wflex_first": round(float(w0), 1),
                "wflex_last": round(float(w1), 1),
                "feat_dist": round(d, 3),
            }
        )
env.close()
json.dump(
    rows, open("results/metrics/debug_so101_joint/first_frame_gap/brittleness.json", "w"), indent=1
)
