"""DIAGNOSTIC: score appearance variants of the sim scene by (a) ResNet backbone feature distance between the
sim render and the real first frame of the same episode and (b) the policy's predicted wrist at the end of the
chunk (real frame ~ +70 = hold; -10..-80 = return to rest). Baseline: real frame of another episode."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

policy, norm, contract = load_checkpoint("outputs/act_joint_so101/checkpoints/020000", device="cpu")
key = contract["camera_keys"][0]
eps = [5, 6, 0]
ds = DerivedTickDataset(".cache/derived_joint", eps, policy.config.chunk_size, normalizer=norm)
arr = ds.arrays
ep = arr["episode_index"]
fr = arr["frame_index"]
joints = arr["joint_state"] if "joint_state" in arr else arr["state"]
m, s = torch.as_tensor(norm.img_mean), torch.as_tensor(norm.img_std)
backbone = policy.model.backbone


def feat(img01):
    with torch.no_grad():
        out = backbone(((img01 - m) / s)[None])
        out = list(out.values())[-1] if isinstance(out, dict) else out
        return F.adaptive_avg_pool2d(out, 1).flatten()


def wflex_end(img01, st):
    with torch.no_grad():
        return float(
            norm.unaction(
                policy.predict_action_chunk(
                    {"observation.state": st[None], key: ((img01 - m) / s)[None]}
                )
            )[0, -1, 3]
        )


firsts = {}
for e in eps:
    t = np.flatnonzero(ep == e)
    f = t[np.argmin(fr[t])]
    it = ds[int(np.flatnonzero(ds.indices == f)[0])]
    firsts[e] = ((it[key] * s + m).clamp(0, 1), it["observation.state"], f)
bp = json.load(open("results/audit/camera_fit/backprojection.json"))["estimates"]["start"]
ball_xy = dict(zip([a for a, _ in bp["episodes_frames"]], bp["points_xy"], strict=False))
base_cfg = yaml.safe_load(open("configs/sim_so101.yaml"))
tmp = Path("outputs/tmp_appearance")
tmp.mkdir(parents=True, exist_ok=True)
variants = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {"current": {}}


def deep(a, b):
    a = copy.deepcopy(a)
    for k, v in b.items():
        a[k] = deep(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


print(f"{'variant':28s} {'feat_dist(mean)':>15s} {'wflex_end(mean)':>15s}   per-episode")
# baselines
d = [
    float(torch.norm(feat(firsts[e][0]) - feat(firsts[o][0]))) for e, o in ((5, 6), (6, 0), (0, 5))
]
w = [wflex_end(firsts[e][0], firsts[e][1]) for e in eps]
print(
    f"{'REAL other-episode baseline':28s} {np.mean(d):15.2f} {np.mean(w):15.1f}   {np.round(d, 1).tolist()} {np.round(w, 0).tolist()}"
)
out_rows = []
for name, ov in variants.items():
    cfg = (
        deep(base_cfg, {"appearance": ov})
        if ov is not None
        else {k: v for k, v in base_cfg.items() if k != "appearance"}
    )
    path = tmp / f"{name}.yaml"
    yaml.safe_dump(cfg, open(path, "w"))
    env = So101PickPlaceEnv(path, render_images=True)
    ds_, ws = [], []
    for e in eps:
        real, st, f = firsts[e]
        q = np.clip(sim_from_real(env.mapping, joints[f]), env.limits[:, 0], env.limits[:, 1])
        env.reset(seed=0, scenario={"joint_positions": q, "ball_xy": ball_xy[e]})
        sim = torch.from_numpy(env.observation()[key]).float()
        ds_.append(float(torch.norm(feat(sim) - feat(real))))
        ws.append(wflex_end(sim, st))
        if e == 5:
            import cv2

            pair = np.concatenate(
                [real.numpy().transpose(1, 2, 0), sim.numpy().transpose(1, 2, 0)], 1
            )
            cv2.imwrite(
                str(tmp / f"{name}_ep5.jpg"),
                cv2.cvtColor((pair * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
    env.close()
    print(
        f"{name:28s} {np.mean(ds_):15.2f} {np.mean(ws):15.1f}   {np.round(ds_, 1).tolist()} {np.round(ws, 0).tolist()}"
    )
    out_rows.append({"variant": name, "feat_dist": ds_, "wflex_end": ws, "overrides": ov})
json.dump(out_rows, open(tmp / "grid_results.json", "w"), indent=1)
