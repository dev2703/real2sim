"""DIAGNOSTIC: what does the joint ACT policy predict at the START of an episode from
(A) the real frame, (B) the sim render of the same pose with the ball at its estimated position,
(C) the sim render with no ball in view, (D) the real END-of-episode frame, (E) the real frame
with the simulator's clamped state. Same normaliser and policy; chunk printed in raw degrees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real

ck = sys.argv[1] if len(sys.argv) > 1 else "outputs/act_joint_so101/checkpoints/020000"
out = Path("results/metrics/debug_so101_joint/first_frame_gap")
out.mkdir(parents=True, exist_ok=True)
policy, norm, contract = load_checkpoint(ck, device="cpu")
key = contract["camera_keys"][0]
bp = json.load(open("results/audit/camera_fit/backprojection.json"))["estimates"]["start"]
ball_xy = {e: xy for (e, _), xy in zip(bp["episodes_frames"], bp["points_xy"], strict=False)}
episodes = [0, 5, 6, 7, 1, 3]  # 4 train (ball estimate available), 2 val
ds = DerivedTickDataset(".cache/derived_joint", episodes, policy.config.chunk_size, normalizer=norm)
arr = ds.arrays
ep_idx = arr["episode_index"]
fr = arr["frame_index"]
joints = arr["joint_state"] if "joint_state" in arr else arr["state"]
img_mean, img_std = torch.as_tensor(norm.img_mean), torch.as_tensor(norm.img_std)
env = So101PickPlaceEnv("configs/sim_so101.yaml", render_images=True)


def sim_image(q_real, ball):
    q6 = np.clip(sim_from_real(env.mapping, q_real), env.limits[:, 0], env.limits[:, 1])
    sc = {"joint_positions": q6}
    if ball is not None:
        sc["ball_xy"] = ball
    env.reset(seed=0, scenario=sc)
    o = env.observation()[key]
    return o, ((torch.from_numpy(o).float() - img_mean) / img_std)


def chunk(img_n, state_n):
    with torch.no_grad():
        c = norm.unaction(
            policy.predict_action_chunk({"observation.state": state_n[None], key: img_n[None]})
        )[0].numpy()
    return c


def summ(c):
    return f"lift {c[0, 1]:6.1f}->{c[-1, 1]:6.1f}  elbow {c[0, 2]:5.1f}->{c[-1, 2]:5.1f}  wflex {c[0, 3]:6.1f}->{c[-1, 3]:6.1f}  pan {c[0, 0]:5.1f}->{c[-1, 0]:5.1f}"


rows = []
for e in episodes:
    ticks = np.flatnonzero(ep_idx == e)
    first, last = ticks[np.argmin(fr[ticks])], ticks[np.argmax(fr[ticks])]
    i_first = int(np.flatnonzero(ds.indices == first)[0])
    i_last = int(np.flatnonzero(ds.indices == last)[0])
    item, item_last = ds[i_first], ds[i_last]
    q_real = np.asarray(joints[first], np.float64)
    label = norm.unaction(item["action"]).numpy()
    ball = ball_xy.get(e)
    sim_o, sim_n = sim_image(q_real, ball if ball is not None else None)
    nob_o, nob_n = sim_image(q_real, [0.55, 0.42])
    sim_state_real = env.real_state() if False else None
    env.reset(
        seed=0,
        scenario={
            "joint_positions": np.clip(
                sim_from_real(env.mapping, q_real), env.limits[:, 0], env.limits[:, 1]
            )
        },
    )
    sim_state_n = torch.from_numpy(norm.state(env.real_state()).astype(np.float32))
    conds = {
        "A_real_first_frame": (item[key], item["observation.state"]),
        "B_sim_render_ball_est": (sim_n, item["observation.state"]),
        "C_sim_render_no_ball": (nob_n, item["observation.state"]),
        "D_real_END_frame": (item_last[key], item["observation.state"]),
        "E_real_first_frame_sim_state": (item[key], sim_state_n),
    }
    print(
        f"\n=== episode {e} ({'train' if ball is not None else 'val'}) first tick {first}; recorded state {np.round(q_real, 1)}; label chunk: {summ(label)}"
    )
    for name, (im, st) in conds.items():
        c = chunk(im, st)
        print(f"  {name:30s} {summ(c)}")
        rows.append(
            {
                "episode": e,
                "cond": name,
                "chunk_first": c[0].round(2).tolist(),
                "chunk_last": c[-1].round(2).tolist(),
                "label_first": label[0].round(2).tolist(),
                "label_last": label[-1].round(2).tolist(),
            }
        )
    real = (item[key] * img_std + img_mean).clamp(0, 1).numpy().transpose(1, 2, 0)
    last_img = (item_last[key] * img_std + img_mean).clamp(0, 1).numpy().transpose(1, 2, 0)
    pair = np.concatenate(
        [real, sim_o.transpose(1, 2, 0), nob_o.transpose(1, 2, 0), last_img], axis=1
    )
    cv2.imwrite(
        str(out / f"ep{e}_real_sim_noball_end.jpg"),
        cv2.cvtColor((pair * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )
env.close()
(out / "report.json").write_text(
    json.dumps({"checkpoint": ck, "note": __doc__, "rows": rows}, indent=2)
)
