"""Vision domain-gap diagnostic: policy predictions on REAL frames vs SIM renders of the same states.

    uv run python -m real2sim.evaluation.domain_gap --checkpoint <dir> --data .cache/derived_joint \
        --episode 1 --out results/metrics/domain_gap_joint.json

For every policy tick of a held-out episode the recorded state is applied to the simulator
(provisional joint mapping), the iphone camera is rendered and letterboxed exactly like the
training transform, and the frozen policy predicts an action chunk from (a) the real frame and
(b) the sim render, with the same recorded state. The first predicted action is compared to the
recorded label in raw units. A large (b)-(a) gap isolates rendering/viewpoint mismatch from
everything else. Labelled diagnostic; no closed loop, no success claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.sim.env import So101PickPlaceEnv


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--config", default="configs/sim_so101.yaml")
    p.add_argument("--stride", type=int, default=5, help="evaluate every Nth tick")
    p.add_argument("--out", required=True)
    p.add_argument("--save-pairs", type=int, default=3, help="save N real/sim image pairs next to --out")
    a = p.parse_args(argv)

    policy, norm, contract = load_checkpoint(a.checkpoint, device="cpu")
    manifest = json.loads((Path(a.data) / "manifest.json").read_text())
    if a.episode not in manifest["split"]["val"]:
        raise SystemExit(f"episode {a.episode} is not a validation episode: {manifest['split']['val']}")
    ds = DerivedTickDataset(a.data, [a.episode], policy.config.chunk_size, normalizer=norm)
    key = contract["camera_keys"][0]
    env = So101PickPlaceEnv(a.config, render_images=True)
    env.reset(seed=0)
    arrays = ds.arrays
    joint_state = arrays["joint_state"] if "joint_state" in arrays else arrays["state"]
    is_joint = len(contract["state_names"]) == 6
    img_mean = torch.as_tensor(norm.img_mean)
    img_std = torch.as_tensor(norm.img_std)

    rows = []
    pairs_saved = 0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(ds), a.stride):
        item = ds[i]
        t = int(ds.indices[i])
        q_real = np.asarray(joint_state[t], np.float64)
        # put the simulator at the recorded joint state and render the policy view
        q_sim = env.mapping.arm_real_to_sim_rad(q_real)
        g_sim = env.mapping.gripper_real_to_sim_rad(q_real[5])
        q6 = np.clip(np.r_[q_sim, g_sim], env.limits[:, 0], env.limits[:, 1])  # rest pose exceeds the MJCF lift limit
        env.reset(seed=0, scenario={"joint_positions": q6})
        obs = env.observation()
        sim_img = torch.from_numpy(obs[key]).float()  # [3,H,W] in [0,1]
        sim_img = (sim_img - img_mean) / img_std
        batch_real = {"observation.state": item["observation.state"][None], key: item[key][None]}
        batch_sim = {"observation.state": item["observation.state"][None], key: sim_img[None]}
        with torch.no_grad():
            pr = norm.unaction(policy.predict_action_chunk(batch_real))[0, 0].numpy()
            ps = norm.unaction(policy.predict_action_chunk(batch_sim))[0, 0].numpy()
        label = norm.unaction(item["action"][0]).numpy()
        rows.append({"tick": t, "err_real": np.abs(pr - label).tolist(), "err_sim": np.abs(ps - label).tolist(),
                     "pred_gap": np.abs(ps - pr).tolist()})
        if pairs_saved < a.save_pairs and i % (a.stride * 20) == 0:
            import cv2

            real = (item[key] * torch.as_tensor(norm.img_std) + torch.as_tensor(norm.img_mean)).clamp(0, 1).numpy()
            pair = np.concatenate([real.transpose(1, 2, 0), obs[key].transpose(1, 2, 0)], axis=1)
            cv2.imwrite(str(out.with_suffix("")) + f"_pair_t{t}.jpg", cv2.cvtColor((pair * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            pairs_saved += 1
    env.close()
    er = np.array([r["err_real"] for r in rows])
    es = np.array([r["err_sim"] for r in rows])
    gap = np.array([r["pred_gap"] for r in rows])
    report = {
        "checkpoint": a.checkpoint, "episode": a.episode, "n_ticks": len(rows), "stride": a.stride,
        "contract": "joint" if is_joint else "cartesian", "action_names": contract["action_names"],
        "note": "Open-loop first-action error vs recorded label, raw units; real frame vs sim render of the same recorded state. Diagnostic only.",
        "mean_abs_err_real_frame": er.mean(0).round(4).tolist(),
        "mean_abs_err_sim_render": es.mean(0).round(4).tolist(),
        "mean_abs_err_real_frame_overall": float(er.mean()), "mean_abs_err_sim_render_overall": float(es.mean()),
        "mean_abs_prediction_gap_sim_vs_real": gap.mean(0).round(4).tolist(),
        "error_ratio_sim_over_real": float(es.mean() / max(er.mean(), 1e-9)),
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("contract", "n_ticks", "mean_abs_err_real_frame_overall", "mean_abs_err_sim_render_overall", "error_ratio_sim_over_real", "mean_abs_err_real_frame", "mean_abs_err_sim_render")}, indent=1), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
