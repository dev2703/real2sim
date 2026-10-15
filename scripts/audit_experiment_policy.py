"""Score unique executed-prefix actions, motion/phase strata and causal baselines."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diag_motion_prediction_audit import metrics, phase_masks
from torch.utils.data import DataLoader, Subset

from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.paired import file_sha256


def audit(checkpoint, data, device, split="val"):
    policy, norm, contract = load_checkpoint(checkpoint, device)
    manifest = json.loads((data / "manifest.json").read_text())
    for key in ("state_names", "action_names", "split", "observation_variant"):
        if contract.get(key) != manifest.get(key):
            raise ValueError(f"contract mismatch: {key}")
    episodes = manifest["split"][split]
    dataset = DerivedTickDataset(data, episodes, policy.config.chunk_size, norm)
    arrays = dataset.arrays
    canonical = dict(arrays, state=arrays.get("canonical_state", arrays["state"]))
    anchors = [i for i, tick in enumerate(dataset.indices)
               if (tick - arrays["ep_start"][tick]) % 2 == 0]
    outputs, ticks, anchor_ticks = [], [], []
    loader = DataLoader(Subset(dataset, anchors), batch_size=16, shuffle=False)
    with torch.inference_mode():
        for batch in loader:
            obs = {k: v.to(device) for k, v in batch.items() if k.startswith("observation.")}
            pred = norm.unaction(policy.predict_action_chunk(obs)[:, :2]).cpu().numpy()
            valid = (~batch["action_is_pad"][:, :2]).numpy()
            grid = np.broadcast_to(batch["tick"].numpy()[:, None], valid.shape)
            outputs.append(pred[valid])
            ticks.extend((grid + np.arange(2)[None])[valid])
            anchor_ticks.extend(grid[valid])
    ticks, anchor_ticks = np.asarray(ticks), np.asarray(anchor_ticks)
    expected = np.flatnonzero(np.isin(arrays["episode_index"], episodes) & ~arrays["label_is_extrapolated"])
    if not np.array_equal(ticks, expected) or policy.config.n_action_steps != 2:
        raise ValueError("prefix coverage/cadence mismatch")
    pred = np.concatenate(outputs)
    if not np.isfinite(pred).all():
        raise FloatingPointError("nonfinite predictions")
    label = arrays["action"][ticks]
    zero = np.zeros_like(label)
    zero[:, -1] = canonical["state"][anchor_ticks, -1]
    persistence = zero.copy()
    previous = np.maximum(anchor_ticks - 1, arrays["ep_start"][anchor_ticks])
    available = anchor_ticks > arrays["ep_start"][anchor_ticks]
    persistence[available, :6] = arrays["action"][previous[available], :6]
    approach = []
    for section in ("train", "val"):
        path = Path(f".cache/paired_cartesian_refit_20261015/{section}/ticks.npy")
        if path.is_file():
            approach.extend(np.load(path).tolist())
    masks = phase_masks(canonical, ticks, np.asarray(approach))
    scores = {name: {phase: metrics(values[mask], label[mask], norm.action_std)
                     for phase, mask in masks.items()}
              for name, values in (("policy", pred), ("zero_motion_hold_gripper", zero),
                                   ("previous_motion_hold_gripper", persistence))}
    episode_scores = {}
    for ep in episodes:
        mask = arrays["episode_index"][ticks] == ep
        episode_scores[str(ep)] = {
            "all": metrics(pred[mask], label[mask], norm.action_std),
            "active": metrics(pred[mask & masks["arm_active"]], label[mask & masks["arm_active"]], norm.action_std),
            "summed_translation_error_m": (pred[mask, :3] - label[mask, :3]).sum(axis=0).tolist(),
        }
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
            "split": split, "episodes": episodes, "valid_actions": len(ticks), "scores": scores,
            "per_episode": episode_scores, "note": "Teacher-forced recorded observations; summed translation error is a drift proxy, not a simulated rollout. Phase names are kinematic proxies. Historical validation has been repeatedly reviewed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("val", "historical_val"), default="val")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = audit(args.checkpoint.resolve(), args.data, args.device, args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"valid_actions": result["valid_actions"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
