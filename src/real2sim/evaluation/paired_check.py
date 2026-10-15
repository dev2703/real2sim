"""Compare frozen parent and paired pilot on held-out real/static-sim approach images."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.paired import PairedViewDataset, file_sha256


def run(checkpoints, pairs, out, per_episode=16, data=Path(".cache/derived_joint")):
    if out.exists():
        raise FileExistsError(out)
    report = {"kind": "HELD_OUT_APPROACH_COUNTERFACTUAL_IMAGES", "checkpoints": {},
              "note": "Same held-out recorded state/labels for each real/sim image. No fitting. "
                      "Static render comparisons do not establish closed-loop success."}
    torch.set_num_threads(2)
    for name, path in checkpoints.items():
        policy, norm, contract = load_checkpoint(path, device="cpu")
        real = DerivedTickDataset(data, contract["split"]["val"], policy.config.chunk_size, norm)
        for key in ("state_names", "action_names", "camera_keys"):
            if contract[key] != real.manifest[key]:
                raise ValueError(f"paired check checkpoint/data {key} mismatch")
        if contract["derived_files_sha256"] != real.manifest["files_sha256"]:
            raise ValueError("paired check checkpoint/data hashes mismatch")
        sim = PairedViewDataset(real, pairs, probability=1, evaluation=True)
        ticks = sim.ticks
        chosen = []
        for ep in contract["split"]["val"]:
            ix = ticks[real.arrays["episode_index"][ticks] == ep]
            if len(ix):
                chosen.extend(ix[np.unique(np.linspace(0, len(ix) - 1, min(per_episode, len(ix))).astype(int))])
        lookup = {int(t): i for i, t in enumerate(real.indices)}
        rows, errors = [], {"real": [], "sim": [], "view_gap": []}
        for tick in chosen:
            a, b = real[lookup[int(tick)]], sim[lookup[int(tick)]]
            batch = {k: torch.stack([a[k], b[k]]) for k in ("observation.state", *contract["camera_keys"])}
            with torch.inference_mode():
                pred = policy.predict_action_chunk(batch).numpy()
            valid = (~a["action_is_pad"][:2]).numpy()
            label = a["action"][:2].numpy()[valid]
            errors["real"].append(abs(pred[0, :2][valid] - label))
            errors["sim"].append(abs(pred[1, :2][valid] - label))
            errors["view_gap"].append(abs(pred[1, :2][valid] - pred[0, :2][valid]))
            raw = norm.unaction(pred)
            rows.append({"tick": int(tick), "episode": a["episode_index"],
                         "frame": int(real.arrays["frame_index"][tick]),
                         "real_prefix_raw": raw[0, :2].tolist(), "sim_prefix_raw": raw[1, :2].tolist(),
                         "label_prefix_raw": norm.unaction(a["action"][:2]).tolist()})
        report["checkpoints"][name] = {"path": str(path), "sha256": file_sha256(Path(path) / "model.safetensors"),
            "samples": len(rows), "prefix_mae_normalized": {k: float(np.concatenate(v).mean()) for k, v in errors.items()},
            "prefix_mae_raw_per_dim": {k: (np.concatenate(v).mean(axis=0) * norm.action_std).tolist() for k, v in errors.items()},
            "rows": rows}
        del policy
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--parent", type=Path, default=Path("outputs/act_joint_so101/checkpoints/020000"))
    p.add_argument("--pairs", type=Path, default=Path(".cache/paired_approach_3988577/val"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path(".cache/derived_joint"))
    args = p.parse_args()
    run({"parent": args.parent, "pilot": args.checkpoint}, args.pairs, args.out, data=args.data)


if __name__ == "__main__":
    main()
