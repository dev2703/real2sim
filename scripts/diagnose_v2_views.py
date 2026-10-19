"""R5 per-camera domain replacement and R4 controlled image-delay sensitivity.

Uses recorded canonical state throughout. Delays here affect images only; this is
explicitly not a measurement of acquisition or actuator latency.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from audit_v2_policy import metrics

from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.paired import PairedViewDataset, file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 4])
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.out.exists():
        raise FileExistsError(a.out)
    if any(d < 0 for d in a.delays):
        raise ValueError("image delays must be nonnegative")
    policy, norm, contract = load_checkpoint(a.checkpoint, a.device)
    m = json.loads((a.data / "manifest.json").read_text())
    if contract["derived_files_sha256"] != m["files_sha256"]:
        raise ValueError("checkpoint data differs")
    ds = DerivedTickDataset(a.data, m["split"]["val"], policy.config.chunk_size, norm)
    pairs = PairedViewDataset(ds, a.pairs, 1, evaluation=True)
    if m["camera_keys"] != contract["camera_keys"]:
        raise ValueError("camera order differs")
    cases = {
        "real": [],
        "both_sim": m["camera_keys"],
        **{f"{k}_sim": [k] for k in m["camera_keys"]},
    }
    pred = {name: [] for name in cases}
    pred.update({f"image_delay_{d}_ticks": [] for d in a.delays})
    labels = []
    ticks = []
    for tick in pairs.ticks:
        if ds.arrays["label_is_extrapolated"][tick]:
            continue
        state = torch.tensor(norm.state(ds.arrays["state"][tick]))[None].to(a.device)
        images = {k: np.asarray(ds.images[k][tick]) for k in m["camera_keys"]}
        simulated = {k: np.asarray(pairs.image_views[k][pairs.lookup[int(tick)]]) for k in images}
        variants = {
            name: {k: simulated[k] if k in replace else images[k] for k in images}
            for name, replace in cases.items()
        }
        for delay in a.delays:
            prior = max(int(ds.arrays["ep_start"][tick]), int(tick) - delay)
            variants[f"image_delay_{delay}_ticks"] = {
                k: np.asarray(ds.images[k][prior]) for k in images
            }
        for name, views in variants.items():
            batch = {
                "observation.state": state,
                **{k: torch.tensor(norm.image(img))[None].to(a.device) for k, img in views.items()},
            }
            kwargs = (
                {"sampling_seed": int(tick)}
                if getattr(policy.config, "type", None) == "small_chunk_head"
                else {}
            )
            with torch.inference_mode():
                action = (
                    norm.unaction(policy.predict_action_chunk(batch, **kwargs))[0, 0].cpu().numpy()
                )
            pred[name].append(action)
        labels.append(ds.arrays["action"][tick])
        ticks.append(int(tick))
    label = np.asarray(labels)
    pred = {k: np.asarray(v) for k, v in pred.items()}
    counterfactual = pairs.manifest.get("counterfactual")
    report = {
        "checkpoint_sha256": file_sha256(a.checkpoint / "model.safetensors"),
        "pairs_manifest_sha256": file_sha256(a.pairs / "manifest.json"),
        "ticks": ticks,
        "scope": "one-step teacher-forced validation observations; held canonical state; identical flow noise across view interventions",
        "counterfactual": counterfactual,
        "errors_against_real_actions": None
        if counterfactual
        else {
            k: metrics(v, label, action_kind=m["action_representation"]) for k, v in pred.items()
        },
        "mean_absolute_action_response_vs_real": {
            k: np.abs(v - pred["real"]).mean(axis=0).tolist() for k, v in pred.items()
        },
        "interpretation": "Object-response direction must be reviewed by phase (ball at approach; bowl at delivery). Response magnitude alone is not localization evidence. Image delays are not physical trace estimates.",
    }
    a.out.mkdir(parents=True)
    np.savez_compressed(a.out / "predictions.npz", ticks=ticks, label=label, **pred)
    (a.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
