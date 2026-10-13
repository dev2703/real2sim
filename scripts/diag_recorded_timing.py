"""Descriptive raw-recording lag audit; never shifts or rewrites training labels."""

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.data.raw import RawDataset


def audit():
    ds = RawDataset(Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v1")
    split = json.loads(Path(".cache/derived_joint/manifest.json").read_text())["split"]
    report = {"lag_frames": list(range(-15, 31)), "positive_means_action_precedes_state": True,
              "note": "Same reference ticks for every lag; moving if either current state or target "
                      "changed >0.2 units. Descriptive alignment only; no label shifts or dynamics "
                      "fitting. Uniform dataset timestamps do not establish wall-clock motor timing.",
              "splits": {}}
    for name in ("train", "val"):
        errors = [[] for _ in report["lag_frames"]]
        for ep in split[name]:
            rows = ds.episode_rows(ep)
            s, a = ds.stack(rows, "observation.state"), ds.stack(rows, "action")
            ref = np.arange(16, len(s) - 30)
            moving = (abs(s[ref] - s[ref - 1]) > .2) | (abs(a[ref] - a[ref - 1]) > .2)
            for ix, lag in enumerate(report["lag_frames"]):
                errors[ix].append(np.where(moving, abs(a[ref] - s[ref + lag]), np.nan))
        mae = np.array([np.nanmean(np.concatenate(v), axis=0) for v in errors])
        ix = np.argmin(mae, axis=0)
        report["splits"][name] = {"moving_mae_by_lag": mae.tolist(),
            "best_lag_frames_descriptive": np.array(report["lag_frames"])[ix].tolist(),
            "best_mae": mae[ix, np.arange(6)].tolist()}
    best = report["splits"]["train"]["best_lag_frames_descriptive"]
    report["validation_at_train_selected_lag"] = [
        report["splits"]["val"]["moving_mae_by_lag"][report["lag_frames"].index(lag)][j]
        for j, lag in enumerate(best)]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
