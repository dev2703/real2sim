"""Run a bounded, matched ball-input pilot with frozen provisional FK labels.

The disabled control has identical parameter shapes and initialization, and four
constant-zero extra inputs. This is a developmental test, not corrected-label
training or reserved evaluation. No paid compute or remote upload is used.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml
from prepare_v2_visual_grounding import prepare_grounding

from real2sim.config import load_yaml
from real2sim.data.raw import sha256_file
from real2sim.policy.train import train
from real2sim.policy.visual_grounding import BallImageFeatures


def run(source: Path, out: Path, data_out: Path, steps: int, device: str):
    if not 1 <= steps <= 512:
        raise ValueError("pilot budget must be 1..512 updates; larger training requires a new experiment")
    if out.exists() or data_out.exists():
        raise FileExistsError("use new output/data directories for an immutable experiment")
    torch.set_num_threads(2)
    out.mkdir(parents=True)
    data_out.mkdir(parents=True)
    # Same upstream architecture and fixed pretrained visual backbone as the old screen.
    base = load_yaml(Path("outputs/v2/learning_screen_20261022/configs/L1_resnet_iphone.yaml"))
    base["scheduler"].update(warmup_steps=max(1, steps // 8))
    base["train"].update(steps=steps, schedule_steps=steps, val_every=steps,
                         save_every=steps, log_every=32, device=device,
                         val_max_batches=None, keep_only_best_and_last=True)
    base["experiment"] = {"kind": "matched_ball_image_input_pilot_v1",
                           "physical_calibration_accepted": False,
                           "target_status": "unchanged provisional FK labels",
                           "promotion": "bounded development rollouts required before reserved evaluation"}
    report = {"kind": "MATCHED_BALL_IMAGE_INPUT_PILOT", "steps": steps,
              "examples_per_model": steps * 16, "training_seed": 0,
              "physical_calibration_accepted": False, "reserved_episodes_used": [],
              "labels_regenerated": False, "models": {},
              "script_sha256": sha256_file(Path(__file__)),
              "limitations": "One seed and small pilot budget on unchanged provisional FK. A detector input tests explicit image grounding; matching a colour contour is not localization ground truth or calibrated 3D geometry. Offline improvement cannot promote a task checkpoint."}
    for name, enabled in (("disabled_control", False), ("ball_input", True)):
        data = data_out / name
        prep = prepare_grounding(source, data, BallImageFeatures(enabled=enabled))
        cfg = copy.deepcopy(base)
        cfg.update(name=name, data=str(data.resolve()))
        cfg_path = out / f"{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        result = train(cfg, data, out / name)
        report["models"][name] = {"preparation": prep, "training": result,
                                  "config_sha256": sha256_file(cfg_path)}
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    print(json.dumps(run(args.source, args.out, args.data_out, args.steps, args.device), indent=2))
