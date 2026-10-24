"""Bounded deterministic capacity gate before further policy generalization work.

The three episodes are training data, one per condition. This measures fitting
under unchanged provisional FK, not geometry acceptance, generalization or success.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from real2sim.config import load_yaml
from real2sim.data.raw import sha256_file
from real2sim.policy.train import train


def gate_from_validation(validation: dict) -> dict:
    per_episode = validation["scores"]["model"]["per_episode"]
    rows = {episode: {"translation_mean_mm": score["translation_mean_mm"],
                      "gripper_mae_fraction": score["gripper_mae"],
                      "pass": score["translation_mean_mm"] < 1. and score["gripper_mae"] < .03}
            for episode, score in per_episode.items()}
    return {"pass": len(rows) == 3 and all(row["pass"] for row in rows.values()),
            "per_episode": rows, "thresholds": {"translation_mean_mm_below": 1., "gripper_mae_fraction_below": .03}}


def run(data: Path, out: Path, device: str):
    if out.exists():
        raise FileExistsError(out)
    base = load_yaml(Path("outputs/v2/learning_screen_20261022/configs/L1_resnet_iphone.yaml"))
    cfg = copy.deepcopy(base)
    cfg.update(name="three_episode_deterministic_capacity", data=str(data.resolve()))
    cfg["model"].update(use_vae=False, kl_weight=0., dropout=0., optimizer_weight_decay=0., optimizer_lr=3e-4)
    cfg["scheduler"].update(warmup_steps=50)
    cfg["train"].update(diagnostic_episodes=[11, 8, 23], steps=1000, schedule_steps=2000,
                         batch_size=8, grad_accum=2, log_every=100, val_every=250,
                         save_every=250, val_max_batches=None, device=device,
                         num_workers=0, exact_resume=True, keep_only_best_and_last=True)
    cfg["experiment"] = {"kind": "three_episode_capacity_gate_v1", "maximum_examples": 32000,
                          "conditions": {"11": "baseline", "8": "ball_shift", "23": "bowl_shift"},
                          "scope": "training-fit only; original provisional FK and current labels; no geometry or task-success claim",
                          "stopping": "check the final checkpoint at16000examples; continue identical optimizer/RNG to32000onlyif per-episode fitting gate fails"}
    torch.set_num_threads(2)
    out.mkdir(parents=True)
    (out / "plan.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "executed_runner.py").write_bytes(Path(__file__).read_bytes())
    dependencies = [Path(__file__), Path("src/real2sim/policy/train.py"),
                    Path("src/real2sim/policy/act.py"), Path("src/real2sim/policy/dataset.py"),
                    Path("src/real2sim/policy/validation.py"), data / "manifest.json", data / "stats.json"]
    report = {"kind": "THREE_TRAINING_EPISODE_CAPACITY_GATE", "physical_calibration_accepted": False,
              "checkpoint_promoted": False, "reserved_episodes_used": [], "validation_episodes_used": [],
              "training_episodes": [11, 8, 23], "source_sha256": {str(p): sha256_file(p) for p in dependencies},
              "stages": [], "limitations": "Training-fit diagnostic under provisional FK. Includes recorded references outside native bounds; their presence is not simulator feasibility or physical calibration. No validation or reserved performance is established."}
    for steps in (1000, 2000):
        cfg["train"]["steps"] = steps
        resume = None if steps == 1000 else (out / "training/checkpoints/001000")
        result = train(cfg, data, out / "training", resume=resume)
        ck = out / "training/checkpoints/last"
        info = json.loads((ck / "run_info.json").read_text())
        gate = gate_from_validation(info["last_val"]["executed_prefix"])
        report["stages"].append({"examples_seen": steps * 16, "result": result,
                                  "checkpoint": str(ck.resolve()), "checkpoint_sha256": sha256_file(ck / "model.safetensors"),
                                  "gate": gate})
        report.update(gate_passed=gate["pass"], total_examples_seen=steps * 16)
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"steps": steps, "gate": gate}), flush=True)
        if gate["pass"]:
            break
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run(args.data, args.out, args.device)
    print(json.dumps({"gate_passed": result["gate_passed"], "total_examples_seen": result["total_examples_seen"]}))
