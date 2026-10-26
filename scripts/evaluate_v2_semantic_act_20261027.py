"""Freeze the SO-trained semantic ACT and declare robot-only execution variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy

import run_v2_semantic_act_20261027 as experiment


def run(source, out, seeds, *, margin=.01, parked_perception=True):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    checkpoint = (source / "checkpoint").resolve()
    (out / "checkpoint").symlink_to(checkpoint, target_is_directory=True)
    torch.set_num_threads(2)
    policy = ACTPolicy.from_pretrained(checkpoint, strict=True, local_files_only=True).eval()
    stats = json.loads((source / "normalization.json").read_text())
    points = np.load(source / "semantic_data.npz")["state"]
    experiment.SEEDS = seeds
    dependencies = [Path(__file__), Path(experiment.__file__), source / "normalization.json",
        source / "semantic_data.npz", checkpoint / "model.safetensors", checkpoint / "config.json",
        experiment.SO_SCENE, experiment.PANDA_SCENE]
    before = {str(p): experiment.digest(p) for p in dependencies}
    report = {"schema": "semantic_act_frozen_execution_20261027_v1", "status": "running",
        "source_training_report": str(source / "report.json"), "seeds": seeds,
        "checkpoint_sha256": experiment.digest(checkpoint / "model.safetensors"),
        "training_robot": "so101", "panda_training_examples": 0, "no_retraining": True,
        "joint_margin_rad": margin, "panda_parked_RGB_then_mapped_start": parked_perception,
        "physical_calibration_accepted": False, "source_sha256_before": before}
    experiment.save(out / "report.json", report)
    report["so101"] = experiment.evaluate(policy, stats, out, "so101", points[0, :3], joint_margin=margin)
    experiment.save(out / "report.json", report)
    report["panda"] = experiment.evaluate(policy, stats, out, "panda", points[0, :3],
        joint_margin=margin, parked_perception=parked_perception)
    after = {str(p): experiment.digest(p) for p in dependencies}
    report.update(status="complete", source_sha256_after=after,
        runtime_source_changes=[p for p in before if before[p] != after[p]],
        checkpoint_unchanged=before[str(checkpoint / "model.safetensors")] == after[str(checkpoint / "model.safetensors")])
    experiment.save(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/v2/semantic_act_transfer_20261027_v2"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[300, 301, 302, 303, 304])
    parser.add_argument("--joint-margin", type=float, default=.01)
    parser.add_argument("--mapped-perception", action="store_true")
    args = parser.parse_args()
    run(args.source, args.out, args.seeds, margin=args.joint_margin,
        parked_perception=not args.mapped_perception)
