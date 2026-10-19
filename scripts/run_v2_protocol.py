"""Local-only v2 preparation, bounded experiment planning, evidence checks and scoring.

Run `python scripts/run_v2_protocol.py --help`. No command uploads data or rents compute.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from run_so101_experiments import queue_lock

from real2sim.config import load_yaml
from real2sim.data.conditions import stratified_split
from real2sim.data.raw import RawDataset
from real2sim.experiments.v2 import (
    experiment_matrix,
    final_summary,
    freeze_bundle,
    gate_failures,
    read_json,
    verify_bundle,
)


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to replace versioned artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    split = sub.add_parser("split")
    split.add_argument("--annotations", type=Path, required=True)
    split.add_argument("--root", type=Path, default=Path.home()/".cache/huggingface/lerobot/local/so101_act_pickplace_v2")
    split.add_argument("--out", type=Path, default=Path("results/v2/split.json"))
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--base", type=Path, default=Path("configs/v2/act_base.yaml"))
    matrix.add_argument("--backbone", choices=("resnet18", "dinov2"), default="resnet18")
    matrix.add_argument("--effective-batch", type=int, default=16)
    matrix.add_argument("--physical-batch", type=int, default=8)
    matrix.add_argument("--head-lr", type=float, default=1e-4)
    matrix.add_argument("--out", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--file", action="append", required=True, help="name=path; include data, task, sim, eval, scenarios and evidence")
    freeze.add_argument("--out", type=Path, required=True)
    gate = sub.add_parser("check")
    gate.add_argument("--evidence", type=Path, required=True)
    gate.add_argument("--before-tiny-fit", action="store_true")
    run = sub.add_parser("train")
    run.add_argument("--matrix", type=Path, required=True)
    run.add_argument("--arm", required=True)
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--resume", type=Path)
    run.add_argument("--runtime-seconds", type=int, required=True)
    score = sub.add_parser("score-final")
    score.add_argument("--rows", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "split":
        ds = RawDataset(args.root.expanduser())
        write(args.out, stratified_split(read_json(args.annotations), ds.episode_indices))
    elif args.command == "matrix":
        arms = experiment_matrix(load_yaml(args.base), backbone=args.backbone, head_lr=args.head_lr,
                                 effective_batch=args.effective_batch, physical_batch=args.physical_batch)
        write(args.out, {"version": "v2_screen_v1", "selection": "development pick-place success; then closure error; then latency",
                         "adaptive_baseline": {"backbone": args.backbone, "batch": args.effective_batch, "head_lr": args.head_lr},
                         "arms": arms})
    elif args.command == "freeze":
        paths = dict(item.split("=", 1) for item in args.file)
        write(args.out, freeze_bundle(paths))
    elif args.command == "check":
        failures = gate_failures(read_json(args.evidence), include_tiny_fit=not args.before_tiny_fit)
        print(json.dumps({"ready": not failures, "missing_or_failed": failures}, indent=2))
        return int(bool(failures))
    elif args.command == "train":
        if args.runtime_seconds <= 0:
            raise ValueError("explicit positive local runtime limit required")
        bundle = read_json(args.bundle)
        verify_bundle(bundle)
        required = {"evidence", "data_manifest", "task", "sim", "eval", "scenarios", "matrix"}
        if not required <= bundle["files"].keys():
            raise ValueError(f"freeze all required inputs: {sorted(required)}")
        if Path(bundle["files"]["matrix"]["path"]).resolve() != args.matrix.resolve():
            raise ValueError("matrix must be the frozen matrix")
        failures = gate_failures(read_json(bundle["files"]["evidence"]["path"]))
        if failures:
            raise ValueError("learning comparison blocked: " + "; ".join(failures))
        arm = next(a for a in read_json(args.matrix)["arms"] if a["id"] == args.arm)
        if arm.get("reuse"):
            raise ValueError(f"reuse existing control {arm['reuse']}; do not charge another screen")
        if arm["family"] != "act":
            raise ValueError("use train_flow_head.py for the matched frozen-feature flow/regression arms")
        data = Path(arm["config"]["data"]).resolve()
        if (data/"manifest.json") != Path(bundle["files"]["data_manifest"]["path"]):
            raise ValueError("arm data differs from frozen data manifest")
        # Hold the same lock for every v2 training subprocess. Timeouts never retry.
        with queue_lock(Path("outputs/v2/training_lock")):
            verify_bundle(bundle)
            args.out.mkdir(parents=True, exist_ok=True)
            if (args.out/"training_result.json").exists():
                raise FileExistsError("completed run exists; reuse it")
            cfgpath = args.out/"input.yaml"
            if cfgpath.exists() and load_yaml(cfgpath) != arm["config"]:
                raise ValueError("existing run config differs")
            cfgpath.write_text(yaml.safe_dump(arm["config"], sort_keys=False))
            cmd = [sys.executable, "-m", "real2sim.policy.train", "--config", str(cfgpath), "--data", str(data), "--out", str(args.out)]
            if args.resume:
                cmd += ["--resume", str(args.resume)]
            with (args.out/"process.log").open("a") as log:
                try:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.runtime_seconds, check=False)
                    status = {"returncode": result.returncode, "completed": result.returncode == 0, "automatic_retry": False}
                except subprocess.TimeoutExpired:
                    status = {"completed": False, "timeout": True, "automatic_retry": False,
                              "resume": "latest fixed-exposure checkpoint; uncheckpointed work is lost"}
                (args.out/"process_status.json").write_text(json.dumps(status, indent=2)+"\n")
            verify_bundle(bundle)
            return int(not status["completed"])
    elif args.command == "score-final":
        write(args.out, final_summary(read_json(args.rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
