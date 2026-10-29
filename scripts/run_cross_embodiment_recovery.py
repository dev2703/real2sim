"""Collect, train and compare recovery heads on a local frozen ACT checkpoint.

The default fits ACT-conditioned and no-ACT heads to the same SO-101/Panda
collections. --train-robots so101 tests source-only recovery transfer to the
requested --robots. Development evaluation is separate from held-out action
validation; reserved benchmark seeds never select a head in this workflow.
--plan-only prints commands without loading a model or creating output files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / ".cache/hf_models_20261027/models--bottowork--so101-act-h256-b16/snapshots/e696c8a078935432e3501e2c28c1fb2598fbff7e/latest"
SOURCE_SCENE = ROOT / "configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml"
RESERVED_SEEDS = set(range(13000, 13100))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    result.add_argument("--source-scene", type=Path, default=SOURCE_SCENE)
    result.add_argument("--robots", nargs="+", choices=("so101", "panda"), default=["so101", "panda"],
                        help="Embodiments used for evaluation")
    result.add_argument("--train-robots", nargs="+", choices=("so101", "panda"),
                        help="Embodiments supplying recovery training/validation; defaults to --robots")
    result.add_argument("--recovery-frame", choices=("native", "source"), default="native",
                        help="Native recovery observations/actions, or virtual SO-101 coordinates for source-only transfer")
    result.add_argument("--collections", nargs="+", type=Path,
                        help="Reuse completed collection directories instead of collecting fresh demonstrations")
    result.add_argument("--train-seeds", nargs="+", type=int, default=list(range(10000, 10004)))
    result.add_argument("--val-seeds", nargs="+", type=int, default=list(range(11000, 11002)))
    result.add_argument("--seeds", nargs="+", type=int, default=list(range(12000, 12004)),
                        help="Development rollout seeds; cannot overlap fitted episodes or reserved seeds")
    result.add_argument("--head-type", choices=("phase", "residual"), default="phase")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--use-act", dest="use_act", action="store_true", default=True)
    mode.add_argument("--no-act", dest="use_act", action="store_false",
                      help="Fit only an observable-history head with ACT inputs masked")
    result.add_argument("--skip-no-act-ablation", action="store_true",
                        help="Fit only the ACT-conditioned head")
    result.add_argument("--steps", type=int, default=4000)
    result.add_argument("--seed", type=int, default=42, help="Training random seed")
    result.add_argument("--temporal-weight", type=float, default=0.)
    result.add_argument("--require-command-labels", action="store_true",
                        help="Also require issued-command phase label provenance for reused collections")
    result.add_argument("--matched-benchmark", action="store_true",
                        help="Evaluate raw ACT, RGB, ACT+RGB and fitted heads on identical resets")
    result.add_argument("--video", action="store_true", help="Save learned rollouts (without --matched-benchmark)")
    result.add_argument("--plan-only", action="store_true")
    return result


def local_path(path):
    path = path.expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def build_plan(args):
    """Construct an inspectable command graph; perform no model imports or writes."""
    out, snapshot, source_scene = map(local_path, (args.out, args.snapshot, args.source_scene))
    train_robots = args.train_robots or args.robots
    if len(set(args.robots)) != len(args.robots) or len(set(train_robots)) != len(train_robots):
        raise ValueError("Embodiments must be unique")
    if args.recovery_frame == "source" and train_robots != ["so101"]:
        raise ValueError("Source recovery frame requires --train-robots so101")
    if args.steps < 1 or not math.isfinite(args.temporal_weight) or args.temporal_weight < 0:
        raise ValueError("Positive training steps and nonnegative temporal weight required")
    if args.head_type != "phase" and (args.temporal_weight or args.require_command_labels):
        raise ValueError("Temporal and command-label options require a phase head")
    if not args.use_act and args.skip_no_act_ablation:
        raise ValueError("--skip-no-act-ablation requires --use-act")
    if args.matched_benchmark and args.video:
        raise ValueError("--video is supported by standalone learned evaluation")
    requested_splits = {name: list(getattr(args, name)) for name in ("train_seeds", "val_seeds", "seeds")}
    for name, seeds in requested_splits.items():
        if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError(f"{name} must contain unique nonnegative seeds")
        if set(seeds) & RESERVED_SEEDS:
            raise ValueError("Reserved seeds 13000–13099 cannot be used for fitting or development selection")
    if set(args.train_seeds) & set(args.val_seeds):
        raise ValueError("Training and validation seeds must be disjoint")
    stages = []

    def add(name, script, options, output, artifact):
        stages.append({"name": name, "command": [sys.executable, str(ROOT / "scripts" / script), *map(str, options)],
                       "output": str(output), "report": str(output / artifact)})

    collection_inputs = []
    fitted_episodes = set()
    if args.collections:
        directories = [local_path(directory) for directory in args.collections]
        if len(set(directories)) != len(directories):
            raise ValueError("Collection directories must be unique")
        covered = set()
        for directory in directories:
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get("status") != "complete" or manifest.get("runtime_source_changes")
                    or manifest.get("base_weights_frozen") is not True
                    or manifest.get("lighting_changed") is not False):
                raise ValueError(f"Incomplete or changed collection: {directory}")
            robot = manifest.get("robot")
            if robot not in train_robots:
                raise ValueError(f"Collection embodiment {robot!r} is outside --train-robots")
            covered.add(robot)
            for partition in ("train", "val"):
                if not (directory / f"{partition}.npz").is_file():
                    raise ValueError(f"Missing {partition}.npz in {directory}")
                fitted_episodes.update(manifest.get("split", {}).get(partition, []))
            collection_inputs.append({"directory": str(directory),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
        if covered != set(train_robots):
            raise ValueError("Supply collection packs for every requested training embodiment")
    else:
        directories = [out / "collections" / robot for robot in train_robots]
        fitted_episodes.update(args.train_seeds)
        fitted_episodes.update(args.val_seeds)
        for robot, directory in zip(train_robots, directories, strict=True):
            add(f"collect_{robot}", "run_recovery_dagger.py",
                ["collect", "--robot", robot, "--out", directory,
                 "--snapshot", snapshot, "--source-scene", source_scene,
                 "--train-seeds", *args.train_seeds, "--val-seeds", *args.val_seeds,
                 "--reference-mode", "tracked", "--expert-execution", "common"], directory, "manifest.json")
    if fitted_episodes & (set(args.seeds) | RESERVED_SEEDS):
        raise ValueError("Development/reserved episodes overlap collection training or validation")
    trainer = "train_phase_recovery_mlp.py" if args.head_type == "phase" else "train_recovery_mlp.py"
    variants = (["learned"] if args.skip_no_act_ablation else ["learned", "no_act"]) if args.use_act else ["no_act"]
    heads = {}
    for variant in variants:
        head = out / "heads" / variant
        heads[variant] = str(head)
        options = ["--data", *directories, "--out", head, "--snapshot", snapshot,
                   "--steps", args.steps, "--seed", args.seed,
                   "--use-act" if variant == "learned" else "--no-act"]
        if args.head_type == "phase":
            options.extend(["--temporal-weight", args.temporal_weight])
            if not args.collections or args.require_command_labels:
                options.append("--require-command-labels")
        add(f"train_{variant}", trainer, options, head, "report.json")
    if args.matched_benchmark:
        directory = out / "benchmark"
        add("matched_development_benchmark", "benchmark_frozen_act.py",
            ["--out", directory, "--robots", *args.robots,
             "--conditions", "raw", "rgb", "hf_rgb", *variants,
             "--seeds", *args.seeds, "--snapshot", snapshot, "--source-scene", source_scene,
             "--controller-factory", "scripts.run_recovery_dagger:controller_factory",
             "--controller-options", json.dumps({"heads": heads, "recovery_frame": args.recovery_frame},
                                                 sort_keys=True)], directory, "report.json")
    else:
        for robot in args.robots:
            for variant, head in heads.items():
                directory = out / "evaluation" / robot / variant
                options = ["evaluate", "--robot", robot, "--head", head, "--out", directory,
                           "--snapshot", snapshot, "--source-scene", source_scene,
                           "--recovery-frame", args.recovery_frame, "--seeds", *args.seeds]
                if args.video:
                    options.append("--video")
                add(f"evaluate_{robot}_{variant}", "run_recovery_dagger.py", options, directory, "report.json")
    return {"schema": "cross_embodiment_recovery_workflow_v1", "output": str(out),
            "snapshot": str(snapshot), "source_scene": str(source_scene), "robots": args.robots,
            "training_robots": list(train_robots),
            "zero_shot_target_robots": [robot for robot in args.robots if robot not in train_robots],
            "recovery_frame": args.recovery_frame,
            "head_type": args.head_type, "heads": heads, "collection_inputs": collection_inputs,
            "development_seeds": list(args.seeds), "reserved_seeds_used": False,
            "checkpoint_selection": "Held-out collection action validation only; rollouts are development evidence",
            "base_weights_frozen": True, "network_access": "offline local checkpoint only", "stages": stages}


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_plan(plan, *, runner=None):
    """Run stages sequentially and retain failed/incomplete stage evidence."""
    runner = subprocess.run if runner is None else runner
    out = Path(plan["output"])
    if out.exists():
        raise FileExistsError(f"Use a new workflow output directory: {out}")
    snapshot = Path(plan["snapshot"])
    for required in (snapshot / "model.safetensors", snapshot / "config.json", Path(plan["source_scene"])):
        if not required.is_file():
            raise FileNotFoundError(required)
    for collection in plan["collection_inputs"]:
        path = Path(collection["directory"]) / "manifest.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != collection["manifest_sha256"]:
            raise RuntimeError(f"Collection manifest changed after planning: {path}")
    out.mkdir(parents=True)
    report = {**plan, "status": "running", "stage_results": []}
    report_path = out / "workflow.json"
    write_report(report_path, report)
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    try:
        for stage in plan["stages"]:
            result = {"name": stage["name"], "status": "running"}
            report["stage_results"].append(result)
            write_report(report_path, report)
            runner(stage["command"], cwd=ROOT, env=environment, check=True)
            artifact = json.loads(Path(stage["report"]).read_text())
            if artifact.get("status") != "complete" or artifact.get("runtime_source_changes"):
                raise RuntimeError(f"Stage {stage['name']} did not complete with unchanged runtime inputs")
            if artifact.get("checkpoint_unchanged") is False:
                raise RuntimeError(f"Stage {stage['name']} changed frozen ACT weights")
            if stage["name"].startswith("train_"):
                # Trainer reports derive these lists from actual NPZ rows. A
                # manifest alone cannot prove development/target exclusion.
                keys = ("train_episodes", "validation_episodes", "training_robot_ids")
                if any(not isinstance(artifact.get(key), list) for key in keys):
                    raise RuntimeError(f"Stage {stage['name']} lacks fitted-episode/embodiment provenance")
                fitted = set(artifact["train_episodes"]) | set(artifact["validation_episodes"])
                if fitted & (set(plan["development_seeds"]) | RESERVED_SEEDS):
                    raise RuntimeError("Actual fitted episodes overlap development or reserved seeds")
                expected_ids = {int(robot == "panda") for robot in plan["training_robots"]}
                if set(artifact["training_robot_ids"]) != expected_ids:
                    raise RuntimeError("Actual training embodiments differ from --train-robots")
                result.update({key: artifact[key] for key in keys})
            result.update(status="complete", report=stage["report"])
            if "summary" in artifact:
                result["summary"] = artifact["summary"]
            if "conditions" in artifact:
                result["conditions"] = artifact["conditions"]
            write_report(report_path, report)
        report["status"] = "complete"
        write_report(report_path, report)
        return report
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if report["stage_results"]:
            report["stage_results"][-1]["status"] = "failed"
        write_report(report_path, report)
        raise


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        plan = build_plan(args)
    except (ValueError, OSError) as exc:
        cli.error(str(exc))
    if args.plan_only:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return plan
    return run_plan(plan)


if __name__ == "__main__":
    main()
