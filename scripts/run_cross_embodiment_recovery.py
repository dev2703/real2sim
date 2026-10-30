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
    result.add_argument("--history-mode", choices=("selected", "achieved"), default="selected",
                        help="Versioned previous-action observation contract used in all collection rounds")
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
    result.add_argument("--transition-weight", type=float, default=0.)
    result.add_argument("--action-routing", choices=("soft", "hard"), default="soft")
    result.add_argument("--history-noise", type=float, default=0.)
    result.add_argument("--history-dropout", type=float, default=0.)
    result.add_argument("--require-command-labels", action="store_true",
                        help="Also require issued-command phase label provenance for reused collections")
    result.add_argument("--dagger-iterations", type=int, default=0,
                        help="Collect learner-visited correction packs and refit after each round")
    result.add_argument("--dagger-beta", nargs="+", type=float,
                        help="Expert execution probability: one repeated value or one per DAgger round (default .5)")
    result.add_argument("--dagger-learner", choices=("learned", "no_act"),
                        help="Previous fitted variant used to visit states; both heads fit the same aggregate packs")
    result.add_argument("--dagger-train-seeds", nargs="+", type=int,
                        help="Training layouts revisited each DAgger round; defaults to --train-seeds")
    result.add_argument("--dagger-val-seeds", nargs="+", type=int,
                        help="Held-out layouts revisited each DAgger round; defaults to --val-seeds")
    result.add_argument("--matched-benchmark", action="store_true",
                        help="Evaluate raw ACT, RGB, ACT+RGB and fitted heads on identical resets")
    result.add_argument("--video", action="store_true", help="Save learned rollouts (without --matched-benchmark)")
    result.add_argument("--plan-only", action="store_true")
    return result


def local_path(path):
    path = path.expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    phase_values = (args.transition_weight, args.history_noise, args.history_dropout)
    if (not all(math.isfinite(value) and value >= 0 for value in phase_values)
            or args.history_dropout > 1):
        raise ValueError("Finite nonnegative phase options and history dropout in 0..1 required")
    if args.head_type != "phase" and (args.temporal_weight or args.require_command_labels
            or any(phase_values) or args.action_routing != "soft"):
        raise ValueError("Temporal and command-label options require a phase head")
    if not args.use_act and args.skip_no_act_ablation:
        raise ValueError("--skip-no-act-ablation requires --use-act")
    if args.matched_benchmark and args.video:
        raise ValueError("--video is supported by standalone learned evaluation")
    if args.dagger_iterations < 0:
        raise ValueError("DAgger iterations must be nonnegative")
    if not args.dagger_iterations and any(value is not None for value in (
            args.dagger_beta, args.dagger_learner, args.dagger_train_seeds, args.dagger_val_seeds)):
        raise ValueError("DAgger options require positive --dagger-iterations")
    requested_splits = {name: list(getattr(args, name)) for name in ("train_seeds", "val_seeds", "seeds")}
    if args.dagger_iterations:
        requested_splits.update(
            dagger_train_seeds=list(args.dagger_train_seeds or args.train_seeds),
            dagger_val_seeds=list(args.dagger_val_seeds or args.val_seeds))
    for name, seeds in requested_splits.items():
        if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError(f"{name} must contain unique nonnegative seeds")
        if set(seeds) & RESERVED_SEEDS:
            raise ValueError("Reserved seeds 13000–13099 cannot be used for fitting or development selection")
    if set(args.train_seeds) & set(args.val_seeds):
        raise ValueError("Training and validation seeds must be disjoint")
    betas = list(args.dagger_beta or [.5])
    if (not all(math.isfinite(beta) and 0 <= beta <= 1 for beta in betas)
            or (args.dagger_iterations and len(betas) not in (1, args.dagger_iterations))):
        raise ValueError("DAgger beta must be in 0..1, with one value or one per iteration")
    if args.dagger_iterations and len(betas) == 1:
        betas *= args.dagger_iterations
    stages = []

    def add(name, script, options, output, artifact, **metadata):
        stages.append({"name": name, "command": [sys.executable, str(ROOT / "scripts" / script), *map(str, options)],
                       "output": str(output), "report": str(output / artifact), **metadata})

    collection_inputs = []
    fitted_episodes = set()
    fitted_train_episodes, fitted_val_episodes = set(), set()
    reference_modes = set()
    observation_contracts = set()
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
            mode = manifest.get("reference_mode")
            observation_names = manifest.get("observation_names")
            inferred_history = ("achieved" if any("previous_achieved_" in name
                for name in (observation_names or [])) else "selected")
            history_mode = manifest.get("history_mode", inferred_history)
            if history_mode != args.history_mode:
                raise ValueError("Collection previous-action history contract differs from --history-mode")
            if observation_names and history_mode != inferred_history:
                raise ValueError("Collection history metadata differs from its observation names")
            if mode is None and observation_names:
                mode = "tracked" if any("fresh_rgb_" in name for name in observation_names) else "initial"
            if mode is not None:
                reference_modes.add(mode)
            if observation_names is not None:
                observation_contracts.add(tuple(observation_names))
            if args.dagger_iterations and (mode is None or mode not in ("initial", "tracked")):
                raise ValueError("DAgger reused collections require a declared RGB-reference contract")
            for partition in ("train", "val"):
                if not (directory / f"{partition}.npz").is_file():
                    raise ValueError(f"Missing {partition}.npz in {directory}")
                episodes = manifest.get("split", {}).get(partition, [])
                if args.dagger_iterations and not episodes:
                    raise ValueError("DAgger reused collections require declared train/validation episodes")
                fitted_episodes.update(episodes)
                (fitted_train_episodes if partition == "train" else fitted_val_episodes).update(episodes)
            collection_inputs.append({"directory": str(directory),
                "manifest_sha256": file_hash(manifest_path),
                "array_sha256": {partition: file_hash(directory / f"{partition}.npz")
                                  for partition in ("train", "val")}})
        if covered != set(train_robots):
            raise ValueError("Supply collection packs for every requested training embodiment")
    else:
        directories = [out / "collections" / robot for robot in train_robots]
        fitted_episodes.update(args.train_seeds)
        fitted_episodes.update(args.val_seeds)
        fitted_train_episodes.update(args.train_seeds)
        fitted_val_episodes.update(args.val_seeds)
        reference_modes.add("tracked")
        for robot, directory in zip(train_robots, directories, strict=True):
            add(f"collect_{robot}", "run_recovery_dagger.py",
                ["collect", "--robot", robot, "--out", directory,
                 "--snapshot", snapshot, "--source-scene", source_scene,
                 "--train-seeds", *args.train_seeds, "--val-seeds", *args.val_seeds,
                 "--history-mode", args.history_mode,
                 "--reference-mode", "tracked", "--expert-execution", "common"], directory, "manifest.json")
    if fitted_episodes & (set(args.seeds) | RESERVED_SEEDS):
        raise ValueError("Development/reserved episodes overlap collection training or validation")
    if fitted_train_episodes & fitted_val_episodes:
        raise ValueError("Training and validation episodes overlap across collection packs")
    if len(reference_modes) > 1 or len(observation_contracts) > 1:
        raise ValueError("Collection RGB-reference/observation contracts differ")
    reference_mode = next(iter(reference_modes), "tracked")
    if args.dagger_iterations:
        dagger_train = set(requested_splits["dagger_train_seeds"])
        dagger_val = set(requested_splits["dagger_val_seeds"])
        if ((dagger_train | fitted_train_episodes) & (dagger_val | fitted_val_episodes)
                or (dagger_train | dagger_val) & set(args.seeds)):
            raise ValueError("DAgger training, validation and development episode roles must stay disjoint")
    trainer = "train_phase_recovery_mlp.py" if args.head_type == "phase" else "train_recovery_mlp.py"
    variants = (["learned"] if args.skip_no_act_ablation else ["learned", "no_act"]) if args.use_act else ["no_act"]
    collecting_variant = args.dagger_learner or variants[0]
    if args.dagger_iterations and collecting_variant not in variants:
        raise ValueError("DAgger collecting learner must be one of the fitted head variants")
    head_history = []

    def fit_heads(iteration):
        parent = out if iteration == 0 else out / "iterations" / f"iteration_{iteration:04d}"
        current = {}
        for variant in variants:
            head = parent / "heads" / variant
            current[variant] = str(head)
            options = ["--data", *directories, "--out", head, "--snapshot", snapshot,
                       "--steps", args.steps, "--seed", args.seed,
                       "--use-act" if variant == "learned" else "--no-act"]
            if args.head_type == "phase":
                options.extend(["--temporal-weight", args.temporal_weight,
                    "--transition-weight", args.transition_weight, "--action-routing", args.action_routing,
                    "--history-noise", args.history_noise, "--history-dropout", args.history_dropout])
                if not args.collections or args.require_command_labels or args.dagger_iterations:
                    options.append("--require-command-labels")
            name = f"train_{variant}" if iteration == 0 else f"train_dagger_{iteration:04d}_{variant}"
            add(name, trainer, options, head, "report.json")
        head_history.append({"iteration": iteration, "heads": current,
                             "collections": list(map(str, directories))})
        return current

    heads = fit_heads(0)
    for iteration in range(1, args.dagger_iterations + 1):
        parent = out / "iterations" / f"iteration_{iteration:04d}"
        learner = heads[collecting_variant]
        additions = []
        for robot in train_robots:
            directory = parent / "collections" / robot
            additions.append(directory)
            add(f"collect_dagger_{iteration:04d}_{robot}", "run_recovery_dagger.py",
                ["collect", "--robot", robot, "--out", directory,
                 "--snapshot", snapshot, "--source-scene", source_scene,
                 "--head", learner, "--beta", betas[iteration - 1], "--iteration", iteration,
                 "--train-seeds", *requested_splits["dagger_train_seeds"],
                 "--val-seeds", *requested_splits["dagger_val_seeds"],
                 "--history-mode", args.history_mode,
                 "--reference-mode", reference_mode, "--expert-execution", "common"],
                 directory, "manifest.json", dagger_collection={"robot": robot,
                    "iteration": iteration, "learner": learner,
                    "beta_teacher_execution": betas[iteration - 1],
                    "reference_mode": reference_mode, "expert_execution": "common",
                    "history_mode": args.history_mode,
                    "train_seeds": requested_splits["dagger_train_seeds"],
                    "val_seeds": requested_splits["dagger_val_seeds"]})
        directories.extend(additions)
        heads = fit_heads(iteration)
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
            "history_mode": args.history_mode,
            "head_type": args.head_type, "heads": heads, "collection_inputs": collection_inputs,
            "head_history": head_history,
            "phase_training": {"action_routing": args.action_routing,
                "temporal_weight": args.temporal_weight, "transition_weight": args.transition_weight,
                "history_noise": args.history_noise, "history_dropout": args.history_dropout},
            "dagger": {"iterations": args.dagger_iterations,
                "expert_execution_probabilities": betas if args.dagger_iterations else [],
                "collecting_learner": collecting_variant if args.dagger_iterations else None,
                "reference_mode": reference_mode,
                "train_seeds": requested_splits.get("dagger_train_seeds", []),
                "validation_seeds": requested_splits.get("dagger_val_seeds", []),
                "same_layout_seeds_revisited_each_iteration": True,
                "both_head_variants_fit_same_aggregate_packs": True,
                "development_evaluation_after_final_fit_only": True},
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
        if file_hash(path) != collection["manifest_sha256"]:
            raise RuntimeError(f"Collection manifest changed after planning: {path}")
    pinned_collections = {}
    for collection in plan["collection_inputs"]:
        directory = Path(collection["directory"])
        pinned_collections[str(directory / "manifest.json")] = collection["manifest_sha256"]
        for partition, digest in collection.get("array_sha256", {}).items():
            pinned_collections[str(directory / f"{partition}.npz")] = digest

    def verify_collections():
        for filename, digest in pinned_collections.items():
            if file_hash(filename) != digest:
                raise RuntimeError(f"Immutable collection changed during workflow: {filename}")

    verify_collections()
    checkpoint_path = snapshot / "model.safetensors"
    checkpoint_hash = file_hash(checkpoint_path)
    out.mkdir(parents=True)
    report = {**plan, "status": "running", "base_checkpoint_sha256": checkpoint_hash,
              "stage_results": []}
    report_path = out / "workflow.json"
    write_report(report_path, report)
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    actual_train_episodes, actual_val_episodes = set(), set()
    try:
        for stage in plan["stages"]:
            result = {"name": stage["name"], "status": "running"}
            report["stage_results"].append(result)
            write_report(report_path, report)
            verify_collections()
            if file_hash(checkpoint_path) != checkpoint_hash:
                raise RuntimeError("Frozen ACT checkpoint changed between workflow stages")
            if "dagger_collection" in stage:
                expected = stage["dagger_collection"]
                if (set(expected["train_seeds"]) & actual_val_episodes
                        or set(expected["val_seeds"]) & actual_train_episodes):
                    raise RuntimeError("DAgger episode roles conflict with actual fitted episodes")
            runner(stage["command"], cwd=ROOT, env=environment, check=True)
            verify_collections()
            if file_hash(checkpoint_path) != checkpoint_hash:
                raise RuntimeError("Frozen ACT checkpoint changed during workflow stage")
            artifact = json.loads(Path(stage["report"]).read_text())
            if artifact.get("status") != "complete" or artifact.get("runtime_source_changes"):
                raise RuntimeError(f"Stage {stage['name']} did not complete with unchanged runtime inputs")
            if artifact.get("checkpoint_unchanged") is False:
                raise RuntimeError(f"Stage {stage['name']} changed frozen ACT weights")
            if "dagger_collection" in stage:
                expected = stage["dagger_collection"]
                exact_keys = ("robot", "iteration", "learner", "beta_teacher_execution",
                              "reference_mode", "expert_execution", "history_mode")
                if (any(artifact.get(key) != expected[key] for key in exact_keys)
                        or artifact.get("base_weights_frozen") is not True
                        or artifact.get("checkpoint_unchanged") is not True
                        or any(artifact.get("split", {}).get(partition) != expected[f"{partition}_seeds"]
                               for partition in ("train", "val"))):
                    raise RuntimeError(f"Stage {stage['name']} differs from planned DAgger collection")
                beta = expected["beta_teacher_execution"]
                execution = "expert_only" if beta == 1 else "learner_only" if beta == 0 else "expert_learner_mixture"
                result.update(collection_execution=execution, dagger_collection=expected)
            if stage["name"].startswith("collect_"):
                directory = Path(stage["output"])
                hashes = {str(directory / name): file_hash(directory / name)
                          for name in ("manifest.json", "train.npz", "val.npz")}
                pinned_collections.update(hashes)
                result["collection_artifact_sha256"] = hashes
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
                current_train = set(artifact["train_episodes"])
                current_val = set(artifact["validation_episodes"])
                if ((current_train | actual_train_episodes) & (current_val | actual_val_episodes)):
                    raise RuntimeError("Actual training/validation episode roles changed across fits")
                actual_train_episodes.update(current_train)
                actual_val_episodes.update(current_val)
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
