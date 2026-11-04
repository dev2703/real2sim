"""Bounded deterministic capacity gate before further policy generalization work.

The three episodes are training data, one per condition. This measures fitting
under unchanged provisional FK, not geometry acceptance, generalization or success.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import yaml

from real2sim.config import load_yaml
from real2sim.data.raw import sha256_file
from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.losses import ReconstructionLoss
from real2sim.policy.train import train
from real2sim.policy.validation import evaluate_executed_prefix, executed_prefix_report

EPISODES = (11, 8, 23)
GATE_VERSION = "three_episode_movement_and_drift_v2"


def gate_from_validation(validation: dict) -> dict:
    """Require every condition to fit and improve both causal baselines.

    Compare stationary *error* to the recorded label, not predicted step size:
    hold's predicted step is zero, so requiring a smaller step is impossible.
    Exact equality is allowed only at the zero-error floor.
    """
    rows = {}
    expected = {str(e) for e in EPISODES}
    scores = validation["scores"]
    valid_episodes = all(set(scores.get(name, {}).get("per_episode", {})) == expected
                         for name in ("model", "hold", "persistence"))
    for episode, score in scores["model"]["per_episode"].items():
        checks = {"translation_fit": math.isfinite(score["translation_mean_mm"]) and score["translation_mean_mm"] < 1.,
                  "gripper_fit": math.isfinite(score["gripper_mae"]) and score["gripper_mae"] < .03}
        comparisons = {}
        for baseline in ("hold", "persistence"):
            reference = scores.get(baseline, {}).get("per_episode", {}).get(episode, {})
            comparisons[baseline] = {}
            for group, metrics in (("movement", ("vector_error_mean",)),
                                   ("stationary_drift", ("summed_error_per_run_mean", "summed_error_per_run_max"))):
                actual, target = score.get(group, {}), reference.get(group, {})
                coverage = actual.get("n", 0) > 0 and actual.get("n") == target.get("n")
                for metric in metrics:
                    value, other = actual.get(metric), target.get(metric)
                    valid = (coverage and value is not None and other is not None
                             and math.isfinite(value) and math.isfinite(other) and value >= 0 and other >= 0)
                    passed = bool(valid and (value < other or value == other == 0.))
                    checks[f"{group}_{metric}_vs_{baseline}"] = passed
                    comparisons[baseline][metric] = {"model": value, "baseline": other, "pass": passed}
        rows[episode] = {"translation_mean_mm": score["translation_mean_mm"],
                         "gripper_mae_fraction": score["gripper_mae"],
                         "pass": all(checks.values()), "checks": checks, "comparisons": comparisons}
    return {"version": GATE_VERSION,
            "pass": valid_episodes and all(row["pass"] for row in rows.values()),
            "expected_training_episodes": list(EPISODES), "episode_coverage_pass": valid_episodes,
            "per_episode": rows, "thresholds": {"translation_mean_mm_below": 1., "gripper_mae_fraction_below": .03},
            "baseline_rule": "Each episode must improve both hold and persistence on moving-vector error and mean/max stationary-run summed error. Exact zero-error ties pass; missing movement or stationary strata fail. Stationary predicted step is descriptive, since hold is identically zero."}


def audit_training_checkpoint(checkpoint: Path, data: Path, device: str) -> dict:
    """One inference pass supplies aggregate and condition-specific motion checks."""
    policy, norm, contract = load_checkpoint(checkpoint, device)
    dataset = DerivedTickDataset(data, list(EPISODES), policy.config.chunk_size, norm)
    if contract["derived_files_sha256"] != dataset.manifest["files_sha256"]:
        raise ValueError("training audit data differs from checkpoint")
    for name, digest in dataset.manifest["files_sha256"].items():
        if Path(name).name != name or sha256_file(data / name) != digest:
            raise ValueError(f"training audit payload changed: {name}")
    if norm.stats != json.loads((data / "stats.json").read_text()):
        raise ValueError("training audit normalization differs from checkpoint")
    report, prediction = evaluate_executed_prefix(policy, dataset, device, batch_size=8)
    for episode in EPISODES:
        mask = dataset.arrays["episode_index"][prediction["tick"]] == episode
        single = executed_prefix_report(prediction["model"][mask], prediction["tick"][mask],
            prediction["anchor_tick"][mask], dataset.arrays, dataset.manifest,
            norm.action_std, [episode], n_action_steps=int(policy.config.n_action_steps))
        for name in ("model", "hold", "persistence"):
            entry = report["scores"][name]["per_episode"][str(episode)]
            for metric in ("movement", "stationary_drift"):
                entry[metric] = single["scores"][name][metric]
    return report


def run(data: Path, out: Path, device: str, *, resume: bool = False,
        supervised_prefix_steps: int | None = None):
    if out.exists() and not resume:
        raise FileExistsError(out)
    if resume and not out.is_dir():
        raise FileNotFoundError(out)
    base = load_yaml(Path("configs/v2/three_episode_fit.yaml"))
    cfg = copy.deepcopy(base)
    cfg.update(name="three_episode_deterministic_capacity", data=str(data.resolve()))
    cfg["model"].update(use_vae=False, kl_weight=0., dropout=0., optimizer_weight_decay=0., optimizer_lr=3e-4)
    cfg["scheduler"].update(warmup_steps=50)
    cfg["train"].update(diagnostic_episodes=list(EPISODES), steps=1000, schedule_steps=2000,
                         batch_size=8, grad_accum=2, log_every=100, val_every=250,
                         save_every=250, val_max_batches=None, device=device,
                         num_workers=0, exact_resume=True, keep_only_best_and_last=True)
    cfg["experiment"] = {"kind": GATE_VERSION, "maximum_examples": 32000,
                          "conditions": {"11": "baseline", "8": "ball_shift", "23": "bowl_shift"},
                          "scope": "training-fit only; original provisional FK and current labels; no geometry or task-success claim",
                          "stopping": "Check the final checkpoint at 16,000 examples; continue identical optimizer/RNG to 32,000 only if per-episode fit/movement/drift gate fails"}
    if supervised_prefix_steps is not None:
        loss = ReconstructionLoss(**(cfg.get("reconstruction_loss", {})
                                    | {"supervised_prefix_steps": supervised_prefix_steps}))
        if loss.supervised_prefix_steps != cfg["model"]["n_action_steps"]:
            raise ValueError("prefix supervision must match the saved executed action count")
        if loss.supervised_prefix_steps > cfg["model"]["chunk_size"]:
            raise ValueError("prefix supervision exceeds the output chunk")
        cfg["name"] += f"_prefix{supervised_prefix_steps}"
        cfg["reconstruction_loss"] = {**cfg.get("reconstruction_loss", {}),
                                      "supervised_prefix_steps": supervised_prefix_steps}
        cfg["experiment"]["supervised_prefix_steps"] = supervised_prefix_steps
        cfg["experiment"]["comparison"] = "Same backbone, initialization seed, labels, statistics, sampler and optimizer schedule; reconstruction supervises only the executed prefix. No target filtering or relabeling."
    torch.set_num_threads(2)
    if resume:
        if load_yaml(out / "plan.yaml") != cfg:
            raise ValueError("resume must retain the original plan, data path and device")
    else:
        out.mkdir(parents=True)
        (out / "plan.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (out / "executed_runner.py").write_bytes(Path(__file__).read_bytes())
    dependencies = [Path(__file__), Path("src/real2sim/policy/train.py"),
                    Path("src/real2sim/policy/act.py"), Path("src/real2sim/policy/dataset.py"),
                    Path("src/real2sim/policy/validation.py"), Path("src/real2sim/policy/losses.py"),
                    data / "manifest.json", data / "stats.json"]
    report = {"kind": "THREE_TRAINING_EPISODE_CAPACITY_GATE", "physical_calibration_accepted": False,
              "checkpoint_promoted": False, "reserved_episodes_used": [], "validation_episodes_used": [],
              "training_episodes": list(EPISODES), "gate_version": GATE_VERSION,
              "supervised_prefix_steps": supervised_prefix_steps,
              "source_sha256": {str(p): sha256_file(p) for p in dependencies},
              "stages": [], "limitations": "Training-fit diagnostic under provisional FK. Includes recorded references outside native bounds; their presence is not simulator feasibility or physical calibration. No validation or reserved performance is established."}
    resume_checkpoint = None
    if resume:
        path = out / "report.json"
        if path.exists():
            report = json.loads(path.read_text())
            if report["gate_version"] != GATE_VERSION:
                raise ValueError("cannot change the gate during a resumed run")
            for name, digest in report["source_sha256"].items():
                if name != str(Path(__file__)) and sha256_file(Path(name)) != digest:
                    raise ValueError(f"resume dependency changed: {name}")
        else:
            report["initial_report_missing"] = "Earlier runner wrote its first report only at a completed stage; recovered source hashes begin at restart. Checkpoint contract verifies original data and statistics."
        resume_checkpoint = (out / "training/checkpoints/last").resolve()
        info = json.loads((resume_checkpoint / "run_info.json").read_text())
        if info.get("diagnostic_episodes") != list(EPISODES) or info.get("scheduler_total_steps") != 2000:
            raise ValueError("checkpoint is not from this bounded training plan")
        snapshot = out / f"resume_runner_{info['step']:06d}.py"
        snapshot.write_bytes(Path(__file__).read_bytes())
        report.setdefault("interruptions", []).append({"resumed_from_step": info["step"],
            "checkpoint": str(resume_checkpoint),
            "training_state_sha256": sha256_file(resume_checkpoint / "training_state.pt"),
            "runner_snapshot": str(snapshot.resolve()), "runner_sha256": sha256_file(snapshot),
            "scope": "Restore saved optimizer, sampler and RNG; recompute any unsaved updates under the original schedule."})
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for steps in (1000, 2000):
        if report.get("gate_passed"):
            break
        if any(stage["examples_seen"] == steps * 16 for stage in report["stages"]):
            continue
        cfg["train"]["steps"] = steps
        restored_step = (json.loads((resume_checkpoint / "run_info.json").read_text())["step"]
                         if resume_checkpoint is not None else 0)
        if restored_step > steps:
            raise ValueError("completed earlier stage audit is missing; do not silently skip the gate")
        if restored_step == steps:
            info = json.loads((resume_checkpoint / "run_info.json").read_text())
            result = {"steps": steps, "device": info["device"],
                      "recovered_completed_checkpoint_for_audit": True}
        else:
            result = train(cfg, data, out / "training", resume=resume_checkpoint)
        ck = (out / "training/checkpoints/last").resolve()
        audited = audit_training_checkpoint(ck, data, result["device"])
        audit_path = out / f"training_motion_{steps:06d}.json"
        audit_path.write_text(json.dumps(audited, indent=2, allow_nan=False) + "\n")
        gate = gate_from_validation(audited)
        report["stages"].append({"examples_seen": steps * 16, "result": result,
                                  "checkpoint": str(ck.resolve()), "checkpoint_sha256": sha256_file(ck / "model.safetensors"),
                                  "training_motion_audit": str(audit_path.resolve()), "gate": gate})
        report.update(gate_passed=gate["pass"], total_examples_seen=steps * 16)
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"steps": steps, "gate": gate}), flush=True)
        resume_checkpoint = ck
        if gate["pass"]:
            break
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume the same saved plan from its last exact checkpoint")
    parser.add_argument("--supervised-prefix-steps", type=int,
                        help="Supervise only this many output actions; must equal n_action_steps")
    args = parser.parse_args()
    result = run(args.data, args.out, args.device, resume=args.resume,
                 supervised_prefix_steps=args.supervised_prefix_steps)
    print(json.dumps({"gate_passed": result["gate_passed"], "total_examples_seen": result["total_examples_seen"]}))
