"""Evaluate a frozen checkpoint on explicit, hashed v2 development/reserved layouts."""

import argparse
import json
from pathlib import Path

from real2sim.config import load_yaml, project_path
from real2sim.data.conditions import manifest_hash
from real2sim.evaluation.run import run_trials
from real2sim.experiments.v2 import gate_failures, read_json, verify_bundle
from real2sim.policy.paired import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--robot", choices=("so101", "panda"), default="so101")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--so101-final",
        type=Path,
        help="required Panda prerequisite: completed SO-101 aggregated final report",
    )
    a = p.parse_args()
    bundle = read_json(a.bundle)
    verify_bundle(bundle)
    files = bundle["files"]
    required = {
        "evidence",
        "sim",
        "eval",
        "scenarios",
        "task",
        "checkpoint",
        "checkpoint_config",
        "checkpoint_stats",
        "checkpoint_contract",
    }
    if not required <= files.keys():
        raise ValueError(f"freeze required artifacts: {sorted(required)}")
    if files["checkpoint"]["sha256"] != file_sha256(a.checkpoint / "model.safetensors"):
        raise ValueError("freeze the checkpoint before evaluation")
    for name, filename in (("checkpoint_config", "config.json"), ("checkpoint_stats", "stats.json"),
                           ("checkpoint_contract", "real2sim_contract.json")):
        if files[name]["sha256"] != file_sha256(a.checkpoint / filename):
            raise ValueError(f"frozen checkpoint metadata differs: {name}")
    failures = gate_failures(read_json(files["evidence"]["path"]))
    if failures:
        raise ValueError("prerequisite evidence failed: " + "; ".join(failures))
    if a.robot == "panda":
        if not a.so101_final:
            raise ValueError("Panda follows a completed SO-101 final evaluation")
        result = read_json(a.so101_final)
        if result.get("n_rollouts") != 270 or result.get("mean_success", 0) < 0.6:
            raise ValueError("SO-101 60% milestone has not been demonstrated")
        # Target-robot scripted verification must be recorded separately from learned rollouts.
        transfer = read_json(files["panda_verification"]["path"])
        if not all(
            transfer.get(k) is True
            for k in ("reachable_starts", "bounded_tracking", "scripted_pick_place")
        ):
            raise ValueError("Panda scripted/control verification is incomplete")
    suite = read_json(files["scenarios"]["path"])
    if suite.get("sha256") != manifest_hash(suite):
        raise ValueError("scenario checksum mismatch")
    sim = load_yaml(files["sim"]["path"])
    task = load_yaml(files["task"]["path"])
    if file_sha256(project_path(sim["task_config"])) != files["task"]["sha256"]:
        raise ValueError("simulator task differs from the frozen task")
    if sim["control_hz"] != 12.5 or sim["physics_hz"] != 500 or not sim.get("strict_initial_state"):
        raise ValueError("v2 evaluation requires strict starts and 12.5/500 Hz control/physics")
    if task["episode"]["timeout_s"] != 45 or task["episode"]["success_hold_s"] != 1:
        raise ValueError("v2 task requires 45-second timeout and one-second released rest")
    if a.out.exists():
        raise FileExistsError("evaluation result exists; reuse it or version a new suite")
    a.out.mkdir(parents=True)
    rows = []
    for case in suite["scenarios"]:
        verify_bundle(bundle)
        summary = run_trials(
            robot=a.robot,
            checkpoint=a.checkpoint,
            config=Path(files["sim"]["path"]),
            eval_config=Path(files["eval"]["path"]),
            n_trials=1,
            seed_start=case["seed"],
            out=a.out / case["scenario_id"],
            video=False,
            controller="learned",
            device=a.device,
            reset_scenarios=[case["reset"]],
        )
        row = {
            "training_seed": a.training_seed,
            "scenario_id": case["scenario_id"],
            "condition": case["condition"],
            "timeout_s": 45,
            "checkpoint_sha256": files["checkpoint"]["sha256"],
            "pick_place_success": summary["summary"]["pick_place_successes"] == 1,
            "historical_ball_in_bowl_success": summary["summary"]["successes"] == 1,
            "abort": summary["summary"]["aborted"] > 0,
        }
        rows.append(row)
        (a.out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (a.out / "COMPLETE.json").write_text(
        json.dumps({"n_rollouts": len(rows), "bundle_sha256": bundle["sha256"]}) + "\n"
    )


if __name__ == "__main__":
    main()
