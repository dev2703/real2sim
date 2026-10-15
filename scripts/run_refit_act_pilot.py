"""Bounded real-demonstration ACT retrain; notify this thread only on completion/failure."""

import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from real2sim.policy.paired import file_sha256

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "outputs/act_refit_pilot_20261014"
TRAIN = ROOT / "outputs/act_joint_refit_stationary_20261014"
RESULTS = ROOT / "results/metrics/paired_refit_pilot_20261014"
PARENT = ROOT / "outputs/act_joint_paired_approach/checkpoints/000500"
PAIRS = ROOT / ".cache/paired_refit_stationary_20261014"


def notify(state):
    callback = json.loads((JOB / "notification.json").read_text())
    receipt = JOB / "delivery.json"
    if receipt.exists():
        return
    message = (
        "The user-requested real-demonstration ACT refit pilot completed or failed. Read "
        "outputs/act_refit_pilot_20261014/status.json and results/metrics/paired_refit_pilot_20261014/. "
        "The objective remains at least 6/10 ACT task success. User explicitly rejected oracle/simulated "
        "demonstrations: keep training labels from real recordings only. This pilot uses corrected "
        "camera renders, reviewed per-episode bowl placement and visible stationary-ball frames closer "
        "to grasp; no synthetic actions, no changed real labels or split. Review matched parent/pilot "
        "real validation, paired-image checks and frozen ten-seed closed-loop results with and without "
        "the original 78-tick response adapter. Diagnose failures before further training. Preserve "
        "hardware, primary benchmark rules, validation split and no-privileged-policy-input constraints. "
        "If success reaches 6/10, confirm on fresh seeds before claiming generalization. No token polling; "
        "notify only when work completes or fails. Prior investigation is in docs/progress.md and "
        "results/audit/grasp_contacts_20261014/. Do not duplicate live jobs or overwrite results."
    )
    try:
        result = subprocess.run([callback["codex_bin"], "queue", "--thread", callback["thread_id"],
                                 "--message", message], capture_output=True, text=True, timeout=45)
        value = {"returncode": result.returncode, "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        value = {"returncode": -1, "error": repr(exc)}
    value["job_status"] = state["status"]
    receipt.write_text(json.dumps(value, indent=2) + "\n")


def main():
    os.chdir(ROOT)
    JOB.mkdir(exist_ok=True, parents=True)
    with (JOB / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (JOB / "status.json").exists() or TRAIN.exists():
            raise FileExistsError("pilot already exists; refusing duplication")
        review = json.loads((PAIRS / "review.json").read_text())
        if review["status"] != "accepted_for_bounded_pilot":
            raise ValueError("pair review has not passed")
        hashes = {str(PARENT / "model.safetensors"): file_sha256(PARENT / "model.safetensors")}
        for p in (RESULTS / "configs").glob("*.yaml"):
            hashes[str(p)] = file_sha256(p)
        for split, digest in review["manifest_sha256"].items():
            p = PAIRS / split / "manifest.json"
            if file_sha256(p) != digest:
                raise ValueError("paired manifest changed after review")
            hashes[str(p)] = digest
        shutil.copyfile(ROOT / "configs/act_joint_refit_stationary.yaml", JOB / "train_config.yaml")
        state = {"status": "running", "pid": os.getpid(), "started_utc": dt.datetime.now(dt.UTC).isoformat(),
                 "completed_phases": [], "frozen_hashes": hashes,
                 "training_kind": "real_recorded_demonstrations_with_paired_render_augmentation_no_oracle_data"}

        def save():
            state["updated_utc"] = dt.datetime.now(dt.UTC).isoformat()
            temp = JOB / "status.tmp.json"
            temp.write_text(json.dumps(state, indent=2) + "\n")
            temp.replace(JOB / "status.json")

        def execute(phase, command):
            state.update(phase=phase, command=[str(x) for x in command], log=str(JOB / f"{phase}.log"))
            save()
            with (JOB / f"{phase}.log").open("w") as log:
                subprocess.run([str(x) for x in command], stdout=log, stderr=subprocess.STDOUT, check=True)
            state["completed_phases"].append(phase)
            save()

        def module(phase, name, *args):
            execute(phase, [sys.executable, "-u", "-m", name, *args])

        try:
            if shutil.disk_usage(ROOT).free < 3 * 1024 ** 3:
                raise RuntimeError("less than 3 GiB free")
            module("train", "real2sim.policy.train", "--config", JOB / "train_config.yaml", "--out", TRAIN)
            best = (TRAIN / "checkpoints/best").resolve()
            state["best_checkpoint"] = str(best)
            module("real_validation", "real2sim.policy.check", "--checkpoint", best, "--data", ".cache/derived_joint",
                   "--device", "cpu", "--out", RESULTS / "real_check.json")
            module("paired_validation", "real2sim.evaluation.paired_check", "--checkpoint", best,
                   "--parent", PARENT, "--pairs", PAIRS / "val", "--out", RESULTS / "paired_check.json")
            cfg = RESULTS / "configs"
            for name, checkpoint in [("parent", PARENT), ("pilot", best)]:
                module(name + "_joint", "real2sim.evaluation.run", "--checkpoint", checkpoint,
                    "--config", cfg / "sim_so101.yaml", "--eval-config", cfg / "eval_so101.yaml",
                    "--n-trials", "10", "--seed-start", "0", "--device", "cpu",
                    "--out", RESULTS / (name + "_joint"), "--experiment-id", "REAL_DEMO_REFIT_" + name.upper())
                execute(name + "_response", [sys.executable, "-u", ROOT / "scripts/eval_act_response.py",
                    "--checkpoint", checkpoint, "--config", cfg / "sim_so101.yaml",
                    "--eval-config", cfg / "eval_so101.yaml", "--out", RESULTS / (name + "_response")])
            for path, digest in hashes.items():
                if file_sha256(Path(path)) != digest:
                    raise RuntimeError(f"frozen artifact changed: {path}")
            state.update(status="completed", phase="completed", frozen_hashes_verified=True,
                summaries={name: json.loads((RESULTS / name / "summary.json").read_text())["summary"]
                           for name in ["parent_joint", "parent_response", "pilot_joint", "pilot_response"]})
        except Exception as exc:
            state.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        finally:
            save()
            notify(state)


if __name__ == "__main__":
    main()
