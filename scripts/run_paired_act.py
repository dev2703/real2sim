"""Bounded paired-image pilot then evaluation; one completion/error callback, no polling."""

import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "outputs/act_paired_approach"
TRAIN = ROOT / "outputs/act_joint_paired_approach"
RESULTS = ROOT / "results/metrics/paired_approach_3988577"


def notify(state, cfg, job=JOB):
    """Single dispatch after work terminates; never wakes the model for progress."""
    receipt = job / "delivery.json"
    if receipt.exists():
        raise FileExistsError("completion delivery already attempted; refusing duplicate")
    message = (
        "The user-approved paired real/sim ACT pilot has terminated. Read outputs/act_paired_approach/status.json "
        "and results/metrics/paired_approach_3988577/. Review the completed real validation, executed-prefix "
        "paired-image checks and ten closed-loop trials against results/metrics/geometry_3988577/. "
        "If failed, diagnose and fix the bounded pipeline without duplicating live jobs or overwriting results. "
        "Report measured outcomes and record them in docs/progress.md. If approach improved but execution "
        "failed, perform the approved recorded-command versus follower-motion diagnostic before proposing "
        "more training. Preserve hardware, data, split, benchmark and privileged-input constraints. "
        "Do not poll while training or launch a larger training run without reviewing this pilot."
    )
    try:
        result = subprocess.run([cfg["codex_bin"], "queue", "--thread", cfg["thread_id"], "--message", message],
                                capture_output=True, text=True, timeout=45)
        delivery = {"returncode": result.returncode, "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        delivery = {"returncode": -1, "error": str(exc)}
    delivery["job_status"] = state["status"]
    receipt.write_text(json.dumps(delivery, indent=2) + "\n")
    return delivery


def main():
    os.chdir(ROOT)
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (JOB / "status.json").exists() or TRAIN.exists() or RESULTS.exists():
            raise FileExistsError("pilot exists; refusing duplicate or overwrite")
        callback = json.loads((JOB / "notification.json").read_text())
        state = {"status": "running", "pid": os.getpid(), "phase": "preflight", "completed_phases": [],
                 "started_utc": dt.datetime.now(dt.UTC).isoformat()}

        def save():
            state["updated_utc"] = dt.datetime.now(dt.UTC).isoformat()
            tmp = JOB / "status.tmp.json"
            tmp.write_text(json.dumps(state, indent=2) + "\n")
            tmp.replace(JOB / "status.json")

        def execute(phase, module, args):
            command = [sys.executable, "-u", "-m", module, *map(str, args)]
            state.update(phase=phase, command=command, log=str(JOB / f"{phase}.log"))
            save()
            with (JOB / f"{phase}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            state["completed_phases"].append(phase)
            save()

        try:
            baseline = json.loads((ROOT / "outputs/geometry_3988577/status.json").read_text())
            if baseline["status"] != "completed":
                raise RuntimeError("corrected geometry baselines must finish first")
            review = json.loads((ROOT / ".cache/paired_approach_3988577/review.json").read_text())
            if review["status"] != "accepted_for_bounded_pilot":
                raise RuntimeError("pair review has not passed")
            if shutil.disk_usage(ROOT).free < 3 * 1024 ** 3:
                raise RuntimeError("less than 3 GiB free before pilot")
            RESULTS.mkdir(parents=True)
            shutil.copyfile(ROOT / "configs/act_joint_paired_approach.yaml", JOB / "train_config.yaml")
            execute("train", "real2sim.policy.train", ["--config", JOB / "train_config.yaml", "--out", TRAIN])
            best = (TRAIN / "checkpoints/best").resolve()
            execute("real_validation", "real2sim.policy.check", ["--checkpoint", best, "--data", ".cache/derived_joint",
                                                                  "--device", "cpu", "--out", RESULTS / "real_check.json"])
            execute("paired_validation", "real2sim.evaluation.paired_check", ["--checkpoint", best, "--out", RESULTS / "paired_check.json"])
            execute("closed_loop", "real2sim.evaluation.run", ["--checkpoint", best,
                "--config", "results/metrics/geometry_3988577/configs/sim_so101.yaml",
                "--eval-config", "results/metrics/geometry_3988577/configs/eval_so101.yaml",
                "--n-trials", "10", "--seed-start", "0", "--device", "cpu",
                "--out", RESULTS / "closed_loop", "--experiment-id", "PAIRED_APPROACH_JOINT_MIXED_IMAGES"])
            state.update(status="completed", phase="completed", best_checkpoint=str(best),
                         training=json.loads((TRAIN / "training_result.json").read_text()),
                         closed_loop=json.loads((RESULTS / "closed_loop/summary.json").read_text())["summary"])
        except Exception as exc:
            state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        finally:
            save()
            state["delivery"] = notify(state, callback)
            save()


if __name__ == "__main__":
    main()
