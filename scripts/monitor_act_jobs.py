"""Hourly local health checks; queue this chat only on issues or new experiment results.

No model calls for healthy unchanged jobs. Create outputs/act_monitor/STOP to stop.
All repair/experiment decisions stay in the existing chat, with its existing permissions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "outputs/act_monitor"
DEFAULT_JOBS = ["outputs/act_remedies", "outputs/act_loss_comparison"]


def utc(timestamp=None):
    return dt.datetime.fromtimestamp(timestamp or time.time(), dt.UTC).isoformat()


def held_lock(path):
    if not path.exists():
        return False
    with path.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
    return False


def inspect_jobs(root, directories, now=None, free_bytes=None):
    now = time.time() if now is None else now
    free_bytes = shutil.disk_usage(root).free if free_bytes is None else free_bytes
    snapshot = {"checked_utc": utc(now), "free_gib": round(free_bytes / 1024 ** 3, 2),
                "jobs": {}, "issues": [], "results": []}
    for directory in directories:
        path = root / directory
        try:
            state = json.loads((path / "status.json").read_text())
            status = state["status"]
            locked = held_lock(path / "job.lock")
            item = {"status": status, "phase": state.get("phase"), "pid": state.get("pid"),
                    "manager_lock_held": locked}
            snapshot["jobs"][directory] = item
            if status in {"running", "waiting"}:
                if not locked:
                    snapshot["issues"].append(f"{directory}: {status} but manager lock is not held")
                log = Path(state["log"]) if state.get("log") else None
                if log is not None:
                    log = log if log.is_absolute() else root / log
                    age = now - log.stat().st_mtime if log.exists() else float("inf")
                    item["log_age_s"] = round(age, 1) if age != float("inf") else None
                    if age > 1800:
                        snapshot["issues"].append(f"{directory}: active phase has no log update for over 30 minutes")
            elif status == "failed":
                snapshot["issues"].append(f"{directory}: failed: {state.get('error', 'see manager log')}")
            elif status != "completed":
                snapshot["issues"].append(f"{directory}: unexpected status {status}")
            for model, comparison in state.get("comparisons", {}).items():
                if "training" in comparison:  # remedies: one result per model
                    snapshot["results"].append(f"{directory}/{model}")
                else:  # loss comparison: one result per model and loss
                    snapshot["results"].extend(f"{directory}/{model}/{loss}" for loss in comparison)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            snapshot["issues"].append(f"{directory}: cannot inspect status: {exc}")
    waiting = [name for name, job in snapshot["jobs"].items() if job["status"] == "waiting"]
    running = any(job["status"] == "running" for job in snapshot["jobs"].values())
    if waiting and not running:
        snapshot["issues"].append("waiting queue has no running predecessor; inspect handoff")
    snapshot["all_completed"] = bool(snapshot["jobs"]) and len(snapshot["jobs"]) == len(directories) and all(
        job["status"] == "completed" for job in snapshot["jobs"].values())
    if not snapshot["all_completed"] and free_bytes < 3 * 1024 ** 3:
        snapshot["issues"].append("less than 3 GiB free; check space before another model starts")
    return snapshot


def notification_key(snapshot):
    # Do not wake the agent merely because another training step or hour elapsed.
    payload = {key: snapshot[key] for key in ("issues", "results", "all_completed")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def needs_notification(snapshot, previous):
    return bool(snapshot["issues"] or set(snapshot["results"]) - set(previous.get("results", []))
                or (snapshot["all_completed"] and not previous.get("all_completed")))


def save_state(state):
    temp = JOB / "status.tmp.json"
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(JOB / "status.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(JOB / "config.json"))
    parser.add_argument("--once", action="store_true", help="read-only inspection; no queue dispatch")
    args = parser.parse_args(argv)
    if args.once:
        print(json.dumps(inspect_jobs(ROOT, DEFAULT_JOBS), indent=2))
        return
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = json.loads(Path(args.config).read_text())
        initial = inspect_jobs(ROOT, cfg["jobs"])
        baseline = initial if not initial["issues"] and not initial["all_completed"] else {"results": []}
        state = {"status": "running", "pid": os.getpid(), "started_utc": utc(),
                 "interval_seconds": 3600, "checks": 0, "last_check": initial,
                 "last_notified": baseline,
                 "last_notification_key": notification_key(initial) if baseline is initial else None}
        next_check = cfg["first_check_epoch"]
        state["next_check_utc"] = utc(next_check)
        save_state(state)
        while not (JOB / "STOP").exists():
            # Sleep in this separate local process, not in an active model turn.
            # Short OS sleeps allow a STOP file to take effect promptly, with no model tokens.
            if time.time() < next_check:
                time.sleep(max(0, min(30, next_check - time.time())))
                continue
            cfg = json.loads(Path(args.config).read_text())  # future approved jobs can be added
            snapshot = inspect_jobs(ROOT, cfg["jobs"])
            state["checks"] += 1
            state["last_check"] = snapshot
            with (JOB / "checks.jsonl").open("a") as log:
                log.write(json.dumps(snapshot) + "\n")
            key = notification_key(snapshot)
            if needs_notification(snapshot, state["last_notified"]) and key != state["last_notification_key"]:
                message = (
                    "Scheduled hourly ACT check requested by the user. Read outputs/act_monitor/status.json "
                    "and checks.jsonl for the latest findings; verify current job states before acting. "
                    "Fix failures in the background experiments and continue the approved queue without "
                    "duplicating live processes or overwriting results. Review newly completed results; "
                    "the user permits small evidence-driven follow-up experiments. Preserve docs/PROJECT_SUMMARY.md's "
                    "data/schema, benchmark and hardware constraints and document decisions/results. "
                    "Do not poll or spend tokens waiting. If healthy, leave jobs running. When all work "
                    "is reviewed and no further justified experiments remain, write outputs/act_monitor/STOP."
                )
                # Persist findings before delivery so a queued turn can read them immediately.
                save_state(state)
                try:
                    result = subprocess.run([cfg["codex_bin"], "queue", "--thread", cfg["thread_id"],
                                             "--message", message], capture_output=True, text=True, timeout=45)
                    delivery = {"utc": utc(), "returncode": result.returncode,
                                "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}
                except (OSError, subprocess.TimeoutExpired) as exc:
                    delivery = {"utc": utc(), "returncode": -1, "error": str(exc)}
                state["last_delivery"] = delivery
                if delivery["returncode"] == 0:
                    state["last_notified"] = snapshot
                    state["last_notification_key"] = key
                with (JOB / "deliveries.jsonl").open("a") as log:
                    log.write(json.dumps(delivery) + "\n")
            elif not snapshot["issues"]:
                # If an issue clears then recurs, report it again; do not suppress it forever.
                state["last_notification_key"] = None
            next_check += 3600
            if next_check <= time.time():  # after machine sleep, never flood missed check-ins
                next_check = time.time() + 3600
            state["next_check_utc"] = utc(next_check)
            save_state(state)
        state.update(status="stopped", stopped_utc=utc(), next_check_utc=None)
        save_state(state)


if __name__ == "__main__":
    main()
