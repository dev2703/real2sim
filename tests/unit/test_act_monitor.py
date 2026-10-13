import fcntl
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/monitor_act_jobs.py"
spec = importlib.util.spec_from_file_location("act_monitor", SCRIPT)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_healthy_progress_does_not_wake_agent(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "run.log").write_text("step 50 loss 0.2\n")
    (job / "status.json").write_text(json.dumps({"status": "running", "phase": "train", "log": str(job / "run.log")}))
    with (job / "job.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        first = monitor.inspect_jobs(tmp_path, ["job"], free_bytes=10 * 1024 ** 3)
        assert first["issues"] == []
        assert not monitor.needs_notification(first, first)
        second = {**first, "checked_utc": "later", "free_gib": 9}
        assert monitor.notification_key(first) == monitor.notification_key(second)
    dead = monitor.inspect_jobs(tmp_path, ["job"], free_bytes=10 * 1024 ** 3)
    assert "manager lock is not held" in dead["issues"][0]
    assert monitor.needs_notification(dead, first)


def test_stale_logs_low_disk_and_failed_jobs_are_reported(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    log = job / "run.log"
    log.touch()
    (job / "status.json").write_text(json.dumps({"status": "running", "log": str(log)}))
    result = monitor.inspect_jobs(tmp_path, ["job"], now=log.stat().st_mtime + 3600, free_bytes=1024 ** 3)
    assert any("30 minutes" in issue for issue in result["issues"])
    assert any("3 GiB" in issue for issue in result["issues"])
    (job / "status.json").write_text(json.dumps({"status": "failed", "error": "test failure"}))
    result = monitor.inspect_jobs(tmp_path, ["job"], free_bytes=10 * 1024 ** 3)
    assert any("test failure" in issue for issue in result["issues"])


def test_results_and_completion_trigger_review_once():
    before = {"issues": [], "results": [], "all_completed": False}
    after = {"issues": [], "results": ["job/joint/l2"], "all_completed": False}
    assert monitor.needs_notification(after, before)
    assert not monitor.needs_notification(after, after)
    completed = {**after, "all_completed": True}
    assert monitor.needs_notification(completed, after)
    assert not monitor.needs_notification(completed, completed)


def test_missing_status_is_an_issue(tmp_path):
    result = monitor.inspect_jobs(tmp_path, ["missing"], free_bytes=10 * 1024 ** 3)
    assert "cannot inspect status" in result["issues"][0]
    assert not result["all_completed"]
