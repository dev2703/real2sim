"""A waiting queue cannot overlap the predecessor, and failed predecessors stop the queue."""

import fcntl
import importlib.util
import json
import select
import subprocess
import sys
from pathlib import Path

import pytest

from real2sim.config import load_yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_act_loss_comparison.py"
spec = importlib.util.spec_from_file_location("loss_queue", SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def test_queue_waits_for_predecessor_lock(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"status": "completed"}))
    with (tmp_path / "job.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        code = """
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('queue', sys.argv[1])
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
print('waiting', flush=True)
queue.wait_for_predecessor(Path(sys.argv[2]))
print('finished', flush=True)
"""
        child = subprocess.Popen([sys.executable, "-u", "-c", code, str(SCRIPT), str(tmp_path)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline().strip() == "waiting"
            assert select.select([child.stdout], [], [], 0.2)[0] == []
            assert child.poll() is None
            fcntl.flock(lock, fcntl.LOCK_UN)
            stdout, stderr = child.communicate(timeout=10)
            assert child.returncode == 0, stderr
            assert stdout.strip() == "finished"
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)


@pytest.mark.parametrize("status", ["failed", "running", "waiting"])
def test_incomplete_predecessor_rejected(tmp_path, status):
    (tmp_path / "job.lock").touch()
    (tmp_path / "status.json").write_text(json.dumps({"status": status}))
    with pytest.raises(RuntimeError, match="did not complete"):
        queue.wait_for_predecessor(tmp_path)


@pytest.mark.parametrize("kind", ["joint", "cartesian"])
@pytest.mark.parametrize("loss", ["l2", "huber"])
def test_only_loss_and_output_retention_change(kind, loss):
    control = load_yaml(f"configs/act_{kind}_augmented.yaml")
    cfg = queue.experiment_config(control, "experiment", {"kind": loss})
    assert cfg["reconstruction_loss"] == {"kind": loss}
    assert cfg["train"].pop("keep_only_best_and_last") is True
    cfg.pop("reconstruction_loss")
    cfg["name"] = control["name"]
    assert cfg == control
