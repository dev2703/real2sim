"""Regenerate SO-101 baselines on a frozen geometry snapshot; preserve old results."""

import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "outputs/geometry_3988577"
RESULTS = ROOT / "results/metrics/geometry_3988577"


def main():
    os.chdir(ROOT)
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (JOB / "status.json").exists() or RESULTS.exists():
            raise FileExistsError("baseline run exists; do not overwrite or duplicate it")
        RESULTS.mkdir(parents=True)
        snapshot = RESULTS / "configs"
        snapshot.mkdir()
        hashes = {}
        for name in ("task.yaml", "sim_so101.yaml", "data.yaml", "eval_so101.yaml"):
            b = (ROOT / "configs" / name).read_bytes()
            (snapshot / name).write_bytes(b)
            hashes[name] = hashlib.sha256(b).hexdigest()
        for name in ("sim_so101.yaml", "data.yaml", "eval_so101.yaml"):
            cfg = yaml.safe_load((snapshot / name).read_text())
            cfg["task_config"] = str(snapshot / "task.yaml")
            if "data_config" in cfg:
                cfg["data_config"] = str(snapshot / "data.yaml")
            (snapshot / name).write_text(yaml.safe_dump(cfg, sort_keys=False))
        state = {"status": "running", "pid": os.getpid(), "completed_phases": [],
                 "started_utc": dt.datetime.now(dt.UTC).isoformat(), "source_config_sha256": hashes}

        def save():
            temp = JOB / "status.tmp.json"
            temp.write_text(json.dumps(state, indent=2) + "\n")
            temp.replace(JOB / "status.json")

        try:
            jobs = [
                ("E0_hold", "real2sim.evaluation.run", ["--controller", "hold"]),
                ("E4_oracle", "real2sim.evaluation.oracle", []),
                ("E1_joint", "real2sim.evaluation.run", ["--checkpoint", "outputs/act_joint_so101/checkpoints/020000"]),
                ("E2_cartesian", "real2sim.evaluation.run", ["--checkpoint", "outputs/act_cartesian/checkpoints/006000"]),
            ]
            summaries = {}
            for phase, module, extra in jobs:
                command = [sys.executable, "-u", "-m", module, "--config", str(snapshot / "sim_so101.yaml"),
                           "--eval-config", str(snapshot / "eval_so101.yaml"), "--n-trials", "10",
                           "--seed-start", "0", "--out", str(RESULTS / phase), *extra]
                state.update(phase=phase, command=command, log=str(JOB / f"{phase}.log"))
                save()
                with (JOB / f"{phase}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                summaries[phase] = json.loads((RESULTS / phase / "summary.json").read_text())["summary"]
                state["completed_phases"].append(phase)
                state["summaries"] = summaries
                save()
            state.update(status="completed", phase="completed")
            (RESULTS / "summary.json").write_text(json.dumps(state, indent=2) + "\n")
        except Exception as exc:
            state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            state["updated_utc"] = dt.datetime.now(dt.UTC).isoformat()
            save()


if __name__ == "__main__":
    main()
