"""Run the two bounded MPS retrains sequentially, then validate and evaluate their best weights.

Designed to survive the chat turn without agent polling. Progress and failures are recorded in
outputs/act_remedies/status.json; each subprocess has a separate log. Existing runs are refused.
"""

from __future__ import annotations

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
JOB = ROOT / "outputs/act_remedies"
RUNS = (
    ("joint", "act_joint_augmented", ".cache/derived_joint", "outputs/act_joint_so101/checkpoints/020000"),
    ("cartesian", "act_cartesian_augmented", ".cache/derived_cartesian", "outputs/act_cartesian/checkpoints/006000"),
)


def main():
    os.chdir(ROOT)
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / "job.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"status": "running", "pid": os.getpid(), "started_utc": dt.datetime.now(dt.UTC).isoformat(),
                 "phase": "starting", "completed_phases": [], "comparisons": {}}

        def save():
            state["updated_utc"] = dt.datetime.now(dt.UTC).isoformat()
            temp = JOB / "status.tmp.json"
            temp.write_text(json.dumps(state, indent=2) + "\n")
            temp.replace(JOB / "status.json")

        def execute(phase, module, args):
            command = [sys.executable, "-u", "-m", module, *args]
            state.update(phase=phase, command=command, log=str(JOB / f"{phase}.log"))
            save()
            with (JOB / f"{phase}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"{phase} failed with exit {result.returncode}; see {phase}.log")
            state["completed_phases"].append(phase)
            save()

        try:
            for _, name, _, _ in RUNS:
                if (ROOT / "outputs" / name).exists():
                    raise FileExistsError(f"refusing to overwrite existing run: outputs/{name}")
            for kind, name, data, baseline in RUNS:
                if shutil.disk_usage(ROOT).free < 3 * 1024 ** 3:
                    raise RuntimeError("less than 3 GiB free; stopped before starting another model")
                output = f"outputs/{name}"
                execute(f"{kind}_train", "real2sim.policy.train", ["--config", f"configs/{name}.yaml", "--out", output])
                best = str((ROOT / output / "checkpoints/best").resolve())
                check_path = f"results/metrics/remedies/{kind}_augmented_check.json"
                execute(f"{kind}_check", "real2sim.policy.check",
                        ["--checkpoint", best, "--data", data, "--device", "cpu", "--out", check_path])
                execute(f"{kind}_closed_loop", "real2sim.evaluation.run",
                        ["--robot", "so101", "--checkpoint", best, "--device", "cpu", "--n-trials", "10",
                         "--seed-start", "0", "--out", f"results/metrics/remedies/{kind}_augmented_closed_loop",
                         "--experiment-id", f"REMEDY_{kind}_augmented_real_only"])
                current = json.loads((ROOT / check_path).read_text())
                previous = json.loads((ROOT / baseline / "run_info.json").read_text())
                loop = json.loads((ROOT / f"results/metrics/remedies/{kind}_augmented_closed_loop/summary.json").read_text())
                state["comparisons"][kind] = {
                    "baseline_checkpoint": baseline, "new_checkpoint": best,
                    "baseline_validation_l1_recorded": previous["last_val"]["l1"],
                    "new_validation_l1": current["val_open_loop_l1_normalised"],
                    "closed_loop": loop["summary"],
                    "training": json.loads((ROOT / output / "training_result.json").read_text()),
                    "note": "New real-only training run, seed 0. Same backbone/dimensions/labels/split/normalization; augmentation, dropout and weight decay changed together. Not an isolated causal ablation."
                }
                save()
            state.update(status="completed", phase="completed")
        except Exception as exc:
            state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            save()
            raise
        save()
        print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
