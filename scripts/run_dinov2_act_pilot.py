"""One bounded DINOv2–Cartesian ACT pilot, then evaluations and one terminal callback."""

import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from real2sim.config import load_yaml
from real2sim.policy.paired import file_sha256

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "outputs/dinov2_act_pilot_20261015"
TRAIN = ROOT / "outputs/act_cartesian_dinov2_pilot_20261015"
RESULTS = ROOT / "results/metrics/dinov2_act_pilot_20261015"
PARENT = ROOT / "outputs/act_cartesian/checkpoints/006000"
PAIRS = ROOT / ".cache/paired_cartesian_refit_20261015"
DATA = ROOT / ".cache/derived_cartesian"
MESSAGE = (
    "The user-approved 2,000-step DINOv2-small Cartesian ACT pilot completed or failed. Read "
    "outputs/dinov2_act_pilot_20261015/status.json and results/metrics/dinov2_act_pilot_20261015/. "
    "Review real validation, paired checks and matched parent/pilot ten-seed SO-101 and Panda "
    "evaluations. Training uses real recorded Cartesian labels with paired-image augmentation only; "
    "no synthetic actions. Final two DINO blocks plus output norm are fine-tuned, earlier blocks frozen; "
    "ACT nonvisual weights warm-start from Cartesian 006000. User chose linear warmup/cosine decay, "
    "peak ACT/DINO LRs 1e-4/1e-5, weight decay .05 and gradient clip 1. Report measured results and "
    "record them in docs/progress.md. Diagnose failures before any further training, do not duplicate "
    "jobs or overwrite results. No automatic longer training. Preserve primary benchmark, data split, "
    "hardware and no-privileged-policy-input constraints. If success >=6/10, confirm on fresh seeds "
    "before claiming generalization. No token polling."
)


def notify(state):
    receipt = JOB / "delivery.json"
    if receipt.exists():
        return
    callback = json.loads((JOB / "notification.json").read_text())
    try:
        p = subprocess.run([callback["codex_bin"], "queue", "--thread", callback["thread_id"],
                            "--message", MESSAGE], capture_output=True, text=True, timeout=45)
        result = {"returncode": p.returncode, "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = {"returncode": -1, "error": repr(exc)}
    result["job_status"] = state["status"]
    receipt.write_text(json.dumps(result, indent=2) + "\n")


def main():
    os.chdir(ROOT)
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (JOB / "status.json").exists() or TRAIN.exists():
            raise FileExistsError("pilot already exists; refusing duplicate job or overwritten results")
        state = {"status": "running", "pid": os.getpid(), "completed_phases": [],
                 "started_utc": dt.datetime.now(dt.UTC).isoformat(),
                 "training_kind": "real_recorded_cartesian_labels_with_paired_images_no_synthetic_actions"}

        def save():
            state["updated_utc"] = dt.datetime.now(dt.UTC).isoformat()
            temp = JOB / "status.tmp.json"
            temp.write_text(json.dumps(state, indent=2) + "\n")
            temp.replace(JOB / "status.json")

        def module(phase, name, *args):
            command = [sys.executable, "-u", "-m", name, *map(str, args)]
            state.update(phase=phase, command=command, log=str(JOB / f"{phase}.log"))
            save()
            env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
            with (JOB / f"{phase}.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                state["child_pid"] = process.pid
                save()
                code = process.wait()
            state.pop("child_pid", None)
            if code:
                raise subprocess.CalledProcessError(code, command)
            state["completed_phases"].append(phase)
            save()

        try:
            save()
            if shutil.disk_usage(ROOT).free < 3 * 1024**3:
                raise RuntimeError("less than 3 GiB free")
            if not (JOB / "notification.json").is_file():
                raise FileNotFoundError("completion notification is not configured")
            shutil.copyfile(ROOT / "configs/act_cartesian_dinov2_pilot.yaml", JOB / "train_config.yaml")
            files = [PARENT / "model.safetensors", JOB / "train_config.yaml",
                     *sorted((RESULTS / "configs").glob("*.yaml")),
                     DATA / "arrays.npz", DATA / "stats.json", DATA / "images_iphone.npy"]
            for split in ("train", "val"):
                files.extend(PAIRS / split / name for name in ("manifest.json", "images.npy", "ticks.npy"))
            files.extend(ROOT.glob("src/real2sim/policy/*.py"))
            model_cfg = load_yaml(JOB / "train_config.yaml")["model"]
            snapshot = ROOT / model_cfg["dinov2_pretrained_path"]
            files.extend(snapshot / name for name in ("config.json", "model.safetensors"))
            files.extend([ROOT / "uv.lock", ROOT / "scripts/run_dinov2_act_pilot.py"])
            hashes = {str(p): file_sha256(p) for p in files}
            state["frozen_hashes"] = hashes
            save()
            module("train", "real2sim.policy.train", "--config", JOB / "train_config.yaml", "--out", TRAIN)
            best = (TRAIN / "checkpoints/best").resolve()
            state["best_checkpoint"] = str(best)
            state["best_checkpoint_sha256"] = file_sha256(best / "model.safetensors")
            module("real_validation", "real2sim.policy.check", "--checkpoint", best, "--data", DATA,
                   "--device", "mps", "--out", RESULTS / "real_check.json")
            module("parent_validation", "real2sim.policy.check", "--checkpoint", PARENT, "--data", DATA,
                   "--device", "mps", "--out", RESULTS / "parent_real_check.json")
            module("paired_validation", "real2sim.evaluation.paired_check", "--checkpoint", best,
                   "--parent", PARENT, "--data", DATA, "--pairs", PAIRS / "val",
                   "--out", RESULTS / "paired_check.json")
            cfg = RESULTS / "configs"
            names = []
            for label, checkpoint in (("parent", PARENT), ("pilot", best)):
                for robot in ("so101", "panda"):
                    name = f"{label}_{robot}"
                    names.append(name)
                    if (RESULTS / name).exists():
                        raise FileExistsError(RESULTS / name)
                    module(name, "real2sim.evaluation.run", "--robot", robot, "--checkpoint", checkpoint,
                           "--config", cfg / f"sim_{robot}.yaml", "--eval-config", cfg / f"eval_{robot}.yaml",
                           "--n-trials", "10", "--seed-start", "0", "--device", "cpu",
                           "--out", RESULTS / name, "--experiment-id", "DINO_PILOT_DIAGNOSTIC_" + name.upper())
            for path, digest in hashes.items():
                if file_sha256(Path(path)) != digest:
                    raise RuntimeError(f"frozen input changed: {path}")
            if file_sha256(best / "model.safetensors") != state["best_checkpoint_sha256"]:
                raise RuntimeError("pilot checkpoint changed during evaluation")
            state.update(status="completed", phase="completed", frozen_hashes_verified=True,
                         summaries={n: json.loads((RESULTS / n / "summary.json").read_text())["summary"] for n in names})
        except Exception as exc:
            state.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        finally:
            save()
            notify(state)


if __name__ == "__main__":
    main()
