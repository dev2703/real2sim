"""Sequential loss ablations that wait for the existing remedies manager without agent polling."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def wait_for_predecessor(directory: Path) -> dict:
    # The remedies manager owns this lock until its final check/evaluation has finished.
    # Block only this background process; no busy loop, agent polling, or concurrent MPS run.
    with (directory / "job.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status = json.loads((directory / "status.json").read_text())
        if status["status"] != "completed":
            raise RuntimeError(f"predecessor did not complete: {status.get('status')} / {status.get('phase')}")
        return status


def experiment_config(control: dict, name: str, loss: dict) -> dict:
    cfg = copy.deepcopy(control)
    if cfg.get("reconstruction_loss", {}).get("kind", "l1") != "l1":
        raise ValueError("control must use L1")
    if not 0 < int(cfg["train"]["steps"]) <= 6000:
        raise ValueError("control exceeds the approved 6000-step budget")
    if loss["kind"] not in {"l2", "huber"}:
        raise ValueError("only the approved L2/Huber ablations may be queued")
    cfg.update(name=name, reconstruction_loss=loss.copy())
    cfg["train"]["keep_only_best_and_last"] = True
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/act_loss_comparison.yaml")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    settings = yaml.safe_load(Path(args.config).read_text())
    job, results = Path(settings["job"]), Path(settings["results"])
    job.mkdir(parents=True, exist_ok=True)
    with (job / "job.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (job / "status.json").exists():
            raise FileExistsError("loss comparison already launched; refusing to overwrite its status")
        state = {"status": "waiting", "phase": "waiting_for_remedies", "pid": os.getpid(),
                 "started_utc": dt.datetime.now(dt.UTC).isoformat(), "settings": settings,
                 "completed_phases": [], "comparisons": {},
                 "note": "Single-seed diagnostic. Same KL beta, but reconstruction scales differ. All checkpoints selected by validation MAE; success assessed separately in simulation."}

        def save():
            state["updated_utc"] = dt.datetime.now(dt.UTC).isoformat()
            temp = job / "status.tmp.json"
            temp.write_text(json.dumps(state, indent=2) + "\n")
            temp.replace(job / "status.json")

        def execute(phase, module, args):
            command = [sys.executable, "-u", "-m", module, *map(str, args)]
            state.update(status="running", phase=phase, command=command, log=str(job / f"{phase}.log"))
            save()
            with (job / f"{phase}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"{phase} failed with exit {result.returncode}; see {phase}.log")
            state["completed_phases"].append(phase)
            save()

        def collect(output, check_path, loop_path):
            checks = json.loads(check_path.read_text())
            if not (checks["contract_ok"] and checks["all_finite"]):
                raise RuntimeError(f"invalid checkpoint check: {check_path}")
            return {"training": json.loads((output / "training_result.json").read_text()),
                    "open_loop": checks, "closed_loop": json.loads(loop_path.read_text()),
                    "training_history": str(output / "training_history.csv")}

        def report():
            results.mkdir(parents=True, exist_ok=True)
            temp = results / "summary.tmp.json"
            temp.write_text(json.dumps(state, indent=2) + "\n")
            temp.replace(results / "summary.json")

        save()
        try:
            predecessor = wait_for_predecessor(Path(settings["predecessor"]))
            state["predecessor_completed_utc"] = predecessor["updated_utc"]
            # Preflight all controls and output paths before spending compute.
            runs = []
            for kind, directory in settings["controls"].items():
                control = Path(directory)
                resolved = yaml.safe_load((control / "resolved_config.yaml").read_text())
                cfg = resolved["config"]
                manifest = json.loads((Path(cfg["data"]) / "manifest.json").read_text())
                if any(manifest[key] != value for key, value in resolved["data_manifest"].items()):
                    raise RuntimeError(f"derived data manifest changed since the L1 control: {control}")
                if not (control / "training_result.json").is_file():
                    raise RuntimeError(f"incomplete L1 control: {control}")
                best = (control / "checkpoints/best").resolve(strict=True)
                if str(best) != predecessor["comparisons"][kind]["new_checkpoint"]:
                    raise RuntimeError("L1 control checkpoint changed after remedies evaluation")
                for loss in settings["losses"]:
                    name = f"act_{kind}_augmented_{loss['kind']}"
                    output = Path("outputs") / name
                    if output.exists():
                        raise FileExistsError(f"refusing to overwrite {output}")
                    runs.append((kind, loss["kind"], output, experiment_config(cfg, name, loss)))
                check_path = results / f"{kind}_l1_check.json"
                execute(f"{kind}_l1_check", "real2sim.policy.check",
                        ["--checkpoint", best, "--data", cfg["data"], "--device", "cpu", "--out", check_path])
                loop = Path(f"results/metrics/remedies/{kind}_augmented_closed_loop/summary.json")
                state["comparisons"][kind] = {"l1": collect(control, check_path, loop)}
                save()
                report()

            (job / "configs").mkdir(exist_ok=True)
            for kind, loss, output, cfg in runs:
                if shutil.disk_usage(ROOT).free < float(settings["minimum_free_gib"]) * 1024 ** 3:
                    raise RuntimeError("insufficient disk space; stopped before starting another model")
                config_path = job / "configs" / f"{output.name}.yaml"
                config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                execute(f"{kind}_{loss}_train", "real2sim.policy.train",
                        ["--config", config_path, "--out", output])
                best = (output / "checkpoints/best").resolve(strict=True)
                check_path = results / f"{kind}_{loss}_check.json"
                execute(f"{kind}_{loss}_check", "real2sim.policy.check",
                        ["--checkpoint", best, "--data", cfg["data"], "--device", "cpu", "--out", check_path])
                loop = results / f"{kind}_{loss}_closed_loop"
                execute(f"{kind}_{loss}_closed_loop", "real2sim.evaluation.run",
                        ["--robot", "so101", "--checkpoint", best, "--device", "cpu", "--n-trials", "10",
                         "--seed-start", "0", "--out", loop,
                         "--experiment-id", f"LOSS_{kind}_{loss}_real_only"])
                state["comparisons"][kind][loss] = collect(output, check_path, loop / "summary.json")
                save()
                report()
            state.update(status="completed", phase="completed")
        except Exception as exc:
            state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            save()
            report()
            raise
        save()
        report()
        print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
