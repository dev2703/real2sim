"""Run the local v2 learning screen with explicitly deferred geometry acceptance.

This exploratory queue never changes the evidence-gated v2 protocol or performs
robot rollouts. Each unique matrix arm receives its declared example budget.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    from scripts.run_so101_experiments import atomic_json, hash_file, now, queue_lock, read_json
except ModuleNotFoundError:
    from run_so101_experiments import atomic_json, hash_file, now, queue_lock, read_json

from real2sim.experiments.v2 import gate_failures

VERSION = "v2_exploratory_learning_screen_v1"
REPO = Path(__file__).resolve().parents[1]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def compile_plan(matrix: dict, data_root: Path, *, device: str, physical_batch: int) -> dict:
    """Keep the published comparisons/exposures; change only execution settings."""
    if physical_batch < 1 or device not in ("cpu", "mps", "cuda"):
        raise ValueError("positive physical batch and explicit local device required")
    rows, jobs, identities, ids = [], [], {}, set()
    for source in matrix["arms"]:
        arm = copy.deepcopy(source)
        name = arm["id"]
        if name in ids or not name or Path(name).name != name:
            raise ValueError("unique, plain arm identifiers required")
        ids.add(name)
        cfg, effective = arm["config"], arm["effective_batch"]
        if effective % physical_batch or arm["family"] not in ("act", "flow", "regression"):
            raise ValueError("physical batch must divide each supported arm's effective batch")
        train = cfg["train"]
        if train["steps"] * effective != arm["examples"] or train["schedule_steps"] < train["steps"]:
            raise ValueError(f"invalid fixed exposure/scheduler in {name}")
        cfg["data"] = str((data_root / Path(cfg["data"]).name).resolve())
        train.update(batch_size=physical_batch, grad_accum=effective // physical_batch,
                     device=device, precision="float32", num_workers=0, exact_resume=True,
                     keep_only_best_and_last=True)
        cfg["model"]["attention_backend"] = "sdpa"
        cfg["experiment"] = {"kind": VERSION, "geometry_acceptance": "deferred_by_user",
                             "physical_grasp_success": "not_evaluated"}
        identity = digest({k: v for k, v in cfg.items() if k not in ("name", "experiment")} |
                          {"family": arm["family"]})
        owner = identities.get(identity)
        if source.get("reuse") and source["reuse"] not in ids:
            raise ValueError(f"alias precedes its control: {name}")
        if owner is None:
            owner = identities[identity] = name
            jobs.append({"id": name, "family": arm["family"], "examples": arm["examples"],
                         "camera_keys": arm["camera_keys"],
                         "action_representation": arm["action_representation"],
                         "controller_semantics": arm["controller_semantics"],
                         "effective_batch": effective, "config": cfg, "identity": identity})
        if source.get("reuse") and owner != source["reuse"]:
            raise ValueError(f"declared alias differs from recomputed training identity: {name}")
        rows.append({"id": name, "stage": arm["stage"], "job": owner,
                     "reuse": owner if owner != name else None})
    if not jobs:
        raise ValueError("empty experiment matrix")
    return {"version": VERSION, "device": device, "physical_batch": physical_batch,
            "matrix_entries": rows,
            # Finish paired heads per backbone, then release their large feature arrays.
            "jobs": [j for j in jobs if j["family"] != "act"] +
                    [j for j in jobs if j["family"] == "act"],
            "scientific_status": "exploratory offline learning; geometry acceptance deferred",
            "physical_grasp_success": "not_evaluated"}


def inspect_inputs(plan: dict, paths: list[Path], *, verify_payloads: bool = True) -> dict:
    """Freeze all metadata and verify each distinct prepared payload once."""
    files = {str(p.resolve()): hash_file(p) for p in paths}
    payloads, seen = {}, set()
    for job in plan["jobs"]:
        data = Path(job["config"]["data"])
        for filename in ("manifest.json", "stats.json"):
            path = data / filename
            files[str(path.resolve())] = hash_file(path)
        manifest = read_json(data / "manifest.json")
        for key in ("camera_keys", "action_representation", "controller_semantics"):
            if manifest.get(key) != job[key]:
                raise ValueError(f"{job['id']} prepared data differs: {key}")
        if manifest.get("policy_hz") != 12.5 or not manifest.get("contract_version"):
            raise ValueError("prepared data must have the v2 12.5 Hz contract")
        split = manifest["split"]
        if not split.get("train") or not split.get("val") or any(
            set(split.get(a, [])) & set(split.get(b, []))
            for a, b in (("train", "val"), ("train", "reserved"), ("val", "reserved"))
        ):
            raise ValueError("prepared split is empty or leaks between partitions")
        for name, expected in manifest["files_sha256"].items():
            if Path(name).name != name:
                raise ValueError("prepared payload names must be plain filenames")
            path = (data / name).resolve()
            if str(path) in payloads and payloads[str(path)] != expected:
                raise ValueError("shared data path has conflicting hashes")
            payloads[str(path)] = expected
            if verify_payloads and path not in seen and hash_file(path) != expected:
                raise ValueError(f"prepared payload checksum mismatch: {path}")
            seen.add(path)
    return {"files": files, "prepared_payloads": payloads}


def latest_checkpoint(job: dict, root: Path) -> tuple[Path | None, int]:
    """Ignore interrupted partial writes; only resume a complete owned checkpoint."""
    best, exposure = None, 0
    for path in sorted(root.glob("attempt-*/checkpoints/[0-9]*")):
        required = ("model.safetensors", "training_state.pt", "run_info.json")
        if path.is_symlink() or not all((path / name).is_file() for name in required):
            continue
        try:
            count = int(read_json(path / "run_info.json")["examples_seen"])
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        if count > exposure:
            best, exposure = path, count
    return best, exposure


class ScreenRunner:
    def __init__(self, args: argparse.Namespace):
        self.args, self.out = args, args.out.resolve()
        self.active: subprocess.Popen | None = None
        self.interrupted = False
        self.plan: dict = {}
        self.frozen: dict = {}
        self.status: dict = {}

    def verify_inputs(self) -> None:
        for path, expected in self.frozen["files"].items():
            if hash_file(Path(path)) != expected:
                raise ValueError(f"frozen learning screen input changed: {path}")
        if shutil.disk_usage(self.out).free < self.args.minimum_free_gib * 1024 ** 3:
            raise RuntimeError("insufficient free space for another training stage")

    def prepare(self) -> None:
        self.plan = compile_plan(read_json(self.args.matrix), self.args.data_root,
                                 device=self.args.device, physical_batch=self.args.physical_batch)
        evidence = read_json(self.args.evidence) if self.args.evidence else {}
        failures = gate_failures(evidence)
        if failures and not self.args.defer_geometry_checks:
            raise ValueError("learning screen requires --defer-geometry-checks for unmet gates: " +
                             "; ".join(failures))
        sources = [self.args.matrix, self.args.task, Path(__file__),
                   REPO / "scripts/train_flow_head.py", REPO / "src/real2sim/experiments/v2.py",
                   *sorted((REPO / "src/real2sim/policy").glob("*.py")),
                   *sorted((REPO / "src/real2sim/geometry").glob("*.py"))]
        if self.args.evidence:
            sources.append(self.args.evidence)
        self.frozen = inspect_inputs(self.plan, sources)
        self.plan["provenance"] = self.frozen
        self.plan["geometry"] = {"deferred_by_user": bool(failures), "gate_failures": failures,
                                 "evidence_path": str(self.args.evidence) if self.args.evidence else None,
                                 "gate_passed": not failures}
        path = self.out / "PLAN.json"
        if path.exists() and read_json(path) != self.plan:
            raise ValueError("learning screen inputs/configs changed; choose a new output directory")
        atomic_json(path, self.plan)
        for job in self.plan["jobs"]:
            path = self.out / "configs" / f"{job['id']}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            body = yaml.safe_dump(job["config"], sort_keys=False)
            if path.exists() and path.read_text() != body:
                raise ValueError(f"prepared training config changed: {path}")
            path.write_text(body)
            self.frozen["files"][str(path)] = hash_file(path)
        # Generated configs are frozen separately to keep PLAN stable across restarts.
        atomic_json(self.out / "INPUTS.json", self.frozen)
        status_path = self.out / "STATUS.json"
        self.status = read_json(status_path) if status_path.exists() else {
            "version": VERSION, "created_at": now(), "status": "prepared",
            "geometry_acceptance": "deferred_by_user" if failures else "passed",
            "physical_grasp_success": "not_evaluated", "jobs": {},
            "matrix_entries": self.plan["matrix_entries"]}
        self.write_status()

    def write_status(self) -> None:
        self.status.update(updated_at=now(), pid=os.getpid(),
                           completed_unique_jobs=sum(r.get("status") == "completed"
                                                     for r in self.status["jobs"].values()),
                           total_unique_jobs=len(self.plan["jobs"]))
        atomic_json(self.out / "STATUS.json", self.status)

    def stop(self, signum, frame) -> None:
        self.interrupted = True
        if self.active and self.active.poll() is None:
            self.active.send_signal(signal.SIGTERM)

    def command(self, command: list[str], log: Path, *, stage: str) -> dict:
        self.verify_inputs()
        if self.interrupted:
            raise InterruptedError("learning screen interrupted")
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS=str(self.args.threads), MKL_NUM_THREADS=str(self.args.threads),
                   OPENBLAS_NUM_THREADS=str(self.args.threads), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
        with log.open("a") as handle:
            self.active = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            self.status.update(active_stage=stage, active_command=command,
                               subprocess_pid=self.active.pid, status="running")
            self.write_status()
            timed_out = False
            try:
                returncode = self.active.wait(timeout=self.args.runtime_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(self.active.pid, signal.SIGTERM)
                try:
                    returncode = self.active.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.active.pid, signal.SIGKILL)
                    returncode = self.active.wait(timeout=10)
            finally:
                self.active = None
        self.verify_inputs()
        return {"command": command, "returncode": returncode, "timed_out": timed_out,
                "wall_s": round(time.monotonic() - started, 3), "finished_at": now(), "log": str(log)}

    def ensure_cache(self, job: dict) -> Path:
        vision = job["config"]["model"]["vision_backbone"]
        root = self.out / "feature_caches" / vision
        for attempt in sorted(root.glob("attempt-*")):
            if all((attempt / name).is_file() for name in
                   ("COMPLETE.json", "manifest.json", "features.npy", "ticks.npy", "backbone.safetensors")):
                return attempt
        data = Path(job["config"]["data"])
        split = read_json(data / "manifest.json")["split"]
        # Cache only train/inner validation anchors, keeping reserved images untouched.
        episodes = sorted(set(split["train"]) | set(split["val"]))
        attempt = root / f"attempt-{len(list(root.glob('attempt-*'))) + 1:03d}"
        command = [sys.executable, "scripts/train_flow_head.py", "cache", "--data", str(data),
                   "--vision-config", str(self.out / "configs" / f"{job['id']}.yaml"),
                   "--out", str(attempt), "--batch-size", str(self.args.physical_batch),
                   "--chunk-size", str(job["config"]["model"]["chunk_size"]),
                   "--device", self.args.device, "--threads", str(self.args.threads),
                   "--episodes", *map(str, episodes)]
        result = self.command(command, self.out / "logs" / f"cache_{vision}_{attempt.name}.log",
                              stage=f"cache_{vision}")
        atomic_json(root / f"{attempt.name}.json", result)
        if result["returncode"] or not (attempt / "COMPLETE.json").is_file():
            raise RuntimeError(f"{vision} feature cache failed; see {result['log']}")
        return attempt

    def train_command(self, job: dict, output: Path, resume: Path | None, cache: Path | None) -> list[str]:
        cfg, train = job["config"], job["config"]["train"]
        if job["family"] == "act":
            command = [sys.executable, "-m", "real2sim.policy.train", "--config",
                       str(self.out / "configs" / f"{job['id']}.yaml"), "--data", cfg["data"],
                       "--out", str(output)]
        else:
            command = [sys.executable, "scripts/train_flow_head.py", "train", "--data", cfg["data"],
                       "--cache", str(cache), "--out", str(output), "--objective", job["family"],
                       "--examples", str(job["examples"]), "--schedule-examples",
                       str(train["schedule_steps"] * job["effective_batch"]),
                       "--batch-size", str(train["batch_size"]), "--grad-accum", str(train["grad_accum"]),
                       "--lr", str(cfg["model"]["optimizer_lr"]), "--weight-decay", "0.05",
                       "--seed", str(train["seed"]), "--device", self.args.device,
                       "--threads", str(self.args.threads), "--validate-every",
                       str(train["val_every"] * job["effective_batch"])]
            command += ["--keep-only-best-and-last"]
        if resume:
            command += ["--resume", str(resume)]
        return command

    def job_complete(self, job: dict, output: Path) -> bool:
        if job["family"] == "act":
            result = output / "training_result.json"
            return result.is_file() and read_json(result).get("steps") == job["config"]["train"]["steps"]
        report = output / "report.json"
        return ((output / "COMPLETE.json").is_file() and report.is_file() and
                read_json(report).get("examples") == job["examples"])

    def release_completed_storage(self, job: dict) -> None:
        """Drop only disposable, queue-owned artifacts after full requested exposure."""
        row = self.status["jobs"][job["id"]]
        output = Path(row["output"])
        if row.get("status") != "completed" or not self.job_complete(job, output):
            raise ValueError("storage release requires a completed full-budget job")
        root = self.out / "runs" / job["id"]
        removed = row.setdefault("released_artifacts", [])
        for checkpoint in root.glob("attempt-*/checkpoints/[0-9]*"):
            if checkpoint.is_symlink() or not checkpoint.resolve().is_relative_to(root.resolve()):
                raise ValueError("refusing to prune optimizer state outside queue-owned run")
            optimizer = checkpoint / "training_state.pt"
            if optimizer.is_file() and not optimizer.is_symlink():
                optimizer.unlink()
                removed.append(str(optimizer))
        row.update(optimizer_pruned=True, exact_resume_after_screen=False,
                   retention="best/last inference weights retained; completed optimizer state released")
        if job["family"] != "act":
            vision = job["config"]["model"]["vision_backbone"]
            pair = [j for j in self.plan["jobs"] if j["family"] != "act" and
                    j["config"]["model"]["vision_backbone"] == vision]
            if all(self.status["jobs"].get(j["id"], {}).get("status") == "completed" for j in pair):
                cache_root = self.out / "feature_caches" / vision
                for cache in cache_root.glob("attempt-*"):
                    if cache.is_symlink() or not cache.is_dir():
                        continue
                    released = []
                    for name in ("features.npy", "ticks.npy"):
                        path = cache / name
                        if path.is_file() and not path.is_symlink():
                            path.unlink()
                            released.append(name)
                    if released:
                        atomic_json(cache / "RELEASED.json", {
                            "at": now(), "removed": released, "reason": "paired full-budget heads completed",
                            "retained": ["manifest.json", "backbone.safetensors"],
                            "regenerate_features_before_further_head_training": True})
        self.write_status()

    def run_job(self, job: dict) -> None:
        name, root = job["id"], self.out / "runs" / job["id"]
        row = self.status["jobs"].setdefault(name, {"status": "pending", "attempts": []})
        if row.get("status") == "completed":
            if not self.job_complete(job, Path(row["output"])):
                raise ValueError(f"completed job artifacts missing or exposure differs: {name}")
            self.release_completed_storage(job)
            return
        root.mkdir(parents=True, exist_ok=True)
        cache = self.ensure_cache(job) if job["family"] != "act" else None
        for _ in range(self.args.max_attempts):
            if self.interrupted:
                raise InterruptedError("learning screen interrupted")
            # A runner may have crashed immediately after the trainer completed.
            for candidate in sorted(root.glob("attempt-*"), reverse=True):
                if self.job_complete(job, candidate):
                    row.update(status="completed", output=str(candidate), recovered=True)
                    self.write_status()
                    self.release_completed_storage(job)
                    return
            resume, exposure = latest_checkpoint(job, root)
            if exposure >= job["examples"]:
                row.update(status="incomplete", error="training budget reached; final evaluation/report incomplete")
                self.write_status()
                return
            attempt = root / f"attempt-{len(list(root.glob('attempt-*'))) + 1:03d}"
            command = self.train_command(job, attempt, resume, cache)
            row.update(status="running", output=str(attempt), resume=str(resume) if resume else None)
            self.write_status()
            result = self.command(command, self.out / "logs" / f"{name}_{attempt.name}.log", stage=name)
            row["attempts"].append(result)
            complete = result["returncode"] == 0 and self.job_complete(job, attempt)
            row.update(status="completed" if complete else "incomplete")
            self.write_status()
            if complete:
                self.release_completed_storage(job)
                return
            if self.interrupted:
                raise InterruptedError("learning screen interrupted")
            # Retry bounded timeouts from a durable checkpoint; errors/OOMs need inspection.
            checkpoint, progressed = latest_checkpoint(job, root)
            if not result["timed_out"] or checkpoint is None or progressed <= exposure:
                return

    def execute(self) -> int:
        self.prepare()
        if self.args.dry_run:
            print(json.dumps({"status": "prepared", "unique_jobs": len(self.plan["jobs"]),
                              "entries": len(self.plan["matrix_entries"]), "out": str(self.out)}))
            return 0
        self.status.update(status="running", started_at=now())
        self.write_status()
        for job in self.plan["jobs"]:
            try:
                self.run_job(job)
            except InterruptedError:
                self.status["status"] = "interrupted"
                self.write_status()
                return 130
            except Exception as error:
                self.status["jobs"].setdefault(job["id"], {}).update(status="failed", error=repr(error))
                self.write_status()
                # Preserve frozen-input and disk guarantees across every subsequent stage.
                self.verify_inputs()
        complete = all(self.status["jobs"].get(j["id"], {}).get("status") == "completed"
                       for j in self.plan["jobs"])
        self.status.update(status="completed" if complete else "incomplete", finished_at=now(),
                           active_stage=None, active_command=None, subprocess_pid=None)
        self.write_status()
        return 0 if complete else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("results/v2/screen_matrix_v4.json"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--physical-batch", type=int, default=2)
    parser.add_argument("--runtime-seconds", type=int, default=172800,
                        help="maximum wall seconds per cache/training attempt")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--minimum-free-gib", type=float, default=3)
    parser.add_argument("--defer-geometry-checks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.physical_batch, args.runtime_seconds, args.max_attempts, args.threads) < 1 or args.minimum_free_gib < 0:
        parser.error("batch, runtime, attempts and threads must be positive; minimum free space nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    runner = ScreenRunner(args)
    with queue_lock(args.out):
        signal.signal(signal.SIGTERM, runner.stop)
        signal.signal(signal.SIGINT, runner.stop)
        try:
            return runner.execute()
        except BaseException as error:
            if runner.status:
                runner.status.update(status="failed", error=repr(error), active_stage=None,
                                     active_command=None, subprocess_pid=None)
                runner.write_status()
            atomic_json(args.out / "FAILED.json", {"at": now(), "error": repr(error),
                        "physical_grasp_success": "not_evaluated"})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
