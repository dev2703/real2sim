"""Run the SO-101 offline experiment queue serially with durable recovery records.

Completion means that training and audits finished. It never opens the geometry,
grasp-success or Panda gates. No physical robot is accessed by this runner.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import yaml

RUNNER_VERSION = "so101_queue_v1"
PROPRIO_ARMS = {
    "P1": "no_velocity", "P2a": "ema_velocity_05", "P2b": "ema_velocity_075",
    "P3": "joints", "P4": "canonical_joints", "H0": "history3",
}
BOUNDARIES = ("A0", "A1R", "A1D", *PROPRIO_ARMS, "A2", "A3", "A4", "F0")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def queue_lock(out: Path):
    """OS lock survives status-file replacement and is released after crashes."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".runner.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"an experiment runner already owns {out}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": now()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def act_config(backbone: str, data: Path, device: str, examples: int = 32000,
               effective_batch: int = 16, dim: int = 512, heads: int = 8) -> dict:
    """Match exposure and schedule fractions, leaving every nonvisual model fresh."""
    if examples < 128 or examples % 64 or effective_batch not in (16, 64):
        raise ValueError("examples must be >=128 and divisible by64; effective batch16 or64")
    if dim % heads or backbone not in ("resnet18", "dinov2") or device not in ("mps", "cuda"):
        raise ValueError("invalid architecture/device")
    template = "configs/act_cartesian_dinov2_pilot.yaml" if backbone == "dinov2" else "configs/act_cartesian.yaml"
    original = yaml.safe_load(Path(template).read_text())
    model = copy.deepcopy(original["model"])
    model.update(chunk_size=20, n_action_steps=2, dim_model=dim, n_heads=heads,
                 dim_feedforward=3200, n_encoder_layers=4, n_decoder_layers=1,
                 optimizer_lr=1e-4, optimizer_lr_backbone=1e-4, optimizer_weight_decay=.05,
                 attention_backend="sdpa", freeze_backbone=True)
    if backbone == "dinov2":
        model["dinov2_trainable_blocks"] = 0
    steps = examples // effective_batch
    # These anchor counts divide by both16 and64 exactly.
    warmup_examples = min(1600, examples // 4)
    return {"name": "fresh_frozen_visual_screen", "data": str(data), "model": model,
            "scheduler": {"warmup_steps": max(1, warmup_examples // effective_batch),
                          "start_factor": .001, "min_lr": 1e-7},
            "train": {"seed": 0, "batch_size": 16, "grad_accum": effective_batch // 16,
                      "steps": steps, "grad_clip_norm": 1., "log_every": max(1, 1024 // effective_batch),
                      "val_every": min(steps, max(1, 4096 // effective_batch)),
                      "save_every": min(steps, max(1, 16384 // effective_batch)),
                      "val_max_batches": None, "num_workers": 0, "device": device,
                      "precision": "bfloat16" if device == "cuda" else "float32",
                      "keep_only_best_and_last": True},
            "experiment": {"sampled_anchor_budget": examples, "effective_batch": effective_batch,
                           "initialization": "fresh nonvisual weights, seed0; pretrained vision frozen",
                           "schedule_clock": "optimizer steps scaled by effective batch; matching examples-seen fractions",
                           "training_data_kind": "real recordings only; identical images and Cartesian targets"}}


def choose_backbone(resnet: dict, dino: dict) -> dict:
    """DINO must improve active error and satisfy explicit closure/idle guardrails."""
    phases = ("arm_active", "arm_stationary", "observed_gripper_closure_proxy")
    if resnet.get("episodes") != dino.get("episodes") or resnet.get("valid_actions") != dino.get("valid_actions"):
        raise ValueError("backbone audits must score the same episodes/actions")
    values = {}
    for name, audit in (("resnet18", resnet), ("dinov2", dino)):
        if audit.get("split") != "val":
            raise ValueError("backbone selection requires inner validation, never historical validation")
        values[name] = {}
        for phase in phases:
            row = audit["scores"]["policy"][phase]
            if row.get("n", 0) <= 0:
                raise ValueError(f"no {phase} samples available for backbone selection")
            error = float(row["translation_vector_error_m"]["mean"])
            if not math.isfinite(error):
                raise ValueError("nonfinite backbone-selection metric")
            values[name][phase] = error
        gripper = float(audit["scores"]["policy"][phases[2]]["mae_raw_per_dim"][6])
        if not math.isfinite(gripper):
            raise ValueError("nonfinite closure gripper error")
        values[name]["closure_gripper_mae"] = gripper
    r, d = values["resnet18"], values["dinov2"]
    if any(resnet["scores"]["policy"][phase]["n"] != dino["scores"]["policy"][phase]["n"] for phase in phases):
        raise ValueError("backbone audits have different phase sample counts")
    criteria = {"active_translation_improves": d["arm_active"] < r["arm_active"],
                "stationary_translation_within_5pct": d["arm_stationary"] <= 1.05 * r["arm_stationary"],
                "closure_translation_within_5pct": d[phases[2]] <= 1.05 * r[phases[2]],
                "closure_gripper_mae_within_5pct": d["closure_gripper_mae"] <= 1.05 * r["closure_gripper_mae"]}
    return {"winner": "dinov2" if all(criteria.values()) else "resnet18", "criteria": criteria,
            "metrics": values, "selection_split": "inner validation", "seed": 0,
            "scientific_status": "single-seed frozen-feature screen; not a task-success result",
            "closure_timing_limit": "closure proxy translation/gripper MAE is scored; this is not a ground-truth contact-time metric"}


def compact_act_run(run: Path, owned_token: str) -> dict:
    """Retain one inference checkpoint; refuse unowned paths and symlink traversal."""
    if read_json(run / "RUNNER_OWNED.json").get("token") != owned_token:
        raise ValueError("refusing to compact a training run not owned by this queue")
    checkpoints = run / "checkpoints"
    if checkpoints.is_symlink():
        raise ValueError("checkpoint directory must not be a symlink")
    best_link = checkpoints / "best"
    best = best_link.resolve(strict=True)
    if best.parent != checkpoints.resolve() or not best.name.isdigit() or len(best.name) != 6:
        raise ValueError("best checkpoint points outside numbered owned checkpoint directories")
    required = ("model.safetensors", "config.json", "real2sim_contract.json", "stats.json", "run_info.json")
    if not all((best / name).is_file() for name in required):
        raise ValueError("best checkpoint is incomplete")
    numbered = [p for p in checkpoints.iterdir() if p.name.isdigit() and len(p.name) == 6]
    if any(p.is_symlink() or not p.is_dir() for p in numbered):
        raise ValueError("unexpected symlink or file among numbered checkpoints")
    removed = []
    optimizer = best / "training_state.pt"
    if optimizer.exists() or optimizer.is_symlink():
        optimizer.unlink()
        removed.append(str(optimizer.relative_to(run)))
    last = checkpoints / "last"
    if last.is_symlink():
        last.unlink()
        removed.append("checkpoints/last symlink")
    elif last.exists():
        raise ValueError("expected last to be a checkpoint symlink")
    for path in numbered:
        if path != best:
            removed.append(str(path.relative_to(run)))
            shutil.rmtree(path)
    report = {"best_checkpoint": str(best), "removed": removed,
              "inference_only": True, "resumable": False,
              "note": "optimizer state and other numbered checkpoints removed after successful training/audits; model weights retained"}
    atomic_json(run / "INFERENCE_ONLY.json", report)
    return report


class Runner:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.data = args.data_root.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        if args.flow_cache.resolve().is_relative_to(self.out):
            raise ValueError("feature cache must live outside the publishable output directory")
        self.token = hashlib.sha256(str(self.out).encode()).hexdigest()
        self.publication_failures = []
        self.current_phase = "initialization"
        self.frozen_files = {}

    def verify_inputs(self) -> None:
        for path, expected in self.frozen_files.items():
            if hash_file(Path(path)) != expected:
                raise ValueError(f"frozen experiment input changed: {path}")
        if shutil.disk_usage(self.out).free < 3 * 1024 ** 3:
            raise RuntimeError("less than 3 GiB free; stopped before another experiment command")

    def notify(self, event: str) -> None:
        """Only terminal invocation events wake the requesting thread, once each."""
        if not self.args.notification_json:
            return
        receipt = self.out / "notifications" / f"{event}.json"
        if receipt.exists() and read_json(receipt).get("returncode") == 0:
            return
        callback = read_json(self.args.notification_json)
        message = (f"The authorized SO-101 experiment runner reached terminal event {event}. "
                   f"Read {self.out}/STAGE_COMPLETE.json, COMPLETE.json or FAILED.json and the stage logs/results. "
                   "A stage completion is not grasp success. Check geometry and observation/label contracts before "
                   "extending training; preserve safety thresholds and keep Panda gated. Do not duplicate a running queue.")
        try:
            result = subprocess.run([callback["codex_bin"], "queue", "--thread", callback["thread_id"],
                                     "--message", message], capture_output=True, text=True, timeout=45)
            delivery = {"returncode": result.returncode, "stdout": result.stdout[-2000:],
                        "stderr": result.stderr[-2000:]}
        except (OSError, subprocess.TimeoutExpired) as error:
            delivery = {"returncode": -1, "error": repr(error)}
        atomic_json(receipt, {"at": now(), "event": event, **delivery})

    def stop_at(self, phase: str) -> bool:
        if self.args.stop_after != phase:
            return False
        self.current_phase = f"stopped_after_{phase}"
        atomic_json(self.out / "STAGE_COMPLETE.json", {"status": "STAGE_COMPLETE", "through": phase,
                    "completed_at": now(), "all_experiments_complete": False,
                    "geometry_gate_passed": False, "panda_gate_passed": False,
                    "note": "requested offline stage boundary reached; later experiments have not been claimed complete"})
        (self.out / "RUNNING.json").unlink(missing_ok=True)
        (self.out / "FAILED.json").unlink(missing_ok=True)
        self.publish()
        self.notify(f"stage_{phase}_complete")
        return True

    def publish(self) -> None:
        if not self.args.publish_repo:
            return
        from huggingface_hub import HfApi
        for attempt in range(3):
            try:
                api = HfApi()
                api.create_repo(repo_id=self.args.publish_repo, repo_type="model", private=True, exist_ok=True)
                if not api.repo_info(self.args.publish_repo, repo_type="model").private:
                    raise ValueError("refusing to upload experiment artifacts to a public repository")
                api.upload_folder(repo_id=self.args.publish_repo, repo_type="model", folder_path=str(self.out),
                                  allow_patterns=["*.json", "configs/**", "runs/**", "results/**", "logs/**", "stages/**"],
                                  ignore_patterns=[".runner.lock", "**/*.tmp", "**/training_state.pt"],
                                  commit_message=f"SO-101 experiment progress: {self.current_phase}")
                atomic_json(self.out / "publication.json", {"repo": self.args.publish_repo, "status": "published",
                                                           "at": now(), "phase": self.current_phase})
                return
            except Exception as error:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    failure = {"phase": self.current_phase, "at": now(), "error": repr(error)}
                    self.publication_failures.append(failure)
                    atomic_json(self.out / "publication.json", {"repo": self.args.publish_repo, "status": "failed",
                                                               "failures": self.publication_failures})
                    print(json.dumps({"publication_warning": failure}), flush=True)

    def command(self, phase: str, command: list[str], *, optional: bool = False) -> bool:
        self.verify_inputs()
        self.current_phase = phase
        stage = self.out / "stages" / phase
        complete = stage / "COMPLETE.json"
        if complete.is_file():
            saved = read_json(complete)
            if saved["command"] != command:
                raise ValueError(f"completed stage command changed: {phase}")
            print(f"already complete: {phase}", flush=True)
            return True
        log = self.out / "logs" / f"{phase}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        record = {"phase": phase, "command": command, "log": str(log), "runner_pid": os.getpid(),
                  "started_at": now(), "status": "RUNNING", "optional": optional}
        process = None
        try:
            with log.open("a") as handle:
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                           env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
                record["subprocess_pid"] = process.pid
                while True:
                    record["heartbeat_at"] = now()
                    atomic_json(stage / "RUNNING.json", record)
                    atomic_json(self.out / "RUNNING.json", record)
                    try:
                        code = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            if code:
                raise subprocess.CalledProcessError(code, command)
            self.verify_inputs()
            record.update(status="COMPLETE", completed_at=now(), returncode=0)
            atomic_json(complete, record)
            (stage / "FAILED.json").unlink(missing_ok=True)
            print(json.dumps({"phase": phase, "status": "COMPLETE", "log": str(log)}), flush=True)
            return True
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            record.update(status="FAILED", failed_at=now(), error=repr(error))
            atomic_json(stage / "FAILED.json", record)
            if optional and isinstance(error, Exception):
                print(json.dumps({"optional_stage_failed": phase, "log": str(log)}), flush=True)
                return False
            raise
        finally:
            (stage / "RUNNING.json").unlink(missing_ok=True)

    def write_config(self, arm: str, backbone: str, variant: str = "canonical",
                     batch: int = 16, dim: int = 512, heads: int = 8) -> Path:
        cfg = act_config(backbone, self.data / variant, self.args.device, self.args.examples, batch, dim, heads)
        cfg["name"] = arm
        target = self.out / "configs" / f"{arm}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and yaml.safe_load(target.read_text()) != cfg:
            raise ValueError(f"refusing to replace changed experiment config {target}")
        if not target.exists():
            target.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return target

    def train_act(self, arm: str, backbone: str, variant: str = "canonical",
                  batch: int = 16, dim: int = 512, heads: int = 8) -> Path:
        config = self.write_config(arm, backbone, variant, batch, dim, heads)
        run = self.out / "runs" / arm
        if run.exists() and not (run / "RUNNER_OWNED.json").is_file():
            raise ValueError(f"refusing existing unowned training output {run}")
        run.mkdir(parents=True, exist_ok=True)
        if not (run / "RUNNER_OWNED.json").exists():
            atomic_json(run / "RUNNER_OWNED.json", {"token": self.token, "arm": arm, "created_at": now()})
        command = [sys.executable, "-m", "real2sim.policy.train", "--config", str(config), "--out", str(run)]
        training_complete = self.out / "stages" / f"{arm}_train" / "COMPLETE.json"
        expected_steps = self.args.examples // batch
        last = run / "checkpoints" / "last"
        if not training_complete.exists() and not (run / "training_result.json").exists() and (last / "run_info.json").is_file():
            info = read_json(last / "run_info.json")
            if info["step"] == expected_steps:
                atomic_json(run / "training_result.json", {**info, "steps": expected_steps,
                            "requested_steps": expected_steps, "recovered_from_final_checkpoint": True})
        if not training_complete.exists() and (run / "training_result.json").exists():
            # Handle a kill after the child saved its result but before the runner flag.
            result = read_json(run / "training_result.json")
            if result.get("steps") != expected_steps:
                raise ValueError("cannot establish training completion from saved result")
            atomic_json(training_complete, {"status": "COMPLETE", "command": command,
                                            "recovered_from": str(run / "training_result.json")})
        if not training_complete.exists():
            if (last / "training_state.pt").is_file():
                # Trainer resumes an interrupted checkpoint; stage stores base command for repeatability.
                self.command(f"{arm}_train_resume", [*command, "--resume", str(last.resolve())])
                atomic_json(training_complete, {"status": "COMPLETE", "command": command,
                                                "recovered_from": str(last),
                                                "stochastic_continuation": "trainer restores optimizer/step; sample/RNG order may differ from uninterrupted run"})
            else:
                self.command(f"{arm}_train", command)
        else:
            self.command(f"{arm}_train", command)
        if read_json(run / "training_result.json").get("steps") != expected_steps:
            raise ValueError(f"{arm} did not finish its matched sampled-anchor budget")
        checkpoint = run / "checkpoints" / "best"
        audit_paths = {}
        for split in ("val", "historical_val"):
            output = self.out / "results" / f"{arm}.{split}.json"
            self.command(f"{arm}_audit_{split}", [sys.executable, "scripts/audit_experiment_policy.py",
                         "--checkpoint", str(checkpoint), "--data", str(self.data / variant),
                         "--out", str(output), "--device", self.args.device, "--split", split])
            if not output.is_file():
                raise FileNotFoundError(output)
            audit_paths[split] = str(output)
        if not (run / "INFERENCE_ONLY.json").exists():
            compact_act_run(run, self.token)
        atomic_json(run / "COMPLETE.json", {"status": "COMPLETE", "arm": arm, "audits": audit_paths,
                                           "checkpoint": str(checkpoint.resolve()), "inference_only": True,
                                           "grasp_success": None, "geometry_gate_passed": False})
        self.current_phase = f"{arm}_complete"
        self.publish()
        return Path(audit_paths["val"])

    def flow(self, dino_config: Path) -> dict:
        cache = self.args.flow_cache.resolve()
        if not (cache / "COMPLETE.json").is_file():
            if cache.exists():
                raise ValueError(f"incomplete feature cache retained; choose a new --flow-cache path: {cache}")
            self.command("F0_cache", [sys.executable, "scripts/train_flow_head.py", "cache", "--data",
                         str(self.data / "canonical"), "--vision-config", str(dino_config), "--out", str(cache),
                         "--device", self.args.device, "--batch-size", "8"])
        smoke = self.out / "runs" / "F0_profile_smoke"
        if not (smoke / "COMPLETE.json").is_file():
            if smoke.exists():
                raise ValueError(f"incomplete flow feasibility run retained at {smoke}")
            self.command("F0_profile_smoke", [sys.executable, "scripts/train_flow_head.py", "train", "--data",
                         str(self.data / "canonical"), "--cache", str(cache), "--objective", "flow",
                         "--examples", "64", "--batch-size", "16", "--grad-accum", "4", "--seed", "0",
                         "--out", str(smoke), "--device", self.args.device])
        smoke_profile = smoke / "profile"
        if not (smoke_profile / "COMPLETE.json").is_file():
            if smoke_profile.exists():
                raise ValueError(f"incomplete flow feasibility profile retained at {smoke_profile}")
            self.command("F0_pretraining_latency", [sys.executable, "scripts/train_flow_head.py", "profile",
                         "--data", str(self.data / "canonical"), "--checkpoint", str(smoke / "checkpoint"),
                         "--out", str(smoke_profile), "--device", self.args.device, "--repeats", "20"])
        latency = read_json(smoke_profile / "profile.json")
        flow_feasible = bool(latency["4_steps"]["under_200ms_p95"])
        feasibility = {"default_4_step_under_200ms_p95": flow_feasible,
                       "profile": str(smoke_profile / "profile.json"), "preflight_anchors": 64,
                       "preflight_excluded_from_comparative_32000_anchor_training": True,
                       "eight_step_under_200ms_p95": bool(latency["8_steps"]["under_200ms_p95"])}
        atomic_json(self.out / "results" / "F0_feasibility.json", feasibility)
        for objective in ("regression", "flow"):
            arm = f"F0_{objective}"
            if objective == "flow" and not flow_feasible:
                atomic_json(self.out / "stages" / arm / "GATED.json", {
                    "status": "GATED", "reason": "4-step frozen-vision inference p95 exceeds 200ms cadence",
                    "feasibility": feasibility, "training_launched": False})
                continue
            run = self.out / "runs" / arm
            if run.exists() and not (run / "COMPLETE.json").is_file():
                raise ValueError(f"incomplete F0 result retained; cannot silently overwrite: {run}")
            if not (run / "COMPLETE.json").is_file():
                self.command(arm, [sys.executable, "scripts/train_flow_head.py", "train", "--data",
                             str(self.data / "canonical"), "--cache", str(cache), "--objective", objective,
                             "--examples", str(self.args.examples), "--batch-size", "16", "--grad-accum", "4",
                             "--seed", "0", "--lr", "0.0001", "--weight-decay", "0.05", "--grad-clip", "1",
                             "--out", str(run), "--device", self.args.device])
            profile = run / "profile"
            if not (profile / "COMPLETE.json").is_file():
                if profile.exists():
                    raise ValueError(f"incomplete flow profile retained at {profile}")
                self.command(f"{arm}_profile", [sys.executable, "scripts/train_flow_head.py", "profile",
                             "--data", str(self.data / "canonical"), "--checkpoint", str(run / "checkpoint"),
                             "--out", str(profile), "--device", self.args.device, "--repeats", "20"])
            self.current_phase = f"{arm}_complete"
            self.publish()
        return feasibility

    def execute(self) -> None:
        manifest_paths = {name: self.data / name / "manifest.json"
                          for name in ["canonical", *PROPRIO_ARMS.values()]}
        checked = set()
        for manifest_path in manifest_paths.values():
            manifest = read_json(manifest_path)
            self.frozen_files[str(manifest_path)] = hash_file(manifest_path)
            if self.args.dry_run:
                continue
            for name, expected in manifest["files_sha256"].items():
                path = (manifest_path.parent / name).resolve(strict=True)
                if path not in checked:
                    if hash_file(path) != expected:
                        raise ValueError(f"data checksum mismatch: {path}")
                    checked.add(path)
                if path.suffix != ".npy":
                    self.frozen_files[str(path)] = expected
        source_files = list(Path("src/real2sim/policy").glob("*.py"))
        source_files += [Path(p) for p in ("src/real2sim/data/labels.py", "src/real2sim/data/resample.py",
                         "scripts/run_so101_experiments.py", "scripts/audit_experiment_policy.py",
                         "scripts/diag_motion_prediction_audit.py", "scripts/profile_act_attention.py",
                         "scripts/train_flow_head.py", "configs/act_cartesian.yaml",
                         "configs/act_cartesian_dinov2_pilot.yaml")]
        self.frozen_files.update({str(path.resolve()): hash_file(path) for path in source_files})
        identity = {"runner_version": RUNNER_VERSION, "device": self.args.device, "examples": self.args.examples,
                    "data_manifests": {key: hash_file(path) for key, path in manifest_paths.items()},
                    "frozen_files": self.frozen_files,
                    "no_panda": True, "profile_batches": self.args.profile_batches,
                    "skip_flow": self.args.skip_flow}
        plan_file = self.out / "QUEUE.json"
        if plan_file.exists() and read_json(plan_file)["identity"] != identity:
            raise ValueError("queue settings/data changed; use a separate output directory")
        plan = {"identity": identity, "created_at": now(), "seed": 0,
                "sequence": ["A0", "A1R", "A1D", "backbone_selection", *PROPRIO_ARMS, "A2", "A3", "A4", "F0"],
                "gates": {"recorded_physical_grasp": False, "learned_so101_success": False, "panda_allowed": False},
                "completion_semantics": "offline commands and audits finished; no grasp success established"}
        if not plan_file.exists():
            atomic_json(plan_file, plan)
        rconfig = self.write_config("A1R", "resnet18")
        dconfig = self.write_config("A1D", "dinov2")
        if self.args.dry_run:
            # Conditional arms are only examples until the backbone selection is observed.
            configs = {arm: str(self.write_config(f"planned_{arm}", "resnet18", variant))
                       for arm, variant in PROPRIO_ARMS.items()}
            for arm, dim, heads in (("A2", 512, 8), ("A3", 384, 8), ("A4", 384, 12)):
                configs[arm] = str(self.write_config(f"planned_{arm}", "resnet18", batch=64, dim=dim, heads=heads))
            atomic_json(self.out / "PLAN_READY.json", {"status": "NOT_STARTED", "examples": self.args.examples,
                                                      "conditional_config_examples": configs})
            print(json.dumps({"status": "PLAN_READY", "out": str(self.out)}), flush=True)
            return
        if (self.out / "COMPLETE.json").is_file():
            print(json.dumps({"status": "ALREADY_COMPLETE", "out": str(self.out)}), flush=True)
            return
        for name, config in (("resnet18", rconfig), ("dinov2", dconfig)):
            for batch in self.args.profile_batches:
                self.command(f"A0_{name}_batch{batch}", [sys.executable, "scripts/profile_act_attention.py",
                             "--config", str(config), "--out", str(self.out / "results" / f"A0_{name}_batch{batch}.json"),
                             "--device", self.args.device, "--batches", str(batch)], optional=batch != 16)
        self.publish()
        if self.stop_at("A0"):
            return
        resnet = self.train_act("A1R", "resnet18")
        if self.stop_at("A1R"):
            return
        dino = self.train_act("A1D", "dinov2")
        selection = choose_backbone(read_json(resnet), read_json(dino))
        selection["audits"] = {"resnet18": str(resnet), "dinov2": str(dino)}
        atomic_json(self.out / "results" / "backbone_selection.json", selection)
        if self.stop_at("A1D"):
            return
        winner = selection["winner"]
        for arm, variant in PROPRIO_ARMS.items():
            self.train_act(arm, winner, variant)
            if self.stop_at(arm):
                return
        for arm, dim, heads in (("A2", 512, 8), ("A3", 384, 8), ("A4", 384, 12)):
            self.train_act(arm, winner, batch=64, dim=dim, heads=heads)
            if self.stop_at(arm):
                return
        flow_result = None if self.args.skip_flow else self.flow(dconfig)
        if self.stop_at("F0"):
            return
        self.current_phase = "queue_complete"
        atomic_json(self.out / "COMPLETE.json", {"status": "COMPLETE", "completed_at": now(),
                    "winner_backbone": winner, "seed": 0, "examples_per_arm": self.args.examples,
                    "flow_skipped": self.args.skip_flow, "geometry_gate_passed": False,
                    "flow_feasibility": flow_result,
                    "learned_grasp_success": None, "panda_gate_passed": False,
                    "note": "offline screen completed; physical success and three-seed finalist confirmation remain separate gates"})
        (self.out / "FAILED.json").unlink(missing_ok=True)
        (self.out / "RUNNING.json").unlink(missing_ok=True)
        (self.out / "STAGE_COMPLETE.json").unlink(missing_ok=True)
        self.publish()
        self.notify("queue_complete")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cuda"), required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/experiments_20261016"))
    parser.add_argument("--data-root", type=Path, default=Path(".cache/experiments_20261016"))
    parser.add_argument("--flow-cache", type=Path, default=Path(".cache/experiments_20261016/dino_features"))
    parser.add_argument("--examples", type=int, default=32000)
    parser.add_argument("--profile-batches", type=int, nargs="+", default=[16])
    parser.add_argument("--publish-repo", help="private Hugging Face model repository for outputs only")
    parser.add_argument("--dry-run", action="store_true", help="write reviewable configs/queue without executing")
    parser.add_argument("--skip-flow", action="store_true", help="explicit partial-queue diagnostic; recorded in completion")
    parser.add_argument("--stop-after", choices=BOUNDARIES, help="stop at an explicit stage boundary; do not claim whole-queue completion")
    parser.add_argument("--notification-json", type=Path, help="optional local codex_bin/thread_id terminal callback metadata")
    args = parser.parse_args(argv)
    if args.examples < 128 or args.examples % 64:
        parser.error("--examples must be >=128 and divisible by64")
    if 16 not in args.profile_batches or any(batch not in (16, 32, 64) for batch in args.profile_batches):
        parser.error("--profile-batches must include16 and contain only16,32,64")
    if args.device == "mps" and args.profile_batches != [16]:
        parser.error("MPS screening is limited to physical batch16; effective64 uses accumulation")
    if args.stop_after == "F0" and args.skip_flow:
        parser.error("cannot stop after F0 while skipping F0")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    runner = Runner(args)
    def stop(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    with queue_lock(runner.out):
        try:
            runner.execute()
        except BaseException as error:
            atomic_json(runner.out / "FAILED.json", {"status": "FAILED", "phase": runner.current_phase,
                        "at": now(), "error": repr(error), "runner_pid": os.getpid(),
                        "grasp_success": None, "panda_gate_passed": False})
            (runner.out / "RUNNING.json").unlink(missing_ok=True)
            if not args.dry_run:
                runner.publish()
                runner.notify(f"failed_{runner.current_phase}")
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
