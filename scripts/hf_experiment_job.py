"""Bounded private Hugging Face experiment job and terminal-only Codex callback.

Bundle/upload are separate from paid launch. A launch reserves its full timeout
cost against the user's $10 cumulative budget, even if it ends early. Credentials
are read through huggingface_hub and passed as Jobs secrets, never saved in files.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "outputs/experiments_20261016_cloud"
REPO = "bottowork/real2sim-experiments-20261016"
TERMINAL = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}


def utc():
    return datetime.now(UTC).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reserve_cost(ledger, hourly, timeout_minutes):
    if not math.isfinite(hourly) or hourly <= 0 or not 1 <= timeout_minutes <= 300:
        raise ValueError("invalid price or timeout; maximum launch duration is five hours")
    maximum = math.ceil(timeout_minutes * hourly / 60 * 100) / 100
    used = sum(entry["maximum_usd"] for entry in ledger.get("reservations", []))
    if used + maximum > 10:
        raise ValueError(f"job would exceed the $10 cap: reserved={used:.2f}, new={maximum:.2f}")
    return {"at_utc": utc(), "hourly_usd": hourly, "timeout_minutes": timeout_minutes,
            "maximum_usd": maximum}


def bundle(kind):
    destination = JOB / f"{kind}.tar.gz"
    if destination.exists():
        raise FileExistsError(destination)
    members = []
    if kind == "data":
        base = ROOT / ".cache/derived_cartesian"
        members.extend((base / name, f".cache/derived_cartesian/{name}")
                       for name in ("images_iphone.npy", "manifest.json", "arrays.npz", "stats.json"))
        for path in sorted((ROOT / ".cache/experiments_20261016").glob("*/*")):
            if path.name in {"arrays.npz", "manifest.json", "stats.json", "images_iphone.npy"}:
                members.append((path, str(path.relative_to(ROOT))))
        prep = ROOT / ".cache/experiments_20261016/preparation.json"
        members.append((prep, str(prep.relative_to(ROOT))))
        snapshot = ROOT / ".cache/huggingface/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056"
        for name in ("model.safetensors", "config.json"):
            members.append((snapshot / name, str((snapshot / name).relative_to(ROOT))))
        resnet = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
        members.append((resnet, ".cache/torch/hub/checkpoints/" + resnet.name))
        for section in ("train", "val"):
            path = ROOT / f".cache/paired_cartesian_refit_20261015/{section}/ticks.npy"
            members.append((path, str(path.relative_to(ROOT))))
    else:
        for directory in ("src", "scripts", "configs"):
            members.extend((p, str(p.relative_to(ROOT))) for p in sorted((ROOT / directory).rglob("*"))
                           if p.is_file() and p.suffix in {".py", ".yaml", ".json"} and "__pycache__" not in p.parts)
        members.extend((ROOT / name, name) for name in ("pyproject.toml", "uv.lock"))
    manifest = {}
    temporary = destination.with_suffix(".partial")
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for path, name in members:
                if path.name == "images_iphone.npy" and "experiments_20261016" in path.parts:
                    info = tarfile.TarInfo(name)
                    info.type = tarfile.SYMTYPE
                    info.linkname = "../../derived_cartesian/images_iphone.npy"
                    archive.addfile(info)
                    manifest[name] = {"symlink": info.linkname}
                else:
                    # Dereference trusted cache symlinks into portable regular weight files.
                    info = archive.gettarinfo(str(path.resolve()), arcname=name)
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                    manifest[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    temporary.replace(destination)
    report = {"at_utc": utc(), "archive": str(destination), "sha256": digest(destination),
              "bytes": destination.stat().st_size, "members": manifest}
    write(JOB / f"{kind}_manifest.json", report)
    return report


BOOTSTRAP = r'''
import os, subprocess, sys, tarfile
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'huggingface_hub', 'uv'], check=True)
from huggingface_hub import hf_hub_download
root = '/workspace/real2sim'
os.makedirs(root, exist_ok=True)
for filename in ('data.tar.gz', 'code.tar.gz'):
    path = hf_hub_download(os.environ['EXPERIMENT_REPO'], 'inputs/' + filename,
                          repo_type='dataset', revision=os.environ['EXPERIMENT_REVISION'])
    with tarfile.open(path) as archive:
        archive.extractall(root, filter='data')
os.chdir(root)
subprocess.run(['uv', 'sync', '--frozen', '--no-dev'], check=True)
os.execv(root + '/.venv/bin/python', [root + '/.venv/bin/python', '-u',
    'scripts/run_so101_experiments.py', '--device', 'cuda',
    '--out', 'outputs/experiments_20261016', '--publish-repo', os.environ['EXPERIMENT_REPO']])
'''


def upload(kind):
    from huggingface_hub import HfApi

    api = HfApi()
    if kind == "data":
        api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(REPO, repo_type="dataset").private:
        raise RuntimeError("experiment repository must be private")
    archive = JOB / f"{kind}.tar.gz"
    manifest = json.loads((JOB / f"{kind}_manifest.json").read_text())
    if digest(archive) != manifest["sha256"]:
        raise ValueError("input archive changed")
    result = api.upload_file(path_or_fileobj=archive, path_in_repo=f"inputs/{kind}.tar.gz",
                             repo_id=REPO, repo_type="dataset", commit_message=f"Private {kind} experiment snapshot")
    write(JOB / f"{kind}_upload.json", {"repo": REPO, "revision": result.oid, "sha256": manifest["sha256"]})


def launch(minutes):
    import httpx
    from huggingface_hub import HfApi, get_token

    if (JOB / "launch.json").exists():
        raise FileExistsError("job already launched; refusing duplicate compute")
    if not all((JOB / f"{kind}_upload.json").is_file() for kind in ("data", "code")):
        raise ValueError("both input snapshots must be uploaded before launch")
    api = HfApi()
    if not api.repo_info(REPO, repo_type="dataset").private:
        raise ValueError("repository is no longer private")
    response = httpx.get("https://huggingface.co/api/jobs/hardware", timeout=30)
    response.raise_for_status()
    hardware = next(row for row in response.json() if row["name"] == "a10g-small")
    if hardware["unitLabel"] != "minute":
        raise ValueError("unrecognized pricing unit")
    ledger_path = JOB / "budget.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"cap_usd": 10, "reservations": []}
    reservation = reserve_cost(ledger, hardware["unitCostUSD"] * 60, minutes)
    ledger["reservations"].append(reservation)
    # Reserve before the request: an ambiguous network failure cannot silently cause a second charge.
    write(ledger_path, ledger)
    write(JOB / "launch_pending.json", {"at_utc": utc(), "reservation": reservation})
    revision = json.loads((JOB / "code_upload.json").read_text())["revision"]
    job = api.run_job(
        image="python:3.12-bookworm", command=["python", "-u", "-c", BOOTSTRAP],
        flavor="a10g-small", timeout=f"{minutes}m", name="so101-five-experiments-20261016",
        env={"EXPERIMENT_REPO": REPO, "EXPERIMENT_REVISION": revision,
             "TORCH_HOME": "/workspace/real2sim/.cache/torch", "TOKENIZERS_PARALLELISM": "false",
             "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
        secrets={"HF_TOKEN": get_token()}, labels={"project": "real2sim", "robot": "so101"},
    )
    record = {"id": job.id, "url": job.url, "namespace": job.owner.name, "repo": REPO,
              "revision": revision, "at_utc": utc(), "reservation": reservation}
    write(JOB / "launch.json", record)
    print(json.dumps(record))


def notify(message):
    receipt = JOB / "delivery.json"
    if receipt.exists() and json.loads(receipt.read_text()).get("returncode") == 0:
        return
    callback = json.loads((JOB / "notification.json").read_text())
    try:
        result = subprocess.run([callback["codex_bin"], "queue", "--thread", callback["thread_id"],
                                 "--message", message], capture_output=True, text=True, timeout=45)
        delivery = {"returncode": result.returncode, "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        delivery = {"returncode": -1, "error": str(exc)}
    write(receipt, {"at_utc": utc(), **delivery})


def watch():
    from huggingface_hub import HfApi, snapshot_download

    launch_info = json.loads((JOB / "launch.json").read_text())
    api = HfApi()
    errors = 0
    with (JOB / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (JOB / "STOP").exists():
            try:
                job = api.inspect_job(job_id=launch_info["id"], namespace=launch_info["namespace"])
                stage = getattr(job.status.stage, "value", job.status.stage)
                state = {"at_utc": utc(), "pid": os.getpid(), "job_id": job.id,
                         "stage": stage, "message": job.status.message, "poll_interval_s": 60}
                write(JOB / "status.json", state)
                errors = 0
                if stage in TERMINAL:
                    logs = list(api.fetch_job_logs(job_id=job.id, namespace=launch_info["namespace"], tail=150))
                    (JOB / "terminal.log").write_text("\n".join(str(line) for line in logs))
                    snapshot_download(REPO, repo_type="dataset", local_dir=JOB / "download",
                                      allow_patterns=["outputs/**", "results/**"])
                    success = stage == "COMPLETED" and (JOB / "download/outputs/experiments_20261016/COMPLETE.json").is_file()
                    write(JOB / ("COMPLETE.json" if success else "FAILED.json"), state)
                    notify("The authorized SO-101 experiment cloud job finished or failed. Read outputs/experiments_20261016_cloud/status.json, terminal.log and downloaded results. Verify the complete top-level experiment flags, review active/phase errors and compare all five tracks. Repair failures without duplicating running jobs or overwriting results. Preserve all safety thresholds. SO-101 recorded physical replay and learned closed-loop success must pass before Panda. Hugging Face cumulative spend cap is $10; budget.json conservatively reserves each launched job's full timeout. Continue the authorized experiments from evidence, and use terminal-only callbacks during training rather than token polling.")
                    return
            except Exception as exc:
                errors += 1
                write(JOB / "monitor_error.json", {"at_utc": utc(), "error": type(exc).__name__, "consecutive": errors})
                if errors >= 10:
                    notify("SO-101 cloud monitor has failed ten consecutive checks. Read outputs/experiments_20261016_cloud/monitor_error.json and launch.json, verify remote job state and restore monitoring. Do not relaunch blindly; the existing GPU job may still be running. Keep cumulative Hugging Face spending within $10 and Panda gated on SO-101 success.")
                    return
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bundle-data", "bundle-code", "upload-data", "upload-code", "launch", "watch", "configure-callback"))
    parser.add_argument("--minutes", type=int, default=300)
    args = parser.parse_args()
    JOB.mkdir(parents=True, exist_ok=True)
    if args.command.startswith("bundle-"):
        result = bundle(args.command.removeprefix("bundle-"))
        print(json.dumps({k: v for k, v in result.items() if k != "members"}))
    elif args.command.startswith("upload-"):
        upload(args.command.removeprefix("upload-"))
    elif args.command == "launch":
        with (JOB / "launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (JOB / "launch_pending.json").exists():
                raise RuntimeError("launch request already attempted; inspect remote jobs before retrying")
            launch(args.minutes)
    elif args.command == "watch":
        watch()
    else:
        thread, binary = os.environ.get("CODEX_THREAD_ID"), shutil.which("codex")
        if not thread or not binary:
            raise RuntimeError("current thread/codex binary unavailable")
        write(JOB / "notification.json", {"thread_id": thread, "codex_bin": binary})


if __name__ == "__main__":
    main()
