"""Run real2sim data preparation + ACT training on a Modal GPU (decision D-009).

    modal run scripts/modal_train.py --mode joint --smoke-steps 2          # cheap GPU smoke test
    modal run scripts/modal_train.py --mode cartesian --config configs/act_cartesian.yaml
    modal volume get real2sim-cache outputs/<run_name> outputs/             # fetch results

The dataset is downloaded on Modal from the pinned Hugging Face revision in configs/data.yaml
(iphone videos, parquet data and metadata only) and prepared with exactly the local code, so the
derived data are reproduced rather than uploaded. Everything persists on the ``real2sim-cache``
volume. GPU spend is bounded by ``--max-hours`` (function timeout).
"""

from __future__ import annotations

import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parents[1]
GPU = "A10G"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "lerobot[dataset]==0.6.1",
        "torch>=2.7,<2.12",
        "torchvision>=0.22,<0.27",
        "numpy>=2.0,<2.3",
        "mujoco==3.15.0",
        "av>=15.0.0,<16.0.0",
        "opencv-python-headless>=4.10",
        "pandas>=2.2",
        "pyarrow>=15",
        "pyyaml>=6",
        "scipy>=1.13",
        "huggingface-hub>=0.34",
    )
    .env({"PYTHONPATH": "/repo/src", "MUJOCO_GL": "egl"})
    .add_local_dir(REPO / "src", "/repo/src")
    .add_local_dir(REPO / "configs", "/repo/configs")
    .add_local_dir(REPO / "assets" / "menagerie" / "robotstudio_so101", "/repo/assets/menagerie/robotstudio_so101")
)
vol = modal.Volume.from_name("real2sim-cache", create_if_missing=True)
app = modal.App("real2sim-train", image=image)
CACHE = pathlib.Path("/cache")


@app.function(gpu=GPU, volumes={str(CACHE): vol}, timeout=6 * 3600, cpu=8, memory=32768)
def prepare_and_train(mode: str, config: str, run_name: str, smoke_steps: int | None, steps: int | None) -> dict:
    import os
    import subprocess

    from huggingface_hub import snapshot_download

    os.chdir("/repo")
    import yaml

    data_cfg = yaml.safe_load(open("configs/data.yaml"))
    src = data_cfg["source"]
    root = snapshot_download(
        repo_id=src["repo_id"],
        repo_type="dataset",
        revision=src["revision"],
        cache_dir=str(CACHE / "hf"),
        allow_patterns=["meta/*", "meta/**", "data/**", "videos/observation.images.iphone/**", "README.md"],
    )
    derived = CACHE / "derived" / f"{mode}_{src['revision'][:8]}"
    if not (derived / "manifest.json").is_file():
        subprocess.run(
            ["python", "-m", "real2sim.data.prepare", "--mode", mode, "--config", "configs/data.yaml",
             "--local-root", root, "--out-dir", str(derived)],
            check=True,
        )
        vol.commit()
    out = CACHE / "outputs" / run_name
    cmd = ["python", "-m", "real2sim.policy.train", "--config", config, "--data", str(derived), "--out", str(out),
           "--device", "cuda", "--num-workers", "6"]
    if smoke_steps:
        cmd += ["--smoke-steps", str(smoke_steps)]
    if steps:
        cmd += ["--steps", str(steps)]
    subprocess.run(cmd, check=True)
    vol.commit()
    hist = (out / "training_history.csv").read_text().splitlines()
    return {"out": f"outputs/{run_name}", "history_tail": hist[-6:], "derived": str(derived)}


@app.local_entrypoint()
def main(mode: str = "joint", config: str = "", run_name: str = "", smoke_steps: int = 0, steps: int = 0, max_hours: float = 3.0):
    config = config or ("configs/act_joint.yaml" if mode == "joint" else "configs/act_cartesian.yaml")
    run_name = run_name or f"modal_{mode}{'_smoke' if smoke_steps else ''}"
    fn = prepare_and_train.with_options(timeout=int(max_hours * 3600))
    res = fn.remote(mode, config, run_name, smoke_steps or None, steps or None)
    print(res)
