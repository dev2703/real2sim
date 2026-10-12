"""G1 smoke test on REAL derived data: forward/backward, finite loss, save, resume, reload, infer.

Requires ``.cache/derived_joint`` (build with ``python -m real2sim.data.prepare --mode joint``);
skipped otherwise so CI without the dataset stays green but honest about what it ran.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from real2sim.config import load_yaml
from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.train import train

DATA = Path(".cache/derived_joint")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (DATA / "manifest.json").is_file(), reason="real derived dataset not built"),
]


def test_real_train_save_resume_reload_infer(tmp_path):
    cfg = load_yaml("configs/act_joint.yaml")
    out = tmp_path / "run"
    r1 = train(cfg, DATA, out, smoke_steps=2, overrides={"num_workers": 0})
    assert r1["last_val"] is not None and np.isfinite(r1["last_val"]["l1"])
    ck2 = out / "checkpoints" / "000002"
    assert (ck2 / "model.safetensors").is_file() and (ck2 / "training_state.pt").is_file()

    # resume to step 3 from the saved optimizer state
    r2 = train(cfg, DATA, out, smoke_steps=3, resume=ck2, overrides={"num_workers": 0})
    assert (out / "checkpoints" / "000003" / "model.safetensors").is_file()
    assert r2["steps"] == 3
    import csv

    with (out / "training_history.csv").open() as f:
        train_steps = [int(r["step"]) for r in csv.DictReader(f) if r["split"] == "train"]
    assert train_steps == sorted(set(train_steps)) == [1, 2, 3], train_steps  # no duplicates on resume
    assert (out / "checkpoints" / "last").resolve().name == "000003"

    policy, norm, contract = load_checkpoint(out / "checkpoints" / "000003", device="cpu")
    assert contract["camera_keys"] == ["observation.images.iphone"]
    assert contract["image_shape_chw"] == [3, 192, 256]
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert contract["split"]["sha256"] == manifest["split"]["sha256"]

    ds = DerivedTickDataset(DATA, manifest["split"]["val"], chunk_size=policy.config.chunk_size, normalizer=norm)
    batch = torch.utils.data.default_collate([ds[0], ds[100]])
    chunk = policy.predict_action_chunk({k: v for k, v in batch.items() if k.startswith("observation.")})
    assert chunk.shape == (2, 20, 6)
    raw = norm.unaction(chunk)
    assert torch.isfinite(raw).all()
    policy.reset()
    a = policy.select_action({k: v for k, v in batch.items() if k.startswith("observation.")})
    assert a.shape == (2, 6)
