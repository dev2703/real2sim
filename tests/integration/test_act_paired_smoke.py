"""Real paired-image warm-start, backward, save/reload and parent immutability."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from real2sim.config import load_yaml
from real2sim.policy.act import load_checkpoint
from real2sim.policy.paired import file_sha256
from real2sim.policy.train import train


@pytest.mark.integration
@pytest.mark.skipif(not Path(".cache/paired_approach_3988577/train/manifest.json").exists(), reason="verified pair cache required")
def test_real_paired_warmstart_smoke():
    cfg = load_yaml("configs/act_joint_paired_approach.yaml")
    parent = Path(cfg["initial_checkpoint"]) / "model.safetensors"
    digest = file_sha256(parent)
    cfg["train"].update(num_workers=0, batch_size=2, device="mps" if torch.backends.mps.is_available() else "cpu")
    with tempfile.TemporaryDirectory(prefix="real2sim-paired-smoke-") as tmp:
        out = Path(tmp)
        result = train(cfg, Path(".cache/derived_joint"), out, smoke_steps=2)
        assert result["steps"] == 2 and result["initial_checkpoint"]["sha256"] == digest
        assert result["paired_images"]["probability"] == .5
        ck = (out / "checkpoints/last").resolve()
        policy, _, contract = load_checkpoint(ck, device="cpu")
        assert contract["training_provenance"]["paired_images"]["training_data_kind"].startswith("real_labels")
        assert all(torch.isfinite(p).all() for p in policy.parameters())
        state = torch.load(ck / "training_state.pt", map_location="cpu", weights_only=False)
        assert state["step"] == 2
        assert json.loads((ck / "run_info.json").read_text())["initial_checkpoint"]["sha256"] == digest
    assert file_sha256(parent) == digest
