"""Actual real-batch L2/Huber optimisation, metadata, saved-weight inference and error checks."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from real2sim.config import load_yaml
from real2sim.policy.check import check
from real2sim.policy.train import train


@pytest.mark.integration
@pytest.mark.skipif(not Path(".cache/derived_joint/manifest.json").exists(), reason="real data required")
@pytest.mark.parametrize("kind", ["l2", "huber"])
def test_real_loss_train_and_reload(tmp_path, kind):
    cfg = load_yaml("configs/act_joint_augmented.yaml")
    cfg["reconstruction_loss"] = {"kind": kind, "huber_delta": 1.0}
    cfg["train"].update(device="mps" if torch.backends.mps.is_available() else "cpu",
                        num_workers=0, keep_only_best_and_last=True)
    out = tmp_path / kind
    result = train(cfg, Path(".cache/derived_joint"), out, smoke_steps=2)
    assert result["steps"] == 2 and np.isfinite(result["best_val_l1"])
    best = out / "checkpoints/best"
    assert json.loads((best / "run_info.json").read_text())["reconstruction_loss"]["kind"] == kind
    metrics = check(best, Path(".cache/derived_joint"), "cpu", max_batches=1)
    assert metrics["all_finite"] and metrics["contract_ok"]
    assert metrics["reconstruction_loss"]["kind"] == kind
    assert metrics["val_open_loop_rmse_normalised"] >= metrics["val_open_loop_l1_normalised"]
    assert np.isfinite(metrics["val_open_loop_p95_abs_normalised"])
    with (out / "training_history.csv").open() as f:
        history = [row for row in csv.DictReader(f) if row["split"] == "train"]
    assert len(history) == 2
    for row in history:
        # Training adds in float32; the CSV components are rounded Python floats.
        assert float(row["loss"]) == pytest.approx(float(row["reconstruction"]) + float(row["weighted_kl"]), rel=2e-7, abs=2e-6)
    cfg["reconstruction_loss"] = {"kind": "l1"}
    with pytest.raises(ValueError, match="different reconstruction loss"):
        train(cfg, Path(".cache/derived_joint"), out, smoke_steps=3, resume=best)
