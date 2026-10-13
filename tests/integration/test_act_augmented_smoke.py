"""Real-batch training with augmentation, early stop, checkpoint choice and reload."""

from pathlib import Path

import pytest
import torch

from real2sim.config import load_yaml
from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.train import train


@pytest.mark.integration
@pytest.mark.skipif(not Path(".cache/derived_joint/manifest.json").exists(), reason="real data required")
def test_augmented_real_training_best_save_and_early_stop(tmp_path):
    cfg = load_yaml("configs/act_joint_augmented.yaml")
    # Zero LR keeps eval loss identical: the second validation must trigger patience=1.
    cfg["model"].update(optimizer_lr=0.0, optimizer_lr_backbone=0.0)
    cfg["train"].update(steps=3, val_every=1, save_every=3, log_every=1, val_max_batches=1,
                        early_stopping_patience=1, early_stopping_min_steps=2, num_workers=0,
                        device="mps" if torch.backends.mps.is_available() else "cpu")
    result = train(cfg, Path(".cache/derived_joint"), tmp_path / "aug")
    assert result["stopped_early"] and result["steps"] == 2
    checkpoints = tmp_path / "aug/checkpoints"
    assert (checkpoints / "best").resolve().name == "000001"
    assert (checkpoints / "last").resolve().name == "000002"
    policy, norm, contract = load_checkpoint(checkpoints / "best", device="cpu")
    ds = DerivedTickDataset(".cache/derived_joint", contract["split"]["val"], 20, norm)
    batch = {k: v[None] for k, v in ds[0].items() if k.startswith("observation.")}
    with torch.inference_mode():
        prediction = policy.predict_action_chunk(batch)
    assert prediction.shape == (1, 20, 6) and torch.isfinite(prediction).all()
