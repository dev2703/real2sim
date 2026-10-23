"""Training persists and selects the new common physical-motion validation report."""

import csv
import json

import numpy as np
import pytest
import torch

from real2sim.policy.train import train
from tests.integration.test_flow_head_smoke import tiny_data  # noqa: F401


@pytest.mark.integration
def test_v2_act_selects_executed_prefix_score_and_persists_units(tiny_data, tmp_path):  # noqa: F811
    data, _, manifest, _ = tiny_data
    manifest.update(contract_version="v2-test", action_representation="cartesian")
    # Give the canonical pose a valid orientation for physically meaningful baselines.
    (data / "manifest.json").write_text(json.dumps(manifest))
    cfg = {"name": "motion_metrics_fixture", "model": {
        "chunk_size": 3, "n_action_steps": 2, "vision_backbone": "resnet18",
        "pretrained_backbone_weights": None, "freeze_backbone": True, "dim_model": 32,
        "n_heads": 4, "dim_feedforward": 64, "n_encoder_layers": 1, "n_decoder_layers": 1,
        "use_vae": False, "dropout": 0., "optimizer_lr": 1e-4, "optimizer_lr_backbone": 1e-5},
        "train": {"seed": 3, "batch_size": 2, "steps": 2, "grad_clip_norm": 1.,
                  "log_every": 1, "val_every": 1, "save_every": 1,
                  "val_max_batches": 1, "num_workers": 0, "device": "cpu"}}
    torch.set_num_threads(1)
    out = tmp_path / "trained"
    result = train(cfg, data, out)
    reports = [json.loads((out / "validation" / f"{step:06d}.json").read_text()) for step in (1, 2)]
    assert all(r["valid_actions"] == 5 for r in reports)  # full episodes despite val_max_batches=1
    expected_step = int(np.argmin([r["offline_selection"]["score"] for r in reports])) + 1
    assert result["best_step"] == expected_step
    assert (out / "checkpoints/best").resolve().name == f"{expected_step:06d}"
    assert result["last_val"]["executed_prefix"]["scores"]["model"]["all"]["gripper_unit"] == "fraction of calibrated range"
    rows = list(csv.DictReader((out / "training_history.csv").open()))
    validation = [r for r in rows if r["split"] == "val"]
    assert len(validation) == 2 and all(r["offline_selection_score"] for r in validation)
