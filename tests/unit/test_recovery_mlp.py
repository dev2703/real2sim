"""Recovery MLP (D-005), correction dataset and training loop on a synthetic correction file.

The synthetic rows exist only to exercise the code path; they are not correction data and the
resulting weights are discarded (plan M6 step 4 collects real corrections later)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from real2sim.config import load_yaml
from real2sim.recovery.dataset import ALL_COLUMNS, EXPERT_COLUMNS, CorrectionDataset
from real2sim.recovery.detector import FEATURE_NAMES
from real2sim.recovery.mlp import MLPSpec, RecoveryMLP
from real2sim.recovery.train_mlp import main as train_main
from real2sim.recovery.train_mlp import train

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "recovery.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_yaml(CONFIG)


def synthetic_corrections(n_rows: int = 200, n_episodes: int = 8, seed: int = 0, provenance="scripted_oracle_sim") -> pd.DataFrame:
    """Rows where the expert action is a smooth function of the features plus noise."""
    rng = np.random.default_rng(seed)
    per = n_rows // n_episodes
    rows = []
    w = rng.normal(size=(29, 7)) * 0.02
    for ep in range(n_episodes):
        start, end = 3, per - 3
        for t in range(per):
            x = rng.normal(size=29)
            x[13:16] = rng.uniform(-0.1, 0.1, 3)
            x[19:21] = rng.uniform(0.5, 1.0, 2)
            x[28] = float(rng.integers(0, 2))
            y = np.tanh(x @ w)
            y[:3] *= 0.03
            y[3:6] *= 0.3
            y[6] = 0.5 + 0.5 * y[6]
            y += rng.normal(size=7) * 1e-3
            row = {
                "correction_episode_id": ep,
                "tick": t,
                "controller_source": "ACT_DET_REC",
                "intervention_start": start,
                "intervention_end": end,
                "reason": "ball_rolled_away",
                "success": True,
                "provenance": provenance,
            }
            row.update({c: float(v) for c, v in zip(EXPERT_COLUMNS, y, strict=True)})
            row.update({c: float(v) for c, v in zip(FEATURE_NAMES, x, strict=True)})
            rows.append(row)
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


# ----------------------------------------------------------------------------- model
def test_spec_matches_d005(cfg):
    spec = MLPSpec.from_config(cfg)
    assert spec.hidden == (256, 256, 128) and spec.input_dim == 29 and spec.output_dim == 7
    assert spec.dropout == 0.1 and spec.input_layernorm and spec.activation == "relu"
    model = RecoveryMLP(spec)
    n_params = sum(p.numel() for p in model.parameters())
    assert 100_000 < n_params < 120_000  # D-005: about 108 k
    assert isinstance(model.net[0], torch.nn.LayerNorm)


def test_forward_shape_and_bounds(cfg):
    model = RecoveryMLP(MLPSpec.from_config(cfg))
    model.set_normalisation(np.zeros(29), np.ones(29), np.zeros(7), np.ones(7) * 100.0)  # huge y_std forces saturation
    x = torch.randn(64, 29) * 10
    y = model(x)
    assert y.shape == (64, 7)
    lo, hi = model.spec.bounds()
    assert torch.all(y >= lo - 1e-7) and torch.all(y <= hi + 1e-7)
    assert torch.any(y[:, :3].abs() == 0.03) or torch.any(y[:, 3:6].abs() == 0.3)  # clamp is active
    assert torch.all(y[:, 6] >= 0) and torch.all(y[:, 6] <= 1)


def test_save_load_round_trip_and_name_mismatch(cfg, tmp_path):
    model = RecoveryMLP(MLPSpec.from_config(cfg))
    model.set_normalisation(np.arange(29) * 0.1, np.ones(29) * 0.5, np.ones(7) * 0.01, np.ones(7) * 0.02)
    model.eval()
    x = torch.randn(5, 29)
    y = model.predict(x, FEATURE_NAMES)
    p = model.save(tmp_path / "m.pt")
    loaded = RecoveryMLP.load(p, feature_names=FEATURE_NAMES)
    assert loaded.feature_names == list(FEATURE_NAMES) and loaded.action_names == model.action_names
    assert torch.allclose(loaded.x_mean, model.x_mean) and torch.allclose(loaded.y_std, model.y_std)
    assert torch.allclose(loaded.predict(x, FEATURE_NAMES), y)

    wrong = list(FEATURE_NAMES)
    wrong[0], wrong[1] = wrong[1], wrong[0]
    with pytest.raises(ValueError, match="feature name"):
        RecoveryMLP.load(p, feature_names=wrong)
    with pytest.raises(ValueError, match="feature name"):
        loaded.predict(x, wrong)
    with pytest.raises(ValueError, match="feature name"):
        loaded.predict(x, FEATURE_NAMES[:28] + ["gt_ball_x"])
    with pytest.raises(ValueError):
        loaded(torch.randn(2, 30))


def test_constant_columns_are_not_amplified_by_normalisation(cfg):
    """Confirmed defect: a 1e-6 std floor mapped a tiny deviation on a constant training column
    (bowl_conf == 1.0, gripper_contact == 0) to |z| ~ 1e4, dominating the LayerNorm."""
    model = RecoveryMLP(MLPSpec.from_config(cfg))
    x_std = np.full(29, 0.1)
    x_std[20] = 0.0  # bowl_conf constant
    x_std[28] = 0.0  # gripper_contact constant
    x_std[0] = 1e-5  # ee_x barely varying -> floored, not exploded
    x_mean = np.zeros(29)
    x_mean[20] = 1.0
    y_std = np.full(7, 0.01)
    y_std[6] = 0.0  # expert gripper constant
    model.set_normalisation(x_mean, x_std, np.zeros(7), y_std)
    assert model.x_const.tolist() == [i in (20, 28) for i in range(29)]
    assert model.y_const.tolist() == [i == 6 for i in range(7)]
    assert float(model.x_std[20]) == 1.0 and float(model.x_std[28]) == 1.0 and float(model.y_std[6]) == 1.0
    assert float(model.x_std[0]) == pytest.approx(1e-3)
    assert model.constant_columns() == {"features": ["bowl_conf", "gripper_contact"], "actions": ["gripper_target"]}

    # realistic companions: every varying feature at a typical |z| ~ 1 spread
    x = torch.as_tensor(np.random.default_rng(0).normal(size=(1, 29)) * 0.1, dtype=torch.float32)
    x[0, 0] = 0.0
    x[0, 20] = 0.99  # inference: bowl_conf 0.99 instead of the constant 1.0
    x[0, 28] = 1.0  # inference: gripper_contact 1 instead of the constant 0
    z = (x - model.x_mean) / model.x_std
    assert float(z[0, 20]) == pytest.approx(-0.01) and float(z[0, 28]) == 1.0
    with torch.no_grad():
        ln = model.net[0](z)
    # before the fix feature 28 was z = 1e6 and LayerNorm gave ~5.3 on it and < 0.2 elsewhere
    assert float(ln[0, 28].abs()) < 2.0 and float(ln[0, :28].abs().max()) > 1.0

    # the prediction stays essentially unchanged by the 0.01 bowl_conf deviation
    model.eval()
    base = x.clone()
    base[0, 20] = 1.0
    diff = (model.predict(x, FEATURE_NAMES) - model.predict(base, FEATURE_NAMES)).abs().max()
    assert float(diff) < 1e-3
    # target normalisation of a constant expert column passes the raw residual through
    y = torch.zeros(1, 7)
    y[0, 6] = 0.02
    assert float(model.normalise_y(y)[0, 6]) == pytest.approx(0.02)

    with pytest.raises(ValueError):
        model.set_normalisation(x_mean, x_std, np.zeros(7), y_std, std_floor=1e-7, const_tol=1e-6)
    with pytest.raises(ValueError):
        model.set_normalisation(x_mean, np.full(29, -1.0), np.zeros(7), y_std)


def test_training_with_constant_columns_keeps_loss_finite_and_logs_them(cfg, tmp_path):
    df = synthetic_corrections(120, 6)
    df["bowl_conf"] = 1.0
    df["gripper_contact"] = 0.0
    df["expert_gripper_target"] = 0.8
    ds = CorrectionDataset(df)
    cfg_fast = dict(cfg)
    cfg_fast["mlp"] = dict(cfg["mlp"], batch_size=32, max_epochs=5)
    log = train(cfg_fast, ds, tmp_path / "out")
    assert np.all(np.isfinite(log["train_loss"])) and np.isfinite(log["best_heldout_loss"])
    assert log["constant_columns"] == {"features": ["bowl_conf", "gripper_contact"], "actions": ["gripper_target"]}
    loaded = RecoveryMLP.load(tmp_path / "out" / "recovery_mlp.pt", feature_names=FEATURE_NAMES)
    assert loaded.constant_columns() == log["constant_columns"]
    x = torch.as_tensor(ds.x[:1], dtype=torch.float32)
    x2 = x.clone()
    x2[0, 28] = 1.0  # never-seen gripper_contact=1 must not blow the input up
    z = (x2 - loaded.x_mean) / loaded.x_std
    assert float(z.abs().max()) < 10.0


# --------------------------------------------------------------------------- dataset
def test_dataset_builds_xy_and_splits_by_episode(cfg, tmp_path):
    df = synthetic_corrections()
    path = tmp_path / "c.csv"
    df.to_csv(path, index=False)
    ds = CorrectionDataset.from_file(path)
    assert ds.x.shape[1] == 29 and ds.y.shape[1] == 7
    assert ds.n_dropped_outside_intervention == 8 * 5  # ticks 0,1,2 before and 23,24 after the window
    assert len(ds) == 200 - ds.n_dropped_outside_intervention
    train_mask, held_mask, split = ds.split(0.25, seed=0)
    assert len(split.heldout_ids) == 2 and not set(split.heldout_ids) & set(split.train_ids)
    ids = np.array(ds.episode_ids)
    assert set(ids[held_mask]) == set(split.heldout_ids) and set(ids[train_mask]) == set(split.train_ids)
    assert ds.split(0.25, seed=0)[2].heldout_ids == split.heldout_ids  # deterministic
    pq = tmp_path / "c.parquet"
    df.to_parquet(pq, index=False)
    assert len(CorrectionDataset.from_file(pq)) == len(ds)


def test_dataset_rejects_bad_provenance_and_missing_columns():
    df = synthetic_corrections(n_rows=40, n_episodes=2)
    df.loc[5, "provenance"] = "made_up_by_agent"
    with pytest.raises(ValueError, match="provenance"):
        CorrectionDataset(df)
    df = synthetic_corrections(n_rows=40, n_episodes=2).drop(columns=["expert_dx"])
    with pytest.raises(ValueError, match="missing columns"):
        CorrectionDataset(df)
    df = synthetic_corrections(n_rows=40, n_episodes=2)
    df["intervention_start"] = 10**6
    with pytest.raises(ValueError, match="intervention"):
        CorrectionDataset(df)


# ---------------------------------------------------------------------------- training
def test_training_decreases_loss_and_early_stops(cfg, tmp_path):
    ds = CorrectionDataset(synthetic_corrections(200, 8))
    cfg_fast = dict(cfg)
    cfg_fast["mlp"] = dict(cfg["mlp"], batch_size=32, max_epochs=400, lr=1e-3)
    cfg_fast["mlp"]["early_stopping"] = dict(cfg["mlp"]["early_stopping"], patience=5)
    log = train(cfg_fast, ds, tmp_path / "out")
    tl = log["train_loss"]
    assert len(tl) >= 10 and tl[-1] < tl[0] and min(tl[-5:]) < tl[0] * 0.5
    assert log["stopped_early"] and log["epochs_run"] < 400
    assert log["best_epoch"] >= 0 and np.isfinite(log["best_heldout_loss"])
    assert (tmp_path / "out" / "recovery_mlp.pt").exists() and (tmp_path / "out" / "train_log.json").exists()
    assert set(log["heldout_episode_ids"]).isdisjoint(log["train_episode_ids"])
    loaded = RecoveryMLP.load(tmp_path / "out" / "recovery_mlp.pt", feature_names=FEATURE_NAMES)
    y = loaded.predict(ds.x[:4], FEATURE_NAMES)
    assert y.shape == (4, 7)


def test_train_cli_on_synthetic_file(tmp_path, capsys):
    path = tmp_path / "c.csv"
    synthetic_corrections(80, 4).to_csv(path, index=False)
    rc = train_main(["--data", str(path), "--config", str(CONFIG), "--out", str(tmp_path / "o"), "--max-epochs", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"epochs_run": 3' in out and (tmp_path / "o" / "recovery_mlp.pt").exists()
