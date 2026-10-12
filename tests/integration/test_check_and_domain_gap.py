"""Checkpoint check CLI and the real-vs-sim-render diagnostic on the real frozen checkpoints.

Skipped (not silently passed) when checkpoints or derived data are absent, e.g. in CI.
"""

import json
from pathlib import Path

import pytest

from real2sim.evaluation import domain_gap
from real2sim.policy.check import check

JOINT_CK = Path("outputs/act_joint_so101/checkpoints/020000")
CART_CK = Path("outputs/act_cartesian/checkpoints/006000")
JOINT_DATA = Path(".cache/derived_joint")
CART_DATA = Path(".cache/derived_cartesian")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not all(p.exists() for p in (JOINT_CK, CART_CK, JOINT_DATA / "manifest.json", CART_DATA / "manifest.json")),
        reason="real checkpoints or derived datasets not present",
    ),
]


def test_check_reports_contract_and_finite_chunk():
    res = check(JOINT_CK, JOINT_DATA, device="cpu", max_batches=1)
    assert res["contract_ok"] and res["all_finite"]
    assert res["chunk_shape"] == [16, 20, 6]
    assert res["camera_keys"] == ["observation.images.iphone"]
    assert res["step"] == 20000
    assert len(res["val_open_loop_l1_raw_per_dim"]) == 6
    assert res["val_samples"] == 16  # max_batches=1 at batch size 16


def test_check_refuses_checkpoint_against_foreign_data():
    """A joint checkpoint checked against the cartesian dataset must fail loudly, not score."""
    with pytest.raises(ValueError, match="differs"):
        check(JOINT_CK, CART_DATA, device="cpu", max_batches=1)


def test_domain_gap_refuses_training_episode(tmp_path):
    manifest = json.loads((JOINT_DATA / "manifest.json").read_text())
    train_ep = manifest["split"]["train"][0]
    with pytest.raises(SystemExit, match="not a validation episode"):
        domain_gap.main(["--checkpoint", str(JOINT_CK), "--data", str(JOINT_DATA),
                         "--episode", str(train_ep), "--out", str(tmp_path / "x.json")])


@pytest.mark.parametrize("ck,data,dim", [(JOINT_CK, JOINT_DATA, 6), (CART_CK, CART_DATA, 7)])
def test_domain_gap_report_schema(tmp_path, ck, data, dim):
    out = tmp_path / "gap.json"
    rc = domain_gap.main(["--checkpoint", str(ck), "--data", str(data), "--episode", "1",
                          "--stride", "200", "--save-pairs", "0", "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert rep["n_ticks"] >= 1
    for key in ("mean_abs_err_real_frame", "mean_abs_err_sim_render", "mean_abs_prediction_gap_sim_vs_real"):
        assert len(rep[key]) == dim and all(v >= 0 for v in rep[key])
    assert rep["error_ratio_sim_over_real"] > 0
    assert "Diagnostic only" in rep["note"]
