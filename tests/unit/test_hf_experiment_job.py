"""Budget ceilings, safe archive portability and terminal-job interpretation."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("hf_experiment_job", Path(__file__).parents[2] / "scripts/hf_experiment_job.py")
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)


def test_timeout_cost_rounded_up_and_cumulative_budget():
    first = job.reserve_cost({}, 1.00002, 300)
    assert first["maximum_usd"] == 5.01
    with pytest.raises(ValueError, match="exceed"):
        job.reserve_cost({"reservations": [first]}, 1.00002, 300)
    second = job.reserve_cost({"reservations": [first]}, 1.00002, 240)
    assert first["maximum_usd"] + second["maximum_usd"] <= 10


@pytest.mark.parametrize("price,minutes", [(0, 60), (-1, 60), (float("nan"), 60),
                                          (float("inf"), 60), (1, 301), (1, 0)])
def test_bad_price_or_timeout_rejected(price, minutes):
    with pytest.raises(ValueError):
        job.reserve_cost({}, price, minutes)


def test_dataset_symlink_resolves_inside_extracted_tree(tmp_path):
    target = tmp_path / ".cache/derived_cartesian/images_iphone.npy"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"test")
    source = tmp_path / ".cache/experiments_20261016/canonical/images_iphone.npy"
    source.parent.mkdir(parents=True)
    source.symlink_to("../../derived_cartesian/images_iphone.npy")
    assert source.resolve() == target
