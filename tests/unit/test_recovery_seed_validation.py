"""Reject rollout seeds that overwrite evidence or create unusable collections."""

import numpy as np
import pytest

from scripts import run_recovery_dagger as recovery

INVALID_SEEDS = [[], [71, 71], [-1], [1.5], [1.0], [True], [np.bool_(True)], ["71"], None]


@pytest.mark.parametrize("seeds", INVALID_SEEDS)
@pytest.mark.parametrize("partition", ["train", "val"])
def test_collection_rejects_invalid_seeds_before_outputs_or_loading(
        tmp_path, monkeypatch, seeds, partition):
    out = tmp_path / "collection"
    monkeypatch.setattr(recovery, "NativeLeRobotACT", lambda *_: pytest.fail("Loaded checkpoint"))
    train, val = (seeds, [11000]) if partition == "train" else ([10000], seeds)
    with pytest.raises(ValueError, match="unique nonnegative integer seeds"):
        recovery.collect("so101", out, train, val)
    assert not out.exists()


@pytest.mark.parametrize("seeds", INVALID_SEEDS)
def test_evaluation_rejects_invalid_seeds_before_outputs_or_loading(tmp_path, monkeypatch, seeds):
    out = tmp_path / "evaluation"
    monkeypatch.setattr(recovery, "load_recovery_head", lambda *_: pytest.fail("Loaded recovery head"))
    with pytest.raises(ValueError, match="unique nonnegative integer seeds"):
        recovery.evaluate("so101", tmp_path / "head", out, seeds)
    assert not out.exists()


@pytest.mark.parametrize("train,val,message", [
    ([10000], [10000], "partitions disjoint"),
    ([13000], [11000], "Reserved evaluation seeds"),
    ([10000], [13099], "Reserved evaluation seeds"),
])
def test_collection_rejects_leaked_episode_roles_before_loading(
        tmp_path, monkeypatch, train, val, message):
    out = tmp_path / "collection"
    monkeypatch.setattr(recovery, "NativeLeRobotACT", lambda *_: pytest.fail("Loaded checkpoint"))
    with pytest.raises(ValueError, match=message):
        recovery.collect("so101", out, train, val)
    assert not out.exists()


@pytest.mark.parametrize("seeds", [[0, 12000], [13000, 13099], [np.int64(71), np.int32(72)]])
def test_valid_evaluation_seeds_including_reserved_cohort_reach_head_loading(
        tmp_path, monkeypatch, seeds):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"frozen weights")

    class HeadReached(Exception):
        pass

    def load_head(*_):
        raise HeadReached

    monkeypatch.setattr(recovery, "load_recovery_head", load_head)
    with pytest.raises(HeadReached):
        recovery.evaluate("so101", tmp_path / "head", tmp_path / "evaluation", seeds, snapshot=snapshot)


def test_collection_accepts_valid_integer_seeds_at_reserved_cohort_boundaries(tmp_path, monkeypatch):
    class CheckpointReached(Exception):
        pass

    def load_checkpoint(*_):
        raise CheckpointReached

    monkeypatch.setattr(recovery, "NativeLeRobotACT", load_checkpoint)
    with pytest.raises(CheckpointReached):
        recovery.collect("so101", tmp_path / "collection", [0, np.int64(12999)], [13100])
