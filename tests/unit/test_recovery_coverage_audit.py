"""Coverage counts preserve failed attempts and reject inflated or mismatched rows."""

import json

import numpy as np
import pytest

from real2sim.artifacts import sha256_file
from scripts.audit_recovery_coverage import (
    distribution,
    normalized_coverage,
    snapshot_integrity,
    validate_partition,
)


def fixture():
    arrays = {"observations": np.zeros((3, 2)), "features": np.zeros((3, 1)),
              "base_actions": np.zeros((3, 7)), "target_actions": np.zeros((3, 7)),
              "robot_id": np.ones(3, dtype=int), "episode_index": np.full(3, 12),
              "expert_phase_id": np.array([1, 2, 2]), "tick": np.array([0, 1, 4])}
    manifest = {"robot": "panda", "split": {"train": [12, 13]},
                "supervised_rows": {"train": 3}, "trials": [
                    {"seed": 12, "partition": "train", "ticks": 8,
                     "supervised_rows": 3, "learner_executed_ticks": 4, "abort": None},
                    {"seed": 13, "partition": "train", "ticks": 0,
                     "supervised_rows": 0, "learner_executed_ticks": 0,
                     "abort": "VirtualSourceProjectionError", "abort_source_projection": {
                         "accepted": False, "position_error_m": .01}}]}
    return arrays, manifest


def test_saved_ticks_and_zero_label_aborted_attempt_are_both_retained():
    arrays, manifest = fixture()
    episodes = validate_partition(arrays, manifest, "train", ["a", "b"], 1)
    assert episodes[0]["missing_ticks_inside_label_span"] == 2
    assert episodes[0]["unlabelled_final_ticks"] == 3
    assert episodes[1]["supervised_rows"] == 0
    assert episodes[1]["abort_source_projection"]["position_error_m"] == .01


def test_missing_scored_attempt_cannot_improve_apparent_coverage():
    arrays, manifest = fixture()
    manifest["trials"].pop()
    with pytest.raises(ValueError, match="Every declared attempt"):
        validate_partition(arrays, manifest, "train", ["a", "b"], 1)


def test_duplicate_ticks_and_wrong_embodiment_are_rejected():
    arrays, manifest = fixture()
    arrays["tick"][-1] = 1
    with pytest.raises(ValueError, match="Duplicate"):
        validate_partition(arrays, manifest, "train", ["a", "b"], 1)
    arrays["robot_id"][0] = 0
    with pytest.raises(ValueError, match="embodiment"):
        validate_partition(arrays, manifest, "train", ["a", "b"], 1)


def test_extrapolation_uses_saved_normalizers_before_inference_clipping():
    report = normalized_coverage(np.array([[0., 1.], [24., 1.]]), np.zeros(2),
                                 np.array([2., 1.]), ["a", "b"])
    assert report["max_abs_training_z"] == 12.
    assert report["rows_any_abs_z_gt_10"] == 1
    assert report["largest_dimensions"][0]["observed_max"] == 24.
    assert distribution([])["median"] is None


def test_snapshot_drift_is_detected_without_requiring_current_runtime_match(tmp_path):
    text = tmp_path / "preserved.py"
    text.write_text("historical = True\n")
    digest = sha256_file(text)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"entries": [
        {"source": "historical/source.py", "sha256": digest, "snapshot": "preserved.py"}]}))
    hashes = {"historical/source.py": digest}
    manifest = {"source_sha256_before": hashes, "source_sha256_after": hashes,
                "executed_sources_manifest": str(ledger)}
    assert snapshot_integrity(tmp_path, manifest)["preserved_text_sources_verified"] == 1
    text.write_text("historical = False\n")
    with pytest.raises(ValueError, match="snapshot checksum"):
        snapshot_integrity(tmp_path, manifest)
