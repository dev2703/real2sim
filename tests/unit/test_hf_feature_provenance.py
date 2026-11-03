"""Prevent fitting an apparently latest head to stale root-checkpoint features."""

import json

import pytest

from real2sim.artifacts import sha256_file
from scripts.train_local_hf_head_20261028 import validate_feature_source


def test_feature_cache_must_match_exact_frozen_weights(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"latest checkpoint")
    report = {"status": "complete", "checkpoint_unchanged": True,
              "runtime_source_changes": [], "weights_sha256": "old root checkpoint"}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different HF checkpoint"):
        validate_feature_source(tmp_path, checkpoint)
    report["weights_sha256"] = sha256_file(checkpoint / "model.safetensors")
    path.write_text(json.dumps(report))
    assert validate_feature_source(tmp_path, checkpoint) == report
    report["status"] = "collecting"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="complete"):
        validate_feature_source(tmp_path, checkpoint)
