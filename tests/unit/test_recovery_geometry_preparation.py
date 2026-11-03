"""Geometry preparation preserves actions, phase labels and dataset roles."""

import json

import numpy as np
import pytest

from real2sim.policy.common_recovery import OBSERVATION_NAMES
from scripts.prepare_recovery_geometry import prepare
from tests.unit.test_recovery_history_conversion import sequence


def pack(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    rows, _ = sequence()
    rows["expert_phase_id"] = np.arange(4, dtype=np.int64)
    for name in ("train", "val"):
        np.savez(directory / f"{name}.npz", **rows)
    manifest = {"status": "complete", "runtime_source_changes": [], "base_weights_frozen": True,
        "lighting_changed": False, "history_mode": "selected", "observation_names": OBSERVATION_NAMES,
        "split": {"train": [10000], "val": [11000]}, "expert_phase_is_audit_label_only": True}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, rows, manifest


def test_geometry_roundtrip_preserves_actions_labels_and_episode_roles(tmp_path):
    directory, rows, original = pack(tmp_path)
    first = prepare(directory, tmp_path / "grasp")
    assert first["observation_geometry"] == "grasp"
    assert first["split"] == original["split"]
    assert first["history_mode"] == "selected"
    assert first["runtime_source_changes"] == []
    assert all("_grasp_relative_" in name for name in first["observation_names"] if "_rgb_" in name)
    with np.load(tmp_path / "grasp/train.npz", allow_pickle=False) as converted:
        for name in rows:
            if name != "observations":
                np.testing.assert_array_equal(converted[name], rows[name])
    prepare(tmp_path / "grasp", tmp_path / "roundtrip", geometry="tcp")
    with np.load(tmp_path / "roundtrip/train.npz", allow_pickle=False) as restored:
        np.testing.assert_allclose(restored["observations"], rows["observations"], atol=1e-7)
    with pytest.raises(FileExistsError):
        prepare(directory, tmp_path / "grasp")


def test_incomplete_pack_cannot_be_prepared_as_valid_training_data(tmp_path):
    directory, _, manifest = pack(tmp_path)
    manifest["status"] = "collecting"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completed unchanged"):
        prepare(directory, tmp_path / "grasp")
    assert not (tmp_path / "grasp").exists()
