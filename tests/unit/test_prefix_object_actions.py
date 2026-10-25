"""Canonical checkpoint/data identity must survive reuse of frozen object masks."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("prefix_object_actions", root / "scripts/diag_v2_prefix_object_actions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(diagnostic):
    return {"camera_keys": [diagnostic.CAMERA], "state_names": diagnostic.CANONICAL_STATE_NAMES,
            "action_names": ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"],
            "action_representation": "cartesian", "split": {"train": [8], "val": [9], "reserved": [10]},
            "files_sha256": {"arrays.npz": "same arrays", "images.npy": "same images", "stats.json": "canonical stats"},
            "image_files": {diagnostic.CAMERA: "images.npy"}}


def test_frozen_measurements_can_share_arrays_and_images_but_not_feature_normalization(diagnostic):
    canonical = manifest(diagnostic)
    grounded = deepcopy(canonical)
    grounded["state_names"] += ["u", "v", "radius", "visible"]
    grounded["visual_grounding"] = {"enabled": True}
    grounded["files_sha256"]["stats.json"] = "grounded stats"
    diagnostic.verify_common_source(canonical, grounded)
    grounded["files_sha256"]["images.npy"] = "changed image"
    with pytest.raises(ValueError, match="source payload differs"):
        diagnostic.verify_common_source(canonical, grounded)


@pytest.mark.parametrize("change", [
    {"visual_grounding": {"enabled": False}}, {"observation_variant": {"version": 1}},
    {"state_names": ["q"] * 13}, {"camera_keys": ["wrist"]},
])
def test_noncanonical_inputs_are_rejected(diagnostic, change):
    with pytest.raises(ValueError, match="canonical 13-state"):
        diagnostic.validate_canonical_manifest(manifest(diagnostic) | change)


def test_checkpoint_normalizer_and_label_mapping_must_match(diagnostic):
    source = manifest(diagnostic)
    checkpoint = deepcopy(source)
    checkpoint["derived_files_sha256"] = checkpoint.pop("files_sha256")
    diagnostic.verify_checkpoint_contract(checkpoint, source, {"mean": [1]}, {"mean": [1]})
    with pytest.raises(ValueError, match="normalization differs"):
        diagnostic.verify_checkpoint_contract(checkpoint, source, {"mean": [0]}, {"mean": [1]})
    checkpoint["derived_files_sha256"]["arrays.npz"] = "other labels"
    with pytest.raises(ValueError, match="payload map differs"):
        diagnostic.verify_checkpoint_contract(checkpoint, source, {}, {})


def test_actual_payload_is_rehashed(diagnostic, tmp_path):
    payload = tmp_path / "arrays.npz"
    payload.write_bytes(b"original labels")
    source = {"files_sha256": {payload.name: diagnostic.sha256_file(payload)}}
    diagnostic.verify_payload(tmp_path, source)
    payload.write_bytes(b"changed labels")
    with pytest.raises(ValueError, match="payload changed"):
        diagnostic.verify_payload(tmp_path, source)
