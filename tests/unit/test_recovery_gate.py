"""A recovery gate must not claim a learned off-switch from positive-only data."""

import json

import numpy as np
import pytest

from real2sim.artifacts import sha256_file
from real2sim.policy.recovery_gate import (
    RecoveryInterventionGate,
    calibrate_threshold,
    classification_metrics,
    correction_needed,
)
from scripts.train_recovery_gate import train as train_gate
from scripts.train_recovery_mlp import train as train_head


def test_correction_labels_use_euclidean_error_and_each_declared_channel():
    base = np.zeros((5, 7))
    expert = base.copy()
    expert[0, :3] = [.0015, .0015, 0.]
    expert[1, 3:6] = [.015, .015, 0.]
    expert[2, 6] = .051
    expert[3] = [.002, 0., 0., .02, 0., 0., .05]
    np.testing.assert_array_equal(correction_needed(base, expert), [True, True, True, False, False])
    with pytest.raises(ValueError, match="matching finite"):
        correction_needed(base, np.full((5, 7), np.nan))


@pytest.mark.parametrize("training,validation", [([1, 1], [1, 1]), ([0, 0], [0, 0]),
                                                   ([0, 1], [1, 1])])
def test_single_class_evidence_cannot_support_disabling_recovery(training, validation):
    probabilities = np.asarray([.1, .9])
    calibration = calibrate_threshold(probabilities, validation, training)
    assert calibration["threshold"] == 0
    assert calibration["decision_mode"] == "always_recover_single_class"
    assert calibration["disabling_recovery_supported_by_data"] is False
    assert (probabilities >= calibration["threshold"]).all()


def test_recall_threshold_is_conservative_for_each_robot_not_only_pooled_rows():
    probabilities = np.array([.1, .3, .5, .7, .9, .2, .4, .6, .8, .95, .12, .13])
    labels = np.asarray([1] * 10 + [0, 0], bool)
    robots = np.asarray([0] * 5 + [1] * 5 + [0, 1])
    calibration = calibrate_threshold(probabilities, labels, [0, 1], minimum_recall=.8, robots=robots)
    assert calibration["decision_mode"] == "learned_threshold"
    for robot in (0, 1):
        mask = robots == robot
        assert classification_metrics(probabilities[mask], labels[mask], calibration["threshold"])["recall"] >= .8
    metrics = classification_metrics([.1, .9], [1, 1], 0.)
    assert metrics["recall"] == 1. and metrics["specificity"] is None
    assert metrics["brier_score"] == pytest.approx(.41)


def artifacts(tmp_path):
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"immutable frozen checkpoint")
    base_hash = sha256_file(checkpoint / "model.safetensors")
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "manifest.json").write_text(json.dumps({
        "status": "complete", "base_checkpoint_sha256": base_hash,
        "lighting_changed": False, "base_weights_frozen": True, "runtime_source_changes": []}))
    rng = np.random.default_rng(4)
    for partition, episode in (("train", 100), ("val", 200)):
        observations = rng.normal(size=(10, 6)).astype(np.float32)
        if partition == "val":
            observations += 7
        base_actions = np.zeros((10, 7), np.float32)
        targets = base_actions.copy()
        targets[:, 0] = .01
        targets[:, 6] = 1.
        np.savez(pack / f"{partition}.npz", observations=observations,
            base_actions=base_actions, target_actions=targets,
            features=rng.normal(size=(10, 4)).astype(np.float32),
            robot_id=np.arange(10) % 2, episode_index=np.full(10, episode))
    head = tmp_path / "head"
    train_head([pack], head, snapshot=checkpoint, steps=2)
    return checkpoint, base_hash, pack, head


def test_gate_artifact_binds_both_models_and_training_only_normalizers(tmp_path):
    checkpoint, base_hash, pack, head = artifacts(tmp_path)
    head_hash = sha256_file(head / "head.pt")
    report = train_gate([pack], head, tmp_path / "gate", snapshot=checkpoint, steps=2)
    assert report["status"] == "complete" and report["runtime_source_changes"] == []
    assert report["training_labels"] == {"n": 10, "positive": 10, "negative": 0}
    assert report["validation_metrics"]["specificity"] is None
    gate = RecoveryInterventionGate(tmp_path / "gate", expected_base_sha256=base_hash,
                                    expected_recovery_head_sha256=head_hash)
    with np.load(pack / "train.npz") as data:
        np.testing.assert_allclose(gate.model.observation_mean.numpy(), data["observations"].mean(0))
        probability = gate.probability(data["observations"][0], data["base_actions"][0], data["features"][0])
        assert 0 <= probability <= 1
        assert gate.needs_recovery(data["observations"][0], data["base_actions"][0], data["features"][0])
        with pytest.raises(ValueError, match="finite"):
            gate.probability(np.full(6, np.nan), data["base_actions"][0], data["features"][0])
    assert sha256_file(head / "head.pt") == head_hash
    with pytest.raises(ValueError, match="different frozen ACT"):
        RecoveryInterventionGate(tmp_path / "gate", expected_base_sha256="other",
                                 expected_recovery_head_sha256=head_hash)
    with pytest.raises(ValueError, match="different recovery MLP"):
        RecoveryInterventionGate(tmp_path / "gate", expected_base_sha256=base_hash,
                                 expected_recovery_head_sha256="other")
    with (tmp_path / "gate/gate.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        RecoveryInterventionGate(tmp_path / "gate", expected_base_sha256=base_hash,
                                 expected_recovery_head_sha256=head_hash)
