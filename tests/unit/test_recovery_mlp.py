"""Keep learned recovery, teacher collection and frozen ACT provenance separate."""

import json

import numpy as np
import pytest
import torch

from real2sim.artifacts import sha256_file
from real2sim.policy.recovery_mlp import RecoveryMLP, RecoveryMLPPolicy, bound_common_action
from scripts.train_recovery_mlp import (
    evaluate,
    load_collections,
    recovery_normalizers,
    robot_episode_weights,
    train,
)


def model(*, use_act=True):
    return RecoveryMLP(np.zeros(6), np.ones(6), np.zeros(4), np.ones(4), use_act=use_act)


def test_untrained_head_preserves_base_and_no_act_removes_both_act_inputs():
    torch.manual_seed(7)
    observations, prior, features = torch.randn(3, 6), torch.randn(3, 7), torch.randn(3, 4)
    torch.testing.assert_close(model()(observations, prior, features), prior)
    direct = model(use_act=False)
    # Nonzero learned weights make this test sensitive to accidental ACT input use.
    torch.nn.init.normal_(direct.network[-1].weight, std=.1)
    first = direct(observations, prior, features)
    second = direct(observations, prior * 100 + 23, features * -100 - 9)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_common_action_caps_are_vector_norms_and_gripper_has_semantic_endpoints():
    bounded = bound_common_action([[.1, .1, .1, 1., 1., 1., -2.],
                                   [0., 0., 0., 0., 0., 0., 2.]])
    assert np.linalg.norm(bounded[0, :3]) == pytest.approx(.03)
    assert np.linalg.norm(bounded[0, 3:6]) == pytest.approx(.3)
    np.testing.assert_array_equal(bounded[:, 6], [0., 1.])
    with pytest.raises(ValueError, match="finite"):
        bound_common_action(np.full(7, np.nan))


def test_unseen_low_variance_inputs_cannot_dominate_the_residual():
    network = RecoveryMLP(np.zeros(6), np.zeros(6), np.zeros(4), np.zeros(4))
    torch.nn.init.normal_(network.network[-1].weight, std=.1)
    actions = torch.zeros(1, 7)
    # A coordinate can be constant on the source arm but variable on the target
    # arm. Saturating its z-score keeps that extrapolation bounded at inference.
    first = network(torch.full((1, 6), 1.), actions, torch.full((1, 4), 1.))
    second = network(torch.full((1, 6), 1000.), actions, torch.full((1, 4), 1000.))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(first).all()


def test_validation_weights_short_episodes_and_target_embodiment_equally():
    robots = np.asarray([0] * 10 + [1] * 2)
    episodes = np.asarray([10] * 8 + [11] * 2 + [20] * 2)
    network = model()
    errors = torch.tensor([0.] * 8 + [1.] * 2 + [.5] * 2)
    tensors = {"observations": torch.zeros(12, 6), "features": torch.zeros(12, 4),
               "base_actions": errors[:, None] * network.action_scales,
               "target_actions": torch.zeros(12, 7)}
    metrics = evaluate(network, tensors, robots, episodes=episodes)
    assert metrics["mean_absolute_normalized_error"] == pytest.approx(.25)
    assert metrics["robot_balanced_mae_normalized"] == pytest.approx(.35)
    assert metrics["robot_episode_balanced_mae_normalized"] == pytest.approx(.5)


def collection(tmp_path, name, *, train_episodes=(10, 11), val_episodes=(20,), base_hash="base",
               iteration=0, evaluation_seeds=()):
    directory = tmp_path / name
    directory.mkdir()
    manifest = {"status": "complete", "base_checkpoint_sha256": base_hash,
                "lighting_changed": False, "base_weights_frozen": True,
                "runtime_source_changes": [], "iteration": iteration,
                "evaluation_seeds": list(evaluation_seeds)}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    rng = np.random.default_rng(42)
    for partition, episodes in (("train", train_episodes), ("val", val_episodes)):
        n = 8 * len(episodes)
        observations = rng.normal(size=(n, 6)).astype(np.float32)
        if partition == "val":
            observations += 5
        prior = np.zeros((n, 7), np.float32)
        targets = np.zeros_like(prior)
        targets[:, 0] = observations[:, 0] * .001
        targets[:, 6] = .5
        np.savez(directory / f"{partition}.npz", observations=observations,
                 base_actions=prior, features=rng.normal(size=(n, 4)).astype(np.float32),
                 target_actions=targets, robot_id=np.arange(n) % 2,
                 episode_index=np.repeat(episodes, 8))
    return directory


def test_dagger_aggregation_rejects_cross_iteration_split_leakage_and_stale_features(tmp_path):
    first = collection(tmp_path, "initial")
    second = collection(tmp_path, "dagger", train_episodes=(20,), val_episodes=(30,), iteration=1)
    with pytest.raises(ValueError, match="overlap"):
        load_collections([first, second], expected_base_sha256="base")
    with pytest.raises(ValueError, match="different frozen ACT"):
        load_collections([first], expected_base_sha256="new base")
    reserved = collection(tmp_path, "reserved", evaluation_seeds=(10,))
    with pytest.raises(ValueError, match="Reserved evaluation"):
        load_collections([reserved], expected_base_sha256="base")


def test_collection_cannot_hide_episode_rows_outside_the_declared_split(tmp_path):
    directory = collection(tmp_path, "undeclared")
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["split"] = {"train": [10], "val": [20]}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="declared train split"):
        load_collections([directory], expected_base_sha256="base")


def test_collection_cannot_hide_target_arm_rows_in_a_source_arm_manifest(tmp_path):
    directory = collection(tmp_path, "wrong_robot")
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["robot"] = "so101"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="declared robot"):
        load_collections([directory], expected_base_sha256="base")


def test_single_arm_binary_indicators_keep_their_natural_scale_for_an_unseen_arm():
    rows = {"observations": np.array([[.1, 1., 0., 1.], [.3, 1., 0., 1.]]),
            "features": np.ones((2, 4))}
    names = ["history_0_tcp_x_m", "history_0_robot_so101", "history_0_robot_panda",
             "history_0_ball_observed"]
    mean, std, _, _ = recovery_normalizers(rows, names)
    assert mean[0] == pytest.approx(.2)
    np.testing.assert_array_equal(mean[1:], 0.)
    np.testing.assert_array_equal(std[1:], 1.)
    unseen_arm = np.array([.2, 0., 1., 0.])
    np.testing.assert_array_equal(((unseen_arm - mean) / std)[1:], [0., 1., 0.])


def test_sampler_balances_robots_then_episodes_despite_unequal_rollout_lengths():
    robots = np.asarray([0] * 10 + [1] * 4)
    episodes = np.asarray([1] * 8 + [2] * 2 + [1] * 4)
    weights = robot_episode_weights(robots, episodes).numpy()
    assert weights[robots == 0].sum() == pytest.approx(.5)
    assert weights[robots == 1].sum() == pytest.approx(.5)
    assert weights[(robots == 0) & (episodes == 1)].sum() == pytest.approx(.25)
    assert weights[(robots == 0) & (episodes == 2)].sum() == pytest.approx(.25)


def test_saved_head_uses_training_only_stats_and_exact_frozen_base_identity(tmp_path):
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"frozen actual checkpoint bytes")
    base_hash = sha256_file(checkpoint / "model.safetensors")
    data = collection(tmp_path, "data", base_hash=base_hash)
    report = train([data], tmp_path / "head", snapshot=checkpoint, steps=2, use_act=False)
    assert report["status"] == "complete" and report["base_weights_frozen"]
    selected = next(row for row in report["history"] if row["step"] == report["best_step"])
    assert report["validation_metrics"]["translation_mae_mm"] == selected["translation_mae_mm"]
    assert report["head"]["validation_metrics"] == report["validation_metrics"]
    loaded = RecoveryMLPPolicy(tmp_path / "head", expected_base_sha256=base_hash)
    assert loaded.model.normalization_clip == 10.
    assert report["best_validation_mae_normalized"] == report["validation_metrics"][
        "robot_episode_balanced_mae_normalized"]
    with np.load(data / "train.npz") as values:
        np.testing.assert_allclose(loaded.model.observation_mean.numpy(), values["observations"].mean(0))
        result = loaded.predict(values["observations"][0], values["base_actions"][0], values["features"][0])
        assert result.shape == (7,) and np.isfinite(result).all()
    with pytest.raises(ValueError, match="different frozen ACT"):
        RecoveryMLPPolicy(tmp_path / "head", expected_base_sha256="another checkpoint")
    # Historical weights did not clip. Loading their existing schema must retain
    # their original prediction contract rather than quietly changing behavior.
    metadata_path = tmp_path / "head/head.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("normalization_clip")
    metadata_path.write_text(json.dumps(metadata))
    legacy = RecoveryMLPPolicy(tmp_path / "head", expected_base_sha256=base_hash)
    assert legacy.model.normalization_clip is None
    with (tmp_path / "head/head.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        RecoveryMLPPolicy(tmp_path / "head", expected_base_sha256=base_hash)
