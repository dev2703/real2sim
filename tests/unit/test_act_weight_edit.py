import json
import shutil

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from real2sim.artifacts import sha256_file
from real2sim.policy.weight_edit import (
    BIAS_KEY,
    WEIGHT_KEY,
    fit_from_cache,
    fit_weight_edit,
    load_feature_rows,
)


def test_ridge_delta_is_folded_into_original_linear_map():
    rng = np.random.default_rng(12)
    x, vx = rng.normal(size=(100, 4)), rng.normal(size=(40, 4))
    w = rng.normal(size=(2, 4)).astype(np.float32)
    b = np.array([.2, -.1], dtype=np.float32)
    correction = np.array([[.1, -.2, .3, .1], [-.2, .1, .2, -.1]])
    intercept = np.array([.4, -.3])
    y, vy = x @ (w + correction).T + b + intercept, vx @ (w + correction).T + b + intercept
    ew, eb, report = fit_weight_edit(w, b, x, y, vx, vy, ridges=(1e-8,), alphas=(0, 1))
    assert report["accepted"]
    np.testing.assert_allclose(ew, w + correction, atol=2e-7)
    np.testing.assert_allclose(eb, b + intercept, atol=2e-7)
    assert report["edited_validation"]["mae_normalized"] < 1e-6
    assert report["task_completion_established"] is False
    assert ew.dtype == np.float32


def test_validation_selects_strength_without_changing_training_solution():
    x = np.linspace(-2, 2, 100)[:, None]
    w, b = np.zeros((1, 1), np.float64), np.zeros(1, np.float64)
    ew, eb, report = fit_weight_edit(w, b, x, 2 * x, x, x,
        ridges=(1e-8,), alphas=(0, .5, 1))
    assert report["selected_alpha"] == .5
    np.testing.assert_allclose(ew, [[1.]], atol=1e-7)
    np.testing.assert_allclose(eb, [0.], atol=1e-7)
    assert report["baseline_validation"]["mae_normalized"] > .9


def test_worse_edit_keeps_exact_original_weights():
    x = np.linspace(-2, 2, 50)[:, None]
    w, b = np.ones((1, 1), np.float32), np.array([.2], np.float32)
    ew, eb, report = fit_weight_edit(w, b, x, -x, x, x + .2)
    assert report["accepted"] is False
    assert report["selected_alpha"] == 0
    assert report["delta_l2"] == 0
    np.testing.assert_array_equal(ew, w)
    np.testing.assert_array_equal(eb, b)


def test_constant_features_remain_finite():
    x = np.ones((20, 3))
    _, _, report = fit_weight_edit(np.zeros((2, 3), np.float32), np.zeros(2, np.float32),
        x, np.ones((20, 2)), x, np.ones((20, 2)))
    assert report["accepted"]
    assert report["edited_validation"]["mae_normalized"] < .02


def test_robot_regression_rejected_even_when_aggregate_mae_improves():
    # The majority arm makes an edit look useful, but the minority arm already
    # matches its validation targets and must not be sacrificed by averaging.
    x = np.zeros((101, 1))
    train_y = np.ones((101, 1))
    validation_y = train_y.copy()
    validation_y[-1] = 0.
    groups = np.r_[np.zeros(100, np.int64), np.ones(1, np.int64)]
    w, b = np.zeros((1, 1), np.float32), np.zeros(1, np.float32)
    _, _, ordinary = fit_weight_edit(w, b, x, train_y, x, validation_y,
                                    ridges=(.01,), alphas=(0, 1))
    ew, eb, grouped = fit_weight_edit(w, b, x, train_y, x, validation_y,
        ridges=(.01,), alphas=(0, 1), train_groups=groups, validation_groups=groups)
    assert ordinary["accepted"]
    assert not grouped["accepted"]
    assert grouped["robot_ids"] == [0, 1]
    candidate = grouped["candidates"][-1]
    assert candidate["mae_normalized"] < grouped["baseline_validation"]["mae_normalized"]
    assert not candidate["no_robot_regression"]
    assert candidate["per_robot"]["1"]["mae_normalized"] > 0
    np.testing.assert_array_equal(ew, w)
    np.testing.assert_array_equal(eb, b)


def test_each_robot_has_equal_total_fitting_weight_despite_different_row_counts():
    x = np.zeros((101, 1))
    groups = np.r_[np.zeros(100, np.int64), np.ones(1, np.int64)]
    train_y = np.r_[np.full(100, 2.), 0.][:, None]
    _, bias, report = fit_weight_edit(np.zeros((1, 1), np.float64), np.zeros(1),
        x, train_y, np.zeros((2, 1)), np.ones((2, 1)), ridges=(1e-8,), alphas=(0, 1),
        train_groups=groups, validation_groups=np.array([0, 1]))
    assert report["accepted"]
    np.testing.assert_allclose(bias, [1.], atol=1e-7)
    assert report["baseline_train"]["per_robot"]["0"]["rows"] == 100
    assert report["baseline_train"]["per_robot"]["1"]["rows"] == 1
    assert report["selection_metric"] == "robot_balanced_mae_normalized"


@pytest.mark.parametrize("train,val", [
    (None, [0, 1]), ([0, 1], None), ([0], [0, 1]), ([0., 1.], [0, 1]),
    ([-1, 0], [-1, 0]), ([0, 1], [0, 0]),
])
def test_invalid_or_unvalidated_robot_groups_are_rejected(train, val):
    with pytest.raises(ValueError, match="robot|groups"):
        fit_weight_edit(np.zeros((1, 1), np.float32), np.zeros(1, np.float32),
            np.zeros((2, 1)), np.ones((2, 1)), np.zeros((2, 1)), np.ones((2, 1)),
            train_groups=train, validation_groups=val)


@pytest.mark.parametrize("kwargs", [
    {"ridges": (0,)}, {"ridges": (float("nan"),)},
    {"alphas": (.5, 1)}, {"alphas": (0, 2)}, {"min_improvement": -1},
])
def test_reject_invalid_search_configuration(kwargs):
    with pytest.raises(ValueError):
        fit_weight_edit(np.zeros((1, 1), np.float32), np.zeros(1, np.float32),
            np.ones((4, 1)), np.ones((4, 1)), np.ones((4, 1)), np.ones((4, 1)), **kwargs)


def _pack(path, seed, *, negative=False, episode=0, robot=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(20, 3, 4)).astype(np.float32)
    y = np.stack([x[..., 0] + .5, x[..., 1] - .3], axis=-1)
    if negative:
        y[:] = 0
    np.savez(path, features=x, target_normalized=y, valid=np.ones((20, 3), bool),
        robot_id=np.full(20, robot, np.int64), episode_index=np.full(20, episode, np.int64),
        tick=np.arange(20))


def _update_feature_hashes(data):
    report = json.loads((data / "report.json").read_text())
    report["feature_files_sha256"] = {name: sha256_file(data / name)
        for name in ("train.npz", "val.npz")}
    (data / "report.json").write_text(json.dumps(report))


@pytest.fixture
def cache(tmp_path):
    snapshot, data = tmp_path / "snapshot", tmp_path / "data"
    snapshot.mkdir()
    data.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"type": "act", "chunk_size": 3}))
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (snapshot / name).write_text(json.dumps({"processor": name}))
    save_file({WEIGHT_KEY: torch.zeros(2, 4), BIAS_KEY: torch.zeros(2),
        "model.backbone.weight": torch.arange(12).reshape(3, 4).float()},
        snapshot / "model.safetensors", metadata={"format": "pt", "custom": "retained"})
    save_file({"mean": torch.ones(2)}, snapshot / "processor.safetensors")
    source_hashes = {str(path.resolve()): sha256_file(path) for path in snapshot.iterdir()}
    (data / "report.json").write_text(json.dumps({"status": "complete",
        "checkpoint_unchanged": True, "runtime_source_changes": [],
        "weights_sha256": sha256_file(snapshot / "model.safetensors"),
        "source_sha256_before": source_hashes, "source_sha256_after": source_hashes,
        "target_source": "simulation teacher"}))
    _pack(data / "train.npz", 12, episode=10)
    _pack(data / "val.npz", 13, episode=20)
    _update_feature_hashes(data)
    return data, snapshot


def test_native_export_changes_only_action_head_and_retains_processors(cache, tmp_path):
    data, snapshot = cache
    out = tmp_path / "edited"
    before = {p.name: sha256_file(p) for p in snapshot.iterdir()}
    report = fit_from_cache(data, out, snapshot, prefix=2, ridges=(.001,))
    assert report["accepted"]
    assert report["base_checkpoint_unchanged"]
    assert report["supervision_source"] == "simulation teacher"
    assert report["train_rows"] == 40
    assert report["train_episodes"] == [[0, 10]]
    assert report["validation_episodes"] == [[0, 20]]
    assert before == {p.name: sha256_file(p) for p in snapshot.iterdir()}
    for name, checksum in before.items():
        if name != "model.safetensors":
            assert sha256_file(out / name) == checksum
    base, edited = load_file(snapshot / "model.safetensors"), load_file(out / "model.safetensors")
    changed = [key for key in base if not torch.equal(base[key], edited[key])]
    assert set(changed) == {WEIGHT_KEY, BIAS_KEY}
    with safe_open(out / "model.safetensors", framework="pt") as payload:
        assert payload.metadata() == {"format": "pt", "custom": "retained"}
    assert report["edited_checkpoint_sha256"] == sha256_file(out / "model.safetensors")
    assert (out / "weight_edit_report.json").is_file()


def test_rejected_export_contains_report_but_no_checkpoint(cache, tmp_path):
    data, snapshot = cache
    _pack(data / "val.npz", 13, negative=True, episode=20)
    _update_feature_hashes(data)
    out = tmp_path / "rejected"
    report = fit_from_cache(data, out, snapshot, prefix=2)
    assert report["accepted"] is False
    assert report["edited_checkpoint_sha256"] is None
    assert not (out / "model.safetensors").exists()
    assert list(out.iterdir()) == [out / "weight_edit_report.json"]


def test_overlapping_episode_rejected_before_export(cache, tmp_path):
    data, snapshot = cache
    _pack(data / "val.npz", 13, episode=10)
    _update_feature_hashes(data)
    with pytest.raises(ValueError, match="overlap"):
        fit_from_cache(data, tmp_path / "out", snapshot, prefix=2)
    assert not (tmp_path / "out").exists()


def test_wrong_base_checkpoint_rejected(cache, tmp_path):
    data, snapshot = cache
    payload = json.loads((data / "report.json").read_text())
    payload["weights_sha256"] = "wrong"
    (data / "report.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different ACT"):
        fit_from_cache(data, tmp_path / "out", snapshot, prefix=2)


def test_decoder_input_features_cannot_be_mistaken_for_action_head_features(cache, tmp_path):
    data, snapshot = cache
    payload = json.loads((data / "report.json").read_text())
    payload["feature_layer"] = "decoder_ffn_input"
    (data / "report.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="action_head input"):
        fit_from_cache(data, tmp_path / "out", snapshot, prefix=2)


def test_incomplete_collection_rejected(cache, tmp_path):
    data, snapshot = cache
    payload = json.loads((data / "report.json").read_text())
    del payload["runtime_source_changes"]
    (data / "report.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="complete"):
        fit_from_cache(data, tmp_path / "out", snapshot, prefix=2)


def test_same_weights_but_changed_processor_rejected(cache, tmp_path):
    data, snapshot = cache
    (snapshot / "policy_preprocessor.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="checkpoint file differs"):
        fit_from_cache(data, tmp_path / "out", snapshot, prefix=2)


def test_modified_feature_pack_rejected(cache, tmp_path):
    data, snapshot = cache
    _pack(data / "train.npz", 99, episode=10)
    with pytest.raises(ValueError, match="feature pack checksum mismatch"):
        fit_from_cache(data, tmp_path / "out", snapshot, prefix=2)


def test_changed_collection_source_checksums_rejected(cache, tmp_path):
    data, snapshot = cache
    report = json.loads((data / "report.json").read_text())
    report["source_sha256_after"][str((snapshot / "config.json").resolve())] = "changed"
    (data / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identical before/after"):
        fit_from_cache(data, tmp_path / "out", snapshot, prefix=2)


def test_aggregate_caches_reuses_same_role_episodes_and_keeps_all_rows(cache, tmp_path):
    data, snapshot = cache
    correction = tmp_path / "correction"
    shutil.copytree(data, correction)
    _pack(correction / "train.npz", 92, episode=10)
    _pack(correction / "val.npz", 93, episode=20)
    _update_feature_hashes(correction)
    report = fit_from_cache([data, correction], tmp_path / "edited", snapshot, prefix=2)
    assert report["accepted"]
    assert report["train_rows"] == report["validation_rows"] == 80
    assert report["train_episodes"] == [[0, 10]]
    assert report["validation_episodes"] == [[0, 20]]
    assert report["data"] == [str(data.resolve()), str(correction.resolve())]
    assert len(report["feature_collections"]) == 2
    assert all(source["train_rows"] == 40 for source in report["feature_collections"])
    assert str((correction / "train.npz").resolve()) in report["source_sha256_before"]


def test_cross_cache_training_validation_overlap_rejected(cache, tmp_path):
    data, snapshot = cache
    correction = tmp_path / "correction"
    shutil.copytree(data, correction)
    _pack(correction / "train.npz", 92, episode=20)
    _pack(correction / "val.npz", 93, episode=30)
    _update_feature_hashes(correction)
    with pytest.raises(ValueError, match="overlap"):
        fit_from_cache([data, correction], tmp_path / "edited", snapshot, prefix=2)
    assert not (tmp_path / "edited").exists()


def test_every_aggregate_cache_checksum_is_validated(cache, tmp_path):
    data, snapshot = cache
    correction = tmp_path / "correction"
    shutil.copytree(data, correction)
    _pack(correction / "train.npz", 92, episode=10)
    with pytest.raises(ValueError, match="feature pack checksum mismatch"):
        fit_from_cache([data, correction], tmp_path / "edited", snapshot, prefix=2)
    assert not (tmp_path / "edited").exists()


def test_aggregate_requires_at_least_one_cache(cache, tmp_path):
    _, snapshot = cache
    with pytest.raises(ValueError, match="at least one"):
        fit_from_cache([], tmp_path / "edited", snapshot, prefix=2)


def test_multi_robot_cache_export_records_coverage_and_per_robot_metrics(cache, tmp_path):
    data, snapshot = cache
    panda = tmp_path / "panda"
    shutil.copytree(data, panda)
    _pack(panda / "train.npz", 92, episode=10, robot=1)
    _pack(panda / "val.npz", 93, episode=20, robot=1)
    _update_feature_hashes(panda)
    report = fit_from_cache([data, panda], tmp_path / "edited", snapshot, prefix=2)
    assert report["accepted"]
    assert report["robot_balanced_fitting"]
    assert report["robot_ids"] == [0, 1]
    for robot, baseline in report["baseline_validation"]["per_robot"].items():
        edited = report["edited_validation"]["per_robot"][robot]
        assert edited["mae_normalized"] <= baseline["mae_normalized"]


@pytest.mark.parametrize("second_scene", ["different_scene.yaml", None])
def test_caches_with_different_or_missing_scene_declarations_cannot_mix(cache, tmp_path, second_scene):
    data, snapshot = cache
    second = tmp_path / "second"
    shutil.copytree(data, second)
    for directory, scene in ((data, "first_scene.yaml"), (second, second_scene)):
        report = json.loads((directory / "report.json").read_text())
        if scene is not None:
            report["source_scene"] = scene
        (directory / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="same source scene"):
        fit_from_cache([data, second], tmp_path / "edited", snapshot, prefix=2)


def test_scene_compatibility_uses_recorded_content_hash_for_relocated_configs(cache, tmp_path):
    data, snapshot = cache
    second = tmp_path / "second"
    shutil.copytree(data, second)
    for index, directory in enumerate((data, second)):
        report = json.loads((directory / "report.json").read_text())
        scene = tmp_path / f"source_{index}.yaml"
        report["source_scene"] = str(scene)
        for key in ("source_sha256_before", "source_sha256_after"):
            report[key][str(scene.resolve())] = "same-recorded-scene-content"
        (directory / "report.json").write_text(json.dumps(report))
    report = fit_from_cache([data, second], tmp_path / "edited", snapshot, prefix=2)
    assert report["accepted"]


@pytest.mark.parametrize("changed_input", ["scripts/evaluate_hf_act_20261027.py", "configs/task.yaml"])
def test_same_scene_yaml_cannot_hide_changed_render_builder_or_task(cache, tmp_path, changed_input):
    data, snapshot = cache
    second = tmp_path / "second"
    shutil.copytree(data, second)
    for index, directory in enumerate((data, second)):
        report = json.loads((directory / "report.json").read_text())
        scene = tmp_path / "same_scene.yaml"
        report["source_scene"] = str(scene)
        for key in ("source_sha256_before", "source_sha256_after"):
            report[key][str(scene.resolve())] = "same-scene-yaml"
            report[key][str((tmp_path / changed_input).resolve())] = f"different-{index}"
        (directory / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="same source scene"):
        fit_from_cache([data, second], tmp_path / "edited", snapshot, prefix=2)


def test_valid_prefix_mask_excludes_unexecuted_queries(tmp_path):
    path = tmp_path / "features.npz"
    x, y = np.arange(12, dtype=np.float32).reshape(2, 3, 2), np.ones((2, 3, 1), np.float32)
    np.savez(path, features=x, target_normalized=y,
        valid=np.array([[True, False, True], [False, True, True]]),
        robot_id=np.array([0, 1]), episode_index=np.array([1, 2]), tick=np.array([0, 3]))
    xx, yy, episodes = load_feature_rows(path, 2)
    np.testing.assert_array_equal(xx, x[[0, 1], [0, 1]])
    assert yy.shape == (2, 1)
    assert episodes == {(0, 1), (1, 2)}
    _, _, _, groups = load_feature_rows(path, 2, with_groups=True)
    np.testing.assert_array_equal(groups, [0, 1])


def test_nonfinite_input_rejected():
    with pytest.raises(ValueError, match="finite"):
        fit_weight_edit(np.zeros((1, 1), np.float32), np.zeros(1, np.float32),
            np.array([[np.nan]]), np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
