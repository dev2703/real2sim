import json

import numpy as np
import pytest

from real2sim.data.labels import PROPRIO_NAMES
from real2sim.data.raw import sha256_file
from scripts.diag_proprioception_noise import summarize_joint_trace
from scripts.prepare_observation_variants import inner_split, prepare_variants


def test_noise_diagnostic_distinguishes_smooth_motion_from_isolated_encoder_spike():
    t = np.arange(600) / 30
    smooth = np.tile((5 * np.sin(2 * np.pi * .3 * t))[:, None], (1, 5))
    clean = summarize_joint_trace(smooth, 30)
    assert clean["isolated_return_spikes_over_1deg"] == [0] * 5
    assert max(clean["power_fraction_above_5hz_position"]) < .001
    corrupted = smooth.copy()
    corrupted[25, 2] += 4
    bad = summarize_joint_trace(corrupted, 30)
    assert bad["isolated_return_spikes_over_1deg"] == [0, 0, 1, 0, 0]
    assert bad["power_fraction_above_5hz_position"][2] > clean["power_fraction_above_5hz_position"][2]


def test_no_low_motion_windows_has_no_noise_estimate():
    moving = np.tile(np.arange(100)[:, None], (1, 5)).astype(float)
    result = summarize_joint_trace(moving, 30)
    assert result["low_motion_windows"] == 0
    assert result["low_motion_noise"] is None


def test_inner_split_preserves_original_validation_and_is_reproducible():
    original = {"train": list(range(28)), "val": [101, 103], "excluded": [55]}
    a = inner_split(original)
    assert a == inner_split(original)
    assert len(a["train"]) == 22 and len(a["val"]) == 6
    assert not set(a["train"]) & set(a["val"])
    assert set(a["train"] + a["val"]) == set(original["train"])
    assert a["historical_val"] == original["val"]


def test_materialized_variants_preserve_targets_and_fit_both_stats_only_on_inner_train(tmp_path):
    source, out = tmp_path / "source", tmp_path / "variants"
    source.mkdir()
    rng = np.random.default_rng(3)
    episodes = np.repeat(np.arange(5), 6)
    state = rng.normal(size=(30, 13)).astype(np.float32)
    action = rng.normal(size=(30, 7)).astype(np.float32)
    # Historical validation would contaminate statistics dramatically if selected.
    state[episodes == 4] += 10000
    action[episodes == 4] += 10000
    extrapolated = np.tile([False] * 5 + [True], 5)
    action[extrapolated] += 20000
    np.savez(source / "arrays.npz", state=state, action=action, episode_index=episodes,
             joint_state=np.ones((30, 6), np.float32), timestamp=np.tile(np.arange(6) / 10, 5),
             label_is_extrapolated=extrapolated)
    np.save(source / "images_iphone.npy", np.zeros((30, 2, 2, 3), np.uint8))
    (source / "stats.json").write_text(json.dumps({"image_normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]}}))
    manifest = {"mode": "cartesian", "policy_hz": 10, "state_names": PROPRIO_NAMES,
                "split": {"train": [0, 1, 2, 3], "val": [4], "excluded": []},
                "image_files": {"observation.images.iphone": "images_iphone.npy"},
                "files_sha256": {f: sha256_file(source / f) for f in ("arrays.npz", "stats.json", "images_iphone.npy")}}
    (source / "manifest.json").write_text(json.dumps(manifest))
    source_hashes = {p.name: sha256_file(p) for p in source.iterdir()}
    report = prepare_variants(source, out)
    train = np.isin(episodes, report["split"]["train"])
    for name in report["variants"]:
        with np.load(out / name / "arrays.npz") as derived:
            np.testing.assert_array_equal(derived["action"], action)
            np.testing.assert_array_equal(derived["canonical_state"], state)
        stats = json.loads((out / name / "stats.json").read_text())
        np.testing.assert_allclose(stats["action"]["mean"], action[train & ~extrapolated].mean(axis=0))
        assert stats["observation.state"]["count"] == train.sum()
        assert stats["train_episodes"] == report["split"]["train"]
        assert (out / name / "images_iphone.npy").resolve() == source / "images_iphone.npy"
    assert source_hashes == {p.name: sha256_file(p) for p in source.iterdir()}
    with pytest.raises(FileExistsError):
        prepare_variants(source, out)
