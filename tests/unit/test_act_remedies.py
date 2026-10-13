"""Ablation isolation, deterministic training augmentation, and validation leak guards."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from real2sim.evaluation.vision_ablation import (
    ImageCondition,
    cross_episode_donors,
    stratified_indices,
)
from real2sim.policy.augmentation import ImageAugmentation
from real2sim.policy.dataset import DerivedTickDataset, Normalizer


@pytest.fixture
def derived(tmp_path):
    geom = dict(pad_top=2, pad_left=0, resized_h=12, resized_w=20)
    manifest = {"camera_keys": ["observation.images.iphone"], "letterbox": geom,
                "image_files": {"observation.images.iphone": "images.npy"},
                "split": {"train": [0], "val": [1, 2]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    stats = {"observation.state": {"mean": [0]*6, "std": [1]*6},
             "action": {"mean": [0]*6, "std": [1]*6},
             "image_normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    np.savez(tmp_path / "arrays.npz", episode_index=np.repeat([0, 1, 2], 4),
             ep_start=np.repeat([0, 4, 8], 4), ep_len=np.full(12, 4),
             state=np.ones((12, 6)), action=np.arange(72).reshape(12, 6))
    images = np.random.default_rng(0).integers(0, 256, (12, 16, 20, 3), dtype=np.uint8)
    images[:, :2] = 114
    images[:, -2:] = 114
    np.save(tmp_path / "images.npy", images)
    return tmp_path, geom


def test_augmentation_seed_padding_and_source_immutable(derived):
    root, geom = derived
    image = np.load(root / "images.npy")[0]
    before = image.copy()
    aug = ImageAugmentation(dict(brightness=.2, contrast=.2, saturation=.15, shift_pixels=4), geom)
    torch.manual_seed(1)
    a = aug(image)
    torch.manual_seed(1)
    b = aug(image)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(a[:2], image[:2])
    np.testing.assert_array_equal(a[-2:], image[-2:])
    assert a.dtype == np.uint8 and a.shape == image.shape
    assert not np.array_equal(a[2:-2], image[2:-2])


def test_dataset_augmentation_only_changes_training_image(derived):
    root, _ = derived
    clean = DerivedTickDataset(root, [0], 3)
    aug = DerivedTickDataset(root, [0], 3, augmentation={"brightness": .3})
    a, b = clean[0], aug[0]
    for key in ("action", "observation.state", "action_is_pad"):
        torch.testing.assert_close(a[key], b[key])
    assert not torch.equal(a["observation.images.iphone"], b["observation.images.iphone"])
    with pytest.raises(ValueError, match="training episodes"):
        DerivedTickDataset(root, [0, 1], 3, augmentation={})
    val = DerivedTickDataset(root, [1], 3)
    torch.testing.assert_close(val[0]["observation.images.iphone"], val[0]["observation.images.iphone"])


def test_donors_are_cross_episode_and_repeatable():
    eps = np.repeat([1, 2, 3], [5, 3, 2])
    donors = cross_episode_donors(eps, 10)
    assert np.all(eps[donors] != eps)
    np.testing.assert_array_equal(donors, cross_episode_donors(eps, 10))
    with pytest.raises(ValueError):
        cross_episode_donors(np.ones(3), 0)


def test_ablation_preserves_state_labels_masks_and_order(derived):
    root, _ = derived
    ds = DerivedTickDataset(root, [1, 2], 3)
    idx = stratified_indices(ds, 2)
    np.testing.assert_array_equal(idx, [0, 3, 4, 7])
    donors = cross_episode_donors(ds.arrays["episode_index"][ds.indices[idx]], 1)
    blank = ImageCondition(ds, idx, "blank")
    shuffled = ImageCondition(ds, idx, "shuffled", donors)
    norm = Normalizer.from_json(root / "stats.json")
    for i in range(len(idx)):
        clean = ds[int(idx[i])]
        for condition in (blank, shuffled):
            changed = condition[i]
            for key in ("action", "observation.state", "action_is_pad"):
                torch.testing.assert_close(clean[key], changed[key])
            assert changed["tick"] == clean["tick"]
        rgb = blank[i][ds.camera_keys[0]].numpy() * norm.img_std + norm.img_mean
        np.testing.assert_allclose(rgb, 114/255, atol=1e-6)
        torch.testing.assert_close(shuffled[i][ds.camera_keys[0]], ds[int(idx[donors[i]])][ds.camera_keys[0]])


def test_stratification_rejects_empty_sample_count():
    with pytest.raises(ValueError):
        stratified_indices(SimpleNamespace(), 0)


@pytest.mark.parametrize("cfg", [{"brightness": float("nan")}, {"shift_pixels": 1.2}, {"rotate": 10}])
def test_invalid_augmentation_refused(cfg):
    with pytest.raises(ValueError):
        ImageAugmentation(cfg, {})
