"""Image-derived features must match offline and live observation paths exactly."""

import json

import cv2
import numpy as np
import pytest

from real2sim.data.labels import ACTION_NAMES
from real2sim.policy.adapters import ContractError, ObservationAdapter, contract_kind
from real2sim.policy.dataset import DerivedTickDataset, Normalizer
from real2sim.policy.visual_grounding import (
    BallImageFeatures,
    append_ball_features,
    ball_features,
    grounded_state_names,
)


def image():
    rgb = np.full((192, 256, 3), 114, np.uint8)
    cv2.circle(rgb, (100, 80), 12, (190, 225, 30), -1)
    return rgb


def contract():
    spec = BallImageFeatures()
    return {"state_names": grounded_state_names(), "action_names": list(ACTION_NAMES),
            "camera_keys": [spec.camera_key], "image_shape_chw": [3, 192, 256],
            "visual_grounding": spec.to_dict()}


def stats():
    return {"observation.state": {"mean": [0] * 17, "std": [1] * 17},
            "action": {"mean": [0] * 7, "std": [1] * 7},
            "image_normalization": {"mean": [0] * 3, "std": [1] * 3}}


def test_detector_missing_and_displaced_ball_are_explicit():
    spec = BallImageFeatures()
    found = ball_features(image(), spec)
    np.testing.assert_allclose(found[:2], [100 / 256, 80 / 192], atol=1e-6)
    assert found[-1] == 1
    np.testing.assert_array_equal(ball_features(np.zeros_like(image()), spec), np.zeros(4))
    moved = np.roll(image(), 20, axis=1)
    assert ball_features(moved, spec)[0] == pytest.approx(found[0] + 20 / 256)


def test_dataset_and_live_adapter_match_and_recompute_from_image(tmp_path):
    spec, norm = BallImageFeatures(), Normalizer(stats())
    state = np.arange(13, dtype=np.float32) / 100
    manifest = {**contract(), "image_files": {spec.camera_key: "images.npy"},
                "split": {"train": [0]}, "image_shape_hwc": [192, 256, 3]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    np.savez(tmp_path / "arrays.npz", state=state[None], action=np.zeros((1, 7)),
             episode_index=[0], ep_start=[0], ep_len=[1])
    np.save(tmp_path / "images.npy", image()[None])
    ds = DerivedTickDataset(tmp_path, [0], 1, norm)

    class Env:
        def canonical_proprio(self, previous):
            return state.copy()

    adapter = ObservationAdapter(contract(), norm)
    obs = {spec.camera_key: image().transpose(2, 0, 1).astype(np.float32) / 255}
    batch = adapter.batch(obs, Env())
    np.testing.assert_array_equal(ds[0]["observation.state"].numpy(), batch["observation.state"][0].numpy())
    np.testing.assert_array_equal(batch["observation.state"][0].numpy(), append_ball_features(state, image(), spec))
    obs[spec.camera_key][:] = 0
    np.testing.assert_array_equal(adapter.batch(obs, Env())["observation.state"][0, -4:], np.zeros(4))
    with pytest.raises(ContractError, match="current observed image"):
        adapter.state(Env())


def test_grounding_contract_rejects_missing_camera_and_wrong_feature_order():
    c = contract()
    c["state_names"] = list(reversed(c["state_names"]))
    with pytest.raises(ContractError, match="visual grounding"):
        contract_kind(c)
    c = contract()
    c["camera_keys"] = ["observation.images.top"]
    with pytest.raises(ContractError, match="visual grounding"):
        contract_kind(c)


def test_disabled_control_has_identical_feature_shape_and_no_image_signal():
    spec = BallImageFeatures(enabled=False)
    np.testing.assert_array_equal(ball_features(image(), spec), np.zeros(4))
    np.testing.assert_array_equal(ball_features(np.roll(image(), 10, axis=1), spec), np.zeros(4))


def test_paired_replacement_cannot_leave_stale_real_image_features():
    from real2sim.policy.paired import PairedViewDataset

    class Base:
        manifest = contract()

    with pytest.raises(ValueError, match="recomputing visual grounding"):
        PairedViewDataset(Base(), "/unused")
