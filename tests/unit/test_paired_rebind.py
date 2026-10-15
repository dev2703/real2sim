import json

import numpy as np
import pytest

from real2sim.data.paired_rebind import rebind
from real2sim.policy.paired import file_sha256


def setup_data(tmp_path, *, shifted=False, different_images=False):
    source, target, pairs = [tmp_path / x for x in ("source", "target", "pairs")]
    manifests = []
    for root, dim in [(source, 6), (target, 13)]:
        root.mkdir()
        frames = [3, 0] if shifted and root == target else [0, 3]
        np.savez(root / "arrays.npz", state=np.zeros((2, dim)), action=np.zeros((2, dim)),
                 episode_index=[0, 0], frame_index=frames)
        value = 1 if different_images and root == target else 0
        np.save(root / "images.npy", np.full((2, 2, 2, 3), value, dtype=np.uint8))
        (root / "stats.json").write_text(json.dumps({"dim": dim}))
        manifest = {"source": {"revision": "fixed"}, "split": {"train": [0], "val": [1]},
                    "camera_keys": ["observation.images.iphone"], "image_shape_hwc": [2, 2, 3],
                    "letterbox": {}, "policy_hz": 10,
                    "image_files": {"observation.images.iphone": "images.npy"},
                    "files_sha256": {name: file_sha256(root / name) for name in
                                     ["arrays.npz", "stats.json", "images.npy"]}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        manifests.append(manifest)
    pairs.mkdir()
    np.save(pairs / "ticks.npy", np.array([0, 1]))
    np.save(pairs / "images.npy", np.zeros((2, 2, 2, 3), dtype=np.uint8))
    (pairs / "manifest.json").write_text(json.dumps({
        "derived_manifest": manifests[0], "split_name": "train",
        "sha256": {name: file_sha256(pairs / name) for name in ("ticks.npy", "images.npy")},
    }))
    return pairs, source, target


def test_rebind_changes_only_manifest_and_preserves_original(tmp_path):
    pairs, source, target = setup_data(tmp_path)
    old = (pairs / "manifest.json").read_bytes()
    out = tmp_path / "out"
    result = rebind(pairs, source, target, out)
    assert (pairs / "manifest.json").read_bytes() == old
    assert (out / "images.npy").resolve() == (pairs / "images.npy").resolve()
    assert result["derived_manifest"] == json.loads((target / "manifest.json").read_text())
    with pytest.raises(FileExistsError):
        rebind(pairs, source, target, out)


@pytest.mark.parametrize("kwargs,match", [({"shifted": True}, "alignment"),
                                         ({"different_images": True}, "images differ")])
def test_rebind_rejects_misaligned_pairs_even_with_valid_hashes(tmp_path, kwargs, match):
    pairs, source, target = setup_data(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        rebind(pairs, source, target, tmp_path / "out")


def test_rebind_rejects_actual_file_tampering(tmp_path):
    pairs, source, target = setup_data(tmp_path)
    (target / "stats.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        rebind(pairs, source, target, tmp_path / "out")
