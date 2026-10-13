"""Protect original labels/splits and reject ambiguous or corrupt image pairs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from real2sim.data.paired_approach import stationary_prefix
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.paired import PairedViewDataset, file_sha256


@pytest.fixture
def data(tmp_path):
    root, pairs = tmp_path / "real", tmp_path / "pairs"
    root.mkdir()
    pairs.mkdir()
    manifest = {"source": {"revision": "fixture"}, "split": {"train": [0], "val": [1]},
                "files_sha256": {"arrays.npz": "fixture"}, "camera_keys": ["observation.images.iphone"],
                "image_files": {"observation.images.iphone": "images.npy"}, "image_shape_hwc": [8, 10, 3]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    stats = {"observation.state": {"mean": [0]*6, "std": [1]*6},
             "action": {"mean": [0]*6, "std": [1]*6},
             "image_normalization": {"mean": [.5]*3, "std": [.2]*3}}
    (root / "stats.json").write_text(json.dumps(stats))
    np.savez(root / "arrays.npz", episode_index=[0, 0, 0, 1, 1, 1],
             ep_start=[0, 0, 0, 3, 3, 3], ep_len=[3]*6,
             state=np.arange(36).reshape(6, 6), action=np.arange(36).reshape(6, 6)*2)
    np.save(root / "images.npy", np.full((6, 8, 10, 3), 50, np.uint8))
    np.save(pairs / "ticks.npy", np.array([0, 1], dtype=np.int64))
    np.save(pairs / "images.npy", np.full((2, 8, 10, 3), 200, np.uint8))
    pm = {"derived_manifest": manifest, "split_name": "train", "camera_key": manifest["camera_keys"][0],
          "sha256": {n: file_sha256(pairs / n) for n in ("ticks.npy", "images.npy")}}
    (pairs / "manifest.json").write_text(json.dumps(pm))
    return root, pairs


def test_pair_only_changes_eligible_image(data):
    root, pairs = data
    original = DerivedTickDataset(root, [0], 2)
    paired = PairedViewDataset(original, pairs, 1)
    for i in range(3):
        a, b = original[i], paired[i]
        for key in ("action", "action_is_pad", "observation.state"):
            torch.testing.assert_close(a[key], b[key])
        assert a["tick"] == b["tick"] and a["episode_index"] == b["episode_index"]
        assert torch.equal(a[original.camera_keys[0]], b[original.camera_keys[0]]) == (i == 2)
    assert np.load(root / "images.npy").max() == 50


def test_pair_rejects_validation_use_and_corruption(data):
    root, pairs = data
    val = DerivedTickDataset(root, [1], 2)
    with pytest.raises(ValueError, match="episode split"):
        PairedViewDataset(val, pairs)
    with pytest.raises(ValueError, match="split"):
        PairedViewDataset(val, pairs, evaluation=True)
    with (pairs / "images.npy").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        PairedViewDataset(DerivedTickDataset(root, [0], 2), pairs)


def test_pair_rejects_cross_split_tick_even_with_valid_checksum(data):
    root, pairs = data
    np.save(pairs / "ticks.npy", np.array([0, 3], dtype=np.int64))
    p = pairs / "manifest.json"
    m = json.loads(p.read_text())
    m["sha256"]["ticks.npy"] = file_sha256(pairs / "ticks.npy")
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="selected split"):
        PairedViewDataset(DerivedTickDataset(root, [0], 2), pairs)


@pytest.mark.parametrize("missing", ["ticks.npy", "images.npy"])
def test_pair_rejects_missing_required_checksums(data, missing):
    root, pairs = data
    path = pairs / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["sha256"][missing]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="must checksum exactly"):
        PairedViewDataset(DerivedTickDataset(root, [0], 2), pairs)


def test_replacement_probability_is_seeded(data):
    root, pairs = data
    ds = PairedViewDataset(DerivedTickDataset(root, [0], 2), pairs, .5)
    torch.manual_seed(0)
    a = [float(ds[0]["observation.images.iphone"].mean()) for _ in range(20)]
    torch.manual_seed(0)
    b = [float(ds[0]["observation.images.iphone"].mean()) for _ in range(20)]
    assert a == b and len(set(a)) == 2


@pytest.mark.parametrize("bad,reason", [(None, "ball_not_visible"), ((30, 0, 100), "ball_moved"),
                                        ((0, 0, 40), "ball_area_changed")])
def test_stationary_prefix_never_resumes_after_uncertainty(bad, reason):
    prefix, _, why = stationary_prefix([(0, 0, 100)]*6 + [bad] + [(0, 0, 100)]*8)
    assert prefix.tolist() == list(range(6)) and why.startswith(reason)


def test_moving_start_is_not_assumed_stationary():
    prefix, centre, reason = stationary_prefix([(i*5, 0, 100) for i in range(10)])
    assert not len(prefix) and centre is None and reason == "initial_ball_not_stationary"


def test_completion_notification_dispatches_once_without_inference_loop(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/run_paired_act.py"
    spec = importlib.util.spec_from_file_location("paired_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="queued", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.notify({"status": "completed"}, {"codex_bin": "codex", "thread_id": "fixture"}, tmp_path)
    assert len(calls) == 1 and calls[0][:2] == ["codex", "queue"]
    with pytest.raises(FileExistsError):
        module.notify({"status": "completed"}, {"codex_bin": "codex", "thread_id": "fixture"}, tmp_path)
    assert len(calls) == 1
