import json

import numpy as np
import pandas as pd
import pytest

from real2sim.data.raw import RawDataset
from real2sim.data.split import SplitManifest, make_episode_split
from real2sim.data.synthetic import JOINTS


def test_metadata(synthetic_root):
    ds = RawDataset(synthetic_root, repo_id="synthetic/test", revision="deadbeef")
    assert ds.codebase_version == "v3.0"
    assert ds.fps == 30
    assert ds.episode_indices == [0, 1, 2]
    assert set(ds.image_keys) == {
        "observation.images.cam_good",
        "observation.images.cam_black",
        "observation.images.cam_frozen",
    }
    assert ds.state_names == JOINTS and ds.action_names == JOINTS
    info = ds.image_info("observation.images.cam_good")
    assert (info["height"], info["width"], info["channels"]) == (96, 128, 3)


def test_rows_are_episode_contiguous(synthetic_root):
    ds = RawDataset(synthetic_root)
    total = 0
    for ep in ds.episode_indices:
        g = ds.episode_rows(ep)
        n = ds.episode_length(ep)
        assert len(g) == n
        assert g["frame_index"].tolist() == list(range(n))
        np.testing.assert_allclose(g["timestamp"].to_numpy(), np.arange(n) / ds.fps, atol=1e-5)
        total += n
    assert total == len(ds.data) == ds.info["total_frames"]


def test_episode_lookup_preserves_interleaved_order_and_empty_schema(synthetic_root):
    ds = RawDataset(synthetic_root)
    # Frame tables need not have contiguous episode blocks or dense IDs.
    ds._data = pd.DataFrame({"episode_index": [9, 2, 9, 2],
                             "frame_index": [3, 7, 4, 8]}, index=[5, 6, 7, 8])
    for episode in (9, 2, 100):
        expected = ds.data[ds.data["episode_index"] == episode].reset_index(drop=True)
        pd.testing.assert_frame_equal(ds.episode_rows(episode), expected)
    rows = ds.episode_rows(9)
    rows.loc[0, "frame_index"] = -1
    assert ds.episode_rows(9)["frame_index"].tolist() == [3, 4]


def test_repeated_episode_reads_do_not_scan_the_episode_column(synthetic_root, monkeypatch):
    ds = RawDataset(synthetic_root)
    ds.episode_rows(0)  # Build the lazy index once.
    original = pd.Series.__eq__

    def reject_scan(series, other):
        if series.name == "episode_index" and len(series) == len(ds.data):
            raise AssertionError("episode lookup scanned the entire frame table")
        return original(series, other)

    monkeypatch.setattr(pd.Series, "__eq__", reject_scan)
    for episode in (0, 1, 2, 99):
        rows = ds.episode_rows(episode)
        assert len(rows) == (0 if episode == 99 else ds.episode_length(episode))


def test_decode_counts_match_episode_lengths(synthetic_root):
    ds = RawDataset(synthetic_root)
    key = "observation.images.cam_good"
    for ep in ds.episode_indices:
        seg = ds.video_segment(key, ep)
        frames = list(ds.iter_frames(key, ep))
        assert len(frames) == ds.episode_length(ep)
        assert frames[0][0] == 0
        assert abs(frames[0][1] - seg.from_ts) < 1e-3
        assert frames[-1][1] < seg.to_ts
        rgb = frames[0][2].to_ndarray(format="rgb24")
        assert rgb.shape == (96, 128, 3)


def test_stride_subsamples(synthetic_root):
    ds = RawDataset(synthetic_root)
    n = ds.episode_length(1)
    frames = list(ds.iter_frames("observation.images.cam_good", 1, stride=5))
    assert len(frames) == (n + 4) // 5
    assert [f[0] for f in frames] == list(range(0, n, 5))


def test_split_is_deterministic_and_leak_free(tmp_path):
    ids = list(range(40))
    a = make_episode_split(ids, n_val=10, seed=0, repo_id="x", revision="y")
    b = make_episode_split(ids, n_val=10, seed=0)
    assert a.train == b.train and a.val == b.val and a.sha256 == b.sha256
    assert (a.n_train, a.n_val) == (30, 10)
    assert not set(a.train) & set(a.val)
    assert sorted(a.train + a.val) == ids
    c = make_episode_split(ids, n_val=10, seed=1)
    assert c.sha256 != a.sha256
    p = tmp_path / "split.json"
    a.to_json(p)
    assert SplitManifest.from_json(p) == a
    with pytest.raises(ValueError):
        make_episode_split(ids, n_val=40)


def test_split_rejects_duplicate_episode_ids():
    with pytest.raises(ValueError, match="duplicates"):
        make_episode_split([1, 1, 2], n_val=1)


def test_unsupported_codebase_version_rejected(tmp_path, synthetic_root):
    bad = tmp_path / "bad"
    (bad / "meta").mkdir(parents=True)
    info = json.loads((synthetic_root / "meta" / "info.json").read_text())
    info["codebase_version"] = "v2.1"
    (bad / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="codebase_version"):
        RawDataset(bad)


def test_file_manifest_hashes_small_files(synthetic_root):
    ds = RawDataset(synthetic_root)
    rows = ds.file_manifest()
    paths = {r["path"] for r in rows}
    assert "meta/info.json" in paths
    info_row = next(r for r in rows if r["path"] == "meta/info.json")
    assert len(info_row["sha256"]) == 64
    video_rows = [r for r in rows if r["path"].endswith(".mp4")]
    assert video_rows and all(r["sha256"] is None for r in video_rows)


def test_exclusion_removes_without_reshuffling():
    ids = list(range(40))
    base = make_episode_split(ids, n_val=10, seed=0)
    ex = make_episode_split(ids, n_val=10, seed=0, exclude=[25, 33, 34])
    assert ex.excluded == [25, 33, 34]
    assert ex.train == [e for e in base.train if e not in (25, 33, 34)]
    assert ex.val == [e for e in base.val if e not in (25, 33, 34)]
    assert not {25, 33, 34} & set(ex.train + ex.val)
    assert ex.sha256 != base.sha256
    with pytest.raises(ValueError, match="not in dataset"):
        make_episode_split(ids, n_val=10, exclude=[99])
