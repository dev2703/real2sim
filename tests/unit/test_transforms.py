import numpy as np
import pandas as pd
import pytest

from real2sim.data.prepare import prepare
from real2sim.data.raw import RawDataset
from real2sim.data.synthetic import make_synthetic_dataset
from real2sim.data.transforms import letterbox_geometry, letterbox_rgb


@pytest.mark.parametrize(
    ("src_h", "src_w", "dst_h", "dst_w", "pad_value"),
    [
        (720, 1280, 192, 256, 114),
        (480, 640, 192, 256, 114),
        (600, 400, 192, 256, 37),
        (96, 128, 48, 64, 0),
        (1, 2000, 192, 256, 200),
        (2000, 1, 192, 256, 255),
    ],
)
def test_letterbox_rgb_out_and_cached_geometry_match_fresh_allocation(src_h, src_w, dst_h, dst_w, pad_value):
    rng = np.random.default_rng(src_h * 1000 + src_w)
    img = rng.integers(0, 256, (src_h, src_w, 3), dtype=np.uint8)

    ref, ref_g = letterbox_rgb(img, dst_h, dst_w, pad_value=pad_value)

    out = np.full((dst_h, dst_w, 3), 77, dtype=np.uint8)
    cached_g = letterbox_geometry(src_w, src_h, dst_w, dst_h)
    got, got_g = letterbox_rgb(img, dst_h, dst_w, pad_value=pad_value, out=out, geometry=cached_g)

    assert got is out
    assert got_g == ref_g
    np.testing.assert_array_equal(got, ref)


def test_letterbox_rgb_recomputes_geometry_when_source_shape_changes():
    rng = np.random.default_rng(7)
    img_a = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    img_b = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

    _, g_a = letterbox_rgb(img_a, 192, 256)
    ref_b, ref_g_b = letterbox_rgb(img_b, 192, 256)

    out = np.empty((192, 256, 3), dtype=np.uint8)
    got_b, got_g_b = letterbox_rgb(img_b, 192, 256, out=out, geometry=g_a)

    assert got_g_b == ref_g_b
    np.testing.assert_array_equal(got_b, ref_b)


def test_letterbox_rgb_rejects_mismatched_out_buffer():
    img = np.zeros((96, 128, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="out shape"):
        letterbox_rgb(img, 48, 64, out=np.zeros((48, 64, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="out shape"):
        letterbox_rgb(img, 48, 64, out=np.zeros((48, 64, 3), dtype=np.float32))


def test_letterbox_rgb_handles_non_contiguous_and_non_uint8_sources():
    rng = np.random.default_rng(11)
    src = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)[:, ::-1, :]
    ref, g = letterbox_rgb(np.ascontiguousarray(src), 64, 64, pad_value=114)

    out = np.empty((64, 64, 3), dtype=np.uint8)
    got, g2 = letterbox_rgb(src, 64, 64, pad_value=114, out=out, geometry=g)
    assert g2 == g
    np.testing.assert_array_equal(got, ref)


def test_prepare_avoids_duplicate_episode_row_scans_and_matches_letterbox_pixels(
    tmp_path, monkeypatch
):
    cams = {"observation.images.cam_a": "moving", "observation.images.cam_b": "moving"}
    raw = make_synthetic_dataset(tmp_path / "raw", cameras=cams, episode_lengths=(20, 21, 22), fps=30)
    cfg = {
        "camera_keys": list(cams),
        "excluded_cameras": [],
        "excluded_episodes": [],
        "image": {"height": 48, "width": 64, "pad_value": 114},
        "policy_hz": 10,
        "split": {"seed": 0, "n_val": 1},
    }

    calls = 0
    orig_episode_rows = RawDataset.episode_rows

    def counted_episode_rows(self, episode_index: int):
        nonlocal calls
        calls += 1
        return orig_episode_rows(self, episode_index)

    monkeypatch.setattr(RawDataset, "episode_rows", counted_episode_rows)
    out = tmp_path / "out"
    manifest = prepare(raw, cfg, out, "s", "r", mode="joint")

    assert calls == 0
    ds = RawDataset(raw)
    for key, rel in manifest["image_files"].items():
        saved = np.load(out / rel, mmap_mode="r")
        idx = 0
        for ep in sorted(manifest["split"]["train"] + manifest["split"]["val"]):
            for _, _, frame in ds.iter_frames(key, ep, stride=manifest["stride"]):
                expected, _ = letterbox_rgb(frame.to_ndarray(format="rgb24"), 48, 64, pad_value=114)
                np.testing.assert_array_equal(saved[idx], expected)
                idx += 1
        assert idx == manifest["n_ticks"]


def test_prepare_preserves_episode_table_errors(tmp_path):
    key = "observation.images.cam_a"
    raw = make_synthetic_dataset(tmp_path / "raw", cameras={key: "moving"}, episode_lengths=(15, 15), fps=30)
    cfg = {
        "camera_keys": [key],
        "excluded_cameras": [],
        "excluded_episodes": [],
        "image": {"height": 48, "width": 64, "pad_value": 114},
        "policy_hz": 10,
        "split": {"seed": 0, "n_val": 1},
    }
    data_file = raw / "data" / "chunk-000" / "file-000.parquet"
    df_data = pd.read_parquet(data_file)
    fractional = df_data.copy()
    fractional["episode_index"] = fractional["episode_index"].replace({0: 0.5})
    fractional.to_parquet(data_file, index=False)
    with pytest.raises(ValueError, match="0 rows but episode table says 15"):
        prepare(raw, cfg, tmp_path / "out_fractional", "s", "r", mode="joint")
    df_data[df_data["episode_index"] != 0].to_parquet(data_file, index=False)
    with pytest.raises(ValueError, match="0 rows but episode table says 15"):
        prepare(raw, cfg, tmp_path / "out_missing_rows", "s", "r", mode="joint")

    ep_file = raw / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df_ep = pd.read_parquet(ep_file)
    pd.concat([df_ep, df_ep.iloc[[0]]], ignore_index=True).to_parquet(ep_file, index=False)
    with pytest.raises(ValueError, match="duplicates"):
        prepare(raw, cfg, tmp_path / "out_dup_ep", "s", "r", mode="joint")
