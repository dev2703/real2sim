from pathlib import Path

import numpy as np
import pytest

from real2sim.config import load_yaml, validate_camera_keys
from real2sim.data.transforms import (
    adjust_intrinsics,
    letterbox_geometry,
    letterbox_rgb,
    to_chw_float,
)

REPO = Path(__file__).resolve().parents[2]


def test_letterbox_720p_to_192x256():
    img = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    out, g = letterbox_rgb(img, 192, 256)
    assert out.shape == (192, 256, 3)
    assert (g.resized_w, g.resized_h) == (256, 144)
    assert (g.pad_left, g.pad_top) == (0, 24)
    assert g.scale == pytest.approx(0.2)
    # padding rows are the pad value, content rows are not
    assert np.all(out[:24] == 114) and np.all(out[-24:] == 114)
    assert not np.all(out[24:-24] == 114)


def test_letterbox_preserves_aspect_for_other_sources():
    g = letterbox_geometry(640, 480, 256, 192)  # 4:3 fills exactly
    assert (g.resized_w, g.resized_h, g.pad_left, g.pad_top) == (256, 192, 0, 0)
    g = letterbox_geometry(1920, 1080, 256, 192)
    assert abs(g.resized_w / g.resized_h - 16 / 9) < 0.02


def test_chw_float_range():
    img = np.full((192, 256, 3), 255, np.uint8)
    t = to_chw_float(img)
    assert t.shape == (3, 192, 256) and t.dtype == np.float32
    assert t.max() == 1.0 and t.min() == 1.0


def test_intrinsics_follow_letterbox():
    K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1]])
    g = letterbox_geometry(1280, 720, 256, 192)
    K2 = adjust_intrinsics(K, g)
    assert K2[0, 0] == pytest.approx(200) and K2[1, 1] == pytest.approx(200)
    assert K2[0, 2] == pytest.approx(128) and K2[1, 2] == pytest.approx(72 + 24)


def test_data_config_selects_single_audited_camera():
    cfg = load_yaml(REPO / "configs" / "data.yaml")
    keys = validate_camera_keys(cfg, ["observation.images.iphone", "observation.images.top"])
    assert keys == ["observation.images.iphone"]
    assert cfg["image"]["height"] == 192 and cfg["image"]["width"] == 256
    assert cfg["policy_hz"] == 10 and cfg["split"]["n_val"] == 10
    with pytest.raises(ValueError, match="not present"):
        validate_camera_keys(cfg, ["observation.images.top"])
    bad = dict(cfg, camera_keys=["observation.images.top"])
    with pytest.raises(ValueError, match="excluded"):
        validate_camera_keys(bad, ["observation.images.top"])
