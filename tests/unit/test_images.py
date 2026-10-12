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
    # cv2.resize pixel-centre convention: x' = (x + 0.5) s - 0.5
    assert K2[0, 2] == pytest.approx(640.5 * 0.2 - 0.5) and K2[1, 2] == pytest.approx(360.5 * 0.2 - 0.5 + 24)


@pytest.mark.parametrize("src_w,src_h", [(1280, 720), (641, 479), (1920, 1080)])
def test_adjusted_intrinsics_predict_where_a_blob_lands(src_w, src_h):
    """Project a point with K, draw a blob there, letterbox; the centroid matches K' projection."""
    K = np.array([[900.0, 0, src_w / 2 + 3.3], [0, 900.0, src_h / 2 - 7.1], [0, 0, 1]])
    p_cam = np.array([0.05, -0.02, 1.0])
    u, v = (K @ p_cam)[:2] / p_cam[2]
    img = np.zeros((src_h, src_w, 3), np.uint8)
    yy, xx = np.mgrid[:src_h, :src_w]
    img[(xx - u) ** 2 + (yy - v) ** 2 <= 30**2] = 255
    out, g = letterbox_rgb(img, 192, 256, pad_value=0)
    w = out[..., 0].astype(np.float64)
    ys, xs = np.mgrid[: out.shape[0], : out.shape[1]]
    cu, cv = (w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()
    K2 = adjust_intrinsics(K, g)
    u2, v2 = (K2 @ p_cam)[:2] / p_cam[2]
    assert abs(u2 - cu) < 0.15 and abs(v2 - cv) < 0.15, ((u2, v2), (cu, cv))


@pytest.mark.parametrize("src", [(1000, 3), (3, 1000), (1, 1), (2000, 1), (1, 2000)])
def test_letterbox_extreme_aspect_ratios_stay_inside(src):
    w, h = src
    g = letterbox_geometry(w, h, 256, 192)
    assert 1 <= g.resized_w <= 256 and 1 <= g.resized_h <= 192
    assert g.pad_left + g.resized_w <= 256 and g.pad_top + g.resized_h <= 192


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


def test_letterbox_rejects_non_positive_sizes():
    with pytest.raises(ValueError):
        letterbox_geometry(0, 10, 256, 192)
    out, g = letterbox_rgb(np.zeros((1, 2000, 3), np.uint8), 192, 256)
    assert out.shape == (192, 256, 3) and g.resized_h == 1
