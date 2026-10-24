"""Known image motion, independently moving mat, occlusion, and reserved isolation."""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("fixed_tracks", root / "scripts/diag_v2_fixed_landmark_tracks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_known_translation_and_occlusion(diagnostic):
    rng = np.random.default_rng(4)
    source = cv2.GaussianBlur(rng.integers(0, 256, (180, 220, 3), dtype=np.uint8), (3, 3), 0)
    matrix = np.array([[1., 0., 13.], [0., 1., -7.]])
    target = cv2.warpAffine(source, matrix, (220, 180))
    seeds = [{"landmark_id": "unique", "region": "base", "reference_px": [90., 90.]}]
    row = diagnostic.track_landmarks(source, target, seeds)[0]
    assert row["accepted"]
    np.testing.assert_allclose(row["target_px"], [103., 83.], atol=.2)
    target[60:110, 80:130] = 120
    assert not diagnostic.track_landmarks(source, target, seeds)[0]["accepted"]


def test_repeated_patch_is_not_an_identity(diagnostic):
    rng = np.random.default_rng(9)
    source = np.full((180, 220, 3), 100, np.uint8)
    source[75:106, 75:106] = rng.integers(0, 256, (31, 31, 3), dtype=np.uint8)
    target = source.copy()
    target[75:106, 110:141] = source[75:106, 75:106]
    row = diagnostic.track_landmarks(source, target, [
        {"landmark_id": "repeat", "region": "foot", "reference_px": [90., 90.]}])[0]
    assert not row["accepted"]
    assert "ambiguous_patch" in row["rejection_reasons"]


def test_mat_relative_motion_survives_camera_stabilization(diagnostic):
    source = np.full((240, 300), 200, np.uint8)
    source[100:] = 35
    target = np.full_like(source, 200)
    target[113:] = 35  # camera moves +5, mat moves an additional +8 pixels
    edge = {"ends": [[50., 100.], [250., 100.]], "normal": [0., 1.], "sign": -1.}
    identity = np.array([[1., 0., 0.], [0., 1., 0.]])
    camera = np.array([[1., 0., 0.], [0., 1., 5.]])
    ref = diagnostic.fit_edge(source, edge, identity)
    shifted = diagnostic.fit_edge(target, edge, camera)
    assert ref["status"] == shifted["status"] == "measured"
    result = diagnostic.compare_edges({"top": ref}, {"top": shifted}, camera)
    assert result["top"]["median_signed_normal_px"] == pytest.approx(8., abs=.1)
    missing = diagnostic.fit_edge(np.full_like(source, 100), edge, identity)
    assert missing["status"] == "insufficient_raw_contrast"
    assert diagnostic.compare_edges({"top": ref}, {"top": missing}, camera)["top"]["status"] == "unmeasured"


def test_heldout_holder_cannot_change_static_fit(diagnostic):
    matrix = np.array([[.98, -.01, 12.], [.01, .98, 4.]])
    points = np.array([[200, 350], [240, 350], [240, 390], [200, 390],
                       [60, 450], [100, 450], [100, 490], [60, 490.]])
    targets = diagnostic.transform(points, matrix)
    rows = [{"region": "base_foot" if i < 4 else "table_clamp", "accepted": True,
             "reference_px": p, "target_px": q} for i, (p, q) in enumerate(zip(points, targets, strict=True))]
    rows.append({"region": "base_holder", "accepted": True,
                 "reference_px": [190, 300], "target_px": [500, 500]})
    fit = diagnostic.fit_similarity(rows, diagnostic.FIT_REGIONS)
    assert fit["supported_regions"] == 2
    np.testing.assert_allclose(fit["matrix_reference_to_target"], matrix, atol=1e-5)
    assert diagnostic.residual_for(rows, "base_holder", matrix)["rms_magnitude_px"] > 100


def test_fresh_selection_never_reads_reserved_lengths(diagnostic):
    class Raw:
        def episode_length(self, episode):
            assert episode != 9
            return 90
    split = {"train": [0, 2], "val": [3], "reserved": [9]}
    selected = diagnostic.make_selection(Raw(), split, 4)
    assert selected == diagnostic.make_selection(Raw(), split, 4)
    assert len(selected) == 9
    assert {r["episode"] for r in selected} == {0, 2, 3}
    with pytest.raises(ValueError, match="overlap"):
        diagnostic.make_selection(Raw(), split | {"reserved": [0]}, 4)
