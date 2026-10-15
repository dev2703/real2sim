import copy

import cv2
import numpy as np
import pytest

from real2sim.data.bowl_placement import detect_bowl_rim, validate_placements


def test_rim_detection_tracks_position():
    for centre in [(165, 123), (184, 146)]:
        image = np.full((192, 256, 3), 25, dtype=np.uint8)
        cv2.ellipse(image, centre, (24, 17), 8, 0, 360, (170, 170, 170), 2)
        # Bright clutter outside the expected bowl region must not win.
        cv2.circle(image, (70, 65), 30, (240, 240, 240), 3)
        result = detect_bowl_rim(image)
        assert result is not None
        np.testing.assert_allclose(result["pixel_policy"], centre, atol=1.)


def test_missing_rim_is_not_fabricated():
    assert detect_bowl_rim(np.zeros((192, 256, 3), np.uint8)) is None


def test_rejects_wrong_image_contract():
    with pytest.raises(ValueError):
        detect_bowl_rim(np.zeros((192, 256, 3), np.float32))


@pytest.fixture
def records():
    manifest = {"files_sha256": {"arrays.npz": "example"}, "split": {"train": [5], "val": [2]}}
    record = {"review_status": "visually_reviewed", "derived_files_sha256": manifest["files_sha256"],
              "episodes": [{"episode": 5, "split": "train", "pixel_policy": [170, 120]},
                           {"episode": 2, "split": "val", "pixel_policy": [180, 140]}]}
    return record, manifest


def test_placements_keep_episode_and_split_identity(records):
    record, manifest = records
    result = validate_placements(record, manifest)
    assert result[5]["pixel_policy"] == [170, 120]
    assert result[2]["pixel_policy"] == [180, 140]


@pytest.mark.parametrize("defect", ["unreviewed", "stale", "duplicate", "split", "nan", "padding"])
def test_rejects_invalid_placement_provenance(records, defect):
    record, manifest = copy.deepcopy(records)
    if defect == "unreviewed":
        record["review_status"] = "pending"
    elif defect == "stale":
        record["derived_files_sha256"] = {}
    elif defect == "duplicate":
        record["episodes"][1] = record["episodes"][0]
    elif defect == "split":
        record["episodes"][0]["split"] = "val"
    else:
        record["episodes"][0]["pixel_policy"][1] = float("nan") if defect == "nan" else 0
    with pytest.raises(ValueError):
        validate_placements(record, manifest)
