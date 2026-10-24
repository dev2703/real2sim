import cv2
import numpy as np
import pytest

from real2sim.geometry.image_measurements import (
    bowl_rim_candidates,
    ellipse_points,
    fit_plane_circle,
)


def test_complete_rim_accepted_partial_arc_does_not_invent_circle():
    full = np.zeros((240, 320, 3), np.uint8)
    cv2.ellipse(full, (160, 90), (90, 50), 5, 0, 360, (255, 255, 255), 2)
    candidates = bowl_rim_candidates(full, [60, 30, 260, 210])
    assert candidates
    assert max(candidates[0]["diameters_px"]) == pytest.approx(180, abs=5)
    partial = np.zeros_like(full)
    cv2.ellipse(partial, (160, 90), (90, 50), 5, 0, 100, (255, 255, 255), 2)
    assert not bowl_rim_candidates(partial, [60, 30, 260, 210])


def test_metric_circle_preserves_frame_translation_and_reports_wrong_shape():
    points = ellipse_points([.2, -.1], [.12, .12], 23)
    result = fit_plane_circle(points)
    assert result["diameter_m"] == pytest.approx(.12)
    np.testing.assert_allclose(result["center_plane_xy_m"], [.2, -.1])
    assert result["circle_rms_m"] < 1e-10
    wrong = fit_plane_circle(ellipse_points([.2, -.1], [.12, .08], 23))
    assert wrong["second_moment_axis_ratio"] < .8
    assert wrong["circle_rms_m"] > .005
