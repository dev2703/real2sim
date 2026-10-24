"""Range audit must recover physical scan coordinates, without refitting edges."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_audit():
    path = Path(__file__).resolve().parents[2] / "scripts/summarize_v2_fixed_landmark_tracks.py"
    spec = importlib.util.spec_from_file_location("fixed_landmark_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scan_offsets_survive_rotated_scaled_capture_coordinates():
    audit = load_audit()
    theta, scale = .12, .97
    matrix = np.array([[scale * np.cos(theta), -scale * np.sin(theta), 14.],
                       [scale * np.sin(theta), scale * np.cos(theta), -9.]])
    spec = {"ends": [[400., 300.], [375., 600.]], "normal": [1., 0.]}
    ends = np.c_[spec["ends"], [1., 1.]] @ matrix.T
    normal = matrix[:, :2] @ spec["normal"]
    normal /= np.linalg.norm(normal)
    indices, offsets = np.array([0, 7, 18, 30]), np.array([-32., -29., 0., 30.])
    points = ends[0] + indices[:, None] / 30 * (ends[1] - ends[0]) + offsets[:, None] * normal
    actual_indices, actual_offsets = audit.reconstruct_scan_offsets(points, spec, matrix, 31)
    np.testing.assert_array_equal(actual_indices, indices)
    np.testing.assert_allclose(actual_offsets, offsets, atol=1e-10)
    edge = {"status": "measured", "points": points, "rays": 31}
    result = audit.audit_edge(edge, spec, matrix, 32, 2)
    assert result["range_limited"]
    assert result["near_search_boundary_rays"] == 2
    with pytest.raises(ValueError, match="declared scan ray"):
        audit.reconstruct_scan_offsets(points + .1 * (ends[1] - ends[0]) / 30,
                                       spec, matrix, 31)


def test_missing_edge_is_not_reported_as_uncensored():
    result = load_audit().audit_edge({"status": "insufficient_line_consensus"}, {}, None, 32, 2)
    assert result["range_limited"] is None
