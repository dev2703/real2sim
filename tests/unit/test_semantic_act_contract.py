"""Semantic ACT sees causal commands and a common robot-independent grasp point."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_v2_semantic_act_20261027 import features, point


def test_command_intent_is_lagged_and_age_cannot_read_future_commands():
    points = np.c_[np.arange(6) * .001, np.zeros((6, 2))]
    intents = np.array([1., 1., 0., 0., 1., 1.])
    actual = features(points, intents, 12.5)
    np.testing.assert_allclose(actual[:, 6], [1, 1, 1, 0, 0, 1])
    np.testing.assert_allclose(actual[:, 7], [0, .08, .16, 0, .08, 0])
    changed = intents.copy()
    changed[3:] = 1. - changed[3:]
    np.testing.assert_array_equal(features(points, changed, 12.5)[:4], actual[:4])
    np.testing.assert_allclose(actual[0, 3:6], 0)
    np.testing.assert_allclose(actual[1:, 3], .0125)


def test_grasp_point_uses_tool_rotation_not_world_offset():
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    pose[:3, 3] = [.2, .1, .05]
    np.testing.assert_allclose(point(pose, np.array([-.01, 0., .03])), [.23, .1, .06])


def test_optional_semantic_payload_has_so_only_stats_and_no_phase():
    root = Path("outputs/v2/semantic_act_transfer_20261027_v2")
    if not (root / "semantic_data.npz").is_file():
        pytest.skip("optional bounded semantic experiment not run")
    source = json.loads((root / "source.json").read_text())
    stats = json.loads((root / "normalization.json").read_text())
    payload = np.load(root / "semantic_data.npz")
    train = np.isin(payload["episode_index"], source["split"]["train"])
    np.testing.assert_allclose(stats["state"]["mean"], payload["state"][train].mean(0))
    np.testing.assert_allclose(stats["objects"]["mean"], payload["objects"][train].mean(0))
    assert np.all(np.asarray(stats["objects"]["std"]) >= .03)
    assert set(payload["action"][:, 3]) == {0., 1.}
    assert payload["state"].shape[1] == 8 and payload["objects"].shape[1] == 4
    assert source["phase_or_privileged_object_inputs"] is False
    assert "SO101" in source["source"]
