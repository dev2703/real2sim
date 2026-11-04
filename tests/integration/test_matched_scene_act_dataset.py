"""Audit the simulated teacher payload's causal command/state contract."""

import json
from pathlib import Path

import numpy as np
import pytest

from real2sim.config import load_yaml
from real2sim.data.prepare import compute_stats
from real2sim.geometry.so101_fk import JointMapping
from real2sim.sim.env import sim_from_real
from scripts.run_v2_matched_scene_act import (
    DEVELOPMENT_SEEDS,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
)

DATA = Path("outputs/v2/matched_scene_act_20261026/data")
pytestmark = [pytest.mark.integration,
    pytest.mark.skipif(not (DATA / "manifest.json").is_file(),
                       reason="optional matched-scene teacher payload is not collected")]


@pytest.fixture
def payload():
    manifest = json.loads((DATA / "manifest.json").read_text())
    arrays = dict(np.load(DATA / "arrays.npz", allow_pickle=False))
    stats = json.loads((DATA / "stats.json").read_text())
    return manifest, arrays, stats


def test_training_statistics_and_seed_partitions_are_causal(payload):
    manifest, arrays, stats = payload
    split = manifest["split"]
    assert set(split["train"]) <= set(TRAIN_SEEDS)
    assert set(split["val"]) <= set(VALIDATION_SEEDS)
    assert not (set(split["train"]) & set(split["val"]))
    assert not (set(np.unique(arrays["episode_index"])) & set(DEVELOPMENT_SEEDS))
    selected = np.isin(arrays["episode_index"], split["train"])
    assert stats["observation.state"] == compute_stats(arrays["state"][selected])
    assert stats["action"] == compute_stats(arrays["action"][selected])
    assert manifest["source"]["repo_id"].startswith("sim_teacher/")
    assert manifest["physical_calibration_accepted"] is False


def test_observation_is_before_action_and_labels_are_commands_not_readbacks(payload):
    manifest, arrays, _ = payload
    mapping = JointMapping.from_task_config(load_yaml("configs/v2/task.scalar_measured_20261026.yaml"))
    observed = np.stack([sim_from_real(mapping, row) for row in arrays["state"]])
    commands = np.stack([sim_from_real(mapping, row) for row in arrays["action"]])
    same_episode = arrays["episode_index"][:-1] == arrays["episode_index"][1:]
    np.testing.assert_allclose(observed[1:][same_episode],
        arrays["achieved_successor_native_rad"][:-1][same_episode], atol=3e-7)
    assert np.max(abs(commands - arrays["achieved_successor_native_rad"])) > .005
    config = load_yaml("configs/v2/sim_so101.waypoint_20261026.yaml")
    caps = np.asarray(config["control"]["max_velocity"]) / manifest["policy_hz"]
    assert np.all(abs(np.diff(commands, axis=0)[same_episode]) <= caps + 3e-7)
    assert not arrays["label_is_extrapolated"].any()


def test_chunks_and_nominal_clock_do_not_cross_teacher_trial_boundaries(payload):
    manifest, arrays, _ = payload
    for episode in np.unique(arrays["episode_index"]):
        ticks = np.flatnonzero(arrays["episode_index"] == episode)
        assert np.array_equal(ticks, np.arange(ticks[0], ticks[-1] + 1))
        assert np.all(arrays["ep_start"][ticks] == ticks[0])
        assert np.all(arrays["ep_len"][ticks] == len(ticks))
        np.testing.assert_allclose(arrays["timestamp"][ticks],
            np.arange(len(ticks)) / manifest["policy_hz"], atol=1e-12)
