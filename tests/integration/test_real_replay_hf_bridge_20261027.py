"""The real-data bridge must reject incompatible references without repair."""

from pathlib import Path

import numpy as np
import pytest

from real2sim.data.raw import RawDataset
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics
from scripts.diag_v2_real_replay_hf_bridge_20261027 import (
    first_strict_start,
    guarded_step,
    native_valid,
)
from scripts.evaluate_hf_act_20261027 import prepare_scene

RAW = Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v2"
pytestmark = [pytest.mark.integration]


def test_bridge_rejects_out_of_range_source_target_without_clipping(tmp_path):
    scene = prepare_scene(tmp_path)
    with So101PickPlaceEnv(scene, render_images=False) as env:
        env.reset(seed=0)
        guard = NativeKinematics(env.model, env.config)
        requested = env.sim_state()
        requested[1] = env.limits[1, 0] - .1
        original = requested.copy()
        assert not native_valid(requested, env.limits)
        _, accepted, reason, _, _, info = guarded_step(env, guard, requested)
        assert not accepted and reason == "target_native_limits"
        np.testing.assert_array_equal(requested, original)
        assert not any(info["control"]["limit_clipped"])


@pytest.mark.skipif(not (RAW / "meta/info.json").exists(), reason="optional raw dataset")
def test_bridge_keeps_full_start_incompatibility_and_labels_suffix(tmp_path):
    raw = RawDataset(RAW)
    states = raw.stack(raw.episode_rows(4), "observation.state")
    unchanged = states.copy()
    scene = prepare_scene(tmp_path)
    with So101PickPlaceEnv(scene, render_images=False) as env:
        assert not native_valid(sim_from_real(env.mapping, states[0]), env.limits)
        with pytest.raises(ValueError, match="upstream model limits"):
            env.reset(seed=0, scenario={"joint_positions": sim_from_real(env.mapping, states[0])})
        frame, reasons = first_strict_start(env, states)
        assert frame == 110 and reasons == {"recorded_native_limits": 110}
        np.testing.assert_allclose(env.real_state(), states[frame], rtol=0, atol=1e-5)
    np.testing.assert_array_equal(states, unchanged)
