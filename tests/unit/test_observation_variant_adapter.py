"""Queued-action ticks must still update causal observation histories exactly once."""

import numpy as np
import pytest

from real2sim.data.labels import ACTION_NAMES
from real2sim.policy.adapters import ObservationAdapter
from real2sim.policy.dataset import Normalizer
from real2sim.policy.observation_variants import (
    ObservationVariant,
    state_names,
    transform_observations,
)


@pytest.mark.parametrize("spec", [ObservationVariant(history=3), ObservationVariant(kind="ema_velocity"),
                                  ObservationVariant(kind="joints"), ObservationVariant(kind="no_velocity")])
def test_streaming_adapter_matches_offline_with_skipped_inference(spec):
    canonical = np.zeros((6, 13), np.float32)
    canonical[:, 0] = np.arange(6) * .001
    canonical[1:, 9] = .01
    canonical[:, 12] = .5
    joints = np.zeros((6, 6), np.float32)
    joints[:, 5] = 50
    class Env:
        tick = 0

        def canonical_proprio(self, previous):
            return canonical[self.tick].copy()

        def ee_pose(self):
            pose = np.eye(4)
            pose[:3, 3] = canonical[self.tick, :3]
            return pose

        def real_state(self):
            return joints[self.tick]

    contract = {"state_names": state_names(spec), "action_names": list(ACTION_NAMES),
                "camera_keys": ["observation.images.iphone"], "image_shape_chw": [3, 192, 256],
                "observation_variant": spec.to_dict(), "policy_hz": 10}
    stats = {"observation.state": {"mean": [0] * len(state_names(spec)), "std": [1] * len(state_names(spec))},
             "action": {"mean": [0] * 7, "std": [1] * 7}, "image_normalization": {"mean": [0] * 3, "std": [1] * 3}}
    adapter = ObservationAdapter(contract, Normalizer(stats))
    expected = transform_observations({"state": canonical, "joint_state": joints,
                                       "episode_index": np.zeros(6, int)}, spec)
    env = Env()
    for tick in range(6):
        env.tick = tick
        if tick % 2 == 0:
            np.testing.assert_allclose(adapter.state(env), expected[tick], atol=1e-7)
            np.testing.assert_array_equal(adapter.state(env), adapter.state(env))
        adapter.advance_tick(env)
    adapter.reset()
    env.tick = 0
    np.testing.assert_array_equal(adapter.state(env), expected[0])
