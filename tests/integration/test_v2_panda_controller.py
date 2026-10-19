"""Persistent Cartesian target parity: rejected IK and episode reset cannot advance targets."""
import numpy as np
import pytest

from real2sim.panda.adapter import PandaCartesianActionAdapter
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.sim.kinematics import IKResult


@pytest.mark.integration
def test_panda_persistent_target_survives_rejection_and_clears_at_reset(monkeypatch):
    with PandaPickPlaceEnv(render_images=False) as env:
        obs, _ = env.reset(seed=0)
        adapter = PandaCartesianActionAdapter(env, target_reference="accepted_target")
        result = adapter.apply(np.array([0, 0, 0, 0, 0, 0, 1.]), obs)
        assert not result.rejected
        initial = adapter._accepted_target.copy()
        # Nonzero feedback after servo movement must retain the same absolute target.
        result = adapter.apply(np.array([0, 0, 0, 0, 0, 0, 1.]), result.observation)
        np.testing.assert_allclose(adapter._accepted_target, initial, atol=1e-7)
        initial = adapter._accepted_target.copy()

        def reject(target, seed, **kwargs):
            return IKResult(seed.copy(), False, .1, .1, 0, "fixture_rejection")
        monkeypatch.setattr(adapter.ik, "solve", reject)
        failed = adapter.apply(np.array([.01, 0, 0, 0, 0, 0, 1.]), result.observation)
        assert failed.rejected
        np.testing.assert_array_equal(adapter._accepted_target, initial)
        adapter.reset_stats()
        assert adapter._accepted_target is None and adapter.stats["ticks"] == 0
