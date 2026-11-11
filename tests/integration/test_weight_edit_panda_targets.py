"""Panda supervision round-trips through the real native execution geometry."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.policy.weight_edit_collection import accepted_panda_target, native_policy_state
from scripts.benchmark_frozen_act import SharedEpisode
from scripts.diag_v2_learned_panda_retarget_20261027 import target_from_source

pytestmark = pytest.mark.integration


def test_applied_panda_target_round_trip_and_observation_use_same_native_interface(tmp_path):
    shared = SharedEpisode("panda", tmp_path / "setup")
    try:
        shared.reset(10160)
        state = native_policy_state(shared)
        assert state.shape == (6,) and np.isfinite(state).all()
        before_q = shared.source.q.copy()
        command = np.r_[shared.env.arm_qpos(), .04]
        command[0] += .001
        _, _, _, _, info = shared.env.step(command)
        applied = np.asarray(info["control"]["applied"])
        label, valid = accepted_panda_target(shared, SimpleNamespace(info=info, ik_rejected=False))
        assert valid
        assert label[5] == pytest.approx(applied[7] / shared.env.max_opening * 90., abs=1e-5)
        applied_pose = shared.guard.forward(applied[:7], finger_opening=applied[7])
        label_pose = target_from_source(shared.source, shared.source.last_target)
        np.testing.assert_allclose(label_pose[:3, 3], applied_pose[:3, 3], atol=3e-3)
        np.testing.assert_array_equal(shared.source.q, before_q)
    finally:
        shared.close()


def test_float32_stop_rounding_masks_panda_label_instead_of_aborting_episode(tmp_path):
    shared = SharedEpisode("panda", tmp_path / "setup")
    try:
        shared.reset(10160)
        prior = shared.source.last_target.copy()
        at_stop = shared.source.q[:5].copy()
        at_stop[0] = shared.source.limits[0, 1]
        # Exercise the real strict native command check after a bounded inverse
        # lands exactly on a stop. Rounding degrees to float32 exceeds this
        # particular shoulder-pan stop by ~47 nanoradians.
        shared.source.project_pose = lambda pose: SimpleNamespace(accepted=True, arm_qpos=at_stop)
        applied = np.r_[shared.env.arm_qpos(), .04]
        result = SimpleNamespace(info={"control": {"applied": applied}}, ik_rejected=False)
        label, valid = accepted_panda_target(shared, result)
        assert not valid
        np.testing.assert_array_equal(label, np.zeros(6))
        np.testing.assert_array_equal(shared.source.last_target, prior)
    finally:
        shared.close()
