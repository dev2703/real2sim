"""Source-schema recovery preserves point motion and measured jaw semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.geometry.so3 import apply_delta, delta_action, exp_so3, rot6d_from_matrix
from real2sim.geometry.so101_fk import JointMapping
from real2sim.panda.model import POINTING_DOWN
from real2sim.policy.common_recovery import FRAME_NAMES, RecoveryObservation, semantic_gripper
from real2sim.policy.source_recovery_frame import SourceRecoveryFrame

SOURCE_OFFSET = np.array([-.010, 0., .030])
PANDA_OFFSET = np.array([-.008, 0., 0.])


class MeasuredPanda:
    max_opening = .08

    def __init__(self):
        self.pose = np.eye(4)
        self.pose[:3, :3] = POINTING_DOWN
        self.pose[:3, 3] = [.2, .1, .15]
        self.semantic = .35

    def ee_pose(self):
        return self.pose.copy()

    def gripper_opening(self):
        return self.semantic * self.max_opening

    def gripper_semantic(self):
        return self.semantic

    def __getattr__(self, name):
        if name in {"info", "data", "model", "ball", "bowl", "privileged_metrics"}:
            raise AssertionError(f"Source recovery requested forbidden {name}")
        raise AttributeError(name)


class Source:
    def __init__(self, env):
        self.mapping = JointMapping()
        self.panda_offset = PANDA_OFFSET.copy()
        self.q = np.zeros(6)
        self.last_target = self.q.copy()
        self.pose = np.eye(4)
        self.pose[:3, :3] = exp_so3(np.array([0., .5, 0.]))
        point = env.pose[:3, 3] + env.pose[:3, :3] @ PANDA_OFFSET
        self.pose[:3, 3] = point + [.0007, -.0002, .0001] - self.pose[:3, :3] @ SOURCE_OFFSET
        self.fk = SimpleNamespace(fk_sim=lambda q: apply_delta(self.pose, np.r_[q[:3], q[3:5], 0.]))
        self.calls = 0
        self.rejected = False

    def source_state(self, measured):
        self.calls += 1
        if self.rejected:
            raise RuntimeError("unrepresentable measured pose")
        measured.ee_pose()
        measured.gripper_opening()
        self.last_projection = {"accepted": True, "position_tolerance_m": .003,
            "position_error_m": np.linalg.norm([.0007, -.0002, .0001])}
        # Deliberately distinct CAD-gap readback to verify semantic override.
        self.q[5] = self.mapping.gripper_real_to_sim_rad(80.)
        return np.zeros(6), self.last_projection["position_error_m"]

    def source_action(self, raw):
        q = np.r_[self.mapping.arm_real_to_sim_rad(raw[:5]),
            self.mapping.gripper_real_to_sim_rad(raw[5])]
        self.last_target = q.copy()
        return q.copy(), q.copy()


def frame():
    env = MeasuredPanda()
    source = Source(env)
    return SourceRecoveryFrame(env, source, source_offset=SOURCE_OFFSET), env, source


def point(pose, offset):
    return pose[:3, 3] + pose[:3, :3] @ offset


def test_proxy_encodes_source_schema_and_measured_panda_jaw_without_object_truth():
    adapter, env, source = frame()
    proxy = adapter.update()
    assert source.calls == 1
    np.testing.assert_array_equal(proxy.ee_pose(), source.pose)
    assert semantic_gripper(proxy, "so101") == pytest.approx(env.semantic)
    observed = RecoveryObservation("so101", 25.).encode(proxy, [.1, .2, .03], [.3, .1, .03], 0)
    values = dict(zip(FRAME_NAMES, observed[-33:], strict=True))
    assert values["robot_so101"] == 1. and values["robot_panda"] == 0.
    assert values["gripper_measured"] == pytest.approx(.35)
    np.testing.assert_allclose(observed[-33 + 3:-33 + 9], rot6d_from_matrix(source.pose[:3, :3]))
    # Callers cannot rewrite the snapshot or mutate its pose through the proxy.
    proxy.ee_pose()[:] = 99
    proxy.sim_state()[:] = 99
    np.testing.assert_array_equal(proxy.ee_pose(), source.pose)


def test_zero_source_action_does_not_execute_the_projection_residual():
    adapter, _, source = frame()
    adapter.update()
    previous_q, previous_target = source.q.copy(), source.last_target.copy()
    mapped = adapter.map_action(np.r_[np.zeros(6), .4])
    np.testing.assert_allclose(mapped, np.r_[np.zeros(6), .4], atol=1e-15)
    assert adapter.diagnostics()["projection_residual_norm_m"] > .0007
    np.testing.assert_array_equal(source.q, previous_q)
    np.testing.assert_array_equal(source.last_target, previous_target)


@pytest.mark.parametrize("native_tilt", [0., .04])
def test_source_world_rotation_moves_the_grasp_point_with_fixed_native_orientation(native_tilt):
    adapter, env, _ = frame()
    env.pose[:3, :3] = exp_so3(np.array([native_tilt, 0., 0.])) @ POINTING_DOWN
    adapter.update()
    action = np.array([.003, -.002, .001, 0., .1, .03, .2])
    mapped = adapter.map_action(action)
    native_target = apply_delta(env.pose, mapped[:6])
    source_target = apply_delta(adapter.source_pose, action[:6])
    source_change = point(source_target, SOURCE_OFFSET) - point(adapter.source_pose, SOURCE_OFFSET)
    native_change = point(native_target, PANDA_OFFSET) - point(env.pose, PANDA_OFFSET)
    np.testing.assert_allclose(native_change, source_change, atol=1e-12)
    np.testing.assert_allclose(native_target[:3, :3], POINTING_DOWN, atol=1e-12)
    assert np.linalg.norm(source_change - action[:3]) > .001
    assert mapped[6] == .2
    assert adapter.diagnostics()["mapping"]["native_action_clipped"] is False


def test_act_prior_uses_source_fk_and_does_not_replace_this_ticks_source_snapshot():
    adapter, env, source = frame()
    adapter.update()
    snapshot = adapter.source_pose.copy()
    raw = np.array([.2, -.1, .1, .5, -.2, 45.])
    prior = adapter.hf_to_common(raw)
    source_target = source.fk.fk_sim(source.last_target[:5])
    expected = np.r_[delta_action(snapshot, source_target), .5]
    np.testing.assert_allclose(prior, expected, atol=1e-12)
    assert np.linalg.norm(prior[:6] - delta_action(env.pose, source_target)) > .1
    np.testing.assert_array_equal(adapter.source_pose, snapshot)
    # update runs independently of ACT chunk caching, on every caller tick.
    adapter.update()
    assert source.calls == 2


def test_native_common_limits_apply_after_mapping_rotational_offset_motion():
    adapter, _, _ = frame()
    adapter.update()
    mapped = adapter.map_action(np.array([.03, 0., 0., 0., .3, 0., 1.]))
    assert np.linalg.norm(mapped[:3]) == pytest.approx(.03)
    assert adapter.diagnostics()["mapping"]["native_action_clipped"] is True


def test_rejected_feedback_discards_previous_snapshot_instead_of_mapping_stale_state():
    adapter, _, source = frame()
    with pytest.raises(RuntimeError, match="Update"):
        adapter.map_action(np.zeros(7))
    adapter.update()
    source.rejected = True
    with pytest.raises(RuntimeError, match="unrepresentable"):
        adapter.update()
    assert adapter.observation_env is None
    with pytest.raises(RuntimeError, match="Update"):
        adapter.hf_to_common(np.zeros(6))
