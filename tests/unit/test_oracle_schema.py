"""G4 oracle-schema guard (plan section 7): the inference adapter serves only the checkpoint's
audited feature slots. New ground-truth slots, foreign camera keys or image shapes are refused
rather than adapted. Also pins the *meaning* of the served slots (one-tick EE velocity) and the
scoring rules of ``TrialTracker`` (no success while the ball is still held)."""

import numpy as np
import pytest
import torch

from real2sim.data.labels import ACTION_NAMES, PROPRIO_NAMES
from real2sim.evaluation.metrics import TrialTracker
from real2sim.policy.adapters import (
    IMAGE_SHAPE_CHW,
    JOINT_STATE_NAMES,
    ChunkExecutor,
    ContractError,
    ObservationAdapter,
    contract_kind,
    normalize_image_chw,
)


class FakeNormalizer:
    def __init__(self, n_state: int):
        self.state_mean = np.zeros(n_state, np.float32)
        self.state_std = np.ones(n_state, np.float32)
        self.img_mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
        self.img_std = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

    def state(self, x):
        return (x - self.state_mean) / self.state_std

    def image(self, hwc_uint8):
        chw = hwc_uint8.transpose(2, 0, 1).astype(np.float32) / 255.0
        return (chw - self.img_mean) / self.img_std


def joint_contract() -> dict:
    return {"camera_keys": ["observation.images.iphone"], "image_shape_chw": [3, 192, 256],
            "state_names": list(JOINT_STATE_NAMES), "action_names": list(JOINT_STATE_NAMES)}


def cartesian_contract() -> dict:
    return {"camera_keys": ["observation.images.iphone"], "image_shape_chw": [3, 192, 256],
            "state_names": list(PROPRIO_NAMES), "action_names": list(ACTION_NAMES)}


def test_accepts_joint_and_cartesian_contracts():
    assert ObservationAdapter(joint_contract(), FakeNormalizer(6)).kind == "joint"
    assert ObservationAdapter(cartesian_contract(), FakeNormalizer(13)).kind == "cartesian"


def test_rejects_extra_ground_truth_slot():
    c = cartesian_contract()
    c["state_names"] = list(PROPRIO_NAMES) + ["ball_xyz"]
    with pytest.raises(ContractError, match="state_names"):
        ObservationAdapter(c, FakeNormalizer(14))
    j = joint_contract()
    j["state_names"] = list(JOINT_STATE_NAMES) + ["ball_x", "ball_y", "ball_z"]
    with pytest.raises(ContractError):
        contract_kind(j)


def test_rejects_wrong_camera_key_and_shape():
    c = joint_contract()
    c["camera_keys"] = ["privileged.ball_position"]
    with pytest.raises(ContractError, match="camera_keys"):
        ObservationAdapter(c, FakeNormalizer(6))
    c = joint_contract()
    c["camera_keys"] = ["observation.images.iphone", "observation.images.iphone"]
    with pytest.raises(ContractError, match="camera_keys"):
        ObservationAdapter(c, FakeNormalizer(6))
    c = joint_contract()
    c["image_shape_chw"] = [3, 480, 640]
    with pytest.raises(ContractError, match="image_shape"):
        ObservationAdapter(c, FakeNormalizer(6))


def test_rejects_normalizer_state_mismatch():
    with pytest.raises(ContractError):
        ObservationAdapter(joint_contract(), FakeNormalizer(13))


def test_image_normalisation_matches_training_path():
    n = FakeNormalizer(6)
    rng = np.random.default_rng(0)
    hwc = rng.integers(0, 256, size=(192, 256, 3), dtype=np.uint8)
    chw01 = hwc.transpose(2, 0, 1).astype(np.float32) / 255.0
    np.testing.assert_allclose(normalize_image_chw(n, chw01), n.image(hwc), atol=1e-6)
    with pytest.raises(ContractError):
        normalize_image_chw(n, np.zeros((3, 144, 256), np.float32))
    with pytest.raises(ContractError):
        normalize_image_chw(n, np.full(IMAGE_SHAPE_CHW, 2.0, np.float32))


# --------------------------------------------------------------------- velocity slot semantics
HZ = 10.0


class FakeEnv:
    """EE moves +0.01 m in x per tick (0.1 m/s at 10 Hz); joint/gripper state constant."""

    control_hz = HZ

    def __init__(self):
        self.p = np.zeros(3)

    def ee_pose(self):
        T = np.eye(4)
        T[:3, 3] = self.p
        return T

    def canonical_proprio(self, prev):  # mirrors So101PickPlaceEnv.canonical_proprio
        vel = np.zeros(3) if prev is None else (self.p - np.asarray(prev, float)) * HZ
        return np.concatenate([self.p, [1, 0, 0, 0, 1, 0], vel, [0.5]]).astype(np.float32)

    def tick(self):
        self.p = self.p + np.array([0.01, 0.0, 0.0])


class FakePolicy:
    class config:
        n_action_steps = 2
        chunk_size = 4

    def reset(self):
        pass

    def predict_action_chunk(self, batch):
        return torch.zeros(1, 4, 7)


class FakeActionAdapter:
    dim = 7

    def __init__(self, env):
        self.env = env

    def apply(self, action):
        from real2sim.policy.adapters import ApplyResult

        self.env.tick()
        return ApplyResult({}, {}, False, 0.0)


class FakeNormalizer13(FakeNormalizer):
    def unaction(self, chunk):
        return chunk


def _obs():
    return {"observation.images.iphone": np.zeros(IMAGE_SHAPE_CHW, np.float32)}


def test_cartesian_velocity_is_one_tick_finite_difference_with_chunked_execution():
    """labels.py: vel_k = (p_k - p_{k-1}) * hz. With n_action_steps=2 the adapter infers every
    other tick but the velocity it serves must still be the per-tick value (0.1 m/s), not 2x."""
    env = FakeEnv()
    ad = ObservationAdapter(cartesian_contract(), FakeNormalizer13(13))
    ex = ChunkExecutor(FakePolicy(), FakeNormalizer13(13), ad, FakeActionAdapter(env))
    ex.reset()
    served = []
    for _ in range(6):
        if not ex.queue:  # the executor is about to infer on this tick: capture what it serves
            served.append(ad.state(env)[9:12].copy())
        ex.step(_obs(), env)
    assert ex.n_policy_calls == 3
    np.testing.assert_allclose(served[0], [0.0, 0.0, 0.0])  # first tick: zero like vel[0]
    for v in served[1:]:
        np.testing.assert_allclose(v, [0.1, 0.0, 0.0], atol=1e-6)
    # state() is a pure read: repeated calls within a tick give the same feature.
    np.testing.assert_allclose(ad.state(env), ad.state(env))


def test_joint_contract_ignores_tick_memory():
    class JointEnv:
        def real_state(self):
            return np.arange(6, dtype=np.float32)

    ad = ObservationAdapter(joint_contract(), FakeNormalizer(6))
    ad.advance_tick(JointEnv())  # must not touch ee_pose for the joint contract
    np.testing.assert_array_equal(ad.state(JointEnv()), np.arange(6))


# --------------------------------------------------------------------- TrialTracker scoring
def _priv(*, held, in_bowl, resting, dist=0.2, lifted=False, z=0.05):
    return {"ball_held": held, "ball_in_bowl": in_bowl, "ball_resting": resting, "lifted": lifted,
            "ee_ball_distance_m": dist, "ball_xyz": [0.0, 0.0, z], "ball_xyz_task": [0.0, 0.0, z],
            "tool_xyz_task": [0.0, 0.0, z]}


def _tracker(**kw):
    base = dict(success_hold_ticks=10, approach_distance_m=0.03, table_top_z=0.0, ball_radius_m=0.0335)
    base.update(kw)
    return TrialTracker(**base)


def test_held_ball_in_bowl_is_not_success():
    t = _tracker()
    for _ in range(5):
        assert not t.update(_priv(held=True, in_bowl=False, resting=False, lifted=True))
    for _ in range(30):
        assert not t.update(_priv(held=True, in_bowl=True, resting=True, lifted=True))
    assert not t.success and t.delivered and not t.released
    assert t.phase_failed() == "release"
    # Release and let it settle: now it counts, after exactly success_hold_ticks released ticks.
    for i in range(10):
        done = t.update(_priv(held=False, in_bowl=True, resting=True, lifted=True))
        assert done == (i == 9)
    assert t.success and t.released and t.phase_failed() == "none"


def test_success_counter_resets_if_regrasped():
    t = _tracker(success_hold_ticks=3)
    t.update(_priv(held=True, in_bowl=True, resting=True))
    t.update(_priv(held=False, in_bowl=True, resting=True))
    t.update(_priv(held=False, in_bowl=True, resting=True))
    t.update(_priv(held=True, in_bowl=True, resting=True))  # picked up again
    assert not t.update(_priv(held=False, in_bowl=True, resting=True))
    assert not t.success


def test_phase_failed_uses_protocol_approach_distance():
    """docs/experiment_protocol.md: 'approach' = never came within 3 cm of the ball."""
    t = _tracker()
    t.update(_priv(held=False, in_bowl=False, resting=True, dist=0.04))
    assert t.phase_failed() == "approach"
    t = _tracker()
    t.update(_priv(held=False, in_bowl=False, resting=True, dist=0.02))
    assert t.phase_failed() == "grasp"
