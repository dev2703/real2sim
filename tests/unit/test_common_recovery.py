"""Observable, causal inputs and shared action meaning for both embodiments."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from real2sim.policy.common_recovery import (
    FRAME_NAMES,
    OBSERVATION_NAMES,
    CommonCartesianExecutor,
    FrozenACTFeatures,
    RecoveryObservation,
    clip_common,
)
from real2sim.policy.recovery_mlp import ACTION_NAMES, bound_common_action


class MeasuredEnvironment:
    """Allow measured robot reads; reject object state and scorer access."""

    def __init__(self, robot="so101"):
        self.robot = robot
        self.pose = np.eye(4)
        self.pose[:3, 3] = [.2, .1, .15]
        self.mapping = SimpleNamespace(gripper_sim_to_real_percent=lambda value: value)
        self.camera_bindings = {"observation.images.top": "top", "observation.images.iphone": "iphone"}

    def ee_pose(self):
        return self.pose.copy()

    def sim_state(self):
        return np.r_[np.zeros(5), 36.]

    def gripper_semantic(self):
        return .7

    def real_state(self):
        return np.zeros(6, np.float32)

    def render(self, _camera):
        return np.zeros((2, 2, 3), np.uint8)

    def __getattr__(self, name):
        if name in {"info", "data", "model", "ball", "bowl", "privileged_metrics"}:
            raise AssertionError(f"Recovery observation requested forbidden {name}")
        raise AttributeError(name)


def current_frame(observation):
    return dict(zip(FRAME_NAMES, observation[-33:], strict=True))


@pytest.mark.parametrize("robot,expected_jaw", [("so101", .4), ("panda", .7)])
def test_history_contains_only_causal_measured_and_rgb_reference_inputs(robot, expected_jaw):
    env = MeasuredEnvironment(robot)
    encoder = RecoveryObservation(robot, 25.)
    ball, bowl = np.array([.05, .33, .02]), np.array([.26, .11, .018])
    first = encoder.encode(env, ball, bowl, 0)
    assert first.shape == (99,) and len(OBSERVATION_NAMES) == 99
    np.testing.assert_array_equal(first[:33], first[33:66])
    np.testing.assert_array_equal(first[33:66], first[66:])
    f0 = current_frame(first)
    assert f0["gripper_measured"] == pytest.approx(expected_jaw)
    assert f0["robot_so101"] == (robot == "so101")
    assert f0["robot_panda"] == (robot == "panda")
    assert f0["initial_rgb_ball_relative_x"] == pytest.approx(-.15)
    assert f0["ball_observation_age_s"] == 0
    selected = np.array([.01, 0., 0., 0., 0., 0., 1.])
    encoder.selected_action(selected, 0)
    selected[0] = 99  # callers cannot mutate stored history after selection.
    env.pose[0, 3] += .004
    second = encoder.encode(env, ball, bowl, 1)
    np.testing.assert_array_equal(second[:66], first[33:])
    f1 = current_frame(second)
    assert f1["velocity_x_mps"] == pytest.approx(.1)
    assert f1["previous_tcp_dx_m"] == pytest.approx(.01)
    assert f1["previous_gripper_open_fraction"] == 1.
    assert f1["seconds_since_selected_gripper_switch"] == pytest.approx(.04)
    assert f1["ball_observation_age_s"] == pytest.approx(.04)


def test_gripper_event_timer_uses_selected_action_history_and_reset_discards_it():
    env, encoder = MeasuredEnvironment(), RecoveryObservation("so101", 25.)
    ball, bowl = [.05, .33, .02], [.26, .11, .018]
    encoder.encode(env, ball, bowl, 0)
    encoder.selected_action(np.r_[np.zeros(6), 1.], 0)
    assert current_frame(encoder.encode(env, ball, bowl, 4))["seconds_since_selected_gripper_switch"] == pytest.approx(.16)
    encoder.selected_action(np.r_[np.zeros(6), .8], 4)
    assert current_frame(encoder.encode(env, ball, bowl, 5))["seconds_since_selected_gripper_switch"] == pytest.approx(.2)
    encoder.selected_action(np.r_[np.zeros(6), 0.], 5)
    assert current_frame(encoder.encode(env, ball, bowl, 6))["seconds_since_selected_gripper_switch"] == pytest.approx(.04)
    encoder.reset()
    reset = encoder.encode(env, ball, bowl, 0)
    assert current_frame(reset)["seconds_since_selected_gripper_switch"] == 0
    assert current_frame(reset)["previous_gripper_open_fraction"] == 0
    np.testing.assert_array_equal(reset[:33], reset[-33:])


def test_native_cartesian_action_uses_world_rotation_and_task_translation():
    env = MeasuredEnvironment()
    env.pose[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    target = env.pose.copy()
    target[:3, :3] = Rotation.from_rotvec([.2, 0., 0.]).as_matrix() @ env.pose[:3, :3]
    target[:3, 3] += [.01, 0., 0.]
    executor = object.__new__(CommonCartesianExecutor)
    executor.env, executor.robot = env, "so101"
    executor.guard = SimpleNamespace(forward=lambda _native: target)
    action = executor.native_to_common(np.r_[np.zeros(5), 45.])
    np.testing.assert_allclose(action[:6], [.01, 0., 0., .2, 0., 0.], atol=1e-12)
    assert action[6] == pytest.approx(.5)
    from real2sim.policy.common_recovery import ACTION_NAMES as COMMON_ACTION_NAMES

    assert COMMON_ACTION_NAMES == ACTION_NAMES


def test_executor_and_head_apply_identical_euclidean_action_limits():
    action = np.array([.1, -.1, .2, 1., 2., -1., 2.])
    np.testing.assert_allclose(clip_common(action), bound_common_action(action), atol=1e-12)
    with pytest.raises(ValueError, match="finite"):
        clip_common(np.full(7, np.inf))


class FakeNativeACT:
    def __init__(self):
        self.config = SimpleNamespace(n_action_steps=3)
        self.camera_keys = ["observation.images.top", "observation.images.iphone"]
        self.action_head = nn.Identity()
        self.policy = SimpleNamespace(model=SimpleNamespace(action_head=self.action_head))
        self.inference_calls = 0

    def reset(self):
        self.inference_calls = 0

    def predict_action_chunk(self, state, images):
        assert state.shape == (6,)
        assert set(images) == set(self.camera_keys)
        ids = np.arange(50) + 1000 * self.inference_calls
        features = torch.as_tensor(np.broadcast_to(ids[:, None], (50, 256)).copy())[None]
        self.action_head(features)
        self.inference_calls += 1
        return np.broadcast_to(ids[:, None], (50, 6)).copy()


def test_frozen_features_follow_the_executed_query_and_reset_drops_old_chunk():
    policy = FakeNativeACT()
    episode = SimpleNamespace(env=MeasuredEnvironment(), source=None, robot="so101")
    stream = FrozenACTFeatures(episode, policy)
    stream.reset()
    for expected in (0, 1, 2, 1000, 1001, 1002, 2000):
        raw, features = stream.next()
        assert raw[0] == features[0] == expected
        assert features.shape == (256,)
    assert stream.calls == 3
    stream.reset()
    raw, features = stream.next()
    assert raw[0] == features[0] == 0 and stream.calls == 1
    assert len(policy.action_head._forward_pre_hooks) == 1
    stream.close()
    assert len(policy.action_head._forward_pre_hooks) == 0
