"""Frozen prefixes hand over once using RGB/robot observations, never scoring."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.policy.frozen_prefix_recovery import FrozenPrefixRecoveryController
from real2sim.policy.hf_recovery import ApproachMonitor


class Robot:
    def __init__(self):
        self.pose = np.eye(4)
        self.pose[:3, 3] = [.1, .2, .3]

    def ee_pose(self):
        return self.pose.copy()

    def __getattr__(self, name):
        if name in {"info", "ball", "bowl", "privileged_metrics", "data", "model"}:
            raise AssertionError(f"Prefix attempted forbidden read: {name}")
        raise AttributeError(name)


class Driver:
    def __init__(self, env, *, uses_act, intervention):
        self.env, self.uses_act, self.intervention = env, uses_act, intervention
        self.steps, self.resets, self.closes = 0, 0, 0
        self.inference_ms = []
        self.act = None
        self.source_frame = None

    def reset(self):
        self.resets += 1
        self.steps = 0
        self.inference_ms = []

    def step(self, obs, env):
        self.steps += 1
        if self.uses_act:
            self.inference_ms.append(2.)
        return SimpleNamespace(detail={"uses_act": self.uses_act,
            "intervention": self.intervention, "driver_step": self.steps})

    def close(self):
        self.closes += 1


class References:
    def __init__(self, ball):
        self.ball, self.ticks = ball, []

    def update(self, tick):
        self.ticks.append(tick)


def wrapper(*, trigger_tick=2, ball=(.4, .2, .3), recovery_uses_act=False):
    env = Robot()
    recovery = Driver(env, uses_act=recovery_uses_act, intervention=True)
    frozen = Driver(env, uses_act=True, intervention=False)
    monitors, references = [], []

    def monitor_factory():
        monitor = SimpleNamespace(calls=[])

        def update(tick, distance):
            monitor.calls.append((tick, distance))
            return "RGB approach stalled" if tick >= trigger_tick else None

        monitor.update = update
        monitors.append(monitor)
        return monitor

    def references_factory():
        tracker = References(None if ball is None else np.asarray(ball))
        references.append(tracker)
        return tracker

    controller = FrozenPrefixRecoveryController(recovery, frozen, monitor_factory,
        references_factory, [.01, 0., 0.])
    return controller, env, recovery, frozen, monitors, references


@pytest.mark.parametrize("recovery_uses_act", [False, True])
def test_handover_latches_and_resets_recovery_history_before_its_first_action(recovery_uses_act):
    controller, env, recovery, frozen, monitors, references = wrapper(recovery_uses_act=recovery_uses_act)
    first = controller.step({}, env).detail
    second = controller.step({}, env).detail
    assert first["prefix_active"] and second["prefix_active"]
    assert first["uses_act"] is True and first["intervention"] is False
    assert recovery.steps == 0 and frozen.steps == 2
    recovery.steps = 99  # stale history must not survive takeover.
    takeover = controller.step({}, env).detail
    assert recovery.resets == 2 and recovery.steps == 1
    assert takeover["prefix_tick"] == 2
    assert takeover["prefix_trigger"]["tick"] == 2
    assert takeover["prefix_trigger"]["reason"] == "RGB approach stalled"
    assert takeover["prefix_trigger"]["distance_m"] == pytest.approx(.29)
    assert takeover["uses_act"] is recovery_uses_act and takeover["intervention"] is True
    assert takeover["learned_recovery_active"] and not takeover["prefix_active"]
    # Progress or missing RGB after takeover cannot return to frozen execution.
    references[-1].ball = None
    controller.step({}, env)
    assert frozen.steps == 2 and controller.recovery_ticks == 2
    assert controller.prefix_ticks == 2 and controller.prefix_policy_calls == 2
    assert references[-1].ticks == [0, 1, 2]
    assert len(monitors[-1].calls) == 3
    assert controller.act is recovery.act and controller.source_frame is recovery.source_frame
    assert takeover["controller_latency_ms"] > 0.
    assert takeover["inference_ms"] == takeover["head_inference_ms"] == 0.
    assert takeover["no_runtime_geometric_teacher"] is True
    # Report consumers cannot mutate the live latch.
    trigger = controller.trigger
    trigger["tick"] = 100
    assert controller.trigger["tick"] == 2


def test_reset_recreates_episode_monitor_and_rgb_state_and_both_drivers():
    controller, env, recovery, frozen, monitors, references = wrapper(trigger_tick=0)
    controller.step({}, env)
    old_monitor, old_references = monitors[-1], references[-1]
    controller.reset()
    assert controller.tick == controller.prefix_ticks == controller.recovery_ticks == 0
    assert controller.trigger is None
    assert monitors[-1] is not old_monitor and references[-1] is not old_references
    assert controller.prefix_policy_calls == 0 and frozen.resets == 2 and recovery.resets == 3
    controller.close()
    controller.close()
    assert recovery.closes == 1 and frozen.closes == 0


def test_handover_carries_observed_rgb_memory_after_reset_before_recovery_action():
    controller, env, recovery, _, _, references = wrapper(trigger_tick=1)
    inherited = []
    recovery.inherit_rgb_references = lambda tracker, tick: inherited.append((tracker, tick, recovery.steps))
    controller.step({}, env)
    controller.step({}, env)
    assert inherited == [(references[-1], 1, 0)]
    assert recovery.steps == 1


def test_missing_ball_uses_monitor_deadline_without_a_configured_goal_fallback():
    controller, env, recovery, frozen, _, _ = wrapper(ball=None)
    controller.monitor_factory = lambda: ApproachMonitor(2., patience_s=.5, max_attempt_s=1.)
    controller.reset()
    assert controller.step({}, env).detail["prefix_active"]
    assert controller.step({}, env).detail["prefix_active"]
    result = controller.step({}, env)
    assert result.detail["prefix_trigger"]["distance_m"] is None
    assert controller.prefix_ticks == 2 and controller.recovery_ticks == 1
    assert frozen.steps == 2 and recovery.steps == 1


def test_distance_uses_rotated_tool_offset_and_rejects_a_different_environment():
    controller, env, _, _, monitors, _ = wrapper(trigger_tick=100, ball=(.1, .21, .3))
    env.pose[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    controller.step({}, env)
    assert monitors[-1].calls[0][1] == pytest.approx(0.)
    with pytest.raises(ValueError, match="different environment"):
        controller.step({}, Robot())


def test_invalid_factories_offsets_or_rgb_values_fail_explicitly():
    controller, env, _, _, _, references = wrapper()
    with pytest.raises(ValueError, match="factories"):
        FrozenPrefixRecoveryController(controller.recovery, controller.frozen, object(), lambda: None, np.zeros(3))
    with pytest.raises(ValueError, match="offset"):
        FrozenPrefixRecoveryController(controller.recovery, controller.frozen, lambda: None, lambda: None, [np.nan]*3)
    references[-1].ball = np.array([np.nan, 0., 0.])
    with pytest.raises(ValueError, match="RGB ball"):
        controller.step({}, env)
