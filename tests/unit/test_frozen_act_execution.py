"""Execution-prefix overrides preserve ordered actions and checkpoint identity."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.artifacts import sha256_file
from scripts import benchmark_frozen_act as benchmark


def driver_setup(monkeypatch):
    class Environment:
        control_hz = 25.
        camera_bindings = {"observation.images.iphone": "iphone"}
        actuator_ids = np.arange(6)
        max_velocity = np.ones(6)

        def __init__(self):
            self.tick = 0
            self.commands = []
            self.data = SimpleNamespace(qpos=np.zeros(6), ctrl=np.zeros(6))

        def real_state(self):
            return np.full(6, self.tick, dtype=float)

        def render(self, camera):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def sim_state(self):
            return self.data.qpos.copy()

        def step(self, command):
            self.commands.append(command.copy())
            self.tick += 1
            return {}, 0., False, False, {"control": {"limit_clipped": np.zeros(6)}}

    env = Environment()

    class Policy:
        config = SimpleNamespace(chunk_size=50, n_action_steps=10)
        camera_keys = ["observation.images.iphone"]

        def __init__(self):
            self.reset()

        def reset(self):
            self.call_ticks = []

        def predict_action_chunk(self, state, images):
            self.call_ticks.append(env.tick)
            first = 100 * (len(self.call_ticks) - 1)
            return np.repeat((first + np.arange(50))[:, None], 6, axis=1)

    policy = Policy()
    shared = SimpleNamespace(env=env, source=None, robot="so101",
        guard=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(6))))
    monkeypatch.setattr(benchmark.transfer.native, "saturate_gripper_only",
                        lambda raw, env: (raw.copy(), raw.copy()))
    monkeypatch.setattr(benchmark, "check_joint_request",
                        lambda guard, current, start, requested, cap, **kwargs:
                        (True, "accepted", requested.copy()))
    return shared, policy


@pytest.mark.parametrize("execute_steps", [10, 50])
def test_prefix_preserves_action_order_and_refreshes_at_requested_cadence(
        monkeypatch, execute_steps):
    shared, policy = driver_setup(monkeypatch)
    driver = benchmark.FrozenDriver(shared, policy, execute_steps=execute_steps)
    inferred = []
    for tick in range(60):
        result = driver.step({}, shared.env)
        if result.detail["native_input"] is not None:
            inferred.append(tick)
        assert result.detail["uses_act"] and not result.detail["intervention"]
    expected_calls = list(range(0, 60, execute_steps))
    assert policy.call_ticks == inferred == expected_calls
    expected_actions = [100 * (tick // execute_steps) + tick % execute_steps
                        for tick in range(60)]
    np.testing.assert_array_equal(np.asarray(shared.env.commands)[:, 0], expected_actions)
    assert policy.config.chunk_size == 50 and policy.config.n_action_steps == 10

    # Reset must discard unexecuted actions and query the fresh observation.
    driver.reset()
    result = driver.step({}, shared.env)
    assert policy.call_ticks == [60]
    np.testing.assert_array_equal(result.detail["raw_hf_action"], np.zeros(6))


@pytest.mark.parametrize("invalid", [0, -1, 51, True, 1.5])
def test_driver_rejects_invalid_execution_prefix(monkeypatch, invalid):
    shared, policy = driver_setup(monkeypatch)
    with pytest.raises(ValueError, match="execution prefix"):
        benchmark.FrozenDriver(shared, policy, execute_steps=invalid)


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "snapshot"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"chunk_size": 50, "n_action_steps": 10}))
    (path / "model.safetensors").write_bytes(b"unchanged frozen ACT")
    return path


def test_execution_override_is_versioned_without_changing_saved_checkpoint(tmp_path, snapshot):
    before = {path.name: sha256_file(path) for path in snapshot.iterdir()}
    plans = {}
    for prefix in (10, 50):
        plans[prefix] = benchmark.run(tmp_path / f"execute_{prefix}", ["so101", "panda"],
            ["raw"], [14000], snapshot=snapshot, execute_steps=prefix, plan_only=True)
        assert plans[prefix]["act_chunk_size"] == 50
        assert plans[prefix]["checkpoint_execution_steps"] == 10
        assert plans[prefix]["act_execution_steps"] == prefix
        assert plans[prefix]["execution_steps_overridden"] is True
        assert plans[prefix]["control_hz"] == 25.
    assert plans[10]["protocol_id"] != plans[50]["protocol_id"]
    assert plans[10]["checkpoint_sha256"] == plans[50]["checkpoint_sha256"]
    assert before == {path.name: sha256_file(path) for path in snapshot.iterdir()}
    with pytest.raises(ValueError, match="protocol/inputs changed"):
        benchmark.run(tmp_path / "execute_10", ["so101", "panda"], ["raw"], [14000],
            snapshot=snapshot, execute_steps=50, plan_only=True)


@pytest.mark.parametrize("invalid", [0, -1, 51, True, 1.5])
def test_plan_rejects_invalid_execution_prefix_before_writing_output(tmp_path, snapshot, invalid):
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="execution prefix"):
        benchmark.run(out, ["so101"], ["raw"], [14000], snapshot=snapshot,
                      execute_steps=invalid, plan_only=True)
    assert not out.exists()
