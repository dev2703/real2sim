"""Reference reports retain clipping before IK as well as native servo constraints."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from real2sim.config import load_yaml

pytestmark = pytest.mark.integration


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reference_inputs(tmp_path):
    # An explicit test calibration makes a legal semantic command exceed the native
    # gripper limit. Neither original geometry nor model limits are edited.
    task = load_yaml("configs/task.yaml")
    task["robot_so101"]["gripper"]["percent_to_joint"] = {
        "offset_rad": -.1745, "scale_rad_per_percent": .03,
    }
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    data = load_yaml("configs/data.yaml")
    data["policy_hz"] = 12.5
    data_path = tmp_path / "data.yaml"
    data_path.write_text(yaml.safe_dump(data))
    cfg = load_yaml("configs/sim_so101.yaml")
    cfg.update(task_config=str(task_path), data_config=str(data_path), control_hz=12.5,
               strict_initial_state=True, initial_qpos_real_deg=[0, 0, 0, 0, 0, 20],
               initial_qpos=[0, 0, 0, 0, 0, .4255])
    config_path = tmp_path / "sim.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    states = np.tile([0., 0., 0., 0., 0., 20.], (5, 1))
    commands = states.copy()
    commands[:, 5] = 100.
    raw = SimpleNamespace(fps=25, episode_rows=lambda _: states,
                          stack=lambda _, key: states if key == "observation.state" else commands)
    case = {
        "episode_index": 0, "condition": "baseline", "start_source_frame": 0,
        "end_source_frame": 4, "preserves_complete_task": True,
        "ball_xy": [.5, .3], "bowl_xy": [.5, -.3],
        "phases_source_frame": {"grasp": 0, "lift": 2, "release": 4},
    }
    return config_path, states, commands, raw, case


@pytest.mark.parametrize("mode", ["joint_reference", "current_delta", "persistent_delta", "absolute_pose"])
def test_v2_replay_preserves_calibration_clipping(reference_inputs, mode):
    config, _, _, raw, case = reference_inputs
    report = script("replay_v2_references").replay(raw, config, case, mode)
    assert report["native_gripper_clipped_ticks"] == 2
    assert report["max_action_saturation"] >= 1 / (6 if mode == "joint_reference" else 7)
    assert report["joint_rate_limited_ticks"] == 2
    for row in report["trace"]:
        detail = row["adapter_detail"]
        assert detail["native_gripper_clipped"]
        assert detail["requested_gripper_rad"] == pytest.approx(2.8255)
        assert detail["bounded_gripper_rad"] == pytest.approx(1.7453292)
        assert row["control"]["rate_limited"][5]
        # Cartesian clipping happens before env.step, so native-only reporting loses it.
        assert row["control"]["limit_clipped"][5] is (mode == "joint_reference")
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mode", ["joint_reference", "persistent_delta"])
def test_v2_control_probe_reports_joint_and_adapter_constraints(reference_inputs, mode):
    config, states, commands, _, _ = reference_inputs
    report, trace = script("diag_v2_kinematic_control").probe(config, states[::2], commands[::2], mode)
    assert report["native_gripper_clipped_ticks"] == 2
    assert report["rate_limited_ticks"] == 2
    assert report["max_action_saturation"] >= 1 / (6 if mode == "joint_reference" else 7)
    assert all(row["adapter_detail"]["native_gripper_clipped"] for row in trace)
    assert all(row["joint_rate_limited"][5] for row in trace)
    assert report["limit_clipped_ticks"] == (2 if mode == "joint_reference" else 0)
    json.dumps({"report": report, "trace": trace}, allow_nan=False)
