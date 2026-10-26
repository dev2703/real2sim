"""Assigned signs must give a coherent, explicitly provisional metric setup."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from real2sim.config import load_yaml
from scripts.prepare_v2_positive_setup import positive_setup


def test_positive_offsets_preserve_table_height_reference_and_dimensions():
    task = load_yaml("configs/v2/task.scalar_measured_20261026.yaml")
    measured = json.loads(Path("configs/v2/measurements.operator_20261026.json").read_text())
    result = positive_setup(task, measured)
    ball = np.array(result["ball"]["start_region"]["center_task_xy"])
    camera = np.array(result["camera_iphone"]["approx_position_task"])
    bowl = np.array(result["bowl"]["center_task_xy"])
    np.testing.assert_allclose(ball, [.0508, .3302])
    np.testing.assert_allclose(camera[:2] - ball, [.7747, .1143])
    np.testing.assert_allclose(bowl - ball, [.3048, 0.])
    assert camera[2] + result["robot_so101"]["base_z_offset"] == pytest.approx(.4064)
    assert result["table"]["mat"]["size_xy"] == pytest.approx([.4064, .3048])
    assert "xyaxes_task" not in result["camera_iphone"]
    assert result["positive_setup_assumptions"]["independently_verified_registration"] is False
    assert result["physical_calibration_accepted"] is False
    assert result["robot_so101"] == task["robot_so101"]
    # Configuration must remain loadable, with no NumPy scalar serialization.
    assert yaml.safe_load(yaml.safe_dump(result)) == result


def test_positive_assignment_does_not_rewrite_existing_scene():
    task = load_yaml("configs/v2/task.scalar_measured_20261026.yaml")
    measured = json.loads(Path("configs/v2/measurements.operator_20261026.json").read_text())
    previous = yaml.safe_dump(task)
    positive_setup(task, measured)
    assert yaml.safe_dump(task) == previous
