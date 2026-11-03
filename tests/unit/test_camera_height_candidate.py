"""A measured table-relative camera height must respect the shared task datum."""

import copy
import json

import pytest

from real2sim.config import load_yaml
from scripts.diag_v2_camera_height import camera_height_candidate


@pytest.mark.parametrize("base_offset", [.012, .0024])
def test_height_is_above_table_not_above_task_origin(base_offset):
    task = load_yaml("configs/v2/task.scalar_measured_20261026.yaml")
    task["robot_so101"]["base_z_offset"] = base_offset
    before = copy.deepcopy(task)
    with open("configs/v2/measurements.clarification_20261026.json") as handle:
        clarification = json.load(handle)
    candidate = camera_height_candidate(task, clarification)
    camera = candidate["camera_iphone"]
    assert camera["approx_position_task"][2] + base_offset == pytest.approx(.4064)
    assert camera["height_above_table_interval_m"] == pytest.approx([.38608, .42672])
    assert camera["xyaxes_task"] == before["camera_iphone"]["xyaxes_task"]
    assert candidate["robot_so101"] == before["robot_so101"]
    assert candidate["physical_calibration_accepted"] is False
    assert task == before
