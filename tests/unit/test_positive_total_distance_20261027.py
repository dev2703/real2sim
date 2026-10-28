"""Total distance and assigned signs must not invent an x-only displacement."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from real2sim.config import load_yaml
from scripts.prepare_v2_positive_total_distance_20261027 import total_distance_setup


def source():
    return (load_yaml("configs/v2/task.scalar_measured_20261026.yaml"),
            json.loads(Path("configs/v2/measurements.operator_20261026.json").read_text()))


def test_total_distance_is_respected_without_x_only_bowl_assumption():
    task, measurements = source()
    candidate = total_distance_setup(task, measurements)
    ball = np.asarray(candidate["ball"]["start_region"]["center_task_xy"])
    bowl = np.asarray(candidate["bowl"]["center_task_xy"])
    delta = bowl - ball
    assert np.linalg.norm(delta) == pytest.approx(.3048)
    assert delta[0] > 0 and delta[1] < 0
    assert abs(delta[0]) < .3048 and abs(delta[1]) < .3048
    np.testing.assert_allclose(bowl, [.2563108824761005, .1051037601737995])


def test_positive_camera_offsets_keep_table_relative_height_and_angle_prior():
    task, measurements = source()
    candidate = total_distance_setup(task, measurements)
    old_ball = np.asarray(task["ball"]["start_region"]["center_task_xy"])
    old_direction = np.asarray(task["bowl"]["center_task_xy"]) - old_ball
    ball = np.asarray(candidate["ball"]["start_region"]["center_task_xy"])
    direction = np.asarray(candidate["bowl"]["center_task_xy"]) - ball
    angle = lambda vector: np.arctan2(vector[1], vector[0])  # noqa: E731
    assert angle(direction) - angle(ball) == pytest.approx(angle(old_direction) - angle(old_ball))
    camera = np.asarray(candidate["camera_iphone"]["approx_position_task"])
    np.testing.assert_allclose(camera[:2] - ball, [.7747, .1143])
    assert camera[2] + candidate["robot_so101"]["base_z_offset"] == pytest.approx(.4064)
    assert candidate["physical_calibration_accepted"] is False
    assert candidate["total_distance_setup_assumptions"]["independently_verified_registration"] is False


def test_conversion_preserves_source_and_robot_mapping_and_serializes():
    task, measurements = source()
    before = copy.deepcopy(task)
    candidate = total_distance_setup(task, measurements)
    assert task == before
    assert candidate["robot_so101"] == before["robot_so101"]
    assert yaml.safe_load(yaml.safe_dump(candidate)) == candidate


def test_degenerate_image_bearing_cannot_manufacture_a_direction():
    task, measurements = source()
    task["bowl"]["center_task_xy"] = task["ball"]["start_region"]["center_task_xy"].copy()
    with pytest.raises(ValueError, match="nondegenerate"):
        total_distance_setup(task, measurements)


def test_revised_camera_measurement_and_uncertainty_are_used_with_current_source():
    task, measurements = source()
    measurements["reported_setup"]["camera from ball"]["value_m"][2] = .45
    measurements["reported_setup"]["ball"]["value_m"][0] = .21
    measurements["source"]["relative_uncertainty"] = .08
    measurements["source"]["sha256"] = "revised_measurement_hash"
    candidate = total_distance_setup(task, measurements)
    camera = candidate["camera_iphone"]
    assert camera["height_above_table_m"] == pytest.approx(.45)
    assert camera["approx_position_task"][2] == pytest.approx(.438)
    assert camera["height_above_table_interval_m"] == pytest.approx([.414, .486])
    assert candidate["ball"]["radius"] == pytest.approx(.21 / (2 * np.pi))
    provenance = candidate["operator_measurements"]
    assert provenance["sha256"] == "revised_measurement_hash"
    assert provenance["relative_uncertainty"] == .08
    assert "camera_iphone.approx_position_task" in provenance["applied_fields"]
    assert "table.mat.size_xy" in provenance["applied_fields"]
