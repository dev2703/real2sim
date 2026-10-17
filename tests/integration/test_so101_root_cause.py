"""Matched controls must preserve source arrays and share a geometry-consistent reference."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from real2sim.geometry.so3 import apply_delta
from real2sim.sim.env import So101PickPlaceEnv

pytestmark = pytest.mark.integration


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    return importlib.import_module("run_so101_root_cause")


def inputs():
    states = np.array([[0., -45., 45., 45., 0., 40.], [.05, -45., 45., 45., 0., 40.],
                       [.1, -45., 45., 45., 0., 40.]])
    joint = {"state": states, "timestamp": np.arange(3) / 10, "frame_index": np.arange(3) * 3}
    cart = {"action": np.zeros((3, 7))}
    cart["action"][:, -1] = .3
    return joint, cart


def test_scene_deltas_reconstruct_reference_without_changing_physics(diagnostic):
    joint, _ = inputs()
    with So101PickPlaceEnv(render_images=False) as env:
        env.reset(seed=0)
        before = env.data.qpos.copy()
        original = joint["state"].copy()
        _, poses, _, deltas = diagnostic.reference_for_scene(env, joint["state"])
        for i, delta in enumerate(deltas):
            np.testing.assert_allclose(apply_delta(poses[i], delta), poses[i + 1], atol=1e-12)
        np.testing.assert_array_equal(env.data.qpos, before)
        np.testing.assert_array_equal(joint["state"], original)


def test_matched_routes_preserve_gripper_command_inputs_and_source_arrays(diagnostic):
    joint, cart = inputs()
    before_joint, before_cart = copy.deepcopy(joint), copy.deepcopy(cart)
    metadata = {"ball_xy": [.35, -.15], "bowl_xy": [.33, .15]}
    review = {"closure": [1, 2], "lift_transport": [2, 3], "release": [3, 4]}
    masks = []
    for mode in diagnostic.MODES:
        result, rows = diagnostic.run_trial(Path("configs/sim_so101.yaml"), joint, cart,
            np.arange(3), metadata, review, mode, [0, 0])
        assert result["ticks"] == 2
        json.dumps({"summary": result, "rows": rows}, allow_nan=False)
        assert [r["recorded_gripper_command_pct"] for r in rows] == [30., 30.]
        assert [r["recorded_gripper_observed_pct"] for r in rows] == [40., 40.]
        masks.append([r["active_reference"] for r in rows])
    assert masks[0] == masks[1] == masks[2]
    for before, after in ((before_joint, joint), (before_cart, cart)):
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])


def test_stationary_control_does_not_misclassify_ball_table_contact(diagnostic):
    probe = importlib.import_module("diag_so101_stationary_hold").probe
    state = inputs()[0]["state"][0]
    fixed = probe(Path("configs/sim_so101.yaml"), state, "fixed_joint_target", ticks=20)
    delta = probe(Path("configs/sim_so101.yaml"), state, "zero_delta_current_pose", ticks=20)
    feedback = probe(Path("configs/sim_so101.yaml"), state, "fixed_initial_pose_feedback", ticks=20)
    assert delta["end_drift_mm"] > fixed["end_drift_mm"] * 5
    assert feedback["end_drift_mm"] < fixed["end_drift_mm"] * 1.1
    for result in (fixed, delta, feedback):
        assert result["ball_non_table_contact_ticks"] == 0
        assert result["ik_rejections"] == 0
        json.dumps(result, allow_nan=False)
