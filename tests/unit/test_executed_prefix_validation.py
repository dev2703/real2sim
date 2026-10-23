"""Execution cadence, causal baselines and motion-sensitive offline selection."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from real2sim.policy.validation import evaluate_executed_prefix, executed_prefix_report


def fixture(kind="cartesian"):
    n = 8
    state = np.zeros((n, 13))
    state[:, 3:9] = [1, 0, 0, 0, 1, 0]
    state[:, -1] = .6
    action = np.zeros((n, 7))
    action[:, -1] = .6
    action[0, 0] = .01
    action[4, 1] = .02
    arrays = {"state": state, "action": action, "episode_index": np.repeat([0, 1], 4),
              "ep_start": np.repeat([0, 4], 4), "ep_len": np.full(n, 4),
              "joint_state": np.zeros((n, 6)),
              "label_is_extrapolated": np.tile([False, False, False, True], 2)}
    manifest = {"split": {"train": [0], "val": [1], "reserved": [2]},
                "action_representation": kind}
    ticks = np.array([4, 5, 6])
    anchors = np.array([4, 4, 6])
    return arrays, manifest, ticks, anchors


def report(pred, *, kind="cartesian"):
    a, m, ticks, anchors = fixture(kind)
    return executed_prefix_report(pred, ticks, anchors, a, m, np.ones(pred.shape[1]), [1])


def test_stopping_reversing_and_stationary_drift_all_affect_selection():
    arrays, _, ticks, _ = fixture()
    label = arrays["action"][ticks]
    accurate = report(label.copy())
    stopped = label.copy()
    stopped[:, :3] = 0
    reversed_motion = label.copy()
    reversed_motion[:, :3] *= -1
    drift = label.copy()
    drift[1:, 0] += .002
    stopped_report, reversed_report, drift_report = map(report, (stopped, reversed_motion, drift))
    perfect_score = accurate["offline_selection"]["score"]
    assert all(r["offline_selection"]["score"] > perfect_score for r in (stopped_report, reversed_report, drift_report))
    assert stopped_report["scores"]["model"]["movement"]["positive_direction_fraction"] == 0
    assert reversed_report["scores"]["model"]["movement"]["positive_direction_fraction"] == 0
    physical = drift_report["scores"]["model"]["all"]
    assert physical["translation_mean_mm"] == pytest.approx(4 / 3)
    stationary = drift_report["scores"]["model"]["stationary_drift"]
    assert stationary["predicted_step_mean"] == 2
    assert stationary["summed_error_per_run_max"] == 4
    assert stationary["contiguous_runs"] == 1


@pytest.mark.parametrize("ticks,anchors", [([4, 4, 6], [4, 4, 6]), ([4, 5, 6, 7], [4, 4, 6, 6]), ([4, 6], [4, 6])])
def test_duplicates_extrapolation_or_missing_actions_are_rejected(ticks, anchors):
    arrays, manifest, _, _ = fixture()
    with pytest.raises(ValueError, match="exactly once"):
        executed_prefix_report(arrays["action"][ticks], ticks, anchors, arrays, manifest, np.ones(7), [1])


def test_reserved_episode_rejected_before_any_image_or_policy_access():
    class ForbiddenPolicy(torch.nn.Module):
        config = SimpleNamespace(n_action_steps=2, chunk_size=3)

        def predict_action_chunk(self, obs):
            pytest.fail("reserved policy inference")

    arrays, manifest, _, _ = fixture()
    arrays["episode_index"][4:] = 2
    dataset = SimpleNamespace(arrays=arrays, manifest=manifest, indices=np.arange(4, 8))
    with pytest.raises(ValueError, match="train/validation"):
        evaluate_executed_prefix(ForbiddenPolicy(), dataset, "cpu")


def test_hold_baseline_uses_anchor_gripper_and_not_future_observation():
    arrays, manifest, ticks, anchors = fixture()
    arrays["state"][5, -1] = .1
    arrays["action"][ticks, -1] = .6
    result = executed_prefix_report(arrays["action"][ticks], ticks, anchors, arrays, manifest, np.ones(7), [1])
    assert result["scores"]["hold"]["all"]["gripper_mae"] == 0
    assert result["valid_actions"] == 3
    assert result["inference_calls"] == 2


def test_absolute_targets_report_movement_relative_to_observed_pose():
    arrays, manifest, ticks, anchors = fixture("cartesian_absolute")
    arrays["state"][:, 0] = .2
    absolute = arrays["state"][ticks, :9].copy()
    absolute[0, 0] += .02
    arrays["action"] = np.c_[arrays["state"][:, :9], arrays["state"][:, -1]]
    arrays["action"][ticks, :9] = absolute
    label = arrays["action"][ticks]
    result = executed_prefix_report(label, ticks, anchors, arrays, manifest, np.ones(10), [1])
    motion = result["scores"]["model"]["movement"]
    assert motion["n"] == 1
    assert motion["positive_direction_fraction"] == 1
    assert "zero_delta_translation_mean_mm" not in result["scores"]["model"]["all"]


def test_joint_targets_keep_degree_and_gripper_percentage_units():
    arrays, manifest, ticks, anchors = fixture("canonical_joint")
    arrays["joint_state"][:, -1] = 60
    arrays["action"] = arrays["joint_state"].copy()
    arrays["action"][4, 1] = 5
    label = arrays["action"][ticks]
    pred = label.copy()
    pred[:, 0] += 2
    pred[:, -1] += 3
    result = executed_prefix_report(pred, ticks, anchors, arrays, manifest, np.ones(6), [1])
    physical = result["scores"]["model"]["all"]
    assert physical["arm_joint_mae_deg"][0] == 2
    assert physical["gripper_mae"] == .03
    assert result["scores"]["model"]["movement"]["unit"] == "deg"


def test_tool_frame_drift_is_integrated_in_base_coordinates():
    arrays, manifest, ticks, anchors = fixture("cartesian_tool")
    arrays["action"][:, :6] = 0
    # Alternate 180-degree tool yaw: identical local drift cancels in base space.
    arrays["state"][5, 3:9] = [-1, 0, 0, 0, -1, 0]
    pred = arrays["action"][ticks].copy()
    pred[:2, 0] = .002
    result = executed_prefix_report(pred, ticks, anchors, arrays, manifest, np.ones(7), [1])
    assert result["scores"]["model"]["stationary_drift"]["summed_error_per_run_max"] == 0


def test_stationary_runs_do_not_integrate_across_episode_boundaries():
    arrays, manifest, _, _ = fixture()
    arrays["action"][:, :6] = 0
    ticks = np.array([0, 1, 2, 4, 5, 6])
    anchors = np.array([0, 0, 2, 4, 4, 6])
    pred = arrays["action"][ticks].copy()
    pred[:, 0] = .001
    result = executed_prefix_report(pred, ticks, anchors, arrays, manifest, np.ones(7), [0, 1])
    assert result["scores"]["model"]["stationary_drift"]["contiguous_runs"] == 2
    assert result["scores"]["model"]["stationary_drift"]["summed_error_per_run_max"] == 3
