"""Historical expert poses reconstruct the same causal feedback as live control."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.geometry.so3 import delta_action, exp_so3, make_T
from real2sim.policy.common_recovery import TRACKED_OBSERVATION_NAMES, RecoveryObservation
from scripts.convert_recovery_history import convert_observations, validate_manifest


def sequence():
    pose = np.eye(4)
    env = SimpleNamespace(ee_pose=lambda: pose.copy(), gripper_semantic=lambda: .7)
    selected = RecoveryObservation("panda", 25.)
    measured = RecoveryObservation("panda", 25., action_history="achieved")
    observations, expected, targets = [], [], []
    for tick, jaw in enumerate((1., 0., 0., 1.)):
        before = make_T(exp_so3(np.array([0., 0., .02 * tick])), np.array([.001 * tick, .2, .1]))
        after = make_T(exp_so3(np.array([0., 0., .02 * (tick + 1)])), np.array([.001 * (tick + 1), .2, .1]))
        pose[:] = before
        observations.append(selected.encode(env, [.1, .2, .03], [.3, .1, .03], tick))
        expected.append(measured.encode(env, [.1, .2, .03], [.3, .1, .03], tick))
        command = np.array([.02, 0., 0., 0., 0., .15, jaw])
        targets.append(command)
        selected.selected_action(command, tick)
        measured.executed_action(np.r_[delta_action(before, after), jaw], tick)
    rows = {"observations": np.stack(observations), "target_actions": np.stack(targets),
            "robot_id": np.ones(4, np.int64), "episode_index": np.full(4, 10000, np.int64),
            "tick": np.arange(4, dtype=np.int64)}
    return rows, np.stack(expected)


def test_reconstructed_history_matches_live_measured_motion_and_issued_jaw():
    rows, expected = sequence()
    original = rows["observations"].copy()
    actual = convert_observations(rows)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(rows["observations"], original)
    assert actual[1, -33 + 23] == pytest.approx(.001)
    assert actual[1, -33 + 28] == pytest.approx(.02)
    assert actual[1, -33 + 29] == 1.
    assert actual[2, -33 + 29] == 0.
    np.testing.assert_array_equal(actual[0, :33], actual[0, -33:])


@pytest.mark.parametrize("invalid", ["gap", "duplicate", "wrong_jaw", "wrong_history"])
def test_missing_or_inconsistent_evidence_cannot_be_silently_reconstructed(invalid):
    rows, _ = sequence()
    if invalid == "gap":
        rows["tick"][1:] += 1
    elif invalid == "duplicate":
        rows["tick"][1] = 0
    elif invalid == "wrong_jaw":
        rows["target_actions"][0, 6] = .3
    else:
        rows["observations"][3, 0] += .01
    with pytest.raises(ValueError):
        convert_observations(rows)


@pytest.mark.parametrize("change", [{"beta_teacher_execution": .5}, {"ik_rejections": 1},
    {"learner_executed_ticks": 1}, {"max_action_saturation": .1}])
def test_mixed_or_rejected_execution_requires_new_live_collection(change):
    manifest = {"status": "complete", "runtime_source_changes": [], "checkpoint_unchanged": True,
        "base_weights_frozen": True, "lighting_changed": False, "beta_teacher_execution": 1.,
        "expert_phase_label_source": "issued_command", "observation_names": TRACKED_OBSERVATION_NAMES,
        "trials": [{"abort": None, "ik_rejections": 0, "learner_executed_ticks": 0,
                    "max_action_saturation": 0.}]}
    if "beta_teacher_execution" in change:
        manifest.update(change)
    else:
        manifest["trials"][0].update(change)
    with pytest.raises(ValueError):
        validate_manifest(manifest)
