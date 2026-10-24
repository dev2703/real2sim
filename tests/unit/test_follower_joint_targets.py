"""Target timing, masks, segment clocks and guarded joint execution semantics."""

import json

import numpy as np
import pytest

from real2sim.data.follower_joint_targets import make_matched_targets, segment_metadata
from real2sim.geometry.so101_fk import JointMapping
from real2sim.policy.dataset import DerivedTickDataset, Normalizer
from real2sim.policy.follower_joint import check_joint_request
from real2sim.policy.validation import executed_prefix_anchors, executed_prefix_report


def source_arrays():
    episode = np.array([0] * 5 + [1] * 3)
    states = np.zeros((8, 6), np.float32)
    states[:, 5] = 40
    states[:, 1] = np.arange(8)
    states[2, 0] = 200
    leader = states.copy()
    leader[:, 5] = np.arange(8) + 10
    leader[3, 0] = 250
    canonical = np.zeros((8, 13), np.float32)
    canonical[:, 3:9] = [1, 0, 0, 0, 1, 0]
    return {"episode_index": episode, "state": canonical, "joint_state": states,
            "joint_action": leader, "action": np.zeros((8, 7), np.float32),
            "ep_start": np.array([0] * 5 + [5] * 3), "ep_len": np.array([5] * 5 + [3] * 3),
            "label_is_extrapolated": np.array([False] * 4 + [True, False, False, True]),
            "timestamp": np.arange(8) / 12.5}


def matched():
    return make_matched_targets(source_arrays(), JointMapping(), np.array([[-2, 2]] * 6), [0, 1])


def test_target_shift_stays_in_episode_and_preserves_current_leader_gripper():
    source = source_arrays()
    variants, report = matched()
    follower, leader = variants.values()
    np.testing.assert_array_equal(follower["action"][0, :5], source["joint_state"][1, :5])
    np.testing.assert_array_equal(follower["action"][4, :5], source["joint_state"][4, :5])
    np.testing.assert_array_equal(follower["action"][:, 5], source["joint_action"][:, 5])
    np.testing.assert_array_equal(leader["action"], source["joint_action"])
    np.testing.assert_array_equal(follower["action_is_invalid"], leader["action_is_invalid"])
    np.testing.assert_array_equal(follower["timestamp"], source["timestamp"])
    np.testing.assert_array_equal(follower["label_is_extrapolated"], source["label_is_extrapolated"])
    assert not report["reference_clipping"]


def test_native_invalid_gaps_become_independent_chunks_without_compressing_rows():
    variants, _ = matched()
    a = variants["follower_next_arm_leader_gripper"]
    np.testing.assert_array_equal(~a["action_is_invalid"], [True, False, False, False, False, True, True, False])
    np.testing.assert_array_equal(a["source_ep_start"], source_arrays()["ep_start"])
    assert a["ep_start"][0] == 0 and a["ep_len"][0] == 1
    assert a["ep_start"][5] == 5 and a["ep_len"][5] == 2
    assert a["segment_index"][0] != a["segment_index"][5]
    ticks, anchors = executed_prefix_anchors(a, [0, 1], 2)
    np.testing.assert_array_equal(ticks, [0, 5, 6])
    np.testing.assert_array_equal(anchors, [0, 5, 5])
    manifest = {"split": {"train": [0], "val": [1], "reserved": [2]}, "action_representation": "canonical_joint"}
    report = executed_prefix_report(a["action"][ticks], ticks, anchors, a, manifest, np.ones(6), [0, 1])
    assert report["valid_actions"] == 3
    assert report["scores"]["model"]["all"]["l1"] == 0


def test_dataset_never_anchors_invalid_observations_or_bridges_masked_gaps(tmp_path):
    a = matched()[0]["follower_next_arm_leader_gripper"]
    manifest = {"camera_keys": ["observation.images.iphone"],
                "image_files": {"observation.images.iphone": "images.npy"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    np.savez(tmp_path / "arrays.npz", **a)
    np.save(tmp_path / "images.npy", np.zeros((8, 192, 256, 3), np.uint8))
    norm = Normalizer({"observation.state": {"mean": [0] * 13, "std": [1] * 13},
                       "action": {"mean": [0] * 6, "std": [1] * 6},
                       "image_normalization": {"mean": [0] * 3, "std": [1] * 3}})
    ds = DerivedTickDataset(tmp_path, [0, 1], 4, norm)
    np.testing.assert_array_equal(ds.indices, [0, 5, 6])
    np.testing.assert_array_equal(ds[0]["action_is_pad"], [False, True, True, True])
    np.testing.assert_array_equal(ds[1]["action_is_pad"], [False, False, True, True])


def test_first_only_valid_run_has_finite_segment_summary():
    starts, lengths, segment = segment_metadata(np.zeros(3, int), [True, False, False])
    np.testing.assert_array_equal(starts, [0, 1, 2])
    np.testing.assert_array_equal(lengths, [1, 1, 1])
    np.testing.assert_array_equal(segment, [0, -1, -1])


class Kinematics:
    limits = np.array([[-1, 1]] * 6)
    config = {"workspace_min": [-1, -1, -1], "workspace_max": [1, 1, 1]}
    collision_interval = None

    def forward(self, q):
        self.q = q.copy()
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        return pose

    def _collision(self):
        return self.collision_interval is not None and self.collision_interval[0] < self.q[0] < self.collision_interval[1]


def test_joint_request_rejects_native_limits_without_silent_clipping():
    requested = np.zeros(6)
    requested[0] = 1.01
    accepted, reason, target = check_joint_request(Kinematics(), np.zeros(6), np.zeros(6), requested, np.ones(6))
    assert not accepted and reason == "target_native_limits"
    np.testing.assert_array_equal(target, np.zeros(6))


def test_joint_request_checks_capped_path_not_only_requested_endpoint():
    ik = Kinematics()
    ik.collision_interval = (.04, .06)
    requested = np.zeros(6)
    requested[0] = .5
    accepted, reason, _ = check_joint_request(ik, np.zeros(6), np.zeros(6), requested, np.full(6, .1), 17)
    assert not accepted and reason == "path_collision"


def test_joint_request_matches_native_rate_cap_about_previous_command():
    start = np.full(6, .2)
    requested = np.full(6, .5)
    accepted, reason, target = check_joint_request(Kinematics(), np.zeros(6), start, requested, np.full(6, .1))
    assert accepted and reason == "accepted"
    np.testing.assert_allclose(target, .3)


@pytest.mark.parametrize("field", ["joint_state", "joint_action"])
def test_nonfinite_joint_evidence_is_rejected(field):
    a = source_arrays()
    a[field][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        make_matched_targets(a, JointMapping(), np.array([[-2, 2]] * 6), [0, 1])
