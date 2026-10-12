"""E0 scripted replay diagnostic (plan M2 step 5) on the real validation episode 1.

Measured on 2026-10-13 (20 ticks, joint mode): median |err| shoulder_pan ~0.0, shoulder_lift ~4.5
(the dataset's -104.5 deg exceeds the +-100 deg model limit), elbow_flex ~16.7 (the clamped
start pose penetrates the sim table and contact forces push the elbow out), wrist_flex ~1.4,
wrist_roll ~0.01 deg. Hence 4 of 5 arm joints are inside 5 deg, not 5 of 5.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from real2sim.evaluation import replay

DATASET = Path("~/.cache/huggingface/lerobot/local/so101_act_pickplace_v1").expanduser()
EPISODE = 1
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

pytestmark = pytest.mark.integration
# Only the physics tests need the dataset; the caption/label guard below is pure and must run
# everywhere (a rename dropping "SCRIPTED REPLAY" must fail even on dataset-less machines).
needs_dataset = pytest.mark.skipif(not (DATASET / "meta" / "info.json").is_file(),
                                   reason="local LeRobot dataset not present")


def test_caption_is_labelled_scripted():
    for mode in replay.MODES:
        caption = replay.caption_for(mode, EPISODE)
        assert "SCRIPTED REPLAY" in caption
        assert "LEARNED" not in caption.upper()
    assert replay.CONTROLLER_SOURCE == "SCRIPTED_REPLAY"


@needs_dataset
def test_joint_replay_20_ticks(tmp_path):
    summary = replay.run_replay(EPISODE, "joint", DATASET, tmp_path / "metrics", tmp_path / "media",
                                max_ticks=20, video_stride=5)
    csv_path = tmp_path / "metrics" / f"replay_ep{EPISODE}_joint.csv"
    assert csv_path.is_file()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20 == summary["n_ticks_replayed"]
    assert {"tick", "cmd_shoulder_lift", "sim_shoulder_lift", "err_shoulder_lift", "ee_pos_err_m"} <= rows[0].keys()
    assert "SCRIPTED REPLAY" in summary["caption"]
    assert summary["controller_source"] == "SCRIPTED_REPLAY"
    medians = np.array([summary["tracking_error"][j]["median_abs"] for j in ARM])
    assert np.all(np.isfinite(medians))
    # Measured: 4 of 5 joints inside 5 deg (elbow_flex ~16.7 deg because of the start-pose table
    # penetration); see the module docstring. Do not raise this to 5/5 without a verified mapping.
    assert int((medians < 5.0).sum()) >= 4, medians.tolist()
    assert (tmp_path / "metrics" / f"replay_ep{EPISODE}_joint.png").is_file()
    assert (tmp_path / "media" / f"replay_ep{EPISODE}_joint.mp4").is_file()
    assert summary["video"]["frames"] == 4
    # The start clamp of shoulder_lift (dataset -104.5 deg vs model limit -100 deg) must be recorded.
    assert [c["joint"] for c in summary["start_pose_clamps"]] == ["shoulder_lift"]


@needs_dataset
def test_cartesian_replay_runs_and_reports_ik(tmp_path):
    summary = replay.run_replay(EPISODE, "cartesian", DATASET, tmp_path / "metrics", None, max_ticks=10)
    assert (tmp_path / "metrics" / f"replay_ep{EPISODE}_cartesian.csv").is_file()
    assert "FK->IK" in summary["caption"] and "SCRIPTED REPLAY" in summary["caption"]
    assert 0.0 <= summary["ik"]["acceptance_rate"] <= 1.0
    assert summary["ik"]["accepted"] + sum(
        v for k, v in summary["ik"]["reasons"].items() if k != "accepted") == 10
