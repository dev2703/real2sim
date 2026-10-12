"""SO-101 -> Panda retargeting baseline (plan M5 step 3) on the real validation episode 1.

Measured on 2026-10-13 (full episode, 598 ticks): IK acceptance 567/598 = 94.8 %, all 31
rejections ``joint_limit`` with joint4 (elbow) at its lower limit near the Panda start pose
(D-015 reach caveat); the offset-corrected relative-motion error grows to 31 cm almost entirely
from the deltas lost on rejected ticks (30.0 of 31.5 cm). No robot-table contact, no grasp
(the ball start is unrelated to the real episode). Nothing here is a policy result.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from real2sim.panda import retarget

DATASET = Path("~/.cache/huggingface/lerobot/local/so101_act_pickplace_v1").expanduser()
EPISODE = 1

pytestmark = pytest.mark.integration
needs_dataset = pytest.mark.skipif(not (DATASET / "meta" / "info.json").is_file(),
                                   reason="local LeRobot dataset not present")


def test_caption_is_labelled_scripted_never_learned():
    caption = retarget.caption_for(EPISODE)
    assert "SCRIPTED REPLAY" in caption
    assert "Panda" in caption
    assert "LEARNED" not in caption.upper()
    assert retarget.CONTROLLER_SOURCE == "SCRIPTED_REPLAY"


def test_contact_classification_ignores_ball_on_table():
    contacts = [{"geom1": "ball_geom", "geom2": "table", "distance_m": -0.0001},
                {"geom1": "geom_12", "geom2": "table", "distance_m": -0.003},
                {"geom1": "bowl_wall_03", "geom2": "geom_40", "distance_m": -0.001},
                {"geom1": "geom_40", "geom2": "geom_41", "distance_m": -0.001}]
    c = retarget.classify_contacts(contacts)
    assert c["robot_table"] == 1 and c["robot_bowl"] == 1 and c["robot_self"] == 1
    assert c["robot_ball"] == 0
    assert c["robot_table_penetration_m"] == pytest.approx(0.003)


@needs_dataset
def test_retarget_20_ticks_of_episode_1(tmp_path):
    summary = retarget.run_retarget(EPISODE, DATASET, tmp_path / "metrics", tmp_path / "media",
                                    max_ticks=20, video_stride=5)
    csv_path = tmp_path / "metrics" / f"panda_retarget_ep{EPISODE}.csv"
    assert csv_path.is_file()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20 == summary["n_ticks_replayed"]
    assert {"tick", "ik_accepted", "ik_reason", "ik_pos_residual_m", "ik_rot_residual_rad",
            "cmd_dx", "cmd_gripper_target", "panda_tcp_x_m", "so101_fk_x_m", "pos_err_raw_m",
            "pos_err_offset_corrected_m", "panda_gripper_opening_m",
            "robot_table_contacts"} <= rows[0].keys()
    assert "SCRIPTED REPLAY" in summary["caption"]
    assert "LEARNED" not in summary["caption"].upper()
    assert summary["controller_source"] == "SCRIPTED_REPLAY"
    ik = summary["ik"]
    assert 0.0 <= ik["acceptance_rate"] <= 1.0
    assert ik["accepted"] + sum(v for k, v in ik["reasons"].items() if k != "accepted") == 20
    # The start poses differ and the offset is reported, not matched: raw error ~ offset norm
    # at tick 0 while the offset-corrected error starts at zero.
    offset = summary["start_pose"]["offset_norm_m"]
    assert offset > 0.1
    assert float(rows[0]["pos_err_raw_m"]) == pytest.approx(offset, abs=1e-3)
    assert float(rows[0]["pos_err_offset_corrected_m"]) < 1e-6
    assert np.isfinite(summary["position_error_cm"]["raw_vs_so101_fk"]["median"])
    assert isinstance(summary["contacts"]["robot_table_any_tick"], bool)
    assert isinstance(summary["ball_outcome"]["ball_in_bowl_final"], bool)
    json_path = tmp_path / "metrics" / f"panda_retarget_ep{EPISODE}.json"
    assert json.loads(json_path.read_text())["caption"] == summary["caption"]
    assert (tmp_path / "metrics" / f"panda_retarget_ep{EPISODE}.png").is_file()
    assert (tmp_path / "media" / f"panda_retarget_ep{EPISODE}.mp4").is_file()
    assert summary["video"]["frames"] == 4
