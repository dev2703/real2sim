"""Edge cases for the data contract: silent misalignment, non-finite values, degenerate geometry.

Each test pins a failure that previously passed silently (see docs/PROJECT_SUMMARY.md, cleanup entry).
"""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from real2sim.config import validate_camera_keys
from real2sim.data import audit
from real2sim.data.labels import episode_labels
from real2sim.data.prepare import prepare
from real2sim.data.split import make_episode_split
from real2sim.data.synthetic import make_synthetic_dataset
from real2sim.geometry.so3 import exp_so3, log_so3, make_T, matrix_from_rot6d
from real2sim.geometry.so101_fk import DEFAULT_MJCF, JointMapping
from real2sim.policy.dataset import DerivedTickDataset

KEY = "observation.images.cam_good"


def joint_cfg():
    return {"camera_keys": [KEY], "excluded_cameras": [], "excluded_episodes": [],
            "image": {"height": 48, "width": 64, "pad_value": 114}, "policy_hz": 10,
            "split": {"seed": 0, "n_val": 1}}


def shift_video_segment(root, episode, frames, fps=30):
    """Make the episode table claim the video segment starts ``frames`` later (a dropped frame)."""
    f = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(f)
    m = df["episode_index"] == episode
    df.loc[m, f"videos/{KEY}/from_timestamp"] += frames / fps
    df.to_parquet(f, index=False)


@pytest.fixture
def fresh_root(tmp_path):
    return make_synthetic_dataset(tmp_path / "ds", cameras={KEY: "moving"}, episode_lengths=(40, 41, 36))


# ------------------------------------------------------------------ prepare alignment
@pytest.mark.parametrize("shift", [1, 2])
def test_prepare_refuses_video_shifted_against_rows(fresh_root, tmp_path, shift):
    """A segment starting 1 or 2 frames late keeps the tick count at stride 3 for some lengths;
    the timestamp check must still refuse it (previously: silent 33-67 ms misalignment)."""
    shift_video_segment(fresh_root, 1, shift)
    with pytest.raises(RuntimeError, match="misalignment|decoded"):
        prepare(fresh_root, joint_cfg(), tmp_path / "out", "s", "r")


def test_prepare_refuses_missing_data_row(fresh_root, tmp_path):
    f = fresh_root / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(f)
    df = df.drop(df[(df.episode_index == 0) & (df.frame_index == 5)].index)
    df.to_parquet(f, index=False)
    with pytest.raises(ValueError, match="rows|contiguous"):
        prepare(fresh_root, joint_cfg(), tmp_path / "out", "s", "r")


def test_prepare_refuses_non_finite_state(fresh_root, tmp_path):
    f = fresh_root / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(f)
    s = np.stack(df["observation.state"].to_numpy())
    s[3, 2] = np.nan  # row 3 is a 10 Hz tick of episode 0
    df["observation.state"] = list(s.astype(np.float32))
    df.to_parquet(f, index=False)
    with pytest.raises(ValueError, match="non-finite state"):
        prepare(fresh_root, joint_cfg(), tmp_path / "out", "s", "r")


def test_prepare_refuses_selection_without_training_ticks(fresh_root, tmp_path):
    m = make_episode_split([0, 1, 2], n_val=1, seed=0)
    with pytest.raises(ValueError, match="no training ticks"):
        prepare(fresh_root, joint_cfg(), tmp_path / "out", "s", "r", episodes=m.val)


# ------------------------------------------------------------------ cartesian masking
@pytest.mark.skipif(not DEFAULT_MJCF.is_file(), reason="Menagerie SO-101 not fetched")
def test_extrapolated_labels_are_masked_and_kept_out_of_stats(fresh_root, tmp_path):
    cfg = joint_cfg() | {"labels": {"cartesian_lead_ticks": 1, "cartesian_gripper_source": "leader_action"},
                         "task_config": "configs/task.yaml"}
    m = prepare(fresh_root, cfg, tmp_path / "c", "s", "r", mode="cartesian")
    a = np.load(tmp_path / "c" / "arrays.npz")
    stats = json.loads((tmp_path / "c" / "stats.json").read_text())
    train = np.isin(a["episode_index"], m["split"]["train"])
    assert stats["action"]["count"] == int((train & ~a["label_is_extrapolated"]).sum())
    ds = DerivedTickDataset(tmp_path / "c", m["split"]["train"], chunk_size=20)
    last = len(ds) - 1  # last tick of the last train episode: its own label is extrapolated
    item = ds[last]
    assert bool(item["action_is_pad"][0]), "extrapolated label must be masked like padding"
    first = ds[0]
    assert not bool(first["action_is_pad"][0])


# ------------------------------------------------------------------ audit
def test_audit_flags_short_video_even_when_table_agrees(tmp_path):
    root = make_synthetic_dataset(tmp_path / "ds", cameras={KEY: "moving"}, episode_lengths=(30, 30))
    # Re-encode the video one frame short of what the table and data rows claim.
    from real2sim.data import synthetic

    info = json.loads((root / "meta" / "info.json").read_text())
    shutil.rmtree(root / "videos")
    other = make_synthetic_dataset(tmp_path / "short", cameras={KEY: "moving"}, episode_lengths=(30, 29))
    shutil.copytree(other / "videos", root / "videos")
    assert synthetic is not None and info["total_frames"] == 60
    out = tmp_path / "audit"
    audit.main(["--local-root", str(root), "--out", str(out), "--n-val", "1"])
    cq = json.loads((out / "camera_quality.json").read_text())
    ep1 = cq["streams"][KEY]["episodes"]["1"]
    assert ep1["label"] != "USABLE", ep1
    assert any("misalignment" in r for r in ep1["reasons"])


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("eps", [0.0, 1e-7, 1e-5, 5e-5, 9e-5])
def test_log_so3_is_accurate_near_pi(eps):
    rng = np.random.default_rng(int(eps * 1e9) + 1)
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        R = exp_so3(axis * (np.pi - eps))
        np.testing.assert_allclose(exp_so3(log_so3(R)), R, atol=1e-8)
        assert abs(np.linalg.norm(log_so3(R)) - (np.pi - eps)) < 1e-8


@pytest.mark.parametrize("x", [np.zeros(6), [1, 0, 0, 2, 0, 0], [0, 0, 0, 0, 1, 0], [np.nan, 0, 0, 0, 1, 0]])
def test_rot6d_degenerate_inputs_raise(x):
    with pytest.raises(ValueError):
        matrix_from_rot6d(np.asarray(x, float))


@pytest.mark.parametrize("kwargs", [
    {"sign": (1, 1, 1, 1)}, {"sign": (1, 1, 1, 1, 0)}, {"sign": (1, 1, 1, 1, 2)},
    {"offset_deg": (0, 0, 0)}, {"offset_deg": (0, 0, 0, 0, np.inf)},
    {"gripper_sim_range_rad": (1.0, -1.0)}, {"gripper_sim_range_rad": (0.0, np.nan)},
    {"gripper_affine_rad": (0.0, 0.0)}, {"gripper_affine_rad": (np.nan, .02)},
    {"gripper_affine_rad": (0.0,)},
])
def test_joint_mapping_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        JointMapping(**kwargs)


def test_joint_mapping_inverse_round_trip_with_signs_and_offsets():
    m = JointMapping(sign=(1, -1, 1, -1, 1), offset_deg=(2, -3, 0, 10, -5))
    q = np.array([10.0, -20.0, 30.0, -40.0, 50.0])
    q_sim = m.arm_real_to_sim_rad(q)
    back = (np.rad2deg(q_sim) - np.asarray(m.offset_deg)) * np.asarray(m.sign)
    np.testing.assert_allclose(back, q, atol=1e-12)
    lo, hi = m.gripper_sim_range_rad
    np.testing.assert_allclose(m.gripper_real_to_sim_rad(np.array([0.0, 100.0])), [lo, hi])


@pytest.mark.parametrize("scale", [np.deg2rad(129.4065934065934) / 100, -.02])
def test_explicit_gripper_affine_round_trip_does_not_rescale_to_native_limits(scale):
    from real2sim.sim.env import real_from_sim, sim_from_real

    m = JointMapping(gripper_affine_rad=(-.1745, scale))
    values = np.array([-5., 0., 30., 60., 100., 110.])
    mapped = m.gripper_real_to_sim_rad(values)
    np.testing.assert_allclose(mapped, -.1745 + values * scale)
    np.testing.assert_allclose(m.gripper_sim_to_real_percent(mapped), values)
    for value in values:
        state = np.r_[10., -20., 30., 40., -5., value]
        np.testing.assert_allclose(real_from_sim(m, sim_from_real(m, state)), state)
    if scale > 0:
        assert mapped[-2] > m.gripper_sim_range_rad[1]


def test_explicit_gripper_mapping_loads_without_altering_legacy_contract():
    from real2sim.config import load_yaml

    task = load_yaml("configs/task.yaml")
    legacy = JointMapping.from_task_config(task)
    assert legacy.gripper_affine_rad is None
    task["robot_so101"]["gripper"]["percent_to_joint"] = {
        "offset_rad": -.1, "scale_rad_per_percent": .022,
    }
    mapped = JointMapping.from_task_config(task)
    assert mapped.gripper_affine_rad == (-.1, .022)
    assert mapped.gripper_real_to_sim_rad(50) == pytest.approx(1.)


# ------------------------------------------------------------------ labels
def _poses(n=5):
    return np.stack([make_T(np.eye(3), np.array([0.3 + 0.01 * k, 0.0, 0.1])) for k in range(n)])


@pytest.mark.parametrize("lead", [0, -1, 1.5])
def test_labels_reject_non_positive_or_fractional_lead(lead):
    with pytest.raises(ValueError, match="lead"):
        episode_labels(_poses(), np.zeros(5), np.zeros(5), hz=10, gripper_source="leader_action", lead=lead)


def test_labels_reject_empty_and_non_finite_inputs():
    with pytest.raises(ValueError, match="no poses"):
        episode_labels(np.zeros((0, 4, 4)), np.zeros(0), np.zeros(0), hz=10, gripper_source="leader_action")
    g = np.zeros(5)
    g[2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        episode_labels(_poses(), np.zeros(5), g, hz=10, gripper_source="leader_action")


def test_labels_single_tick_episode_is_fully_extrapolated():
    lab = episode_labels(_poses(1), np.array([40.0]), np.array([2.0]), hz=10, gripper_source="leader_action")
    assert lab.label_is_extrapolated.tolist() == [True]
    np.testing.assert_allclose(lab.action[0, :6], 0)
    np.testing.assert_allclose(lab.proprio[0, 9:12], 0)


def test_labels_lead_longer_than_episode():
    lab = episode_labels(_poses(3), np.zeros(3), np.zeros(3), hz=10, gripper_source="leader_action", lead=5)
    assert lab.label_is_extrapolated.all()


# ------------------------------------------------------------------ config and split
@pytest.mark.parametrize("excluded", [None, []])
def test_excluded_cameras_null_is_treated_as_empty(excluded):
    cfg = {"camera_keys": ["a"], "excluded_cameras": excluded}
    assert validate_camera_keys(cfg, ["a"]) == ["a"]


@pytest.mark.parametrize("excluded", [["a"], [{"name": "a"}], [None]])
def test_malformed_excluded_cameras_raise_clearly(excluded):
    with pytest.raises(ValueError, match="mappings"):
        validate_camera_keys({"camera_keys": ["b"], "excluded_cameras": excluded}, ["b"])


def test_split_refuses_exclusions_that_remove_all_training():
    base = make_episode_split(list(range(6)), n_val=2, seed=0)
    with pytest.raises(ValueError, match="no training"):
        make_episode_split(list(range(6)), n_val=2, seed=0, exclude=base.train)


def test_split_n_val_bounds():
    with pytest.raises(ValueError):
        make_episode_split([0, 1, 2], n_val=3)
    with pytest.raises(ValueError):
        make_episode_split([0, 1, 2], n_val=-1)
    m = make_episode_split([0, 1, 2], n_val=0)
    assert m.val == [] and m.train == [0, 1, 2]
