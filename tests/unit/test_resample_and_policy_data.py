import json

import numpy as np
import pytest
import yaml

from real2sim.data.prepare import prepare_joint
from real2sim.data.resample import chunk_indices, episode_tick_rows, policy_stride
from real2sim.policy.dataset import DerivedTickDataset


def test_policy_stride():
    assert policy_stride(30, 10) == 3
    assert policy_stride(30, 30) == 1
    with pytest.raises(ValueError):
        policy_stride(30, 7)


def test_tick_rows_start_at_zero_and_stay_inside():
    r = episode_tick_rows(10, 3)
    assert r.tolist() == [0, 3, 6, 9]
    assert episode_tick_rows(9, 3).tolist() == [0, 3, 6]


def test_chunks_never_cross_episodes():
    # two episodes of 4 and 3 ticks
    ep_start = np.array([0, 0, 0, 0, 4, 4, 4])
    ep_len = np.array([4, 4, 4, 4, 3, 3, 3])
    idx, pad = chunk_indices(ep_start, ep_len, 2, 5)
    assert idx.tolist() == [2, 3, 3, 3, 3]
    assert pad.tolist() == [False, False, True, True, True]
    idx, pad = chunk_indices(ep_start, ep_len, 4, 5)
    assert idx.tolist() == [4, 5, 6, 6, 6]
    assert pad.tolist() == [False, False, False, True, True]
    assert all(i >= 4 for i in idx)


@pytest.fixture(scope="module")
def derived(tmp_path_factory, synthetic_root):
    cfg = {
        "camera_keys": ["observation.images.cam_good"],
        "excluded_cameras": [],
        "excluded_episodes": [],
        "image": {"height": 48, "width": 64, "pad_value": 114},
        "policy_hz": 10,
        "split": {"seed": 0, "n_val": 1},
    }
    out = tmp_path_factory.mktemp("derived")
    manifest = prepare_joint(synthetic_root, cfg, out, "synthetic/test", "abc")
    return out, manifest


def test_prepare_joint_outputs(derived, synthetic_root):
    out, m = derived
    # synthetic episodes have 40, 50, 36 rows -> 14, 17, 12 ticks at stride 3
    assert m["n_ticks"] == 14 + 17 + 12
    assert m["stride"] == 3 and m["label_semantics"].startswith("joint: recorded leader target")
    a = np.load(out / "arrays.npz")
    assert (a["frame_index"] % 3 == 0).all()
    imgs = np.load(out / m["image_files"]["observation.images.cam_good"], mmap_mode="r")
    assert imgs.shape == (m["n_ticks"], 48, 64, 3)
    stats = json.loads((out / "stats.json").read_text())
    train_eps = set(m["split"]["train"])
    mask = np.isin(a["episode_index"], list(train_eps))
    np.testing.assert_allclose(stats["action"]["mean"], a["action"][mask].mean(0), rtol=1e-5)
    assert stats["action"]["count"] == int(mask.sum())  # stats from train only
    assert not set(m["split"]["train"]) & set(m["split"]["val"])


def test_dataset_items(derived):
    out, m = derived
    ds = DerivedTickDataset(out, m["split"]["train"], chunk_size=20)
    item = ds[len(ds) - 1]  # last tick of an episode -> mostly padding
    assert item["observation.images.cam_good"].shape == (3, 48, 64)
    assert item["observation.state"].shape == (6,)
    assert item["action"].shape == (20, 6)
    assert item["action_is_pad"][0].item() is False and item["action_is_pad"][1:].all()
    assert set(np.unique([ds[i]["episode_index"] for i in range(len(ds))])) == set(m["split"]["train"])


def test_act_config_files_parse():
    cfg = yaml.safe_load(open("configs/act_joint.yaml"))
    assert cfg["model"]["dim_model"] == 512 and cfg["train"]["batch_size"] == 16
    assert cfg["model"]["chunk_size"] == 20 and cfg["model"]["n_action_steps"] == 2


def _cart_cfg(source):
    return {
        "camera_keys": ["observation.images.cam_good"],
        "excluded_cameras": [],
        "excluded_episodes": [],
        "image": {"height": 48, "width": 64, "pad_value": 114},
        "policy_hz": 10,
        "split": {"seed": 0, "n_val": 1},
        "labels": {"cartesian_lead_ticks": 1, "cartesian_gripper_source": source},
        "task_config": "configs/task.yaml",
    }


def test_cartesian_prepare_refuses_until_d013(tmp_path, synthetic_root):
    from real2sim.data.prepare import prepare

    with pytest.raises(ValueError, match="D-013"):
        prepare(synthetic_root, _cart_cfg("PENDING"), tmp_path / "c", "s", "r", mode="cartesian")


def test_cartesian_prepare_shapes(tmp_path, synthetic_root):
    from real2sim.data.prepare import prepare
    from real2sim.geometry.so101_fk import DEFAULT_MJCF

    if not DEFAULT_MJCF.is_file():
        pytest.skip("Menagerie SO-101 not fetched")
    m = prepare(synthetic_root, _cart_cfg("leader_action"), tmp_path / "c", "s", "r", mode="cartesian")
    a = np.load(tmp_path / "c" / "arrays.npz")
    assert a["state"].shape == (m["n_ticks"], 13) and a["action"].shape == (m["n_ticks"], 7)
    assert a["joint_state"].shape == (m["n_ticks"], 6)
    assert m["label_diagnostics"]["extrapolated_ticks"] == 3  # one per episode
    assert len(m["state_names"]) == 13 and m["action_names"][-1] == "gripper_target"
    ds = DerivedTickDataset(tmp_path / "c", m["split"]["train"], chunk_size=20)
    assert ds[0]["action"].shape == (20, 7) and ds[0]["observation.state"].shape == (13,)
