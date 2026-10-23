"""Relabeling changes FK-dependent arrays while preserving immutable sampled evidence."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from real2sim.config import load_yaml
from real2sim.data.labels import episode_labels
from real2sim.data.prepare import compute_stats
from real2sim.data.raw import sha256_file
from real2sim.geometry.so101_fk import JointMapping, So101FK

SPEC = importlib.util.spec_from_file_location("regenerate_v2_fk_labels", Path("scripts/regenerate_v2_fk_labels.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def source_data(tmp_path):
    task = load_yaml("configs/v2/task.exploratory_20261022.yaml")
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    cfg = {"task_config": str(task_path), "policy_hz": 12.5,
           "labels": {"cartesian_gripper_source": "leader_action", "cartesian_lead_ticks": 1},
           "camera_keys": ["observation.images.iphone"], "contract_version": "corrected-v2"}
    q = np.array([[0, -20, 30, 0, 0, 30], [1, -21, 31, 0, 2, 20], [2, -22, 32, 0, 4, 10]], np.float32)
    joints = np.tile(q, (3, 1))
    fk = So101FK(mapping=JointMapping.from_task_config(task))
    labels = [episode_labels(fk.fk_real(q), q[:, 5], q[:, 5], hz=12.5,
                             gripper_source="leader_action", leader_poses=fk.fk_real(q)) for _ in range(3)]
    arrays = {"joint_state": joints, "joint_action": joints.copy(),
              "episode_index": np.repeat([0, 1, 2], 3), "frame_index": np.tile([0, 2, 4], 3),
              "timestamp": np.tile([0, .08, .16], 3),
              "state": np.concatenate([a.proprio for a in labels]).astype(np.float32),
              "action": np.concatenate([a.action for a in labels]).astype(np.float32),
              "label_is_extrapolated": np.concatenate([a.label_is_extrapolated for a in labels]),
              "leader_follower_pos_err": np.zeros(9, np.float32), "jump_flag": np.zeros(9, bool)}
    return task, task_path, cfg, arrays


@pytest.mark.parametrize("mode", ["cartesian", "cartesian_tool", "cartesian_absolute", "canonical_joint"])
def test_changed_mapping_recomputes_states_and_preserves_raw_rows_and_boundaries(source_data, mode):
    task, task_path, cfg, arrays = source_data
    task["robot_so101"]["real_to_sim"]["offset_deg"][0] = 10
    task_path.write_text(yaml.safe_dump(task))
    result, _ = MODULE.regenerate_arrays(arrays, cfg, mode)
    assert np.linalg.norm(result["state"][0, :3] - arrays["state"][0, :3]) > .001
    for key in ("joint_state", "joint_action", "timestamp", "frame_index", "episode_index", "label_is_extrapolated"):
        np.testing.assert_array_equal(arrays[key], result[key])
    np.testing.assert_array_equal(result["state"][[0, 3, 6], 9:12], 0)
    if mode in ("cartesian", "cartesian_tool"):
        np.testing.assert_array_equal(result["action"][[2, 5, 8], :6], 0)
    if mode == "canonical_joint":
        np.testing.assert_array_equal(result["action"], arrays["joint_action"])


def test_regeneration_versions_lineage_reuses_images_and_keeps_reserved_out_of_stats(source_data, tmp_path):
    _, task_path, cfg, arrays = source_data
    source, out = tmp_path / "source", tmp_path / "new"
    source.mkdir()
    # Reserved states must not affect new training statistics.
    arrays["joint_state"][6:, 0] = 100
    np.savez(source / "arrays.npz", **arrays)
    np.save(source / "images_iphone.npy", np.zeros((9, 2, 2, 3), np.uint8))
    (source / "stats.json").write_text(json.dumps({"observation.state": {}, "action": {}, "train_episodes": [0]}))
    split = {"train": [0], "val": [1], "reserved": [2]}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    cfg["split"] = {"fixed_manifest": str(split_path)}
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump(cfg))
    manifest = {"contract_version": "old-v1", "mode": "cartesian", "policy_hz": 12.5,
                "camera_keys": cfg["camera_keys"], "split": split,
                "image_files": {cfg["camera_keys"][0]: "images_iphone.npy"},
                "files_sha256": {p.name: sha256_file(p) for p in source.iterdir()}, "action_names": []}
    (source / "manifest.json").write_text(json.dumps(manifest))
    report = MODULE.regenerate(source, config, out)
    assert report["reserved_ticks_preserved_but_not_scored"] == 3
    assert (out / "images_iphone.npy").is_symlink()
    assert sha256_file(out / "task.yaml") == sha256_file(task_path)
    with np.load(out / "arrays.npz") as loaded:
        expected = compute_stats(loaded["state"][:3])
    assert json.loads((out / "stats.json").read_text())["observation.state"] == expected
    new = json.loads((out / "manifest.json").read_text())
    assert new["contract_version"] == "corrected-v2"
    assert new["physical_calibration_accepted"] is False
    assert new["split"] == split
    assert sha256_file(source / "arrays.npz") == manifest["files_sha256"]["arrays.npz"]
    with pytest.raises(FileExistsError):
        MODULE.regenerate(source, config, out)
    cfg["contract_version"] = "old-v1"
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="new explicit contract_version"):
        MODULE.regenerate(source, config, tmp_path / "different")
