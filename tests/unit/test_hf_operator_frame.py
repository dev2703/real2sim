"""A diagnostic frame fit must preserve scalar geometry and exclude validation."""

import json

import numpy as np
import yaml

from real2sim.config import load_yaml
from scripts import diagnose_hf_operator_frame as diagnostic


def test_training_only_yaw_preserves_measurements_and_lighting(monkeypatch, tmp_path):
    source = load_yaml("configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml")
    task = load_yaml(source["task_config"])
    task["ball"]["start_region"]["center_task_xy"] = [0., 1.]
    task["bowl"]["center_task_xy"] = [1., 1.]
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    source["task_config"] = str(task_path)
    source_path = tmp_path / "sim.yaml"
    source_path.write_text(yaml.safe_dump(source))
    snapshot = tmp_path / "checkpoint"
    snapshot.mkdir()
    (snapshot / "experiment.json").write_text(json.dumps({"train_episodes": [1, 2], "val_episodes": [99]}))
    (snapshot / "model.safetensors").write_bytes(b"frozen mock weights")
    dataset = tmp_path / "data"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text("{}")

    class Dataset:
        def __init__(self, _path):
            pass

        def episode_rows(self, episode):
            assert episode in [1, 2], "validation rows must not enter frame derivation"
            return episode

        def stack(self, rows, key):
            states = np.zeros((3, 6), float)
            if key == "action":
                states[:, 5] = [60., 40., 20.]
            return states

    class FK:
        def __init__(self, **_kwargs):
            pass

        def fk_real(self, _state):
            pose = np.eye(4)
            pose[0, 3] = 1.
            return pose

    monkeypatch.setattr(diagnostic, "RawDataset", Dataset)
    monkeypatch.setattr(diagnostic, "So101FK", FK)
    candidate_path = diagnostic.derive(snapshot, dataset, source_path, tmp_path / "candidate")
    candidate_config = load_yaml(candidate_path)
    candidate = load_yaml(candidate_config["task_config"])
    fit = candidate["operator_frame_hypothesis"]
    np.testing.assert_allclose(fit["operator_to_native_yaw_degrees"], -90.)
    np.testing.assert_allclose(candidate["ball"]["start_region"]["center_task_xy"], [1., 0.], atol=1e-12)
    before_separation = np.linalg.norm(np.asarray(task["bowl"]["center_task_xy"]) - [0., 1.])
    after_separation = np.linalg.norm(np.asarray(candidate["bowl"]["center_task_xy"]) - [1., 0.])
    np.testing.assert_allclose(after_separation, before_separation)
    np.testing.assert_allclose(candidate["table"]["mat"]["yaw_deg"], -90.)
    assert candidate["table"]["mat"]["size_xy"] == task["table"]["mat"]["size_xy"]
    assert candidate["robot_so101"] == task["robot_so101"]
    assert candidate_config["appearance"]["lighting"] == source["appearance"]["lighting"]
    assert [episode["episode"] for episode in fit["episodes"]] == [1, 2]
    assert fit["validation_episodes_not_used"] == [99]
    assert not fit["distances_scaled"] and not fit["joint_offsets_changed"]
