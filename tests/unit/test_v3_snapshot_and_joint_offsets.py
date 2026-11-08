"""Task joint offsets survive evaluation setup, and supported CLIs default to v3."""

from pathlib import Path

import numpy as np
import yaml

from real2sim.config import load_yaml, project_path
from scripts.evaluate_hf_act_20261027 import SNAPSHOT, V3_REVISION
from scripts.evaluate_hf_act_panda_20261028 import prepare_source
from scripts.run_cross_embodiment_recovery import SNAPSHOT as PIPELINE_SNAPSHOT
from scripts.train_local_hf_head_20261028 import SNAPSHOT as HEAD_SNAPSHOT
from scripts.train_recovery_mlp import SNAPSHOT as TRAIN_SNAPSHOT

CONFIRMED_V3 = "076452e6024c778a2bffabe99ecf33776de141e1"
HISTORICAL_V2 = "e696c8a078935432e3501e2c28c1fb2598fbff7e"
MEASURED = Path("configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml")


def _scene_with_offsets(tmp_path, offsets):
    config = load_yaml(MEASURED)
    task = load_yaml(project_path(config["task_config"]))
    task["robot_so101"]["real_to_sim"]["offset_deg"] = list(offsets)
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    data = load_yaml(project_path(config["data_config"]))
    data_path = tmp_path / "data.yaml"
    data["task_config"] = str(task_path)
    data_path.write_text(yaml.safe_dump(data))
    config["task_config"] = str(task_path)
    config["data_config"] = str(data_path)
    scene = tmp_path / "sim.yaml"
    scene.write_text(yaml.safe_dump(config))
    return scene


def test_task_offset_deg_is_preserved_when_no_override_is_passed(tmp_path):
    expected = [1.5, -2.0, 3.25, -4.0, 0.5]
    scene = _scene_with_offsets(tmp_path, expected)
    out = tmp_path / "prepared"
    out.mkdir()
    _, task = prepare_source(out, source_scene=scene)
    saved = load_yaml(out / "source" / "task.yaml")
    np.testing.assert_allclose(task["robot_so101"]["real_to_sim"]["offset_deg"], expected)
    np.testing.assert_allclose(saved["robot_so101"]["real_to_sim"]["offset_deg"], expected)

    overridden = [0.0, 0.0, 0.0, 0.0, 9.0]
    replaced_out = tmp_path / "overridden"
    replaced_out.mkdir()
    _, replaced = prepare_source(replaced_out, overridden, source_scene=scene)
    np.testing.assert_allclose(replaced["robot_so101"]["real_to_sim"]["offset_deg"], overridden)


def test_supported_cli_default_snapshot_is_confirmed_v3_revision():
    assert V3_REVISION == CONFIRMED_V3
    assert SNAPSHOT.name == CONFIRMED_V3
    assert SNAPSHOT.is_absolute()
    assert "models--bottowork--act_so101_pickplace_v3_varied_kaggle" in SNAPSHOT.parts
    assert PIPELINE_SNAPSHOT == SNAPSHOT
    assert HISTORICAL_V2 in TRAIN_SNAPSHOT.parts
    assert HISTORICAL_V2 in HEAD_SNAPSHOT.parts
