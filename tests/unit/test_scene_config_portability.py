"""Runtime scenes must load their task from the current checkout."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from real2sim import config
from real2sim.panda import model as panda
from real2sim.sim import model as so101


@pytest.mark.parametrize("loader,relative_path", [
    (so101.load_scene, "configs/v2/positive_setup_20261026/sim_so101.yaml"),
    (so101.load_scene, "configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml"),
    (panda.load_scene, "configs/v2/measured_registration_20261028/sim_panda_positive_all_scalars.yaml"),
])
def test_scene_loads_task_from_relocated_checkout(tmp_path, monkeypatch, loader, relative_path):
    original_root = config.REPO_ROOT
    original_model, original_config, original_task = loader(original_root / relative_path)
    relocated = tmp_path / "checkout"
    shutil.copytree(original_root / "configs", relocated / "configs")
    # Assets stay byte-identical; the copied task has a marker to detect accidentally
    # loading the original checkout's still-existing absolute path.
    (relocated / "assets").symlink_to(original_root / "assets", target_is_directory=True)
    task_relative = Path(original_config["task_config"])
    if task_relative.is_absolute():
        task_relative = task_relative.relative_to(original_root)
    task_path = relocated / task_relative
    copied_task = config.load_yaml(task_path)
    copied_task["checkout_marker"] = "relocated"
    task_path.write_text(yaml.safe_dump(copied_task))
    monkeypatch.setattr(config, "REPO_ROOT", relocated)

    relocated_model, _, relocated_task = loader(relocated / relative_path)

    assert relocated_task.pop("checkout_marker", None) == "relocated"
    assert relocated_task == original_task
    for field in ("body_pos", "body_mass", "body_inertia", "geom_size", "geom_friction",
                  "jnt_range", "actuator_ctrlrange", "cam_pos", "cam_quat"):
        np.testing.assert_array_equal(getattr(relocated_model, field), getattr(original_model, field))
