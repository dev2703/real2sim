"""Test-only optical mounts: render ordering and rigid-body motion on both embodiments."""

import copy

import mujoco
import numpy as np
import pytest
import yaml

from real2sim.config import load_yaml
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.sim.env import So101PickPlaceEnv


@pytest.mark.integration
@pytest.mark.parametrize(
    "robot,body,cls",
    [("so101", "gripper", So101PickPlaceEnv), ("panda", "hand", PandaPickPlaceEnv)],
)
def test_link_camera_renders_declared_views_and_moves_with_robot(tmp_path, robot, body, cls):
    config = copy.deepcopy(load_yaml(f"configs/sim_{robot}.yaml"))
    task = load_yaml(config["task_config"])
    local = np.array([0.01, 0.005, 0.02])
    task["additional_cameras"] = [
        dict(
            key="observation.images.top",
            name="wrist",
            fovy_deg=70,
            mounts={
                robot: dict(
                    body=body, position_body_m=local.tolist(), xyaxes_body=[1, 0, 0, 0, 1, 0]
                )
            },
        )
    ]
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    data = load_yaml(config["data_config"])
    data["camera_keys"] = ["observation.images.top", "observation.images.iphone"]
    data["excluded_cameras"] = []
    dp = tmp_path / "data.yaml"
    dp.write_text(yaml.safe_dump(data))
    config.update(task_config=str(task_path), data_config=str(dp))
    cp = tmp_path / "sim.yaml"
    cp.write_text(yaml.safe_dump(config))
    with cls(cp) as env:
        obs, _ = env.reset(seed=0)
        assert [k for k in obs if k.startswith("observation.images.")] == data["camera_keys"]
        assert all(obs[k].shape == (3, 192, 256) for k in data["camera_keys"])
        camera = env.model.camera("wrist").id
        link = env.model.body(body).id
        before = env.data.cam_xpos[camera].copy()
        env.data.qpos[env.qadr[0]] += 0.1
        mujoco.mj_forward(env.model, env.data)
        assert np.linalg.norm(env.data.cam_xpos[camera] - before) > 1e-4
        R = env.data.xmat[link].reshape(3, 3)
        np.testing.assert_allclose(
            R.T @ (env.data.cam_xpos[camera] - env.data.xpos[link]), local, atol=1e-10
        )
