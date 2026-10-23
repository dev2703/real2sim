"""Scene diagnostics must preserve the checkpoint's policy interface and production gate."""

import copy
import importlib.util
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
import yaml

from real2sim.config import load_yaml
from real2sim.evaluation.run import check_contract_timing_and_image
from real2sim.policy.adapters import ContractError
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics


@pytest.fixture
def ablation():
    path = Path(__file__).resolve().parents[2] / "scripts/diag_v2_geometry_policy_ablation.py"
    spec = importlib.util.spec_from_file_location("geometry_ablation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tasks(ablation):
    original = load_yaml("configs/v2/task.exploratory_20261022.yaml")
    camera = json.loads(Path("results/v2/camera_render_candidate_20261023.json").read_text())
    scene = next(case for case in json.loads(Path(
        "results/v2/ball_support_scene_candidates_20261023.json").read_text())["cases"]
                 if case["case_id"] == "image_support_10mm")
    return original, ablation.make_task(original, camera, scene)


@pytest.mark.parametrize("path,value", [
    (("robot_so101", "real_to_sim", "offset_deg"), [0., 0., 0., 0., 1.]),
    (("robot_so101", "gripper", "sim_range_rad"), [-.1, 1.7]),
    (("frames", "T_task_so101base", "xyz"), [0., 0., .01]),
    (("ball", "radius"), .032),
    (("episode", "timeout_s"), 2.),
    (("camera_wrist",), None),
])
def test_unlisted_changes_rejected(ablation, tasks, path, value):
    original, candidate = tasks
    parent = candidate
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match="unlisted"):
        ablation.validate_task_ablation(original, candidate)


def test_scene_cannot_claim_accepted_calibration(ablation, tasks):
    original, candidate = tasks
    candidate["physical_calibration_accepted"] = True
    with pytest.raises(ValueError, match="physically unaccepted"):
        ablation.validate_task_ablation(original, candidate)


def test_matched_observed_layout_preserves_predeclared_height_and_camera(ablation):
    original = load_yaml("configs/v2/task.exploratory_20261022.yaml")
    camera = json.loads(Path("results/v2/camera_render_candidate_20261023.json").read_text())
    scene = next(case for case in json.loads(Path(
        "results/v2/ball_support_scene_candidates_20261023.json").read_text())["cases"]
                 if case["case_id"] == "image_support_10mm")
    observed = {"camera_candidate": "late", "physical_calibration_accepted": False,
                "ball_center_on_declared_support_plane_native_m": [.25, .02, .0411],
                "support": {**scene["support"], "center_task_xy": [.268, .023]}}
    candidate = ablation.make_task(original, camera, scene, "late", observed)
    assert candidate["ball"]["start_region"]["center_task_xy"] == [.25, .02]
    assert candidate["ball"]["support"]["height"] == .010
    assert candidate["camera_iphone"]["intrinsics_px"] == camera[
        "candidates"]["late"]["camera"]["intrinsics_px"]
    assert candidate["bowl"]["center_task_xy"] == scene["bowl_xy"]
    with pytest.raises(ValueError, match="camera candidate disagree"):
        ablation.make_task(original, camera, scene, "early", observed)
    observed["support"]["height"] = .020
    with pytest.raises(ValueError, match="predeclared support height"):
        ablation.make_task(original, camera, scene, "late", observed)
    observed["support"]["height"] = .010
    observed["ball_center_on_declared_support_plane_native_m"][2] = .0511
    with pytest.raises(ValueError, match="declared support plane"):
        ablation.make_task(original, camera, scene, "late", observed)


def test_real_scene_preserves_fk_but_production_guard_stays_closed(ablation, tasks, tmp_path):
    original, candidate = tasks
    reference_path = Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml")
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(candidate))
    config = load_yaml(reference_path)
    config["task_config"] = str(task_path)
    config["ik"]["workspace_min"][2] = -.0024
    config_path = tmp_path / "sim.yaml"
    config_path.write_text(yaml.safe_dump(config))
    contract = json.loads(Path(
        "outputs/v2/learning_screen_20261022/runs/L1_resnet_iphone/attempt-001/"
        "checkpoints/004000/real2sim_contract.json").read_text())
    saved_contract = copy.deepcopy(contract)
    with closing(So101PickPlaceEnv(reference_path, render_images=False)) as reference:
        with closing(So101PickPlaceEnv(config_path, render_images=False)) as env:
            ablation.validate_scene_interface(reference, env, contract)
            with pytest.raises(ContractError, match="calibration differs"):
                check_contract_timing_and_image(contract, env)
            reference.reset(seed=0)
            env.reset(seed=0)
            np.testing.assert_array_equal(reference.canonical_proprio(), env.canonical_proprio())
            before, after = (NativeKinematics(e.model, e.config) for e in (reference, env))
            for q in np.random.default_rng(72).uniform(
                    env.limits[:, 0], env.limits[:, 1], size=(8, 6)):
                np.testing.assert_allclose(before.forward(q), after.forward(q), atol=1e-12)
            env.control_hz = 10.
            with pytest.raises(ContractError, match="timing or joint units"):
                ablation.validate_scene_interface(reference, env, contract)
    assert contract == saved_contract
    assert original["robot_so101"]["real_to_sim"] == candidate["robot_so101"]["real_to_sim"]
