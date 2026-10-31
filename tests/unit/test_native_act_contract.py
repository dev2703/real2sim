"""Native runner contracts retain checkpoint camera roles and honest identity."""

from types import SimpleNamespace

import pytest

from real2sim.config import load_yaml
from real2sim.policy.native_lerobot_act import (
    native_checkpoint_identity,
    native_scene_contract,
)
from scripts.evaluate_hf_act_20261027 import prepare_scene


def config(wrist="top", shape=(3, 360, 640)):
    images = {f"observation.images.{wrist}": SimpleNamespace(shape=shape),
              "observation.images.iphone": SimpleNamespace(shape=shape)}
    return SimpleNamespace(input_features={"observation.state": SimpleNamespace(shape=(6,)), **images},
        output_features={"action": SimpleNamespace(shape=(6,))}, image_features=images,
        n_obs_steps=1, chunk_size=50, n_action_steps=10, temporal_ensemble_coeff=None, repo_id=None)


@pytest.mark.parametrize("wrist", ["top", "wrist"])
def test_checkpoint_camera_roles_and_sizes_drive_source_scene(tmp_path, wrist):
    model = config(wrist, (3, 180, 320))
    # Saved camera order is meaningful even when global precedes wrist.
    model.image_features = dict(reversed(list(model.image_features.items())))
    contract = native_scene_contract(model)
    scene = load_yaml(prepare_scene(tmp_path, native_contract=contract))
    task, data = load_yaml(scene["task_config"]), load_yaml(scene["data_config"])
    assert data["camera_keys"] == list(model.image_features)
    assert task["additional_cameras"][0]["key"] == f"observation.images.{wrist}"
    assert scene["render"] == {"height": 180, "width": 320}
    assert contract["policy_hz"] == data["policy_hz"] == scene["control_hz"] == 25.
    assert task["additional_cameras"][0]["physical_calibration_accepted"] is False


@pytest.mark.parametrize("invalid", ["state", "action", "missing_state", "camera", "shape",
                                     "rgb", "ensemble", "observations", "prefix"])
def test_unsupported_native_interfaces_fail_before_robot_execution(invalid):
    model = config()
    if invalid == "state":
        model.input_features["observation.state"].shape = (7,)
    elif invalid == "action":
        model.output_features["action"].shape = (7,)
    elif invalid == "missing_state":
        model.input_features.pop("observation.state")
    elif invalid == "camera":
        model.image_features["observation.images.unknown"] = model.image_features.pop("observation.images.top")
    elif invalid == "shape":
        model.input_features["observation.images.iphone"].shape = (3, 180, 320)
    elif invalid == "rgb":
        for feature in model.image_features.values():
            feature.shape = (1, 360, 640)
    elif invalid == "ensemble":
        model.temporal_ensemble_coeff = .01
    elif invalid == "observations":
        model.n_obs_steps = 2
    else:
        model.n_action_steps = 0
    with pytest.raises(ValueError, match="native"):
        native_scene_contract(model)


def test_requested_cache_snapshot_identity_uses_its_own_revision(tmp_path):
    snapshot = tmp_path / "models--bottowork--act_so101_pickplace_v3_varied_kaggle" / "snapshots" / "v3-revision"
    identity = native_checkpoint_identity(snapshot, config())
    assert identity["repo_id"] == "bottowork/act_so101_pickplace_v3_varied_kaggle"
    assert identity["revision"] == "v3-revision"
    assert identity["checkpoint_identity_basis"] == "huggingface_cache_path"


def test_unidentified_local_checkpoint_does_not_claim_default_hf_identity(tmp_path):
    identity = native_checkpoint_identity(tmp_path / "my-checkpoint", config())
    assert identity["repo_id"] is None and identity["revision"] is None
    assert identity["checkpoint_identity_basis"] == "unidentified_local_checkpoint"


def test_local_checkpoint_config_repo_is_reported_without_inventing_revision(tmp_path):
    model = config()
    model.repo_id = "my-org/my-model"
    identity = native_checkpoint_identity(tmp_path / "my-checkpoint", model)
    assert identity["repo_id"] == model.repo_id and identity["revision"] is None
    assert identity["checkpoint_identity_basis"] == "checkpoint_config"
