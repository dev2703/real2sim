"""Execution-reference ablations must be explicit, recorded and refused when inapplicable."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from real2sim.evaluation import run
from real2sim.policy.adapters import CartesianActionAdapter, ContractError, JointActionAdapter
from tests.unit.test_oracle_schema import FakeNormalizer, cartesian_contract, joint_contract


@pytest.fixture
def fake_checkpoint(tmp_path, monkeypatch):
    from real2sim.policy import act

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"unchanged frozen checkpoint")
    env = SimpleNamespace(
        mapping=SimpleNamespace(), model=object(), config={"robot": "so101"}, control_hz=10,
        data_config={"image": {"height": 192, "width": 256}}, max_episode_steps=200,
        task={"episode": {"success_hold_s": .2}, "ball": {"radius": .0335}},
        table_top=.75, initial_qpos=np.zeros(6), close=lambda: None)
    contract = cartesian_contract()
    contract.update(policy_hz=10., letterbox={"src_w": 1280, "src_h": 720,
                                             "dst_w": 256, "dst_h": 192})
    policy = SimpleNamespace(config=SimpleNamespace(chunk_size=20, n_action_steps=2))
    loads = []

    def load(path, device):
        loads.append((path, device))
        return policy, FakeNormalizer(len(contract["state_names"])), contract

    monkeypatch.setattr(act, "load_checkpoint", load)
    monkeypatch.setattr(run, "NativeKinematics", lambda model, config: object())
    return checkpoint, env, contract, loads


def eval_config(reference=None):
    config = {"cartesian": {"max_step_m": .03, "max_step_rad": .3},
              "phase": {"approach_distance_m": .03},
              "video": {"fps": 10, "upscale": 2, "codec": "mp4v"}}
    if reference is not None:
        config["cartesian"]["target_reference"] = reference
    return config


@pytest.mark.parametrize("reference", [None, "current_pose", "accepted_target"])
def test_so101_cartesian_config_reaches_adapter_and_preserves_checkpoint(fake_checkpoint, reference):
    checkpoint, env, contract, _ = fake_checkpoint
    original = copy.deepcopy(contract)
    config = eval_config(reference)
    original_config = copy.deepcopy(config)
    executor, loaded_contract, metadata = run.build_learned_controller(env, checkpoint, config, "cpu")
    expected = reference or "current_pose"
    assert isinstance(executor.action_adapter, CartesianActionAdapter)
    assert executor.action_adapter.target_reference == expected
    assert metadata["target_reference"] == expected
    assert metadata["execution_interface_ablation"] == (expected == "accepted_target")
    assert metadata["checkpoint_contract_modified"] is False
    assert loaded_contract == original == contract
    assert config == original_config
    assert (checkpoint / "model.safetensors").read_bytes() == b"unchanged frozen checkpoint"


def test_joint_checkpoint_refuses_persistent_target_but_keeps_default(fake_checkpoint):
    checkpoint, env, contract, _ = fake_checkpoint
    contract.update(joint_contract())
    with pytest.raises(ContractError, match="requires a Cartesian checkpoint"):
        run.build_learned_controller(env, checkpoint, eval_config("accepted_target"), "cpu")
    executor, _, metadata = run.build_learned_controller(env, checkpoint, eval_config(), "cpu")
    assert isinstance(executor.action_adapter, JointActionAdapter)
    assert metadata["target_reference"] is None
    assert metadata["execution_interface_ablation"] is False


def test_panda_builder_refuses_persistent_mode_before_checkpoint_loading(fake_checkpoint):
    checkpoint, env, _, loads = fake_checkpoint
    with pytest.raises(ContractError, match="cannot be applied to Panda"):
        run.build_learned_controller(env, checkpoint, eval_config("accepted_target"), "cpu",
                                     robot="panda")
    assert loads == []


@pytest.mark.parametrize("robot,controller", [("so101", "hold"), ("panda", "learned"),
                                             ("panda", "hold")])
def test_run_trials_rejects_inapplicable_mode_before_env_creation(tmp_path, monkeypatch,
                                                                 robot, controller):
    config = tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump(eval_config("accepted_target")))

    def forbid_env(*args, **kwargs):
        raise AssertionError("inapplicable ablation must fail before simulator construction")

    monkeypatch.setattr(run, "make_env", forbid_env)
    with pytest.raises(ContractError, match="requires a learned SO-101 Cartesian controller"):
        run.run_trials(robot=robot, checkpoint=None, config=Path("absent.yaml"), eval_config=config,
                       n_trials=1, seed_start=0, out=tmp_path / "out", video=False,
                       controller=controller, device="cpu")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("invalid", ["typo", "", 123, None])
def test_unknown_or_null_reference_fails_loudly(fake_checkpoint, invalid):
    checkpoint, env, _, loads = fake_checkpoint
    config = eval_config()
    config["cartesian"]["target_reference"] = invalid
    with pytest.raises(ValueError, match="cartesian.target_reference"):
        run.build_learned_controller(env, checkpoint, config, "cpu")
    assert loads == []


@pytest.mark.parametrize("reference,expected_id", [
    (None, "E2_cart_ACT_so101"),
    ("accepted_target", "DIAGNOSTIC_E2_cart_ACT_so101_persistent_target"),
])
def test_summary_marks_effective_execution_interface_separately_from_contract(
        fake_checkpoint, tmp_path, monkeypatch, reference, expected_id):
    checkpoint, env, contract, _ = fake_checkpoint
    config = tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump(eval_config(reference)))
    monkeypatch.setattr(run, "make_env", lambda *args: env)
    captions = []

    def no_physics(*args, **kwargs):
        captions.append(kwargs["caption_prefix"])

    monkeypatch.setattr(run, "run_trial", no_physics)
    summary = run.run_trials(robot="so101", checkpoint=checkpoint,
                             config=Path("configs/sim_so101.yaml"), eval_config=config,
                             n_trials=1, seed_start=0, out=tmp_path / "out", video=False,
                             controller="learned", device="cpu")
    saved = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert saved["experiment_id"] == summary["experiment_id"] == expected_id
    assert saved["execution_interface"]["target_reference"] == (reference or "current_pose")
    assert saved["execution_interface"]["ablation"] == (reference == "accepted_target")
    assert saved["execution_interface"]["checkpoint_contract_modified"] is False
    assert saved["contract"] == contract
    assert ("execution ablation" in captions[0]) == (reference == "accepted_target")


def test_hold_metadata_does_not_claim_cartesian_execution(fake_checkpoint, tmp_path, monkeypatch):
    _, env, _, loads = fake_checkpoint
    config = tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump(eval_config("current_pose")))
    monkeypatch.setattr(run, "make_env", lambda *args: env)
    monkeypatch.setattr(run, "run_trial", lambda *args, **kwargs: None)
    summary = run.run_trials(robot="so101", checkpoint=None,
                             config=Path("configs/sim_so101.yaml"), eval_config=config,
                             n_trials=1, seed_start=0, out=tmp_path / "out", video=False,
                             controller="hold", device="cpu")
    assert summary["execution_interface"]["target_reference"] is None
    assert not summary["execution_interface"]["ablation"]
    assert loads == []
