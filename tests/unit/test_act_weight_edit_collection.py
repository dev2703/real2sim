"""Supervision follows applied teacher commands and causal chunk offsets."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from real2sim.geometry.so101_fk import JointMapping
from real2sim.policy.weight_edit_collection import (
    accepted_native_target,
    accepted_panda_target,
    collect_anchor_at,
    episode_pack,
    native_policy_state,
    saved_action_statistics,
    stack_records,
    validate_robots,
    validate_rollin_contract,
    validate_rollin_options,
    validate_seed_split,
)
from real2sim.sim.env import real_from_sim, sim_from_real


def test_pack_aligns_queries_to_executed_ticks_and_masks_rejections_and_terminal_tail():
    # A deliberately nonstandard chunk/feature geometry catches hardcoded ACT dimensions.
    commands = np.arange(18, dtype=np.float32).reshape(3, 6)
    mean, std = np.arange(6, dtype=np.float32), np.full(6, 2, np.float32)
    anchors = [{"features": np.full((4, 7), tick), "raw_state": np.zeros(6),
                "tick": tick, "episode_index": 40} for tick in (0, 2)]
    pack = episode_pack(anchors, commands, [True, False, True], chunk_size=4,
                        feature_dim=7, mean=mean, std=std)
    np.testing.assert_array_equal(pack["valid"], [[True, False, True, False], [True, False, False, False]])
    np.testing.assert_allclose(pack["target_normalized"][0, 0], (commands[0] - mean) / std)
    np.testing.assert_allclose(pack["target_normalized"][0, 2], (commands[2] - mean) / std)
    np.testing.assert_allclose(pack["target_normalized"][1, 0], (commands[2] - mean) / std)
    assert np.all(pack["target_normalized"][~pack["valid"]] == 0)
    np.testing.assert_array_equal(pack["tick"], [0, 2])
    np.testing.assert_array_equal(pack["episode_index"], [40, 40])
    assert pack["features"].shape == (2, 4, 7)


@pytest.mark.parametrize("invalid", ["features", "state", "tick", "command", "std"])
def test_invalid_supervision_fails_instead_of_silently_entering_training(invalid):
    anchor = {"features": np.zeros((3, 5)), "raw_state": np.zeros(6),
              "tick": 0, "episode_index": 1}
    command, std = np.zeros((1, 6)), np.ones(6)
    if invalid == "features":
        anchor["features"][0, 0] = np.nan
    elif invalid == "state":
        anchor["raw_state"] = np.zeros(7)
    elif invalid == "tick":
        anchor["tick"] = 1
    elif invalid == "command":
        command[0, 0] = np.inf
    else:
        std[0] = 0
    with pytest.raises(ValueError):
        episode_pack([anchor], command, [True], chunk_size=3, feature_dim=5,
                     mean=np.zeros(6), std=std)


def test_empty_pack_retains_checkpoint_dimensions():
    pack = stack_records([], chunk_size=12, feature_dim=32, action_dim=6)
    assert pack["features"].shape == (0, 12, 32)
    assert pack["target_normalized"].shape == (0, 12, 6)
    assert pack["valid"].dtype == np.bool_


def test_decoder_memory_pack_preserves_spatial_tokens_and_causal_query_mask():
    memory = np.arange(20, dtype=np.float32).reshape(4, 5)
    positions = memory / 10
    anchor = {"features": np.zeros((3, 5), np.float32), "raw_state": np.zeros(6),
              "encoder_memory": memory, "encoder_positions": positions,
              "tick": 1, "episode_index": 7}
    pack = episode_pack([anchor], np.zeros((2, 6)), [True, True],
                        chunk_size=3, feature_dim=5, mean=np.zeros(6), std=np.ones(6))
    np.testing.assert_array_equal(pack["encoder_memory"][0], memory)
    np.testing.assert_array_equal(pack["encoder_positions"][0], positions)
    np.testing.assert_array_equal(pack["valid"], [[True, False, False]])
    assert pack["features"].shape == (1, 3, 5)


@pytest.mark.parametrize("invalid", ["missing", "dimension", "shape", "nan"])
def test_decoder_memory_pack_rejects_inconsistent_or_nonfinite_tokens(invalid):
    anchor = {"features": np.zeros((3, 5)), "raw_state": np.zeros(6),
              "encoder_memory": np.zeros((4, 5)), "encoder_positions": np.zeros((4, 5)),
              "tick": 0, "episode_index": 7}
    if invalid == "missing":
        del anchor["encoder_positions"]
    elif invalid == "dimension":
        anchor["encoder_memory"] = np.zeros((4, 6))
    elif invalid == "shape":
        anchor["encoder_positions"] = np.zeros((3, 5))
    else:
        anchor["encoder_memory"][0, 0] = np.nan
    with pytest.raises(ValueError, match="[Ee]ncoder"):
        episode_pack([anchor], np.zeros((1, 6)), [True], chunk_size=3,
                     feature_dim=5, mean=np.zeros(6), std=np.ones(6))


@pytest.mark.parametrize("train,val", [([1], [1]), ([1, 1], [2]), ([], [2]), ([1], []), ([-1], [2])])
def test_episode_roles_reject_overlap_duplicates_and_missing_split(train, val):
    with pytest.raises(ValueError):
        validate_seed_split(train, val)


def test_episode_roles_accept_disjoint_whole_episodes():
    validate_seed_split([10090, 10091], [11090])


@pytest.mark.parametrize("robots", [[], ["panda", "panda"], ["unknown"]])
def test_collection_rejects_ambiguous_or_unsupported_robots(robots):
    with pytest.raises(ValueError, match="robots"):
        validate_robots(robots)


def test_panda_pack_keeps_robot_identity_separate_from_so101_same_seed():
    anchor = {"features": np.zeros((2, 3)), "raw_state": np.zeros(6),
              "tick": 0, "episode_index": 10}
    kwargs = dict(chunk_size=2, feature_dim=3, mean=np.zeros(6), std=np.ones(6))
    source = episode_pack([anchor], np.zeros((1, 6)), [True], robot="so101", **kwargs)
    panda = episode_pack([anchor], np.zeros((1, 6)), [True], robot="panda", **kwargs)
    assert source["episode_index"][0] == panda["episode_index"][0] == 10
    assert source["robot_id"][0] == 0 and panda["robot_id"][0] == 1


def test_panda_policy_input_uses_benchmark_endpoint_jaw_readback():
    source = SimpleNamespace(source_state=lambda env: (np.arange(6, dtype=np.float32), .001))
    env = SimpleNamespace(gripper_opening=lambda: .04, max_opening=.08)
    state = native_policy_state(SimpleNamespace(robot="panda", env=env, source=source))
    np.testing.assert_array_equal(state, [0, 1, 2, 3, 4, 45])


@pytest.mark.parametrize("rejected,representable", [(True, True), (False, False)])
def test_panda_rejections_mask_labels_and_preserve_causal_command_prior(rejected, representable):
    def unexpected_command(raw):
        pytest.fail("Rejected teacher command must not advance source history")

    source = SimpleNamespace(project_pose=lambda pose: SimpleNamespace(accepted=representable),
                             source_action=unexpected_command)
    shared = SimpleNamespace(env=SimpleNamespace(max_opening=.08), source=source,
                             guard=SimpleNamespace(forward=lambda q, **kw: np.eye(4)))
    result = SimpleNamespace(info={"control": {"applied": [0.] * 7 + [.04]}},
                             ik_rejected=rejected)
    target, valid = accepted_panda_target(shared, result)
    assert not valid
    np.testing.assert_array_equal(target, np.zeros(6))


def test_labels_use_applied_command_inverse_mapping_and_keep_rejection_mask():
    mapping = JointMapping(sign=(-1., 1., -1., 1., 1.), offset_deg=(10., 20., 30., 40., 50.),
                           gripper_sim_range_rad=(.1, 1.5))
    applied = np.array([.2, .3, .4, .5, .6, .8])
    result = SimpleNamespace(info={"control": {"applied_radians": applied,
                                               "requested_radians": applied + 1.}}, ik_rejected=False)
    target, valid = accepted_native_target(SimpleNamespace(mapping=mapping), result)
    np.testing.assert_allclose(target, real_from_sim(mapping, applied), rtol=1e-6)
    np.testing.assert_allclose(sim_from_real(mapping, target), applied, rtol=1e-6)
    assert valid
    result.ik_rejected = True
    _, valid = accepted_native_target(SimpleNamespace(mapping=mapping), result)
    assert not valid


def write_processor(directory):
    payload = {"steps": [{"registry_name": "unnormalizer_processor",
        "config": {"norm_map": {"ACTION": "MEAN_STD"},
                   "features": {"action": {"shape": [6]}}},
        "state_file": "actual_saved_statistics.safetensors"},
        {"registry_name": "device_processor", "config": {"device": "cpu"}}]}
    (directory / "policy_postprocessor.json").write_text(json.dumps(payload))
    save_file({"action.mean": torch.arange(6, dtype=torch.float32),
               "action.std": torch.full((6,), 3., dtype=torch.float32)},
              directory / "actual_saved_statistics.safetensors")
    return payload


def test_saved_statistics_follow_actual_processor_state_reference(tmp_path):
    write_processor(tmp_path)
    mean, std = saved_action_statistics(tmp_path, 6)
    np.testing.assert_array_equal(mean, np.arange(6))
    np.testing.assert_array_equal(std, np.full(6, 3.))


@pytest.mark.parametrize("invalid", ["mode", "shape", "extra_transform", "state_reference", "std"])
def test_incompatible_saved_postprocessor_is_rejected(tmp_path, invalid):
    payload = write_processor(tmp_path)
    step = payload["steps"][0]
    if invalid == "mode":
        step["config"]["norm_map"]["ACTION"] = "MIN_MAX"
    elif invalid == "shape":
        step["config"]["features"]["action"]["shape"] = [7]
    elif invalid == "extra_transform":
        payload["steps"].append({"registry_name": "unrecognized_action_transform"})
    elif invalid == "state_reference":
        step["state_file"] = "../outside_snapshot.safetensors"
    else:
        save_file({"action.mean": torch.zeros(6), "action.std": torch.zeros(6)},
                  tmp_path / step["state_file"])
    (tmp_path / "policy_postprocessor.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        saved_action_statistics(tmp_path, 6)


@pytest.mark.parametrize("path,steps", [(None, 50), ("edited", 0), ("edited", -1), (None, True)])
def test_prefix_requires_snapshot_and_positive_integer_steps_together(path, steps):
    with pytest.raises(ValueError):
        validate_rollin_options(path, steps)


def test_default_and_explicit_rollin_options_are_valid():
    validate_rollin_options(None, 0)
    validate_rollin_options("edited", 50)


def test_queries_start_at_handoff_and_do_not_learn_prefix_commands():
    start, stride = 3, 2
    query_ticks = [tick for tick in range(7) if collect_anchor_at(tick, stride, start)]
    assert query_ticks == [3, 5]
    commands = np.arange(42, dtype=np.float32).reshape(7, 6)
    # Deliberately distinctive prefix values would contaminate fitting if the
    # real tick indices were shifted when anchors began at handoff.
    commands[:start] = 10000.
    anchors = [{"features": np.zeros((3, 4)), "raw_state": np.zeros(6),
                "tick": tick, "episode_index": 10090} for tick in query_ticks]
    pack = episode_pack(anchors, commands, [False] * start + [True] * 4,
                        chunk_size=3, feature_dim=4, mean=np.zeros(6), std=np.ones(6))
    np.testing.assert_array_equal(pack["tick"], [3, 5])
    np.testing.assert_array_equal(pack["target_normalized"][0], commands[3:6])
    np.testing.assert_array_equal(pack["target_normalized"][1, :2], commands[5:7])
    assert not np.any(pack["target_normalized"] == 10000.)
    assert not pack["valid"][1, 2]


def rollin_pair(directory):
    policies = []
    for name in ("base", "rollin"):
        path = directory / name
        path.mkdir()
        config = {"input_features": {"observation.state": {"shape": [6]}},
                  "output_features": {"action": {"shape": [6]}},
                  "normalization_mapping": {"ACTION": "MEAN_STD"}}
        (path / "config.json").write_text(json.dumps(config))
        (path / "policy_preprocessor.json").write_text('{"saved": "identical"}')
        (path / "policy_postprocessor.json").write_text('{"saved": "identical"}')
        (path / "policy_postprocessor_state.safetensors").write_bytes(b"identical-saved-state")
        # Model weights are allowed to differ; the decoder feature base remains
        # separately pinned to its original checkpoint identity.
        (path / "model.safetensors").write_bytes(name.encode())
        policies.append(SimpleNamespace(path=path, scene_contract={"camera_keys": ["wrist", "iphone"]}))
    return policies


def test_rollin_allows_changed_weights_through_identical_native_interface(tmp_path):
    base, rollin = rollin_pair(tmp_path)
    validate_rollin_contract(base, rollin)


@pytest.mark.parametrize("invalid", ["scene", "units", "processor", "processor_missing"])
def test_rollin_rejects_changed_scene_units_or_saved_processing(tmp_path, invalid):
    base, rollin = rollin_pair(tmp_path)
    if invalid == "scene":
        rollin.scene_contract["camera_keys"] = ["different"]
    elif invalid == "units":
        config_path = rollin.path / "config.json"
        config = json.loads(config_path.read_text())
        config["normalization_mapping"]["ACTION"] = "MIN_MAX"
        config_path.write_text(json.dumps(config))
    elif invalid == "processor":
        (rollin.path / "policy_postprocessor_state.safetensors").write_bytes(b"changed-stats")
    else:
        (rollin.path / "policy_preprocessor.json").unlink()
    with pytest.raises(ValueError):
        validate_rollin_contract(base, rollin)
