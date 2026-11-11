"""Cached decoder replay remains native and exports only the intended tensors."""

import json

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from real2sim.artifacts import sha256_file
from real2sim.policy.decoder_weight_edit import DecoderTail, fit_decoder_edit, fit_from_cache


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def native_parts(activation="relu", pre_norm=False):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTDecoder

    config = ACTConfig(dim_model=8, n_heads=2, dim_feedforward=16,
                       n_decoder_layers=2, dropout=.2, pre_norm=pre_norm,
                       feedforward_activation=activation)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        decoder, head = ACTDecoder(config).eval(), nn.Linear(8, 2).eval()
    state = {**{f"model.decoder.{key}": value.clone() for key, value in decoder.state_dict().items()},
             **{f"model.action_head.{key}": value.clone() for key, value in head.state_dict().items()},
             "model.encoder.frozen_test_tensor": torch.arange(5, dtype=torch.float32)}
    saved = {"type": "act", "dim_model": 8, "dim_feedforward": 16,
             "n_decoder_layers": 2, "pre_norm": pre_norm, "feedforward_activation": activation,
             "chunk_size": 3}
    return saved, decoder, head, state


@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_replay_equals_actual_native_decoder_subgraph(activation):
    config, decoder, head, state = native_parts(activation)
    captured = []
    handle = decoder.layers[-1].linear1.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone()))
    generator = torch.Generator().manual_seed(15)
    features = torch.randn(3, 2, 8, generator=generator)
    memory = torch.randn(7, 2, 8, generator=generator)
    with torch.inference_mode():
        expected = head(decoder(features, memory))
        handle.remove()
        tail = DecoderTail(config, state)
        actual = tail(captured[0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not tail.norm3.weight.requires_grad
    assert not tail.decoder_norm.weight.requires_grad
    assert len(tail.trainable_checkpoint_keys) == 6


def synthetic_problem():
    config, _, _, state = native_parts()
    teacher = DecoderTail(config, state)
    with torch.no_grad():
        teacher.linear1.weight[0] += .6
        teacher.linear2.weight[:, 0] += .5
        teacher.action_head.weight *= 1.6
        teacher.action_head.bias += .3
    generator = torch.Generator().manual_seed(17)
    x, vx = torch.randn(128, 8, generator=generator), torch.randn(48, 8, generator=generator)
    with torch.inference_mode():
        y, vy = teacher(x), teacher(vx)
    return config, state, x.numpy(), y.numpy(), vx.numpy(), vy.numpy()


@pytest.mark.parametrize("loss", ["mse", "l1"])
def test_decoder_fit_learns_nonlinear_target_without_modifying_input_state(loss):
    config, state, x, y, vx, vy = synthetic_problem()
    original = {key: value.clone() for key, value in state.items()}
    edits, report = fit_decoder_edit(state, config, x, y, vx, vy,
        steps=250, batch_size=64, lr=.005, l2=.01, validation_every=10, patience=25, loss=loss)
    assert report["accepted"]
    assert report["edited_validation"]["mae_normalized"] < report["baseline_validation"]["mae_normalized"] * .4
    assert report["task_completion_established"] is False
    assert report["runtime_teacher"] is False
    assert report["loss"] == loss
    assert any("linear1" in key and not torch.equal(tensor, state[key]) for key, tensor in edits.items())
    assert any("linear2" in key and not torch.equal(tensor, state[key]) for key, tensor in edits.items())
    for key, value in state.items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_minority_robot_regression_keeps_exact_original_tensors():
    config, _, _, state = native_parts()
    state["model.action_head.weight"].zero_()
    state["model.action_head.bias"].zero_()
    x = np.ones((11, 8), np.float32)
    y, vy = np.ones((11, 2), np.float32), np.ones((11, 2), np.float32)
    vy[-1] = 0.
    groups = np.r_[np.zeros(10, np.int64), np.ones(1, np.int64)]
    edits, report = fit_decoder_edit(state, config, x, y, x, vy,
        train_groups=groups, validation_groups=groups, steps=20,
        validation_every=5, patience=4, lr=.01)
    assert not report["accepted"]
    assert any(not row["no_robot_regression"] for row in report["history"][1:])
    for key, value in edits.items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


@pytest.mark.parametrize("options", [
    {"steps": 0}, {"batch_size": True}, {"lr": float("nan")}, {"l2": -1},
    {"patience": 0}, {"seed": -1}, {"train_groups": [0, 1]},
    {"loss": "huber"},
])
def test_invalid_fit_options_rejected(options):
    config, state, x, y, vx, vy = synthetic_problem()
    with pytest.raises(ValueError):
        fit_decoder_edit(state, config, x, y, vx, vy, **options)


def test_pre_norm_features_cannot_replay_residual_without_separate_cache():
    config, _, _, state = native_parts(pre_norm=True)
    with pytest.raises(ValueError, match="pre_norm=False"):
        DecoderTail(config, state)


@pytest.fixture
def decoder_cache(tmp_path):
    config, state, x, y, vx, vy = synthetic_problem()
    snapshot, data = tmp_path / "snapshot", tmp_path / "features"
    snapshot.mkdir()
    data.mkdir()
    (snapshot / "config.json").write_text(json.dumps(config))
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (snapshot / name).write_text(json.dumps({"saved": name}))
    save_file(state, snapshot / "model.safetensors", metadata={"format": "pt", "custom": "retained"})
    save_file({"action.mean": torch.ones(2)}, snapshot / "saved_processor.safetensors")
    for partition, features, targets, episode in (("train", x[:126], y[:126], 10),
                                                  ("val", vx, vy, 20)):
        rows = len(features) // 3
        np.savez(data / f"{partition}.npz", features=features.reshape(rows, 3, 8),
            target_normalized=targets.reshape(rows, 3, 2), valid=np.ones((rows, 3), bool),
            robot_id=np.arange(rows) % 2, episode_index=np.full(rows, episode, np.int64),
            tick=np.arange(rows))
    checksums = {str(path.resolve()): sha256_file(path) for path in snapshot.iterdir()}
    report = {"status": "complete", "feature_layer": "decoder_ffn_input",
        "checkpoint_unchanged": True, "runtime_source_changes": [],
        "weights_sha256": sha256_file(snapshot / "model.safetensors"),
        "source_sha256_before": checksums, "source_sha256_after": checksums,
        "feature_files_sha256": {name: sha256_file(data / name) for name in ("train.npz", "val.npz")}}
    (data / "report.json").write_text(json.dumps(report))
    return snapshot, data


def test_native_export_preserves_processors_and_all_tensors_outside_whitelist(decoder_cache, tmp_path):
    snapshot, data = decoder_cache
    before = {path.name: sha256_file(path) for path in snapshot.iterdir()}
    out = tmp_path / "edited"
    report = fit_from_cache(data, out, snapshot, prefix=3, steps=150,
                           lr=.005, l2=.01, validation_every=10, patience=15)
    assert report["accepted"]
    assert report["robot_ids"] == [0, 1]
    assert report["base_checkpoint_unchanged"]
    assert before == {path.name: sha256_file(path) for path in snapshot.iterdir()}
    for name, checksum in before.items():
        if name != "model.safetensors":
            assert sha256_file(out / name) == checksum
    original, edited = load_file(snapshot / "model.safetensors"), load_file(out / "model.safetensors")
    changed = [key for key in original if not torch.equal(original[key], edited[key])]
    assert set(changed) == set(report["changed_tensor_names"])
    assert set(changed) == set(report["trainable_tensor_names"])
    with safe_open(out / "model.safetensors", framework="pt") as handle:
        assert handle.metadata() == {"format": "pt", "custom": "retained"}
    assert (out / "decoder_weight_edit_report.json").is_file()


def test_action_head_features_are_rejected_by_decoder_fitter(decoder_cache, tmp_path):
    snapshot, data = decoder_cache
    report = json.loads((data / "report.json").read_text())
    report["feature_layer"] = "action_head"
    (data / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="decoder_ffn_input"):
        fit_from_cache(data, tmp_path / "edited", snapshot, prefix=3)


def test_overlap_rejected_before_export(decoder_cache, tmp_path):
    snapshot, data = decoder_cache
    with np.load(data / "val.npz") as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["episode_index"][:] = 10
    np.savez(data / "val.npz", **arrays)
    report = json.loads((data / "report.json").read_text())
    report["feature_files_sha256"]["val.npz"] = sha256_file(data / "val.npz")
    (data / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="overlap"):
        fit_from_cache(data, tmp_path / "edited", snapshot, prefix=3)
