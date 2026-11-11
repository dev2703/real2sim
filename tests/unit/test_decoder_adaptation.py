"""Native attention adaptation replays cached image tokens without a teacher."""

import copy
import json

import numpy as np
import pytest
import torch
from lerobot.policies.act.modeling_act import ACTDecoder
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from real2sim.artifacts import sha256_file
from real2sim.policy.decoder_adaptation import (
    NativeDecoder,
    fit_decoder,
    fit_from_cache,
    validate_pack,
)


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def native_parts(activation="relu"):
    from lerobot.policies.act.configuration_act import ACTConfig

    config = ACTConfig(dim_model=8, n_heads=2, dim_feedforward=16,
                       n_decoder_layers=1, dropout=.2, chunk_size=3, n_action_steps=3,
                       feedforward_activation=activation)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        decoder, head, query = ACTDecoder(config).eval(), nn.Linear(8, 2).eval(), nn.Embedding(3, 8)
    state = {**{f"model.decoder.{key}": value.clone() for key, value in decoder.state_dict().items()},
             **{f"model.action_head.{key}": value.clone() for key, value in head.state_dict().items()},
             "model.decoder_pos_embed.weight": query.weight.detach().clone(),
             "model.encoder.frozen_test_tensor": torch.arange(5, dtype=torch.float32)}
    saved = {"type": "act", "dim_model": 8, "dim_feedforward": 16, "n_heads": 2,
             "n_decoder_layers": 1, "dropout": .2, "pre_norm": False,
             "feedforward_activation": activation, "chunk_size": 3}
    return saved, decoder, head, query, state


def pack(model, rows, seed, episode):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 3, 8)).astype(np.float32)
    memory = rng.normal(size=(rows, 7, 8)).astype(np.float32)
    positions = rng.normal(size=(rows, 7, 8)).astype(np.float32)
    with torch.inference_mode():
        targets = model(torch.from_numpy(x), torch.from_numpy(memory), torch.from_numpy(positions)).numpy()
    return dict(features=x, encoder_memory=memory, encoder_positions=positions,
                target_normalized=targets, valid=np.ones((rows, 3), bool),
                robot_id=np.arange(rows, dtype=np.int64) % 2,
                episode_index=np.full(rows, episode, np.int64), tick=np.arange(rows))


@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_cached_memory_replays_native_decoder_bit_exactly(activation):
    config, decoder, head, query, state = native_parts(activation)
    model = NativeDecoder(config, state)
    data = pack(model, 2, 4, 10)
    x, memory, positions = (torch.from_numpy(data[key]).transpose(0, 1)
                            for key in ("features", "encoder_memory", "encoder_positions"))
    with torch.inference_mode():
        expected = head(decoder(x, memory, decoder_pos_embed=query.weight.unsqueeze(1),
                                encoder_pos_embed=positions)).transpose(0, 1)
    np.testing.assert_array_equal(data["target_normalized"], expected.numpy())
    assert any("multihead_attn" in key for key in model.edited_tensors())
    assert "model.decoder_pos_embed.weight" in model.edited_tensors()


def problem():
    config, _, _, _, state = native_parts()
    teacher = NativeDecoder(config, state)
    with torch.no_grad():
        teacher.decoder.layers[0].multihead_attn.in_proj_weight *= 1.15
        teacher.action_head.weight *= 1.3
        teacher.action_head.bias += .25
    return config, state, pack(teacher, 24, 10, 100), pack(teacher, 12, 11, 200)


def test_attention_adaptation_learns_without_mutating_original_state():
    config, state, train, val = problem()
    before = {key: value.clone() for key, value in state.items()}
    edits, report = fit_decoder(state, config, train, val, steps=200, batch_size=8,
                               lr=.003, l2=.01, validation_every=10, patience=20)
    assert report["accepted"]
    assert report["edited_validation"]["mae_normalized"] < report["baseline_validation"]["mae_normalized"] * .5
    assert report["task_completion_established"] is False
    assert report["runtime_teacher"] is False
    assert any("multihead_attn" in key and not torch.equal(value, state[key]) for key, value in edits.items())
    for key in state:
        torch.testing.assert_close(state[key], before[key], rtol=0, atol=0)


def test_padded_queries_do_not_affect_gradient_updates_or_metrics():
    config, state, train, val = problem()
    for data in (train, val):
        data["valid"][:, -1] = False
    changed_train, changed_val = copy.deepcopy(train), copy.deepcopy(val)
    changed_train["target_normalized"][:, -1] = 1e30
    changed_val["target_normalized"][:, -1] = -1e30
    options = dict(steps=15, batch_size=8, lr=.002, validation_every=5, patience=10, loss="mse")
    original, original_report = fit_decoder(state, config, train, val, **options)
    masked, masked_report = fit_decoder(state, config, changed_train, changed_val, **options)
    for key in original:
        torch.testing.assert_close(original[key], masked[key], rtol=0, atol=0)
    assert original_report["edited_validation"] == masked_report["edited_validation"]


def test_baseline_candidate_blocks_minority_robot_regression():
    config, _, _, _, state = native_parts()
    state["model.action_head.weight"].zero_()
    state["model.action_head.bias"].zero_()
    data = pack(NativeDecoder(config, state), 4, 8, 100)
    data["features"][:] = 1
    data["encoder_memory"][:] = 1
    data["encoder_positions"][:] = 1
    data["target_normalized"][:] = 1
    val = copy.deepcopy(data)
    val["target_normalized"][val["robot_id"] == 1] = 0
    edits, report = fit_decoder(state, config, data, val, steps=20, batch_size=16,
                               lr=.01, validation_every=5, patience=4)
    assert not report["accepted"]
    for key, tensor in edits.items():
        torch.testing.assert_close(tensor, state[key], rtol=0, atol=0)


@pytest.mark.parametrize("option", [
    {"steps": 0}, {"batch_size": True}, {"lr": float("nan")}, {"l2": -1},
    {"patience": 0}, {"seed": -1}, {"loss": "unknown"},
])
def test_invalid_fit_configuration_rejected(option):
    config, state, train, val = problem()
    with pytest.raises(ValueError):
        fit_decoder(state, config, train, val, **option)


def test_zero_native_decoder_inputs_are_valid_and_terminal_anchors_are_filtered():
    config, state, train, _ = problem()
    train["features"][:] = 0
    train["valid"][0] = False
    checked = validate_pack(train, prefix=2)
    assert len(checked["features"]) == len(train["features"]) - 1
    assert not checked["valid"][:, 2].any()
    assert NativeDecoder(config, state)(*(torch.from_numpy(checked[key]) for key in
        ("features", "encoder_memory", "encoder_positions"))).shape == (23, 3, 2)


@pytest.fixture
def cache(tmp_path):
    config, state, train, val = problem()
    snapshot, data = tmp_path / "snapshot", tmp_path / "features"
    snapshot.mkdir()
    data.mkdir()
    (snapshot / "config.json").write_text(json.dumps(config))
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (snapshot / name).write_text(json.dumps({"saved": name}))
    save_file(state, snapshot / "model.safetensors", metadata={"format": "pt", "custom": "retained"})
    save_file({"action.mean": torch.ones(2)}, snapshot / "saved_processor.safetensors")
    np.savez_compressed(data / "train.npz", **train)
    np.savez_compressed(data / "val.npz", **val)
    sources = {str(path.resolve()): sha256_file(path) for path in snapshot.iterdir()}
    report = dict(status="complete", checkpoint_unchanged=True, runtime_source_changes=[],
                  weights_sha256=sha256_file(snapshot / "model.safetensors"),
                  feature_layer="decoder_memory", source_sha256_before=sources,
                  source_sha256_after=sources, domain_randomization={"profile": "test"},
                  feature_files_sha256={name: sha256_file(data / name) for name in ("train.npz", "val.npz")})
    (data / "report.json").write_text(json.dumps(report))
    return data, snapshot


def test_native_export_preserves_encoder_sidecars_and_safetensors_metadata(cache, tmp_path):
    data, snapshot = cache
    out = tmp_path / "edited"
    before = {path.name: sha256_file(path) for path in snapshot.iterdir()}
    report = fit_from_cache(data, out, snapshot, prefix=3, steps=40, lr=.003,
                           batch_size=8, validation_every=10, patience=10)
    assert report["accepted"]
    assert report["training_seeds"] == [100]
    assert report["validation_seeds"] == [200]
    assert before == {path.name: sha256_file(path) for path in snapshot.iterdir()}
    base, edited = load_file(snapshot / "model.safetensors"), load_file(out / "model.safetensors")
    torch.testing.assert_close(base["model.encoder.frozen_test_tensor"], edited["model.encoder.frozen_test_tensor"], rtol=0, atol=0)
    changed = [key for key in base if not torch.equal(base[key], edited[key])]
    assert set(changed) == set(report["changed_tensor_names"])
    assert all(key.startswith(("model.decoder.", "model.decoder_pos_embed.", "model.action_head.")) for key in changed)
    for name, digest in before.items():
        if name != "model.safetensors":
            assert sha256_file(out / name) == digest
    with safe_open(out / "model.safetensors", framework="pt") as handle:
        assert handle.metadata() == {"format": "pt", "custom": "retained"}


def update_hashes(data):
    report = json.loads((data / "report.json").read_text())
    report["feature_files_sha256"] = {name: sha256_file(data / name) for name in ("train.npz", "val.npz")}
    (data / "report.json").write_text(json.dumps(report))


def test_train_validation_overlap_rejected_before_export(cache, tmp_path):
    data, snapshot = cache
    with np.load(data / "val.npz") as payload:
        pack = {key: payload[key] for key in payload.files}
    pack["episode_index"][:] = 100
    np.savez_compressed(data / "val.npz", **pack)
    update_hashes(data)
    with pytest.raises(ValueError, match="identities overlap"):
        fit_from_cache(data, tmp_path / "edited", snapshot, prefix=3)


def test_modified_memory_cache_rejected(cache, tmp_path):
    data, snapshot = cache
    (data / "train.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fit_from_cache(data, tmp_path / "edited", snapshot, prefix=3)


def test_mixed_domain_randomization_profiles_rejected(cache, tmp_path):
    import shutil

    data, snapshot = cache
    second = tmp_path / "second"
    shutil.copytree(data, second)
    report = json.loads((second / "report.json").read_text())
    report["domain_randomization"] = {"profile": "different"}
    (second / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="domain randomization profile"):
        fit_from_cache([data, second], tmp_path / "edited", snapshot, prefix=3)


def test_applied_and_planned_goal_supervision_cannot_be_pooled(cache, tmp_path):
    import shutil

    data, snapshot = cache
    second = tmp_path / "second"
    shutil.copytree(data, second)
    report = json.loads((second / "report.json").read_text())
    report["target_kind"] = "planned_native_goal"
    (second / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="target_kind supervision"):
        fit_from_cache([data, second], tmp_path / "edited", snapshot, prefix=3)


def test_goal_supervision_semantics_reach_native_export_report(cache, tmp_path):
    data, snapshot = cache
    provenance = json.loads((data / "report.json").read_text())
    provenance.update(target_kind="planned_native_goal", target_source="Training-only native phase goals")
    (data / "report.json").write_text(json.dumps(provenance))
    report = fit_from_cache(data, tmp_path / "edited", snapshot, prefix=3, steps=10,
                           validation_every=5, patience=2, lr=.003)
    assert report["target_kind"] == "planned_native_goal"
    assert report["supervision_source"] == ["Training-only native phase goals"]


def test_progress_logging_is_explicit_and_reports_validation(capsys):
    config, state, train, val = problem()
    fit_decoder(state, config, train, val, steps=1, validation_every=1, progress=True)
    row = json.loads(capsys.readouterr().out)
    assert row["step"] == 1
    assert set(row["per_robot_validation_mae"]) == {"0", "1"}
    assert np.isfinite(row["validation_mae"])
