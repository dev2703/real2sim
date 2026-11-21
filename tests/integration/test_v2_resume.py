"""CPU training continuation parity and two-view checkpoint reload, no downloads."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from real2sim.data.prepare import compute_stats
from real2sim.policy.act import build_act_config, load_checkpoint
from real2sim.policy.paired import file_sha256
from real2sim.policy.train import train, validate_exact_resume_model_config


@pytest.fixture
def tiny_data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    rng = np.random.default_rng(23)
    n, per_episode = 24, 6
    state = rng.normal(0, .2, (n, 13)).astype(np.float32)
    state[:, -1] = .5
    action = rng.normal(0, .002, (n, 7)).astype(np.float32)
    action[:, -1] = .4
    ep = np.repeat(np.arange(4), per_episode)
    frame = np.tile(np.arange(per_episode), 4)
    extrapolated = frame == per_episode - 1
    action[extrapolated] = 0
    arrays = dict(state=state, canonical_state=state.copy(), action=action, episode_index=ep,
                  frame_index=frame, row_index=np.arange(n), ep_start=ep * per_episode,
                  ep_len=np.full(n, per_episode), label_is_extrapolated=extrapolated)
    np.savez(root / "arrays.npz", **arrays)
    images = rng.integers(0, 256, (n, 28, 42, 3), dtype=np.uint8)
    np.save(root / "images_iphone.npy", images)
    stats = {"observation.state": compute_stats(state[ep < 2]),
             "action": compute_stats(action[(ep < 2) & ~extrapolated]),
             "image_normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]},
             "train_episodes": [0, 1], "computed_on": "inner training episodes only"}
    (root / "stats.json").write_text(json.dumps(stats))
    manifest = {
        "prepare_version": "observation_variants_v1", "camera_keys": ["observation.images.iphone"],
        "image_shape_hwc": [28, 42, 3], "image_files": {"observation.images.iphone": "images_iphone.npy"},
        "state_names": [str(i) for i in range(12)] + ["gripper_obs"],
        "action_names": ["dx", "dy", "dz", "d_rx", "d_ry", "d_rz", "gripper_target"],
        "split": {"train": [0, 1], "val": [2], "historical_val": [3], "seed": 42, "sha256": "split"},
        "source_split": {"train": [0, 1, 2], "val": [3]},
        "letterbox": {}, "source": {}, "policy_hz": 10, "label_semantics": "test",
        "observation_variant": {"kind": "canonical", "history": 1},
        "experiment_split": {"purpose": "smoke"},
        "files_sha256": {name: file_sha256(root / name) for name in
                         ("arrays.npz", "stats.json", "images_iphone.npy")},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def tiny_act_config(trajectory_weight=0.):
    return {
        "name": "resume_fixture",
        "reconstruction_loss": {"trajectory_weight": trajectory_weight},
        "model": dict(
            chunk_size=3,
            n_action_steps=2,
            vision_backbone="resnet18",
            pretrained_backbone_weights=None,
            freeze_backbone=True,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            use_vae=True,
            latent_dim=4,
            n_vae_encoder_layers=1,
            dropout=0.1,
            kl_weight=1.0,
            optimizer_lr=1e-4,
            optimizer_lr_backbone=1e-5,
            optimizer_weight_decay=0.05,
        ),
        "train": dict(
            seed=9,
            batch_size=2,
            grad_accum=1,
            steps=3,
            schedule_steps=3,
            grad_clip_norm=1.0,
            log_every=1,
            val_every=1,
            save_every=1,
            val_max_batches=1,
            num_workers=0,
            device="cpu",
            exact_resume=True,
        ),
        "scheduler": dict(warmup_steps=1, start_factor=0.001, min_lr=1e-7),
    }

@pytest.mark.integration
@pytest.mark.parametrize("trajectory_weight", [0., .1])
def test_two_view_act_resume_matches_uninterrupted_training(tiny_data, tmp_path, trajectory_weight):
    data, manifest = tiny_data
    images = np.load(data / "images_iphone.npy")
    np.save(data / "images_top.npy", images[:, ::-1].copy())
    manifest["camera_keys"].append("observation.images.top")
    manifest["image_files"]["observation.images.top"] = "images_top.npy"
    manifest["files_sha256"]["images_top.npy"] = file_sha256(data / "images_top.npy")
    manifest["action_representation"] = "cartesian_tool"
    (data / "manifest.json").write_text(json.dumps(manifest))
    cfg = tiny_act_config(trajectory_weight)
    torch.set_num_threads(1)
    train(cfg, data, tmp_path / "full")
    cfg["train"]["steps"] = 1
    train(cfg, data, tmp_path / "split")
    cfg["train"]["steps"] = 3
    train(cfg, data, tmp_path / "split", resume=tmp_path / "split/checkpoints/000001")
    full = load_file(tmp_path / "full/checkpoints/000003/model.safetensors")
    resumed = load_file(tmp_path / "split/checkpoints/000003/model.safetensors")
    assert full.keys() == resumed.keys()
    for k, v in full.items():
        torch.testing.assert_close(v, resumed[k], atol=0, rtol=0, msg=k)
    policy, _, contract = load_checkpoint(tmp_path / "split/checkpoints/000003")
    assert contract["camera_keys"] == ["observation.images.iphone", "observation.images.top"]
    batch = {
        "observation.state": torch.zeros(1, 13),
        **{k: torch.zeros(1, 3, 28, 42) for k in contract["camera_keys"]},
    }
    assert policy.predict_action_chunk(batch).shape == (1, 3, 7)


def run_fingerprints(out):
    return {str(path.relative_to(out)): file_sha256(path) for path in out.rglob("*")
            if path.is_file()}


@pytest.fixture
def trained_run(tiny_data, tmp_path):
    data, _ = tiny_data
    cfg = tiny_act_config()
    cfg["train"]["steps"] = 2
    out = tmp_path / "run"
    torch.set_num_threads(1)
    train(cfg, data, out)
    return cfg, data, out


@pytest.mark.integration
def test_train_requires_resume_to_reuse_a_run(trained_run):
    cfg, data, out = trained_run
    before = run_fingerprints(out)
    cfg["train"]["seed"] += 1
    with pytest.raises(FileExistsError, match="resume"):
        train(cfg, data, out)
    assert run_fingerprints(out) == before


@pytest.mark.integration
@pytest.mark.parametrize("setting", ["grad_accum", "batch_size"])
def test_rejected_exact_resume_preserves_run_artifacts(trained_run, setting):
    cfg, data, out = trained_run
    before = run_fingerprints(out)
    cfg["train"]["steps"] = 3
    cfg["train"][setting] += 1
    with pytest.raises(ValueError, match="accumulation|physical batch"):
        train(cfg, data, out, resume=out / "checkpoints/000002")
    assert run_fingerprints(out) == before


@pytest.mark.integration
def test_resume_from_older_checkpoint_preserves_later_checkpoints(trained_run, tmp_path):
    cfg, data, out = trained_run
    before = run_fingerprints(out)
    cfg["train"]["steps"] = 3
    checkpoint = out / "checkpoints/000001"
    with pytest.raises(FileExistsError, match="later checkpoint"):
        train(cfg, data, out, resume=checkpoint)
    assert run_fingerprints(out) == before

    destination = tmp_path / "continued"
    result = train(cfg, data, destination, resume=checkpoint)
    assert result["steps"] == 3
    assert (destination / "checkpoints/000003/model.safetensors").is_file()
    assert run_fingerprints(out) == before


@pytest.mark.integration
@pytest.mark.parametrize("setting,value", [
    ("dropout", .75), ("n_heads", 8), ("kl_weight", 3.),
    ("n_action_steps", 1), ("optimizer_lr", 2e-4),
])
def test_exact_resume_rejects_changed_model_semantics(trained_run, tmp_path, setting, value):
    cfg, data, out = trained_run
    before = run_fingerprints(out)
    cfg["train"]["steps"] = 3
    cfg["model"][setting] = value
    checkpoint = out / "checkpoints/000002"
    with pytest.raises(ValueError, match=f"exact resume.*{setting}"):
        train(cfg, data, out, resume=checkpoint)
    assert run_fingerprints(out) == before

    destination = tmp_path / "rejected"
    with pytest.raises(ValueError, match=f"exact resume.*{setting}"):
        train(cfg, data, destination, resume=checkpoint)
    assert not destination.exists()


@pytest.mark.integration
def test_exact_resume_accepts_effective_defaults_and_execution_metadata(trained_run, tmp_path):
    cfg, data, out = trained_run
    checkpoint = out / "checkpoints/000002"
    config_path = checkpoint / "config.json"
    saved_config = json.loads(config_path.read_text())
    saved_config.update(push_to_hub=True, repo_id="fixture/original", tags=["fixture"])
    del saved_config["dropout"]
    config_path.write_text(json.dumps(saved_config))
    before = run_fingerprints(out)
    cfg["train"]["steps"] = 3
    del cfg["model"]["dropout"]  # Omitted 0.1 default has the same effective semantics.
    manifest = json.loads((data / "manifest.json").read_text())
    act_cfg = build_act_config(cfg["model"], manifest, "cpu")
    act_cfg.device = "cuda:99"  # Check device relocation without requiring another accelerator.
    validate_exact_resume_model_config(act_cfg, checkpoint)
    result = train(cfg, data, tmp_path / "continued", resume=checkpoint)
    assert result["steps"] == 3
    assert run_fingerprints(out) == before
