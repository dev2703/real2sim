"""Small CPU cache/train/profile lifecycle; no network, MuJoCo or GPU required."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from transformers import Dinov2Config, Dinov2Model

from real2sim.data.prepare import compute_stats
from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.paired import file_sha256

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/train_flow_head.py"


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
    dino = Dinov2Config(hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
                        image_size=28, patch_size=14, mlp_ratio=2)
    vision = tmp_path / "dino"
    Dinov2Model(dino).save_pretrained(vision)
    config = tmp_path / "vision.yaml"
    config.write_text(yaml.safe_dump({"model": {"dinov2_config": dino.to_dict(),
                      "dinov2_pretrained_path": str(vision), "dinov2_model_id": "tiny-test",
                      "dinov2_revision": "offline-fixture"}}))
    return root, config, manifest, stats


def command(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, timeout=90)


@pytest.mark.integration
@pytest.mark.parametrize("objective", ["flow", "regression"])
def test_cache_train_portable_reload_and_profile(tiny_data, tmp_path, objective, monkeypatch):
    data, vision_config, manifest, stats = tiny_data
    cache = tmp_path / "cache"
    result = command("cache", "--data", data, "--vision-config", vision_config, "--out", cache,
                     "--device", "cpu", "--batch-size", 4)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (cache / "COMPLETE.json").exists() and not (cache / "RUNNING.json").exists()
    out = tmp_path / objective
    result = command("train", "--data", data, "--cache", cache, "--out", out, "--objective", objective,
                     "--device", "cpu", "--batch-size", 2, "--grad-accum", 2, "--examples", 8)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / "COMPLETE.json").exists() and not (out / "RUNNING.json").exists()
    report = json.loads((out / "report.json").read_text())
    assert report["examples"] == 8 and report["updates"] == 2
    assert report["split"] == manifest["split"]
    # Central loader dispatch, exact common split/statistics, and raw-image inference.
    policy, norm, contract = load_checkpoint(out / "checkpoint")
    assert norm.stats == stats and contract["split"] == manifest["split"]
    assert contract["observation_variant"] == manifest["observation_variant"]
    assert contract["experiment_split"] == manifest["experiment_split"]
    dataset = DerivedTickDataset(data, [2], chunk_size=20, normalizer=norm)
    item = dataset[0]
    obs = {key: value[None] for key, value in item.items() if key.startswith("observation.")}
    prediction = policy.predict_action_chunk(obs, sampling_seed=[12])
    assert prediction.shape == (1, 20, 7) and torch.isfinite(prediction).all()
    # Exercise root's shared offline audit, including its contract checks.
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("flow_smoke_audit", ROOT / "scripts/audit_experiment_policy.py")
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)
    audit = audit_module.audit(out / "checkpoint", data, "cpu")
    assert audit["valid_actions"] == 5
    profile = out / "profile"
    result = command("profile", "--data", data, "--checkpoint", out / "checkpoint", "--out", profile,
                     "--device", "cpu", "--repeats", 2)
    assert result.returncode == 0, result.stdout + result.stderr
    measured = json.loads((profile / "profile.json").read_text())
    key = "4_steps" if objective == "flow" else "regression"
    assert measured[key]["median_ms"] > 0 and (profile / "COMPLETE.json").exists()


@pytest.mark.integration
def test_cache_failure_writes_flag_without_success(tmp_path):
    out = tmp_path / "failed"
    result = command("cache", "--data", tmp_path / "missing", "--out", out, "--device", "cpu")
    assert result.returncode != 0
    failure = json.loads((out / "FAILED.json").read_text())
    assert "FileNotFoundError" in failure["error"]
    assert not (out / "COMPLETE.json").exists() and not (out / "RUNNING.json").exists()
