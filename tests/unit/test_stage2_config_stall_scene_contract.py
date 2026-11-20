"""Stage 2 regression tests: v2 config refs, fresh RGB stall window, and scene compatibility."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from real2sim.artifacts import sha256_file
from real2sim.config import REPO_ROOT, project_path
from real2sim.policy.frozen_prefix_recovery import FrozenPrefixRecoveryController
from real2sim.policy.hf_recovery import ApproachMonitor
from scripts import train_phase_recovery_mlp as phase_trainer
from scripts import train_recovery_mlp as residual_trainer


def test_every_v2_yaml_task_config_resolves_to_existing_file():
    v2_dir = REPO_ROOT / "configs" / "v2"
    yamls = sorted(v2_dir.glob("*.yaml"))
    assert yamls
    missing = []
    for path in yamls:
        data = yaml.safe_load(path.read_text()) or {}
        if isinstance(data, dict) and "task_config" in data:
            target = project_path(data["task_config"])
            if not target.is_file():
                missing.append(f"{path.name} -> {data['task_config']}")
    assert not missing, f"Broken v2 task_config refs: {missing}"
    for name in ("data.yaml", "eval_so101.yaml", "sim_so101.yaml"):
        payload = yaml.safe_load((v2_dir / name).read_text())
        assert payload["task_config"] == "configs/v2/task.exploratory_20261022.yaml"


@pytest.mark.parametrize("report_missing", [True, False])
def test_approach_monitor_expires_old_samples_across_missing_frame_gaps(report_missing):
    monitor = ApproachMonitor(2, patience_s=2., max_attempt_s=20.)
    # Far-away samples at t=0,1 must expire so they neither stretch patience
    # nor masquerade as 20cm progress when fresh samples arrive at t=9..12.
    assert monitor.update(0, .50) is None
    assert monitor.update(1, .50) is None
    if report_missing:
        for tick in range(2, 9):
            assert monitor.update(tick, None) is None
    for tick in range(9, 13):
        assert monitor.update(tick, .30) is None
    reason = monitor.update(13, .30)
    assert reason is not None and "progress" in reason


def test_approach_monitor_missing_frames_do_not_delay_deadline():
    monitor = ApproachMonitor(2, patience_s=2., max_attempt_s=5.)
    for tick in range(10):
        assert monitor.update(tick, .30 if tick % 2 else None) is None
    reason = monitor.update(10, None)
    assert reason is not None and "elapsed" in reason


def test_approach_monitor_true_stall_and_normal_progress():
    stalled = ApproachMonitor(25, patience_s=2., max_attempt_s=5., progress_m=.003)
    progressed = ApproachMonitor(25, patience_s=2., max_attempt_s=5., progress_m=.003)
    for tick in range(50):
        assert stalled.update(tick, .30 - tick * .00004) is None
        assert progressed.update(tick, .30 - tick * .001) is None
    assert "progress" in stalled.update(50, .298)
    assert progressed.update(50, .25) is None


class _DummyEnv:
    def __init__(self):
        self.pose = np.eye(4)

    def ee_pose(self):
        return self.pose.copy()


class _DummyDriver:
    def __init__(self, env):
        self.env = env
        self.inference_ms = []
        self.steps = 0

    def reset(self):
        self.inference_ms = []
        self.steps = 0

    def step(self, obs, env):
        del obs, env
        self.steps += 1
        return SimpleNamespace(detail={})

    def close(self):
        pass


class _RetainedStaleTracker:
    """Mimics RGBReferenceTracker retaining an old ball pose when occluded."""

    def __init__(self):
        self.ball = np.array([.50, 0., 0.])
        self.visible = {"ball": False}
        self.last_seen = {"ball": None}
        self.refresh_ticks = 5

    def update(self, tick):
        if tick == 0:
            self.visible["ball"] = True
            self.last_seen["ball"] = 0
        elif tick == 5:
            # Ball occluded: tracker keeps stale self.ball and last_seen=0, visible=False.
            self.visible["ball"] = False


def test_frozen_prefix_rejects_stale_retained_target_as_fresh_progress():
    env = _DummyEnv()
    monitor = ApproachMonitor(2, patience_s=3., max_attempt_s=4.)
    tracker = _RetainedStaleTracker()
    recorded_distances = []
    orig_update = monitor.update

    def spy_update(tick, distance_m):
        recorded_distances.append((tick, distance_m))
        return orig_update(tick, distance_m)

    monitor.update = spy_update
    controller = FrozenPrefixRecoveryController(
        _DummyDriver(env),
        _DummyDriver(env),
        lambda: monitor,
        lambda: tracker,
        np.zeros(3),
    )

    for tick in range(9):
        if tick >= 5:
            # Arm moves closer to the stale retained target during occlusion.
            env.pose[0, 3] = .05 * (tick - 4)
        result = controller.step({}, env)
        if tick < 8:
            assert result.detail["prefix_active"] is True

    # Ticks 0..4 use the current refresh-cycle target; ticks 5..8 must drop stale retained distances.
    assert all(d is not None for _, d in recorded_distances[:5])
    assert all(d is None for _, d in recorded_distances[5:9])
    assert controller.trigger is not None
    assert "elapsed" in controller.trigger["reason"]


def test_frozen_prefix_rejects_aged_last_seen_even_when_visible_flag_stays_true():
    env = _DummyEnv()
    tracker = SimpleNamespace(
        ball=np.array([.50, 0., 0.]),
        visible={"ball": True},
        last_seen={"ball": 0},
        refresh_ticks=5,
        update=lambda _tick: None,
    )
    seen_distances = []

    def make_monitor():
        return SimpleNamespace(
            update=lambda tick, d: seen_distances.append((tick, d)) or None
        )

    controller = FrozenPrefixRecoveryController(
        _DummyDriver(env),
        _DummyDriver(env),
        make_monitor,
        lambda: tracker,
        np.zeros(3),
    )
    for _ in range(7):
        controller.step({}, env)
    assert all(d is not None for _, d in seen_distances[:5])
    assert all(d is None for _, d in seen_distances[5:])


def _make_snapshot(tmp_path: Path) -> tuple[Path, str]:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"frozen-act-checkpoint")
    return snapshot, sha256_file(snapshot / "model.safetensors")


def _make_pack(
    tmp_path: Path,
    name: str,
    *,
    base_hash: str,
    train_episodes: tuple[int, ...],
    val_episodes: tuple[int, ...],
    manifest_extra: dict | None = None,
    iteration: int = 0,
) -> Path:
    pack = tmp_path / name
    pack.mkdir()
    manifest = {
        "status": "complete",
        "base_checkpoint_sha256": base_hash,
        "lighting_changed": False,
        "base_weights_frozen": True,
        "runtime_source_changes": [],
        "expert_phase_is_audit_label_only": True,
        "expert_phase_label_source": "issued_command",
        "iteration": iteration,
        "split": {"train": list(train_episodes), "val": list(val_episodes)},
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (pack / "manifest.json").write_text(json.dumps(manifest))
    rng = np.random.default_rng(17 + iteration)
    for partition, episodes in (("train", train_episodes), ("val", val_episodes)):
        n = 6 * len(episodes)
        obs = rng.normal(size=(n, 6)).astype(np.float32)
        base = np.zeros((n, 7), np.float32)
        targets = rng.normal(size=(n, 7)).astype(np.float32) * .01
        features = rng.normal(size=(n, 4)).astype(np.float32)
        robot_id = np.arange(n, dtype=np.int64) % 2
        episode_index = np.repeat(np.asarray(episodes, dtype=np.int64), 6)
        expert_phase_id = np.tile(np.asarray([1, 2, 3], dtype=np.int64), 2 * len(episodes))
        np.savez(
            pack / f"{partition}.npz",
            observations=obs,
            base_actions=base,
            features=features,
            target_actions=targets,
            robot_id=robot_id,
            episode_index=episode_index,
            expert_phase_id=expert_phase_id,
        )
    return pack


@pytest.mark.parametrize("trainer_module", [residual_trainer, phase_trainer])
def test_matching_scene_metadata_including_dagger_learner_files_is_accepted(tmp_path, trainer_module):
    snapshot, base_hash = _make_snapshot(tmp_path)
    scene_hashes_iter0 = {
        "/repo/configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml": "scene-hash-1",
        "/repo/assets/menagerie/robotstudio_so101/so101.xml": "xml-hash-1",
        "/repo/src/real2sim/sim/cameras.py": "cam-hash-1",
    }
    scene_hashes_iter1 = {
        **scene_hashes_iter0,
        "/tmp/iter0/heads/learned/head.json": "learner-json-hash",
        "/tmp/iter0/heads/learned/head.pt": "learner-pt-hash",
    }
    common_meta = {
        "source_scene": "configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml",
        "scene_sha256": "scene-hash-1",
        "scene_contract": {"camera_keys": ["observation.images.iphone", "observation.images.wrist"]},
    }
    pack0 = _make_pack(
        tmp_path, "pack0", base_hash=base_hash, train_episodes=(10,), val_episodes=(20,),
        iteration=0, manifest_extra={**common_meta, "source_sha256_before": scene_hashes_iter0},
    )
    pack1 = _make_pack(
        tmp_path, "pack1", base_hash=base_hash, train_episodes=(11,), val_episodes=(21,),
        iteration=1, manifest_extra={**common_meta, "source_sha256_before": scene_hashes_iter1},
    )
    out = tmp_path / "head_out"
    report = trainer_module.train([pack0, pack1], out, snapshot=snapshot, steps=2, use_act=False)
    head = json.loads((out / "head.json").read_text())
    assert report["allow_mixed_scenes"] is False
    assert head["allow_mixed_scenes"] is False
    assert len(report["collection_scenes"]) == 2
    assert head["collection_scenes"] == report["collection_scenes"]
    assert report["collection_scenes"][0]["source_render_sha256"] == report["collection_scenes"][1]["source_render_sha256"]


@pytest.mark.parametrize("trainer_module", [residual_trainer, phase_trainer])
def test_identical_external_paths_and_legacy_schema_packs_are_accepted(tmp_path, trainer_module):
    snapshot, base_hash = _make_snapshot(tmp_path)
    external_a = "/machine-a/ext/custom_scene.yaml"
    external_b = "/machine-b/ext/custom_scene.yaml"
    pack_a = _make_pack(
        tmp_path, "ext_a", base_hash=base_hash, train_episodes=(10,), val_episodes=(20,),
        iteration=0, manifest_extra={
            "source_scene": external_a,
            "scene_sha256": "ext-scene-hash-1",
            "source_sha256_before": {
                external_a: "ext-scene-hash-1",
                "/machine-a/repo/assets/menagerie/robotstudio_so101/so101.xml": "xml-hash-1",
            },
        },
    )
    pack_b = _make_pack(
        tmp_path, "ext_b", base_hash=base_hash, train_episodes=(11,), val_episodes=(21,),
        iteration=1, manifest_extra={
            "source_scene": external_b,
            "scene_sha256": "ext-scene-hash-1",
            "source_sha256_before": {
                external_b: "ext-scene-hash-1",
                "/machine-b/repo/assets/menagerie/robotstudio_so101/so101.xml": "xml-hash-1",
            },
        },
    )
    out_ext = tmp_path / "ext_head"
    report_ext = trainer_module.train([pack_a, pack_b], out_ext, snapshot=snapshot, steps=2, use_act=False)
    assert report_ext["allow_mixed_scenes"] is False
    assert [entry["source_scene"] for entry in report_ext["collection_scenes"]] == [external_a, external_b]

    # Legacy pack (no top-level source_scene/scene_sha256) with matching source_sha256_before
    # should combine cleanly with a new-schema pack for the same scene.
    shared_hashes = {
        "/repo/configs/v2/weight_space_20261111/sim_so101.yaml": "scene-hash-1",
        "/repo/assets/menagerie/robotstudio_so101/so101.xml": "xml-hash-1",
    }
    legacy_pack = _make_pack(
        tmp_path, "legacy", base_hash=base_hash, train_episodes=(12,), val_episodes=(22,),
        iteration=0, manifest_extra={"source_sha256_before": shared_hashes},
    )
    new_pack = _make_pack(
        tmp_path, "new_schema", base_hash=base_hash, train_episodes=(13,), val_episodes=(23,),
        iteration=1, manifest_extra={
            "source_scene": "configs/v2/weight_space_20261111/sim_so101.yaml",
            "scene_sha256": "scene-hash-1",
            "source_sha256_before": shared_hashes,
        },
    )
    out_mix = tmp_path / "legacy_new_head"
    report_mix = trainer_module.train([legacy_pack, new_pack], out_mix, snapshot=snapshot, steps=2, use_act=False)
    assert report_mix["allow_mixed_scenes"] is False
    assert len(report_mix["collection_scenes"]) == 2


@pytest.mark.parametrize("conflict_meta", [
    {"scene_sha256": "different-scene-hash"},
    {"scene_sha256": "scene-hash-1", "scene_contract": {"camera_keys": ["observation.images.top"]}},
    {"scene_sha256": "scene-hash-1", "source_sha256_before": {"configs/v2/sim_so101.yaml": "render-hash-2"}},
    {},  # One pack carries scene metadata and the other omits it.
])
@pytest.mark.parametrize("trainer_module", [residual_trainer, phase_trainer])
def test_conflicting_scene_metadata_rejected_by_default(tmp_path, trainer_module, conflict_meta):
    snapshot, base_hash = _make_snapshot(tmp_path)
    pack0 = _make_pack(
        tmp_path, "pack0", base_hash=base_hash, train_episodes=(10,), val_episodes=(20,),
        manifest_extra={
            "source_scene": "configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml",
            "scene_sha256": "scene-hash-1",
            "source_sha256_before": {"configs/v2/sim_so101.yaml": "render-hash-1"},
        },
    )
    if conflict_meta:
        conflict_payload = {
            "source_scene": "configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml",
            "source_sha256_before": {"configs/v2/sim_so101.yaml": "render-hash-1"},
            **conflict_meta,
        }
    else:
        conflict_payload = {}
    pack1 = _make_pack(
        tmp_path, "pack1", base_hash=base_hash, train_episodes=(11,), val_episodes=(21,),
        iteration=1, manifest_extra=conflict_payload,
    )
    with pytest.raises(ValueError, match="allow-mixed-scenes"):
        trainer_module.train([pack0, pack1], tmp_path / "out", snapshot=snapshot, steps=2, use_act=False)


@pytest.mark.parametrize("script_name", ["scripts/train_recovery_mlp.py", "scripts/train_phase_recovery_mlp.py"])
def test_allow_mixed_scenes_cli_flag_opts_in_and_records_metadata(tmp_path, script_name):
    snapshot, base_hash = _make_snapshot(tmp_path)
    pack0 = _make_pack(
        tmp_path, "pack0", base_hash=base_hash, train_episodes=(10,), val_episodes=(20,),
        manifest_extra={"source_scene": "scene_a.yaml", "scene_sha256": "hash-a"},
    )
    pack1 = _make_pack(
        tmp_path, "pack1", base_hash=base_hash, train_episodes=(11,), val_episodes=(21,),
        iteration=1, manifest_extra={"source_scene": "scene_b.yaml", "scene_sha256": "hash-b"},
    )
    out = tmp_path / "mixed_head"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / script_name),
            "--data",
            str(pack0),
            str(pack1),
            "--out",
            str(out),
            "--snapshot",
            str(snapshot),
            "--steps",
            "2",
            "--no-act",
            "--allow-mixed-scenes",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    report = json.loads((out / "report.json").read_text())
    head = json.loads((out / "head.json").read_text())
    assert report["allow_mixed_scenes"] is True
    assert head["allow_mixed_scenes"] is True
    assert [entry["scene_sha256"] for entry in report["collection_scenes"]] == ["hash-a", "hash-b"]
    assert head["collection_scenes"] == report["collection_scenes"]
