"""Recovery collection and ACT handoffs retain the requested camera interface."""

from types import SimpleNamespace

import pytest
import torch
import yaml

from real2sim.artifacts import sha256_file
from real2sim.config import load_yaml
from scripts import evaluate_hf_recovery as rgb_recovery
from scripts import run_recovery_dagger as dagger
from scripts.benchmark_frozen_act import SharedEpisode


@pytest.mark.parametrize("robot", ["so101", "panda"])
@pytest.mark.parametrize("mode", ["collect", "act_recovery", "no_act_after_prefix"])
def test_recovery_uses_checkpoint_camera_names_order_and_dimensions(
        tmp_path, monkeypatch, robot, mode):
    snapshot = tmp_path / "checkpoint"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"frozen weights")
    contract = {
        "camera_keys": ["observation.images.iphone", "observation.images.wrist"],
        "wrist_key": "observation.images.wrist",
        "image_shape_chw": [3, 180, 320],
    }
    checkpoint = SimpleNamespace(policy=torch.nn.Linear(1, 1),
        config=SimpleNamespace(dim_model=8), scene_contract=contract)
    monkeypatch.setattr(dagger, "NativeLeRobotACT", lambda *_: checkpoint)
    monkeypatch.setattr(dagger, "source_files", lambda *_: {})
    monkeypatch.setattr(dagger, "snapshot_sources", lambda *_: "unused")
    monkeypatch.setattr(dagger, "load_recovery_head", lambda *_: SimpleNamespace(
        metadata={"use_act": mode == "act_recovery"}))
    captured = []

    class ScenesChecked(Exception):
        """Stop after real CPU scene construction, before rendering/inference."""

    def inspect_scene(*args, **kwargs):
        shared = SharedEpisode(*args, **kwargs)
        try:
            # Check source and target scenes: Panda's virtual SO-101 must use
            # the same saved camera interface as the actual rollout scene.
            for env in (shared.source.source, shared.env):
                assert env.camera_keys == contract["camera_keys"]
                assert env.config["render"] == {"height": 180, "width": 320}
                assert "observation.images.wrist" in env.camera_bindings
            captured.append(args[1].name)
        finally:
            shared.close()
        if mode != "collect" or len(captured) == 2:
            raise ScenesChecked
        return shared

    monkeypatch.setattr(dagger, "SharedEpisode", inspect_scene)
    out = tmp_path / "recovery"
    with pytest.raises(ScenesChecked):
        if mode == "collect":
            dagger.collect(robot, out, [10000], [11000], snapshot=snapshot)
        else:
            dagger.evaluate(robot, tmp_path / "head", out, [12000], snapshot=snapshot,
                            act_prefix=mode == "no_act_after_prefix")
    assert captured == (["live", "expert_shadow"] if mode == "collect" else ["setup"])


@pytest.mark.parametrize("robot", ["so101", "panda"])
def test_rgb_recovery_uses_requested_cameras_hold_duration_and_video_size(
        tmp_path, monkeypatch, robot):
    snapshot = tmp_path / "checkpoint"
    snapshot.mkdir()
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"frozen weights")
    contract = {
        "camera_keys": ["observation.images.iphone", "observation.images.wrist"],
        "wrist_key": "observation.images.wrist", "image_shape_chw": [3, 180, 320],
    }
    checkpoint = SimpleNamespace(policy=torch.nn.Linear(1, 1), scene_contract=contract,
        weights_sha256=sha256_file(weights), reset=lambda: None)
    monkeypatch.setattr(rgb_recovery, "NativeLeRobotACT", lambda *_: checkpoint)

    scene = load_yaml(rgb_recovery.MEASURED_SCENE)
    task = load_yaml(scene["task_config"])
    task["episode"]["success_hold_s"] = .12
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    scene["task_config"] = str(task_path)
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text(yaml.safe_dump(scene))
    captured = {}
    tracker = rgb_recovery.TrialTracker

    def capture_tracker(**kwargs):
        captured.update(kwargs)
        return tracker(**kwargs)

    class VideoReached(Exception):
        """Stop after real CPU reset, before creating a renderer or encoder."""

    def capture_video(path, fps, frame_hw):
        captured.update(video_fps=fps, video_shape=frame_hw)
        raise VideoReached

    # Perception is irrelevant to scene/scoring construction. Keep this
    # regression headless while executing real reset and camera registration.
    monkeypatch.setattr(rgb_recovery, "StaticBowlRecovery", lambda *_args, **_kwargs:
        SimpleNamespace(reset=lambda: None, _acquire=lambda: True,
                        perception_log=[{"targets": {}}]))
    monkeypatch.setattr(rgb_recovery, "TrialTracker", capture_tracker)
    monkeypatch.setattr(rgb_recovery, "CaptionedVideo", capture_video)
    out = tmp_path / "evaluation"
    with pytest.raises(VideoReached):
        rgb_recovery.run(robot, out, [71], snapshot=snapshot, source_scene=scene_path, video=True)

    assert captured["success_hold_ticks"] == 3
    assert captured["video_fps"] == 25.
    assert captured["video_shape"] == (180, 320)
    for data_path in [out / "source/data.yaml"] + ([out / "panda_data.yaml"] if robot == "panda" else []):
        assert load_yaml(data_path)["camera_keys"] == contract["camera_keys"]
