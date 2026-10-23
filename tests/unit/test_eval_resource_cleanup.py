"""Evaluation errors must release renderer and video resources exactly once."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.evaluation import run


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    env = SimpleNamespace(
        task={"episode": {"success_hold_s": .2}, "ball": {"radius": .0335}},
        config={"robot": "so101"}, data_config={"image": {"height": 2, "width": 2}},
        control_hz=10, max_episode_steps=1, table_top=.75, initial_qpos=np.zeros(6),
        model=SimpleNamespace(opt=SimpleNamespace(
            impratio=1., noslip_iterations=0, noslip_tolerance=1e-6, solver=2, cone=0)),
        close_calls=0,
    )

    def close():
        env.close_calls += 1

    obs = {run.IPHONE_KEY: np.zeros((3, 2, 2), np.float32)}
    env.close = close
    env.reset = lambda **kwargs: (obs, {})
    metrics = {
        "ee_ball_distance_m": .2, "ball_held": False, "lifted": False,
        "ball_in_bowl": False, "ball_resting": True, "ball_xyz": [0, 0, .8],
        "ball_xyz_task": [0, 0, .05], "tool_xyz_task": [0, 0, .25],
    }
    env.step = lambda action: (
        obs, 0., False, False,
        {"control": {"limit_clipped": [False] * 6}, "privileged_metrics": metrics},
    )
    eval_cfg = {
        "phase": {"approach_distance_m": .03},
        "video": {"fps": 10, "upscale": 1, "codec": "mp4v"},
    }
    monkeypatch.setattr(run, "make_env", lambda *args: env)
    monkeypatch.setattr(run, "load_yaml", lambda path: eval_cfg if path.name == "eval.yaml"
                        else {"robot": "so101"})
    videos = []

    class Video:
        def __init__(self, *args):
            self.close_calls = 0
            videos.append(self)

        def write(self, *args):
            pass

        def close(self):
            self.close_calls += 1

    monkeypatch.setattr(run, "CaptionedVideo", Video)

    def evaluate(**overrides):
        args = dict(
            robot="so101", checkpoint=None, config=Path("sim.yaml"), eval_config=Path("eval.yaml"),
            n_trials=1, seed_start=0, out=tmp_path / "out", video=False,
            controller="hold", device="cpu",
        )
        args.update(overrides)
        return run.run_trials(**args)

    return SimpleNamespace(env=env, videos=videos, config=eval_cfg, evaluate=evaluate)


def fail(exc):
    def raise_error(*args, **kwargs):
        raise exc

    return raise_error


def test_successful_trials_close_environment_once_and_each_video_once(evaluation):
    summary = evaluation.evaluate(n_trials=2, video=True)
    assert summary["summary"]["n"] == 2
    assert summary["summary"]["aborted"] == 0
    assert summary["resolved"]["contact_solver"] == {
        "impratio": 1., "noslip_iterations": 0, "noslip_tolerance": 1e-6, "solver": 2, "cone": 0,
    }
    assert [trial["ticks"] for trial in summary["trials"]] == [1, 1]
    assert evaluation.env.close_calls == 1
    assert [video.close_calls for video in evaluation.videos] == [1, 1]


def test_missing_checkpoint_closes_environment(evaluation):
    with pytest.raises(ValueError, match="checkpoint is required"):
        evaluation.evaluate(controller="learned")
    assert evaluation.env.close_calls == 1


def test_failed_checkpoint_loading_closes_environment_once(evaluation, monkeypatch):
    monkeypatch.setattr(run, "build_learned_controller", fail(FileNotFoundError("missing model")))
    with pytest.raises(FileNotFoundError, match="missing model"):
        evaluation.evaluate(controller="learned", checkpoint=Path("missing"))
    assert evaluation.env.close_calls == 1


def test_tracker_setup_failure_closes_environment(evaluation):
    evaluation.config["phase"]["approach_distance_m"] = 0
    with pytest.raises(ValueError, match="approach_distance_m must be positive"):
        evaluation.evaluate()
    assert evaluation.env.close_calls == 1


@pytest.mark.parametrize("exc", [run.ContractError("bad contract"), TypeError("bad runtime type")])
def test_propagated_rollout_error_closes_environment_and_video(evaluation, exc):
    evaluation.env.step = fail(exc)
    with pytest.raises(type(exc), match=str(exc)):
        evaluation.evaluate(video=True)
    assert evaluation.env.close_calls == 1
    assert [video.close_calls for video in evaluation.videos] == [1]


def test_physics_abort_still_records_every_failed_trial(evaluation):
    evaluation.env.step = fail(RuntimeError("physics failed"))
    summary = evaluation.evaluate(n_trials=2, video=True)
    assert summary["summary"]["n"] == summary["summary"]["aborted"] == 2
    assert [trial["abort"] for trial in summary["trials"]] == ["physics failed"] * 2
    assert [trial["phase_failed"] for trial in summary["trials"]] == ["aborted"] * 2
    assert evaluation.env.close_calls == 1
    assert [video.close_calls for video in evaluation.videos] == [1, 1]


def test_writer_setup_failure_closes_environment(evaluation, monkeypatch):
    monkeypatch.setattr(run, "CaptionedVideo", fail(RuntimeError("writer could not open")))
    with pytest.raises(RuntimeError, match="writer could not open"):
        evaluation.evaluate(video=True)
    assert evaluation.env.close_calls == 1


def test_writer_close_failure_still_closes_environment(evaluation, monkeypatch):
    monkeypatch.setattr(run.CaptionedVideo, "close", fail(OSError("writer close failed")))
    with pytest.raises(OSError, match="writer close failed"):
        evaluation.evaluate(video=True)
    assert evaluation.env.close_calls == 1
