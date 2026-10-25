"""Capacity-gate checks must require success on all three training conditions."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("tiny_fit_runner", Path("scripts/run_v2_three_episode_fit.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report(translation, gripper):
    episodes = {str(e): {"translation_mean_mm": x, "gripper_mae": g,
                         "movement": {"n": 20, "vector_error_mean": .5},
                         "stationary_drift": {"n": 10, "summed_error_per_run_mean": .02,
                                              "summed_error_per_run_max": .04}}
                for e, x, g in zip((11, 8, 23), translation, gripper, strict=True)}
    result = {"scores": {"model": {"per_episode": episodes}}}
    for name in ("hold", "persistence"):
        baseline = deepcopy(episodes)
        for row in baseline.values():
            row["movement"]["vector_error_mean"] = 2.
            row["stationary_drift"].update(summed_error_per_run_mean=.1, summed_error_per_run_max=.2)
        result["scores"][name] = {"per_episode": baseline}
    return result


def test_all_conditions_must_meet_strict_physical_thresholds():
    assert module.gate_from_validation(report([.2, .9, .8], [.01, .02, .029]))["pass"]
    assert not module.gate_from_validation(report([.2, 1., .8], [.01, .02, .02]))["pass"]
    assert not module.gate_from_validation(report([.2, .9, .8], [.01, .03, .02]))["pass"]


def test_two_conditions_cannot_pass_three_condition_gate():
    value = report([.2, .9, .8], [.01, .02, .02])
    del value["scores"]["model"]["per_episode"]["23"]
    assert not module.gate_from_validation(value)["pass"]


@pytest.mark.parametrize("baseline", ["hold", "persistence"])
@pytest.mark.parametrize("group,metric", [
    ("movement", "vector_error_mean"),
    ("stationary_drift", "summed_error_per_run_mean"),
    ("stationary_drift", "summed_error_per_run_max"),
])
def test_good_aggregate_fit_cannot_hide_a_condition_worse_than_either_baseline(baseline, group, metric):
    value = report([.2, .3, .4], [.01] * 3)
    value["scores"][baseline]["per_episode"]["23"][group][metric] = 0.
    assert not module.gate_from_validation(value)["pass"]


@pytest.mark.parametrize("group", ["movement", "stationary_drift"])
def test_missing_or_mismatched_motion_strata_fail(group):
    value = report([.2] * 3, [.01] * 3)
    value["scores"]["model"]["per_episode"]["11"][group]["n"] = 0
    assert not module.gate_from_validation(value)["pass"]
    value["scores"]["model"]["per_episode"]["11"][group]["n"] = 999
    assert not module.gate_from_validation(value)["pass"]


def test_zero_stationary_error_floor_can_tie_but_nonzero_ties_fail():
    value = report([.2] * 3, [.01] * 3)
    for score in value["scores"].values():
        for row in score["per_episode"].values():
            row["stationary_drift"].update(summed_error_per_run_mean=0., summed_error_per_run_max=0.)
    assert module.gate_from_validation(value)["pass"]
    for score in value["scores"].values():
        score["per_episode"]["11"]["stationary_drift"]["summed_error_per_run_mean"] = .1
    assert not module.gate_from_validation(value)["pass"]


def test_nonfinite_error_fails():
    value = report([.2] * 3, [.01] * 3)
    value["scores"]["model"]["per_episode"]["11"]["movement"]["vector_error_mean"] = float("nan")
    assert not module.gate_from_validation(value)["pass"]


def test_wrong_episode_set_fails():
    value = report([.2] * 3, [.01] * 3)
    for score in value["scores"].values():
        score["per_episode"]["12"] = score["per_episode"].pop("11")
    assert not module.gate_from_validation(value)["pass"]


@pytest.mark.parametrize("prefix", [None, 2])
def test_interrupted_run_restores_saved_checkpoint_and_preserves_bounded_plan(tmp_path, monkeypatch, prefix):
    data, out = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    for name in ("manifest.json", "stats.json"):
        (data / name).write_text("{}\n")
    calls = []
    real_load_yaml = module.load_yaml
    source_config = Path("outputs/v2/learning_screen_20261022/configs/L1_resnet_iphone.yaml")

    def load_test_config(path):
        # The experiment's generated base config is intentionally not committed.
        # Keep real loading for the saved temporary plan so resume checks still
        # exercise YAML serialization and the original-plan comparison.
        if path == source_config:
            return {"model": {"chunk_size": 25, "n_action_steps": 2}, "scheduler": {}, "train": {}}
        return real_load_yaml(path)

    def fake_train(cfg, source, destination, resume=None):
        calls.append({"steps": cfg["train"]["steps"], "resume": resume,
                      "schedule": cfg["train"]["schedule_steps"],
                      "supervised_prefix_steps": cfg.get("reconstruction_loss", {}).get("supervised_prefix_steps")})
        step = 500 if len(calls) == 1 else cfg["train"]["steps"]
        checkpoint = destination / "checkpoints" / f"{step:06d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "run_info.json").write_text(json.dumps({"step": step,
            "diagnostic_episodes": [11, 8, 23], "scheduler_total_steps": 2000, "device": "cpu"}))
        (checkpoint / "model.safetensors").write_bytes(b"test model")
        (checkpoint / "training_state.pt").write_bytes(b"test optimizer, RNG and sampler state")
        last = checkpoint.parent / "last"
        if last.is_symlink():
            last.unlink()
        last.symlink_to(checkpoint.name)
        if len(calls) == 1:
            raise RuntimeError("simulated interruption after the saved 500-step checkpoint")
        return {"steps": step, "device": "cpu"}

    def fake_audit(checkpoint, source, device):
        values = report([.2] * 3, [.01] * 3)
        if checkpoint.name == "001000":
            values["scores"]["model"]["per_episode"]["23"]["gripper_mae"] = .04
        return values

    monkeypatch.setattr(module, "train", fake_train)
    monkeypatch.setattr(module, "audit_training_checkpoint", fake_audit)
    monkeypatch.setattr(module, "load_yaml", load_test_config)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.run(data, out, "cpu", supervised_prefix_steps=prefix)
    original_plan = (out / "plan.yaml").read_bytes()
    assert (out / "report.json").exists()
    result = module.run(data, out, "cpu", resume=True, supervised_prefix_steps=prefix)
    assert (out / "plan.yaml").read_bytes() == original_plan
    assert [call["steps"] for call in calls] == [1000, 1000, 2000]
    assert [call["schedule"] for call in calls] == [2000] * 3
    assert [call["supervised_prefix_steps"] for call in calls] == [prefix] * 3
    assert calls[1]["resume"].name == "000500"
    assert calls[2]["resume"].name == "001000"
    assert result["interruptions"][0]["resumed_from_step"] == 500
    assert [stage["examples_seen"] for stage in result["stages"]] == [16000, 32000]
    assert result["gate_passed"]
    with pytest.raises(ValueError, match="original plan"):
        module.run(data, out, "mps", resume=True, supervised_prefix_steps=prefix)
    with pytest.raises(ValueError, match="original plan"):
        module.run(data, out, "cpu", resume=True, supervised_prefix_steps=2 if prefix is None else None)
