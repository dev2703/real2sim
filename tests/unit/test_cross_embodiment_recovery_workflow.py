"""Workflow wiring keeps model selection, matched resets and failures explicit."""

import json
import subprocess
from pathlib import Path

import pytest

from scripts.run_cross_embodiment_recovery import build_plan, main, parser, run_plan


def options(tmp_path, *extra):
    return parser().parse_args(["--out", str(tmp_path / "workflow"), *extra])


def collection(tmp_path, robot, *, train=(10000,), val=(11000,)):
    directory = tmp_path / robot
    directory.mkdir()
    manifest = {"status": "complete", "runtime_source_changes": [],
                "robot": robot, "base_weights_frozen": True, "lighting_changed": False,
                "split": {"train": list(train), "val": list(val)}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    for partition in ("train", "val"):
        (directory / f"{partition}.npz").write_bytes(b"validated by the trainer")
    return directory


def test_fresh_workflow_trains_both_heads_and_compares_identical_resets(tmp_path):
    plan = build_plan(options(tmp_path, "--matched-benchmark"))
    assert [stage["name"] for stage in plan["stages"]] == [
        "collect_so101", "collect_panda", "train_learned", "train_no_act",
        "matched_development_benchmark"]
    for stage in plan["stages"][:2]:
        assert stage["command"][-4:] == ["--reference-mode", "tracked", "--expert-execution", "common"]
    assert "--use-act" in plan["stages"][2]["command"]
    assert "--no-act" in plan["stages"][3]["command"]
    assert all("--require-command-labels" in stage["command"] for stage in plan["stages"][2:4])
    command = plan["stages"][-1]["command"]
    assert command[command.index("--conditions") + 1:command.index("--seeds")] == [
        "raw", "rgb", "hf_rgb", "learned", "no_act"]
    assert json.loads(command[-1])["heads"] == plan["heads"]
    assert json.loads(command[-1])["recovery_frame"] == "native"
    assert plan["reserved_seeds_used"] is False
    assert plan["training_robots"] == ["so101", "panda"]
    assert plan["zero_shot_target_robots"] == []
    assert plan["recovery_frame"] == "native"
    assert not (tmp_path / "workflow").exists()


def test_no_act_only_reuses_both_robot_collections(tmp_path):
    packs = [collection(tmp_path, robot) for robot in ("so101", "panda")]
    plan = build_plan(options(tmp_path, "--collections", *map(str, packs), "--no-act"))
    assert [stage["name"] for stage in plan["stages"]] == [
        "train_no_act", "evaluate_so101_no_act", "evaluate_panda_no_act"]
    assert "--no-act" in plan["stages"][0]["command"]
    assert len(plan["collection_inputs"]) == 2
    assert all(len(item["manifest_sha256"]) == 64 for item in plan["collection_inputs"])


@pytest.mark.parametrize("reuse", [False, True])
def test_source_only_training_evaluates_both_arms_and_labels_target_exclusion(tmp_path, reuse):
    extra = ["--train-robots", "so101", "--skip-no-act-ablation"]
    if reuse:
        extra.extend(["--collections", str(collection(tmp_path, "so101"))])
    plan = build_plan(options(tmp_path, *extra))
    assert plan["training_robots"] == ["so101"]
    assert plan["zero_shot_target_robots"] == ["panda"]
    assert plan["robots"] == ["so101", "panda"]
    assert [stage["name"] for stage in plan["stages"]] == (
        ([] if reuse else ["collect_so101"])
        + ["train_learned", "evaluate_so101_learned", "evaluate_panda_learned"])
    training = next(stage for stage in plan["stages"] if stage["name"] == "train_learned")
    assert not any("collections/panda" in item for item in training["command"])


@pytest.mark.parametrize("matched", [False, True])
def test_source_recovery_frame_is_forwarded_to_all_evaluators(tmp_path, matched):
    extra = ["--train-robots", "so101", "--recovery-frame", "source"]
    if matched:
        extra.append("--matched-benchmark")
    plan = build_plan(options(tmp_path, *extra))
    assert plan["recovery_frame"] == "source"
    assert plan["training_robots"] == ["so101"]
    if matched:
        assert json.loads(plan["stages"][-1]["command"][-1])["recovery_frame"] == "source"
    else:
        evaluations = [stage for stage in plan["stages"] if stage["name"].startswith("evaluate_")]
        assert len(evaluations) == 4
        for stage in evaluations:
            command = stage["command"]
            assert command[command.index("--recovery-frame") + 1] == "source"


@pytest.mark.parametrize("train_robots", [["so101", "panda"], ["panda"]])
def test_source_recovery_frame_rejects_training_with_target_arm(tmp_path, train_robots):
    with pytest.raises(ValueError, match="requires --train-robots so101"):
        build_plan(options(tmp_path, "--train-robots", *train_robots, "--recovery-frame", "source"))


@pytest.mark.parametrize("extra, message", [
    (["--train-seeds", "11000"], "disjoint"),
    (["--seeds", "10000"], "overlap"),
    (["--seeds", "13000"], "Reserved"),
    (["--val-seeds", "13001"], "Reserved"),
    (["--seeds", "12000", "12000"], "unique"),
    (["--head-type", "residual", "--temporal-weight", "0.1"], "phase head"),
    (["--temporal-weight", "nan"], "nonnegative"),
    (["--no-act", "--skip-no-act-ablation"], "requires --use-act"),
])
def test_invalid_splits_or_head_options_fail_before_work(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        build_plan(options(tmp_path, *extra))


def test_reused_collections_need_both_embodiments_and_development_separation(tmp_path):
    so101 = collection(tmp_path, "so101")
    with pytest.raises(ValueError, match="every requested training embodiment"):
        build_plan(options(tmp_path, "--collections", str(so101)))
    panda = collection(tmp_path, "panda", train=(12000,))
    with pytest.raises(ValueError, match="overlap"):
        build_plan(options(tmp_path, "--collections", str(so101), str(panda)))


def test_plan_only_works_without_checkpoint_and_writes_no_files(tmp_path, capsys):
    out = tmp_path / "workflow"
    plan = main(["--out", str(out), "--snapshot", str(tmp_path / "missing"), "--plan-only"])
    assert json.loads(capsys.readouterr().out) == plan
    assert not out.exists()


def executable_plan(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("model.safetensors", "config.json"):
        (snapshot / name).write_text("local checkpoint")
    scene = tmp_path / "scene.yaml"
    scene.write_text("scene")
    plan = build_plan(options(tmp_path, "--snapshot", str(snapshot), "--source-scene", str(scene),
                              "--robots", "so101", "--skip-no-act-ablation"))
    return plan


def test_run_records_every_stage_and_enforces_offline_execution(tmp_path):
    plan = executable_plan(tmp_path)
    commands = []

    def runner(command, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        commands.append(command)
        stage = plan["stages"][len(commands) - 1]
        path = Path(stage["report"])
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"status": "complete", "runtime_source_changes": [],
                                    "summary": {"n": 1}, "train_episodes": [10000],
                                    "validation_episodes": [11000], "training_robot_ids": [0]}))

    report = run_plan(plan, runner=runner)
    assert report["status"] == "complete"
    assert commands == [stage["command"] for stage in plan["stages"]]
    assert all(stage["summary"] == {"n": 1} for stage in report["stage_results"])
    with pytest.raises(FileExistsError, match="new workflow"):
        run_plan(plan, runner=runner)


def test_failed_stage_stops_pipeline_and_keeps_failure_report(tmp_path):
    plan = executable_plan(tmp_path)
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(3, command)

    with pytest.raises(subprocess.CalledProcessError):
        run_plan(plan, runner=runner)
    report = json.loads((Path(plan["output"]) / "workflow.json").read_text())
    assert report["status"] == "failed"
    assert report["stage_results"] == [{"name": "collect_so101", "status": "failed"}]
    assert len(calls) == 1


def test_incomplete_or_mutated_stage_cannot_advance(tmp_path):
    plan = executable_plan(tmp_path)

    def runner(_command, **_kwargs):
        path = Path(plan["stages"][0]["report"])
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"status": "complete", "runtime_source_changes": ["changed.py"]}))

    with pytest.raises(RuntimeError, match="unchanged runtime"):
        run_plan(plan, runner=runner)


@pytest.mark.parametrize("metadata, message", [
    ({"train_episodes": [12000]}, "Actual fitted episodes overlap"),
    ({"validation_episodes": [13000]}, "Actual fitted episodes overlap"),
    ({"training_robot_ids": [0, 1]}, "Actual training embodiments differ"),
    ({"train_episodes": None}, "lacks fitted-episode/embodiment provenance"),
])
def test_actual_trainer_rows_are_checked_before_any_rollout(tmp_path, metadata, message):
    plan = executable_plan(tmp_path)
    calls = []

    def runner(command, **_kwargs):
        stage = plan["stages"][len(calls)]
        calls.append(command)
        artifact = {"status": "complete", "train_episodes": [10000],
                    "validation_episodes": [11000], "training_robot_ids": [0]}
        if stage["name"].startswith("train_"):
            artifact.update(metadata)
        path = Path(stage["report"])
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(artifact))

    with pytest.raises(RuntimeError, match=message):
        run_plan(plan, runner=runner)
    assert len(calls) == 2
    report = json.loads((Path(plan["output"]) / "workflow.json").read_text())
    assert report["status"] == "failed"
    assert report["stage_results"][-1]["name"] == "train_learned"
