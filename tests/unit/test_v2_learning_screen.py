import sys
from pathlib import Path

import pytest

from scripts.run_so101_experiments import atomic_json, hash_file, read_json
from scripts.run_v2_learning_screen import (
    ScreenRunner,
    compile_plan,
    inspect_inputs,
    latest_checkpoint,
    parse_args,
)


def matrix():
    return read_json(Path("results/v2/screen_matrix_v4.json"))


def fixture_runner(tmp_path, *extra):
    inputs = matrix()
    inputs["arms"] = inputs["arms"][:1]
    matrix_path = tmp_path / "matrix.json"
    atomic_json(matrix_path, inputs)
    task = tmp_path / "task.yaml"
    task.write_text("calibration: provisional\n")
    data_root = tmp_path / "data"
    data = data_root / "cartesian_iphone"
    data.mkdir(parents=True)
    (data / "payload.bin").write_bytes(b"frozen real records")
    atomic_json(data / "manifest.json", {
        "camera_keys": ["observation.images.iphone"],
        "action_representation": "cartesian", "controller_semantics": "accepted_target",
        "policy_hz": 12.5, "contract_version": "v2",
        "split": {"train": [0], "val": [1], "reserved": [2]},
        "files_sha256": {"payload.bin": hash_file(data / "payload.bin")},
    })
    atomic_json(data / "stats.json", {"train_episodes": [0]})
    args = parse_args(["--matrix", str(matrix_path), "--task", str(task),
                       "--data-root", str(data_root), "--out", str(tmp_path / "queue"),
                       "--device", "cpu", "--minimum-free-gib", "0", *extra])
    return ScreenRunner(args)


def test_complete_matrix_preserves_exposures_and_deduplicates_controls(tmp_path):
    original = matrix()
    plan = compile_plan(original, tmp_path, device="mps", physical_batch=2)
    assert len(plan["matrix_entries"]) == 24
    assert len(plan["jobs"]) == 22
    assert {r["id"]: r["reuse"] for r in plan["matrix_entries"] if r["reuse"]} == {
        "L4_dino_frozen_control": "L2_dino_frozen",
        "L5_cartesian": "L1_resnet_iphone_top",
    }
    for job in plan["jobs"]:
        source = next(a for a in original["arms"] if a["id"] == job["id"])
        train = job["config"]["train"]
        assert train["steps"] * train["batch_size"] * train["grad_accum"] == 64000
        assert train["schedule_steps"] * job["effective_batch"] == 320000
        assert train["val_every"] * job["effective_batch"] == 8000
        assert train["batch_size"] == 2 and train["exact_resume"]
        assert train["keep_only_best_and_last"] and train["precision"] == "float32"
        assert job["config"]["model"]["dim_model"] == source["config"]["model"]["dim_model"]
        assert job["config"]["model"]["chunk_size"] == 25
    assert plan["physical_grasp_success"] == "not_evaluated"
    assert original == matrix()


def test_forged_alias_and_unequal_exposure_are_rejected(tmp_path):
    original = matrix()
    original["arms"][-1]["reuse"] = original["arms"][0]["id"]
    with pytest.raises(ValueError, match="alias differs"):
        compile_plan(original, tmp_path, device="cpu", physical_batch=2)
    original = matrix()
    original["arms"][0]["examples"] = 63_999
    with pytest.raises(ValueError, match="fixed exposure"):
        compile_plan(original, tmp_path, device="cpu", physical_batch=2)


def test_missing_geometry_requires_explicit_override_and_never_passes_evidence(tmp_path):
    runner = fixture_runner(tmp_path)
    with pytest.raises(ValueError, match="defer-geometry-checks"):
        runner.prepare()
    runner.args.defer_geometry_checks = True
    runner.prepare()
    assert runner.plan["geometry"]["deferred_by_user"]
    assert runner.plan["geometry"]["gate_failures"]
    assert not runner.plan["geometry"]["gate_passed"]
    assert runner.status["physical_grasp_success"] == "not_evaluated"


def test_preparation_is_restartable_but_frozen_config_mutation_is_rejected(tmp_path):
    runner = fixture_runner(tmp_path, "--defer-geometry-checks")
    runner.prepare()
    ScreenRunner(runner.args).prepare()
    runner.args.task.write_text("calibration: changed\n")
    with pytest.raises(ValueError, match="input changed"):
        runner.verify_inputs()
    with pytest.raises(ValueError, match="inputs/configs changed"):
        ScreenRunner(runner.args).prepare()


def test_preparation_verifies_payload_hash_and_reserved_split(tmp_path):
    runner = fixture_runner(tmp_path, "--defer-geometry-checks")
    plan = compile_plan(matrix(), runner.args.data_root, device="cpu", physical_batch=2)
    plan["jobs"] = [job for job in plan["jobs"] if job["id"] == "L1_resnet_iphone"]
    data = runner.args.data_root / "cartesian_iphone"
    (data / "payload.bin").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        inspect_inputs(plan, [])
    manifest = read_json(data / "manifest.json")
    manifest["files_sha256"]["payload.bin"] = hash_file(data / "payload.bin")
    manifest["split"]["reserved"] = [0]
    atomic_json(data / "manifest.json", manifest)
    with pytest.raises(ValueError, match="leaks"):
        inspect_inputs(plan, [])


def test_checkpoint_discovery_ignores_partial_writes(tmp_path):
    job = compile_plan(matrix(), tmp_path, device="cpu", physical_batch=2)["jobs"][0]
    root = tmp_path / "run"
    complete = root / "attempt-001/checkpoints/000500"
    complete.mkdir(parents=True)
    for filename in ("model.safetensors", "training_state.pt"):
        (complete / filename).write_text("saved")
    atomic_json(complete / "run_info.json", {"examples_seen": 8000})
    partial = root / "attempt-002/checkpoints/001000"
    partial.mkdir(parents=True)
    atomic_json(partial / "run_info.json", {"examples_seen": 16000})
    assert latest_checkpoint(job, root) == (complete, 8000)


def test_dry_run_launches_nothing_and_records_no_completions(tmp_path, monkeypatch):
    runner = fixture_runner(tmp_path, "--defer-geometry-checks", "--dry-run")
    monkeypatch.setattr(runner, "command", lambda *a, **k: pytest.fail("launched in dry run"))
    assert runner.execute() == 0
    assert runner.status["status"] == "prepared"
    assert runner.status["completed_unique_jobs"] == 0


def test_command_timeout_kills_child_and_records_real_returncode(tmp_path):
    runner = fixture_runner(tmp_path, "--defer-geometry-checks", "--runtime-seconds", "1")
    runner.prepare()
    result = runner.command([sys.executable, "-c", "import time; time.sleep(60)"],
                            tmp_path / "timeout.log", stage="timeout")
    assert result["timed_out"] and result["returncode"] != 0
    assert result["wall_s"] < 15
    assert runner.active is None


def test_exact_exposure_and_no_duplicate_training_on_restart(tmp_path, monkeypatch):
    runner = fixture_runner(tmp_path, "--defer-geometry-checks")
    calls = []

    def fake_command(command, log, *, stage):
        calls.append(command)
        output = Path(command[command.index("--out") + 1])
        atomic_json(output / "training_result.json", {"steps": 4000})
        return {"command": command, "returncode": 0, "timed_out": False}

    monkeypatch.setattr(runner, "command", fake_command)
    assert runner.execute() == 0
    assert runner.status["status"] == "completed"
    assert runner.execute() == 0
    assert len(calls) == 1
    assert runner.status["completed_unique_jobs"] == 1


def test_incomplete_training_does_not_claim_finished_experiment(tmp_path, monkeypatch):
    runner = fixture_runner(tmp_path, "--defer-geometry-checks")

    def fake_command(command, log, *, stage):
        output = Path(command[command.index("--out") + 1])
        atomic_json(output / "training_result.json", {"steps": 500})
        return {"command": command, "returncode": 0, "timed_out": False}

    monkeypatch.setattr(runner, "command", fake_command)
    assert runner.execute() == 1
    assert runner.status["status"] == "incomplete"
    assert runner.status["completed_unique_jobs"] == 0


def test_cached_head_commands_share_schedule_and_match_accumulation(tmp_path):
    runner = fixture_runner(tmp_path, "--defer-geometry-checks")
    plan = compile_plan(matrix(), tmp_path, device="cpu", physical_batch=2)
    jobs = [job for job in plan["jobs"] if job["family"] != "act"]
    assert len(jobs) == 4
    for job in jobs:
        command = runner.train_command(job, tmp_path / job["id"], tmp_path / "resume", tmp_path / "cache")
        values = dict(zip(command[3::2], command[4::2], strict=False))
        assert values["--examples"] == "64000"
        assert values["--schedule-examples"] == "320000"
        assert int(values["--batch-size"]) * int(values["--grad-accum"]) == job["effective_batch"]
        assert command[-2:] == ["--resume", str(tmp_path / "resume")]


def test_completed_storage_release_preserves_weights_and_unfinished_optimizer(tmp_path):
    runner = fixture_runner(tmp_path, "--defer-geometry-checks")
    runner.prepare()
    job = runner.plan["jobs"][0]
    output = runner.out / "runs" / job["id"] / "attempt-001"
    checkpoint = output / "checkpoints/004000"
    checkpoint.mkdir(parents=True)
    for name in ("model.safetensors", "training_state.pt", "run_info.json"):
        (checkpoint / name).write_text("retained")
    atomic_json(checkpoint / "run_info.json", {"step": 4000, "examples_seen": 64000, "best_step": 4000})
    runner.status["jobs"][job["id"]] = {"status": "incomplete", "output": str(output)}
    with pytest.raises(ValueError, match="completed full-budget"):
        runner.release_completed_storage(job)
    assert (checkpoint / "training_state.pt").is_file()
    atomic_json(output / "training_result.json", {"steps": 4000})
    runner.status["jobs"][job["id"]]["status"] = "completed"
    runner.release_completed_storage(job)
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "run_info.json").is_file()
    assert not (checkpoint / "training_state.pt").exists()
    assert runner.status["jobs"][job["id"]]["optimizer_pruned"]
    assert not runner.status["jobs"][job["id"]]["exact_resume_after_screen"]


def test_head_checkpoint_retention_preserves_best_last_and_unrelated_files(tmp_path):
    from scripts.train_flow_head import prune_head_checkpoints

    checkpoints = tmp_path / "checkpoints"
    for name in ("000008000", "000016000", "000024000"):
        path = checkpoints / name
        path.mkdir(parents=True)
        (path / "model.safetensors").write_text(name)
    unrelated = checkpoints / "review.txt"
    unrelated.write_text("user note")
    assert prune_head_checkpoints(checkpoints, 8000, 24000) == ["000016000"]
    assert (checkpoints / "000008000/model.safetensors").is_file()
    assert (checkpoints / "000024000/model.safetensors").is_file()
    assert unrelated.read_text() == "user note"
    (checkpoints / "000032000").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="unexpected"):
        prune_head_checkpoints(checkpoints, 8000, 24000)


def test_shared_training_lock_prevents_a_second_queue_with_different_output(tmp_path, monkeypatch):
    import scripts.run_v2_learning_screen as module
    from scripts.run_so101_experiments import queue_lock

    runner = fixture_runner(tmp_path, "--defer-geometry-checks")
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "parse_args", lambda argv: runner.args)
    called = []
    monkeypatch.setattr(module.ScreenRunner, "execute", lambda self: called.append(True) or 0)
    with queue_lock(tmp_path / "outputs/v2/training_lock"):
        with pytest.raises(RuntimeError, match="already owns"):
            module.main([])
    assert not called
    assert module.main([]) == 0
    assert called == [True]


def test_act_full_checkpoint_crash_recovers_report_and_best_across_attempts(tmp_path, monkeypatch):
    runner = fixture_runner(tmp_path, "--defer-geometry-checks")
    runner.prepare()
    job = runner.plan["jobs"][0]
    root = runner.out / "runs" / job["id"]
    for attempt, step, examples in (("attempt-001", 500, 8000), ("attempt-002", 4000, 64000)):
        path = root / attempt / "checkpoints" / f"{step:06d}"
        path.mkdir(parents=True)
        for name in ("model.safetensors", "training_state.pt"):
            (path / name).write_text("saved")
        atomic_json(path / "run_info.json", {"step": step, "examples_seen": examples,
                    "best_step": 500, "best_val_l1": .2, "last_val": {"l1": .25}})
    monkeypatch.setattr(runner, "command", lambda *a, **k: pytest.fail("recovery trained again"))
    runner.run_job(job)
    row = runner.status["jobs"][job["id"]]
    assert row["status"] == "completed"
    assert row["best_checkpoint"] == str(root / "attempt-001/checkpoints/000500")
    recovered = read_json(root / "attempt-002/training_result.json")
    assert recovered["additional_training_examples"] == 0 and recovered["steps"] == 4000
    assert recovered["last_val"] == {"l1": .25}


def test_full_budget_head_checkpoint_is_finalized_without_more_examples(tmp_path, monkeypatch):
    runner = fixture_runner(tmp_path, "--defer-geometry-checks")
    runner.prepare()
    job = runner.plan["jobs"][0]
    job["family"] = "regression"
    job["config"]["model"]["vision_backbone"] = "dinov2"
    checkpoint = runner.out / "runs" / job["id"] / "attempt-001/checkpoints/000064000"
    checkpoint.mkdir(parents=True)
    for name in ("model.safetensors", "training_state.pt"):
        (checkpoint / name).write_text("saved")
    atomic_json(checkpoint / "run_info.json", {"examples_seen": 64000})
    monkeypatch.setattr(runner, "ensure_cache", lambda job: tmp_path / "cache")

    def fake_finalize(command, log, *, stage):
        assert command[-1] == "--finalize-only"
        assert command[command.index("--resume") + 1] == str(checkpoint)
        output = Path(command[command.index("--out") + 1])
        atomic_json(output / "report.json", {"examples": 64000})
        atomic_json(output / "COMPLETE.json", {"completed": True})
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(runner, "command", fake_finalize)
    runner.run_job(job)
    assert runner.status["jobs"][job["id"]]["status"] == "completed"


def test_finalize_only_rejects_missing_resume_before_accessing_data():
    from argparse import Namespace

    from scripts.train_flow_head import train_head

    with pytest.raises(ValueError, match="full-budget resume"):
        train_head(Namespace(finalize_only=True, resume=None))
