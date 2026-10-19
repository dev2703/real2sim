import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_so101_experiments import (
    Runner,
    act_config,
    atomic_json,
    choose_backbone,
    compact_act_run,
    parse_args,
    queue_lock,
    read_json,
)


def make_runner(tmp_path, *extra):
    return Runner(parse_args(["--device", "mps", "--out", str(tmp_path / "queue"),
                              "--flow-cache", str(tmp_path / "features"), *extra]))


def sample_audit(active=.004, stationary=.0005, closure=.003, gripper=.05):
    return {"split": "val", "episodes": [15, 20], "valid_actions": 200,
            "scores": {"policy": {
                phase: {"n": 50, "translation_vector_error_m": {"mean": value},
                        "mae_raw_per_dim": [0.] * 6 + [gripper]}
                for phase, value in (("arm_active", active), ("arm_stationary", stationary),
                                      ("observed_gripper_closure_proxy", closure))}}}


def test_fresh_configs_match_anchor_exposure_and_schedule():
    small = act_config("resnet18", Path("example"), "mps")
    large = act_config("resnet18", Path("example"), "cuda", effective_batch=64, dim=384, heads=12)
    assert small["train"]["steps"] == 2000 and large["train"]["steps"] == 500
    for key in ("val_every", "save_every", "log_every"):
        assert small["train"][key] * 16 == large["train"][key] * 64
    assert small["scheduler"]["warmup_steps"] * 16 == large["scheduler"]["warmup_steps"] * 64
    assert large["train"]["precision"] == "bfloat16"
    assert small["train"]["precision"] == "float32"
    assert large["model"]["n_encoder_layers"] == 4
    assert large["model"]["n_decoder_layers"] == 1
    assert large["model"]["chunk_size"] == 20
    assert large["model"]["n_action_steps"] == 2
    assert large["model"]["dim_model"] == 384 and large["model"]["n_heads"] == 12
    for cfg in (small, large, act_config("dinov2", Path("example"), "cuda")):
        assert cfg["model"]["freeze_backbone"]
        assert cfg["model"]["attention_backend"] == "sdpa"
        assert not {"initial_checkpoint", "paired_images", "augmentation"} & cfg.keys()
        assert cfg["model"]["optimizer_lr"] == 1e-4


def test_backbone_selection_requires_active_improvement_and_all_guardrails():
    resnet, dino = sample_audit(), sample_audit(active=.0039)
    assert choose_backbone(resnet, dino)["winner"] == "dinov2"
    for candidate in (sample_audit(active=.004), sample_audit(active=.003, stationary=.0006),
                      sample_audit(active=.003, closure=.004), sample_audit(active=.003, gripper=.06)):
        assert choose_backbone(resnet, candidate)["winner"] == "resnet18"
    invalid = copy.deepcopy(dino)
    invalid["split"] = "historical_val"
    with pytest.raises(ValueError, match="inner validation"):
        choose_backbone(resnet, invalid)
    invalid = copy.deepcopy(dino)
    invalid["episodes"] = [100]
    with pytest.raises(ValueError, match="same episodes"):
        choose_backbone(resnet, invalid)


def owned_checkpoints(tmp_path):
    run = tmp_path / "owned"
    atomic_json(run / "RUNNER_OWNED.json", {"token": "my-token"})
    for number in ("000256", "002000"):
        checkpoint = run / "checkpoints" / number
        checkpoint.mkdir(parents=True)
        for name in ("model.safetensors", "config.json", "stats.json", "real2sim_contract.json",
                     "run_info.json", "training_state.pt"):
            (checkpoint / name).write_text(name)
    (run / "checkpoints" / "best").symlink_to("000256")
    (run / "checkpoints" / "last").symlink_to("002000")
    return run


def test_compaction_preserves_best_weights_and_unrelated_files(tmp_path):
    run = owned_checkpoints(tmp_path)
    unrelated = run / "checkpoints" / "review.txt"
    unrelated.write_text("user note")
    report = compact_act_run(run, "my-token")
    assert report["inference_only"] and not report["resumable"]
    assert (run / "checkpoints/best/model.safetensors").read_text() == "model.safetensors"
    assert not (run / "checkpoints/best/training_state.pt").exists()
    assert not (run / "checkpoints/002000").exists()
    assert unrelated.read_text() == "user note"


def test_compaction_rejects_unowned_or_external_symlink_before_deleting(tmp_path):
    run = owned_checkpoints(tmp_path)
    with pytest.raises(ValueError, match="not owned"):
        compact_act_run(run, "wrong")
    target = run / "checkpoints" / "best"
    target.unlink()
    outside = tmp_path / "outside" / "001000"
    outside.mkdir(parents=True)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        compact_act_run(run, "my-token")
    assert (run / "checkpoints/000256/training_state.pt").exists()
    assert (run / "checkpoints/002000").exists()


def test_process_flags_and_optional_failure_preserve_queue(tmp_path):
    runner = make_runner(tmp_path)
    command = [sys.executable, "-c", "print('verified')"]
    assert runner.command("works", command)
    result = read_json(runner.out / "stages/works/COMPLETE.json")
    assert result["subprocess_pid"] and result["returncode"] == 0
    assert not (runner.out / "stages/works/RUNNING.json").exists()
    assert runner.command("works", command)  # idempotent; no duplicate process
    assert not runner.command("optional_oom", [sys.executable, "-c", "raise SystemExit(3)"], optional=True)
    assert (runner.out / "stages/optional_oom/FAILED.json").exists()
    with pytest.raises(subprocess.CalledProcessError):
        runner.command("required", [sys.executable, "-c", "raise SystemExit(2)"])
    assert (runner.out / "stages/required/FAILED.json").exists()
    assert not (runner.out / "stages/required/COMPLETE.json").exists()


def test_queue_lock_prevents_duplicate_live_runner(tmp_path):
    with queue_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with queue_lock(tmp_path):
                pass
    with queue_lock(tmp_path):
        pass


def test_stop_boundary_never_claims_full_completion(tmp_path):
    runner = make_runner(tmp_path, "--stop-after", "A1D")
    assert not runner.stop_at("A1R")
    assert runner.stop_at("A1D")
    result = read_json(runner.out / "STAGE_COMPLETE.json")
    assert result["through"] == "A1D" and not result["all_experiments_complete"]
    assert not result["panda_gate_passed"]
    assert not (runner.out / "COMPLETE.json").exists()


def test_dry_run_writes_configs_without_launching_or_claiming_completion(tmp_path, monkeypatch):
    root = tmp_path / "data"
    variants = ["canonical", "no_velocity", "ema_velocity_05", "ema_velocity_075", "joints", "canonical_joints", "history3"]
    for name in variants:
        atomic_json(root / name / "manifest.json", {"kind": name, "files_sha256": {}})
    runner = make_runner(tmp_path, "--dry-run", "--data-root", str(root))
    monkeypatch.setattr(runner, "command", lambda *a, **k: pytest.fail("dry run launched a subprocess"))
    runner.execute()
    assert read_json(runner.out / "PLAN_READY.json")["status"] == "NOT_STARTED"
    assert not (runner.out / "COMPLETE.json").exists()
    assert not (runner.out / "RUNNING.json").exists()
    assert (runner.out / "configs/A1D.yaml").exists()


def test_callback_delivers_once_per_terminal_event(tmp_path, monkeypatch):
    callback = tmp_path / "notification.json"
    callback.write_text(json.dumps({"codex_bin": "/fake/codex", "thread_id": "thread"}))
    runner = make_runner(tmp_path, "--notification-json", str(callback))
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "queued", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    runner.notify("stage_A1D_complete")
    runner.notify("stage_A1D_complete")
    assert len(calls) == 1
    assert calls[0][:4] == ["/fake/codex", "queue", "--thread", "thread"]


def test_cli_rejects_unsupported_mps_batch_and_wrong_budget(tmp_path):
    with pytest.raises(SystemExit):
        make_runner(tmp_path, "--profile-batches", "16", "64")
    with pytest.raises(SystemExit):
        make_runner(tmp_path, "--examples", "100")
    with pytest.raises(SystemExit):
        make_runner(tmp_path, "--skip-flow", "--stop-after", "F0")
