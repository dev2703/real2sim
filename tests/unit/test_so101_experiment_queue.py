"""Equal exposure, selection guardrails and safe terminal-stage dispatch."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("so101_queue", Path(__file__).parents[2] / "scripts/run_so101_experiments.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def test_effective_batch_preserves_exposure_and_schedule():
    small = queue.act_config("resnet18", Path("data"), "mps", effective_batch=16)
    large = queue.act_config("resnet18", Path("data"), "mps", effective_batch=64, dim=384, heads=12)
    assert small["train"]["steps"] * 16 == large["train"]["steps"] * 64 == 32000
    assert small["scheduler"]["warmup_steps"] * 16 == large["scheduler"]["warmup_steps"] * 64
    assert large["train"]["batch_size"] == 16 and large["train"]["grad_accum"] == 4
    assert large["model"]["dim_model"] % large["model"]["n_heads"] == 0
    assert not {"initial_checkpoint", "paired_images", "initialization"} & large.keys()
    assert large["model"]["n_encoder_layers"] == 4
    assert large["model"]["n_decoder_layers"] == 1


def score(active=.004, stationary=.0005, closure=.004, grip=.1):
    rows = {}
    for name, value in (("arm_active", active), ("arm_stationary", stationary),
                        ("observed_gripper_closure_proxy", closure)):
        rows[name] = {"n": 10, "translation_vector_error_m": {"mean": value},
                      "mae_raw_per_dim": [0] * 6 + [grip]}
    return {"episodes": [1], "valid_actions": 20, "split": "val", "scores": {"policy": rows}}


def test_backbone_selection_requires_active_improvement_and_guardrails():
    assert queue.choose_backbone(score(), score(active=.003))["winner"] == "dinov2"
    assert queue.choose_backbone(score(), score(stationary=.0001))["winner"] == "resnet18"
    assert queue.choose_backbone(score(), score(active=.003, grip=.2))["winner"] == "resnet18"
    other = score(active=.003)
    other["split"] = "historical_val"
    with pytest.raises(ValueError, match="inner validation"):
        queue.choose_backbone(score(), other)


def test_stage_boundary_never_writes_whole_queue_completion(tmp_path):
    args = SimpleNamespace(out=tmp_path, data_root=tmp_path / "data", flow_cache=tmp_path.parent / "cache",
                           stop_after="A1D", publish_repo=None, notification_json=None)
    runner = queue.Runner(args)
    assert runner.stop_at("A1D")
    report = queue.read_json(tmp_path / "STAGE_COMPLETE.json")
    assert not report["all_experiments_complete"]
    assert not report["panda_gate_passed"]
    assert not (tmp_path / "COMPLETE.json").exists()


def test_changed_frozen_file_is_rejected_before_launch(tmp_path):
    args = SimpleNamespace(out=tmp_path, data_root=tmp_path / "data", flow_cache=tmp_path.parent / "cache")
    runner = queue.Runner(args)
    source = tmp_path / "source.py"
    source.write_text("old")
    runner.frozen_files = {str(source): queue.hash_file(source)}
    source.write_text("new")
    with pytest.raises(ValueError, match="frozen experiment input changed"):
        runner.verify_inputs()
