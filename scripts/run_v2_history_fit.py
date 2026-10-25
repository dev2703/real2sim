"""One controlled history-four prefix fit with function-preserving initialization."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml
from run_v2_three_episode_fit import (
    EPISODES,
    GATE_VERSION,
    audit_training_checkpoint,
    gate_from_validation,
)

from real2sim.config import load_yaml
from real2sim.data.raw import sha256_file
from real2sim.policy.act import build_act_config, build_policy, save_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.train import seed_everything, train

STATE_WEIGHT = "model.encoder_robot_state_input_proj.weight"


def expand_state_weights(reference, candidate, history):
    """Copy every shared tensor; older observations/age/valid start with zero weight."""
    if set(reference) != set(candidate):
        raise ValueError("history model parameters differ beyond state projection")
    expanded = {}
    for name, value in reference.items():
        shape = candidate[name].shape
        if name == STATE_WEIGHT:
            if value.ndim != 2 or value.shape[1] != 13 or shape != (value.shape[0], 15 * history):
                raise ValueError("unexpected canonical/history state projection")
            output = torch.zeros_like(candidate[name])
            offset = (history - 1) * 15
            output[:, offset:offset + 13] = value
            expanded[name] = output
        else:
            if value.shape != shape:
                raise ValueError(f"unexpected architecture difference: {name}")
            expanded[name] = value.clone()
    if STATE_WEIGHT not in expanded:
        raise ValueError("missing state projection")
    return expanded


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(str((name, tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialize(cfg, data, source, directory):
    original_manifest = json.loads((source / "manifest.json").read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    variant = manifest["observation_variant"]
    if variant["kind"] != "canonical" or variant["history"] != 4 or variant["policy_hz"] != 12.5:
        raise ValueError("requires the declared four-observation canonical history")
    canonical = DerivedTickDataset(source, list(EPISODES), 25)
    history = DerivedTickDataset(data, list(EPISODES), 25)
    if canonical.normalizer.stats["action"] != history.normalizer.stats["action"]:
        raise ValueError("action normalization changed")
    seed_everything(int(cfg["train"]["seed"]))
    reference = build_policy(build_act_config(cfg["model"], original_manifest, "cpu")).eval()
    candidate = build_policy(build_act_config(cfg["model"], manifest, "cpu")).eval()
    candidate.load_state_dict(expand_state_weights(reference.state_dict(), candidate.state_dict(), 4), strict=True)
    differences = []
    for episode in EPISODES:
        index = next(i for i, tick in enumerate(canonical.indices)
                     if canonical.arrays["episode_index"][tick] == episode)
        a, b = canonical[index], history[index]
        assert int(a["tick"]) == int(b["tick"])
        a = {key: value[None] for key, value in a.items() if key.startswith("observation.")}
        b = {key: value[None] for key, value in b.items() if key.startswith("observation.")}
        with torch.inference_mode():
            old, new = reference.predict_action_chunk(a), candidate.predict_action_chunk(b)
        torch.testing.assert_close(old, new, atol=2e-6, rtol=2e-6)
        differences.append(float((old - new).abs().max()))
    proof = {"method": "Canonical seed0 initialization copied into shared tensors; current-state projection copied; older slots/metadata start at zero.",
             "canonical_initial_state_sha256": state_digest(reference.state_dict()),
             "expanded_initial_state_sha256": state_digest(candidate.state_dict()),
             "parity_episodes": list(EPISODES), "max_abs_normalized_prediction_difference": max(differences),
             "architecture_change": "State projection 13 to60 inputs; backbone and all shared initial tensors identical.",
             "physical_calibration_accepted": False}
    save_checkpoint(directory, candidate, data / "stats.json", manifest,
                    {"step": 0, "seed": cfg["train"]["seed"], "history_initialization": proof})
    return proof


def run(data, out, resume=False):
    if out.exists() and not resume:
        raise FileExistsError(out)
    baseline = Path("outputs/v2/three_episode_prefix2_20261025")
    cfg = copy.deepcopy(load_yaml(baseline / "plan.yaml"))
    cfg.update(name="three_episode_prefix2_history4", data=str(data.resolve()),
               initial_checkpoint=str((out / "initial_checkpoint").resolve()))
    cfg["train"]["steps"] = 2000
    cfg["experiment"] = {"kind": "causal_history4_capacity_v1", "maximum_examples": 32000,
        "conditions": {"11": "baseline", "8": "ball_shift", "23": "bowl_shift"},
        "scope": "Training fit under provisional FK; four causal state observations; no geometry or task-success acceptance.",
        "comparison": "Same data labels/images/action normalization/backbone/shared initial weights, prefix2 objective, seed0, optimizer and 2000-update horizon. State input expands13to60; no additional loss or output gate.",
        "stopping": "Fixed2000updates/32000examples; evaluate finalcheckpoint with unchanged per-condition gate. No outcome-conditioned extra run."}
    torch.set_num_threads(2)
    dependencies = [Path(__file__), Path("scripts/prepare_v2_history.py"), Path("scripts/run_v2_three_episode_fit.py"),
                    *[Path("src/real2sim/policy") / name for name in
                      ("train.py", "act.py", "dataset.py", "observation_variants.py", "adapters.py", "validation.py", "losses.py")],
                    data / "manifest.json", data / "stats.json", baseline / "plan.yaml"]
    if resume:
        if load_yaml(out / "plan.yaml") != cfg:
            raise ValueError("resume plan differs")
        report = json.loads((out / "report.json").read_text())
        for path, digest in report["source_sha256"].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"resume dependency changed: {path}")
        checkpoint = (out / "training/checkpoints/last").resolve()
        report.setdefault("resumes", []).append({"checkpoint": str(checkpoint),
            "training_state_sha256": sha256_file(checkpoint / "training_state.pt")})
    else:
        out.mkdir(parents=True)
        (out / "executed_runner.py").write_bytes(Path(__file__).read_bytes())
        (out / "plan.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        report = {"schema": "history4_capacity_run_v1", "gate_version": GATE_VERSION,
                  "training_episodes": list(EPISODES), "reserved_episodes_used": [],
                  "physical_calibration_accepted": False, "checkpoint_promoted": False,
                  "source_sha256": {str(p): sha256_file(p) for p in dependencies},
                  "initialization": initialize(cfg, data, Path(load_yaml(baseline / "plan.yaml")["data"]), out / "initial_checkpoint")}
        checkpoint = None
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    training_cfg = copy.deepcopy(cfg)
    if checkpoint is not None:
        training_cfg.pop("initial_checkpoint")
    result = train(training_cfg, data, out / "training", resume=checkpoint)
    checkpoint = (out / "training/checkpoints/last").resolve()
    audited = audit_training_checkpoint(checkpoint, data, result["device"])
    path = out / "training_motion_002000.json"
    path.write_text(json.dumps(audited, indent=2, allow_nan=False) + "\n")
    report.update(training_result=result, checkpoint=str(checkpoint),
                  checkpoint_sha256=sha256_file(checkpoint / "model.safetensors"),
                  training_motion_audit=str(path.resolve()), gate=gate_from_validation(audited),
                  total_examples_seen=32000, complete=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"gate": report["gate"], "initialization": report["initialization"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".cache/v2_history4_20261025"))
    parser.add_argument("--out", type=Path, default=Path("outputs/v2/three_episode_history4_20261025"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.data, args.out, args.resume)
