"""Frozen paired ACT with isolated early-window response adapter; no retraining."""

import argparse
import json
from pathlib import Path

from real2sim.evaluation import run
from real2sim.evaluation.response_adapter import ApproachResponseAdapter
from real2sim.policy.paired import file_sha256
from real2sim.sim.follower_response import FollowerResponse


def main(out):
    if out.exists():
        raise FileExistsError("refusing to overwrite response evaluation")
    out.mkdir(parents=True)
    checkpoint = Path("outputs/act_joint_paired_approach/checkpoints/000500")
    model_path = Path("results/audit/follower_response_approach_20261013/model.json")
    prep_path = Path(".cache/paired_approach_3988577/preparation.json")
    record = json.loads(model_path.read_text())
    model = FollowerResponse(**record["model"])
    prep = json.loads(prep_path.read_text())
    train = set(record["fit_episodes"])
    if train & set(record["primary_split"]["val"]):
        raise ValueError("response fit overlaps held-out episodes")
    if record.get("fit_scope") != "approach":
        raise ValueError("requires the frozen approach-specific response")
    prefixes = [r for r in prep["episodes"] if r["split"] == "train"]
    if {r["episode"] for r in prefixes} != train:
        raise ValueError("training approach provenance mismatch")
    # End at the last known state in the shortest TRAINING stationary-ball prefix.
    # This is a fixed early window, not a claim of online phase recognition.
    active_ticks = min(r["eligible_ticks"] for r in prefixes) - 1
    provenance = {"kind": "FROZEN_ACT_EARLY_WINDOW_RESPONSE_DIAGNOSTIC",
        "active_ticks": active_ticks, "horizon_seconds": active_ticks / 10,
        "horizon_rule": "shortest training stationary-ball prefix minus one tick; fixed before evaluation",
        "transition": "original joint adapter after horizon; native rate/limit clamps retained",
        "velocity_feedback": "last measured policy-tick displacement divided by three raw ticks",
        "response_model_sha256": file_sha256(model_path), "prefixes_sha256": file_sha256(prep_path),
        "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
        "note": "No privileged gate/inputs or new training. Fixed window may not match policy phase. "
                "Online measured-state feedback differs from precomputed response replay; test it explicitly."}
    (out / "protocol.json").write_text(json.dumps(provenance, indent=2) + "\n")
    original = run.build_learned_controller
    adapters = []

    def build(env, checkpoint, eval_cfg, device, robot="so101"):
        ctrl, contract, meta = original(env, checkpoint, eval_cfg, device, robot)
        if meta["kind"] != "joint" or robot != "so101":
            raise ValueError("response diagnostic requires SO-101 joint ACT")
        adapter = ApproachResponseAdapter(env, model, active_ticks=active_ticks)
        ctrl.action_adapter = adapter
        adapters.append(adapter)
        return ctrl, contract, meta

    # Local process-only dependency replacement; production evaluation stays untouched.
    run.build_learned_controller = build
    try:
        result = run.run_trials(robot="so101", checkpoint=checkpoint,
            config=Path("results/metrics/geometry_3988577/configs/sim_so101.yaml"),
            eval_config=Path("results/metrics/geometry_3988577/configs/eval_so101.yaml"),
            n_trials=10, seed_start=0, out=out, video=False, controller="learned", device="cpu",
            experiment_id="DIAGNOSTIC_PAIRED_ACT_EARLY_RESPONSE")
        if file_sha256(model_path) != provenance["response_model_sha256"]:
            raise RuntimeError("response model changed during evaluation")
        if file_sha256(checkpoint / "model.safetensors") != provenance["checkpoint_sha256"]:
            raise RuntimeError("ACT checkpoint changed during evaluation")
        print(json.dumps(result["summary"]))
    finally:
        run.build_learned_controller = original
        for i, adapter in enumerate(adapters):
            (out / f"response_traces_{i}.json").write_text(json.dumps(adapter.episodes) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/metrics/act_response_20261013"))
    main(parser.parse_args().out)
