"""Test frozen matched-scene ACT with explicit gripper actuator saturation."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import run_v2_matched_scene_act as experiment
import torch

from real2sim.policy.bounded_joint import BoundedGripperJointActionAdapter


def run(checkpoint, scene, out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    sources = [Path(__file__), Path(experiment.__file__),
        Path("src/real2sim/policy/bounded_joint.py"),
        Path("src/real2sim/policy/follower_joint.py"),
        Path("src/real2sim/policy/adapters.py"), scene,
        checkpoint / "model.safetensors"]
    hashes = {str(p.resolve()): experiment.sha256(p) for p in sources}
    report = {"schema": "bounded_gripper_act_20261027_v1", "status": "evaluating",
        "checkpoint": str(checkpoint), "scene": str(scene), "weights_retrained": False,
        "change": "Only gripper target saturated to its unchanged native/percentage actuator interval before original whole-command guards",
        "arm_target_clipping": False, "teacher_or_corrective_ik": False,
        "source_sha256_before": hashes}
    experiment.write_json(out / "report.json", report)
    with patch.object(experiment, "CommandPathGuard", BoundedGripperJointActionAdapter):
        report["development"] = experiment.evaluate(checkpoint, scene, out / "development", range(300, 305))
        report["training_seed_diagnostic"] = experiment.evaluate(checkpoint, scene,
            out / "training_seed", [100])
    report["source_sha256_after"] = {p: experiment.sha256(p) for p in hashes}
    report["runtime_source_changes"] = [p for p in hashes if hashes[p] != report["source_sha256_after"][p]]
    report["status"] = "complete"
    experiment.write_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/v2/matched_scene_act_20261026/training/checkpoints/002000"))
    parser.add_argument("--scene", type=Path, default=Path("configs/v2/sim_so101.waypoint_20261026.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.scene, args.out)
