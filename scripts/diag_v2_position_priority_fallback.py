"""Matched learned rollout with a numerical IK fallback and unchanged hard gates.

The fallback retries only residual failures at 1/4 and 1/16 of the configured
orientation optimization weight. It never changes the requested target, pose
tolerances, workspace, joint limits, collision checks or simulator physics.
This diagnostic does not install a production controller change.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from real2sim.policy.paired import file_sha256
from real2sim.sim.kinematics import NativeKinematics


class PositionPriorityFallback(NativeKinematics):
    instances = []
    scales = (.25, .0625)

    def __init__(self, model, config):
        super().__init__(model, config)
        self.retries = []
        for scale in self.scales:
            candidate = copy.deepcopy(config)
            candidate["ik"]["orientation_weight_m_per_rad"] *= scale
            self.retries.append(NativeKinematics(model, candidate))
        self.trace = []
        self.instances.append(self)

    @staticmethod
    def describe(result, seed):
        return {"reason": result.reason, "accepted": result.accepted,
                "position_error_m": result.position_error_m,
                "orientation_error_rad": result.orientation_error_rad,
                "maximum_required_joint_step_rad": float(np.max(np.abs(result.qpos[:5] - seed[:5])))}

    def solve(self, target, seed, *, gripper=None):
        initial = super().solve(target, seed, gripper=gripper)
        row = {"primary": self.describe(initial, seed), "retries": []}
        result = initial
        if initial.reason == "residual_rejected":
            for scale, solver in zip(self.scales, self.retries, strict=True):
                attempt = solver.solve(target, seed, gripper=gripper)
                row["retries"].append({"optimization_weight_scale": scale,
                                       **self.describe(attempt, seed)})
                if attempt.accepted:
                    result = attempt
                    break
        row["fallback_recovered"] = result.accepted and not initial.accepted
        self.trace.append(row)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/v2/learning_screen_20261022/runs/L1_resnet_iphone/attempt-001/checkpoints/004000"))
    parser.add_argument("--config", type=Path, default=Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("configs/v2/eval_so101.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import real2sim.evaluation.run as evaluation

    original = evaluation.NativeKinematics
    script = Path(__file__).with_name("diag_v2_full_policy_rollouts.py")
    spec = importlib.util.spec_from_file_location("diagnostic_full_rollouts", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = sys.argv
    sys.argv = [str(script), "--checkpoint", str(args.checkpoint), "--config", str(args.config),
                "--eval-config", str(args.eval_config), "--seeds", *map(str, args.seeds),
                "--out", str(args.out)]
    evaluation.NativeKinematics = PositionPriorityFallback
    try:
        module.main()
    finally:
        evaluation.NativeKinematics = original
        sys.argv = argv
    path = args.out / "report.json"
    report = json.loads(path.read_text())
    trace = [row for instance in PositionPriorityFallback.instances for row in instance.trace]
    report["kind"] = "FULL_LEARNED_POLICY_NUMERICAL_IK_FALLBACK_ABLATION"
    report["numerical_execution_ablation"] = {
        "optimization_weight_scales": PositionPriorityFallback.scales,
        "original_hard_gates_unchanged": True, "requested_targets_unchanged": True,
        "physics_unchanged": True, "production_source_unchanged": True,
        "privileged_guidance": False,
        "fallback_calls": sum(bool(row["retries"]) for row in trace),
        "fallback_recoveries": sum(row["fallback_recovered"] for row in trace),
        "script_sha256": file_sha256(Path(__file__)),
        "trace": trace,
    }
    report["checkpoint"]["execution_interface_ablation"] = True
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("NUMERICAL_FALLBACK " + json.dumps({key: value for key, value in
          report["numerical_execution_ablation"].items() if key != "trace"}), flush=True)


if __name__ == "__main__":
    main()
