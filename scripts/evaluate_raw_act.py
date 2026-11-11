"""Evaluate native ACT without controller handoffs; require 20/20 on both arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real2sim.evaluation.raw_act import DEFAULT_SCENE, ROBOTS, audit_report, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--source-scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--robots", choices=ROBOTS, nargs="+", default=list(ROBOTS))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=15000)
    parser.add_argument("--training-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--validation-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--execute-steps", type=int)
    parser.add_argument("--initial-gripper-percent", type=float, default=1.2)
    parser.add_argument("--domain-randomization", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.audit_report is not None:
        print(json.dumps(audit_report(args.audit_report)))
        return
    if args.snapshot is None or args.out is None:
        parser.error("--snapshot and --out are required for evaluation")
    if args.count < 1 or args.seed_start < 0:
        parser.error("--count must be positive and --seed-start nonnegative")
    report = run(args.out, snapshot=args.snapshot, source_scene=args.source_scene,
        robots=args.robots, seeds=tuple(range(args.seed_start, args.seed_start + args.count)),
        execute_steps=args.execute_steps, initial_gripper_percent=args.initial_gripper_percent,
        training_seeds=args.training_seeds, validation_seeds=args.validation_seeds,
        domain_randomization=args.domain_randomization, plan_only=args.plan_only)
    print(json.dumps(report.get("gate", {"protocol_id": report["protocol_id"]})))


if __name__ == "__main__":
    main()
