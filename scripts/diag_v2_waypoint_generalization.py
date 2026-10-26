"""Paired bowl-shift development trials for the fixed geometric controller.

This wrapper changes reset layouts only. Goals still come from rendered RGB;
the static collision map is updated to the declared layout by the environment.
It does not test ACT weights, unregistered maps, or physical camera calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


def run(robot: str, out: Path, *, video: bool = False):
    source = Path(__file__).with_name("run_visual_waypoint.py")
    spec = importlib.util.spec_from_file_location("fixed_waypoint_runner", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if robot == "so101":
        from real2sim.sim.env import So101PickPlaceEnv as environment
    elif robot == "panda":
        from real2sim.panda.env import PandaPickPlaceEnv as environment
    else:
        raise ValueError("robot must be so101 or panda")

    seeds = list(range(10, 20))
    # Declared before running any shifted trial. Match ball RNG and bowl offsets
    # across embodiments; no failed example is replaced or excluded.
    offsets = np.array([[-.03, 0.], [.03, 0.], [0., -.03], [0., .03]])
    schedule = {seed: offsets[(seed - 10) % len(offsets)] for seed in seeds}
    original_reset = environment.reset
    actual_scenarios = {}

    def reset_with_layout(env, *, seed=None, scenario=None):
        if scenario is not None or seed not in schedule:
            raise ValueError("diagnostic requires declared seeds and unchanged configured joints")
        bowl_xy = np.asarray(env._bowl_xy_default) + schedule[seed]
        actual_scenarios[seed] = {"bowl_xy": bowl_xy.tolist(),
                                  "bowl_offset_xy_m": schedule[seed].tolist()}
        return original_reset(env, seed=seed, scenario={"bowl_xy": bowl_xy})

    # A scoped wrapper leaves the maintained runner/controller source and
    # configuration bytes unchanged. Original reset owns the actual scene update.
    with patch.object(environment, "reset", reset_with_layout):
        report = runner.run(robot=robot, config=runner.DEFAULT_CONFIGS[robot],
                            seeds=seeds, out=out, video=video, settle_start=False)
    metadata = {
        "schema": "fixed_waypoint_bowl_shift_schedule_v1", "robot": robot,
        "controller_changed": False, "learned_policy": False,
        "frozen_act_transfer_demonstrated": False,
        "seeds": seeds, "schedule": actual_scenarios,
        "paired_comparison": "Same seeded ball positions as the new unshifted seed10..19 runs; bowl offsets repeat -x,+x,-y,+y by30mm.",
        "map_assumption": "Environment reset updates exact static bowl collision geometry; RGB supplies controller goals.",
        "initial_joints": "configured, unchanged; no settling/reset replacement",
        "scoring": "unchanged held lift, released resting placement",
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "full_report": str(out / "report.json"),
        "summary": report["summary"],
    }
    (out / "scenario_protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("so101", "panda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    run(args.robot, args.out, video=args.video)
