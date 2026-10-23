"""Matched support-height sensitivity; never choose physical geometry by success.

This creates separate provisional scenes. It does not adopt any support dimensions
or modify a calibrated task. Grasp control is privileged; physics, gains, native
limits, collision acceptance and task scoring remain fixed across scenes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml
from diag_so101_long_hold import trial

from real2sim.config import load_yaml


def variants(config, out, heights_mm):
    base = load_yaml(config)
    task = load_yaml(base["task_config"])
    for height in heights_mm:
        scene, geometry = copy.deepcopy(base), copy.deepcopy(task)
        if height == 0:
            geometry["ball"].pop("support", None)
        else:
            geometry["ball"]["support"] = {"type": "box", "height": height / 1000,
                "size_xy": [.10, .10],
                "center_task_xy": geometry["ball"]["start_region"]["center_task_xy"]}
        directory = out / f"support_{height:02d}mm"
        directory.mkdir()
        task_path = directory / "task.yaml"
        task_path.write_text(yaml.safe_dump(geometry, sort_keys=False))
        scene["task_config"] = str(task_path)
        scene_path = directory / "sim.yaml"
        scene_path.write_text(yaml.safe_dump(scene, sort_keys=False))
        yield height, scene_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heights-mm", type=int, nargs="+", default=[0, 5, 10, 20])
    parser.add_argument("--grasp-x-mm", type=int, nargs="+", default=[-10, -20])
    parser.add_argument("--grasp-tilts-deg", type=float, nargs="+", default=[10, 15, 20, 25])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--modes", nargs="+", default=["fixed_close"],
                        choices=["fixed_close", "encoder_width"])
    parser.add_argument("--seconds", type=float, default=30.)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.seconds < 20 or min(args.heights_mm) < 0:
        parser.error("requires at least 20s and nonnegative support heights")
    args.out.mkdir(parents=True)
    sources = [Path(__file__), Path("scripts/diag_so101_long_hold.py"),
        Path("scripts/diag_so101_feasible_carry.py"),
        Path("scripts/diag_so101_contact_retention.py"),
        Path("src/real2sim/sim/ball_support.py"), Path("src/real2sim/sim/model.py"),
        Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/kinematics.py"),
        Path("src/real2sim/evaluation/oracle.py"), args.config,
        Path(load_yaml(args.config)["task_config"])]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = {"kind": "PRIVILEGED_SUPPORT_RETENTION_SENSITIVITY_NOT_CALIBRATION",
        "source_sha256_at_start": before,
        "protocol": {"support_heights_mm": args.heights_mm,
            "grasp_x_mm": args.grasp_x_mm, "grasp_z_mm": 30, "seeds": args.seeds,
            "grasp_tilts_deg": args.grasp_tilts_deg,
            "modes": args.modes, "hold_seconds": args.seconds,
            "physical_support_height_verified": False,
            "select_physical_height_by_retention": False,
            "support_model": "rigid 100x100mm box proxy; cloth compliance unmodeled",
            "unchanged": "actuators, gains, friction, limits, collision gates, scoring",
            "min_force_each_jaw_n": .01, "max_ball_tool_displacement_m": .01,
            "min_ball_lift_above_initial_rest_m": .03}, "trials": []}
    path = args.out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for height, config in variants(args.config, args.out, args.heights_mm):
        for depth in args.grasp_x_mm:
            for mode in args.modes:
                for seed in args.seeds:
                    result = trial(config, seed, mode=mode, seconds=args.seconds,
                                   grasp_x=depth / 1000, grasp_tilts_deg=args.grasp_tilts_deg)
                    result["support_height_mm"] = height
                    report["trials"].append(result)
                    path.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps({"height_mm": height, "grasp_x_mm": depth,
                        "mode": mode, "seed": seed, "acquired": result["hold_acquired"],
                        "hold": result.get("hold"), "candidates":
                        result["grasp_search"]["candidate_count"]}), flush=True)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report["source_changed_during_run"] = [p for p in before if before[p] != after[p]]
    report["source_sha256_at_end"] = after
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
