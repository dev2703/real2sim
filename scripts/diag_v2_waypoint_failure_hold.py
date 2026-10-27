"""Test closed-gripper failure hold without modifying the fixed controller source."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from real2sim.artifacts import sha256_file as digest
from real2sim.policy import visual_waypoint
from real2sim.policy.waypoint_failure_hold import FailureHoldWaypointController


def run(out: Path):
    source = Path(__file__).with_name("run_visual_waypoint.py")
    spec = importlib.util.spec_from_file_location("failure_hold_runner", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with patch.object(visual_waypoint, "VisualWaypointController", FailureHoldWaypointController):
        report = runner.run(robot="so101", config=Path("configs/v2/positive_setup_20261026/sim_so101.yaml"),
            seeds=[0, 1, 2, 3, 4], out=out, video=True, settle_start=False)
    retained = []
    for trial in report["trials"]:
        trace = Path(trial["trace"])
        last = None
        first_hold_tick = None
        for line in trace.read_text().splitlines():
            record = json.loads(line)
            last = record
            if first_hold_tick is None and record["detail"].get("failure_hold_armed"):
                first_hold_tick = record["tick"]
        retained.append({"seed": trial["seed"], "failure_hold_first_tick": first_hold_tick,
                         "held_at_final_tick": last["scoring_only"]["ball_held"],
                         "lifted_at_final_tick": last["scoring_only"]["lifted"],
                         "final_ball_xyz_task": last["scoring_only"]["ball_xyz_task"],
                         "ball_escaped_table": trial["ball_escaped_table"],
                         "pick_place_success": trial["pick_place_success"],
                         "controller_failure": trial["controller_failure"],
                         "trace": str(trace), "trace_sha256": digest(trace),
                         "video": trial["video_path"]})
    metadata = {"schema": "waypoint_failure_hold_diagnostic_v1", "controller_variant": "FailureHoldWaypointController",
                "learned_policy": False, "placement_claim": False,
                "inputs_for_failure_handling": "phase and accepted native command history only; no object/contact/success flags",
                "changes_to_scene_grasp_ik_or_physics": False,
                "source_sha256": {str(path): digest(path) for path in (Path(__file__), source,
                    Path("src/real2sim/artifacts.py"), Path("src/real2sim/policy/waypoint_failure_hold.py"))},
                "full_report": str(out / "report.json"), "trials": retained}
    (out / "failure_hold_protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out)
