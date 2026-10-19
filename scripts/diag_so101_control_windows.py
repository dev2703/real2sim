"""Replay motion inside native joint limits to separate initialization/clipping from drift.

The longest contiguous in-limit window is selected from each episode BEFORE any outcome is
observed. By default ball and bowl are moved away to diagnose unloaded tracking. --objects task
keeps the same task objects/common offset as the full replay, testing a feasible-start workaround.
Neither experiment validates calibration. All controller bounds/collision checks remain active.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import yaml
from run_so101_experiments import atomic_json, hash_file, now, queue_lock, read_json
from run_so101_root_cause import reference_for_scene, run_trial

from real2sim.sim.env import So101PickPlaceEnv

MODES = ("joint_reference", "cartesian_delta", "cartesian_persistent_target")
CONFIG = Path("results/audit/geometry_control_20261016/g2_fitted_xy_local/sim_f0.6_d1.0_b12.0.yaml")


def longest_valid_window(qpos, limits, min_ticks=30):
    """Return [start, stop) for the longest contiguous valid run; earliest wins a tie."""
    qpos, limits = np.asarray(qpos), np.asarray(limits)
    valid = (np.isfinite(qpos).all(axis=1)
             & (qpos >= limits[:, 0]).all(axis=1) & (qpos <= limits[:, 1]).all(axis=1))
    edges = np.diff(np.r_[False, valid, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts) == 0:
        raise ValueError("No in-limit window")
    selected = int(np.argmax(stops - starts))
    start, stop = int(starts[selected]), int(stops[selected])
    if stop - start - 1 < min_ticks:
        raise ValueError("No sufficiently long in-limit motion window")
    return start, stop


def run(out, config, objects="away"):
    paths = [Path(f".cache/derived_{kind}/arrays.npz") for kind in ("joint", "cartesian")]
    for path in paths:
        manifest = read_json(path.with_name("manifest.json"))
        if hash_file(path) != manifest["files_sha256"]["arrays.npz"]:
            raise ValueError(f"Source checksum mismatch: {path}")
    joint, cart = (dict(np.load(path, allow_pickle=False)) for path in paths)
    for key in ("episode_index", "frame_index", "timestamp"):
        np.testing.assert_array_equal(joint[key], cart[key])
    reference_path = Path("results/audit/geometry_control_20261016/g0_complete/report.json")
    references = read_json(reference_path)
    sources = paths + [Path(__file__), Path("scripts/run_so101_root_cause.py"),
        Path("scripts/diag_geometry_contact_sweep.py"), Path("scripts/diag_recorded_grasp_control.py"),
        Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/kinematics.py"),
        Path("src/real2sim/policy/adapters.py"), Path("src/real2sim/evaluation/metrics.py"),
        config, Path(yaml.safe_load(config.read_text())["task_config"]), reference_path]
    hashes = {str(path): hash_file(path) for path in sources}
    report = {"kind": ("IN_LIMIT_UNLOADED_RECORDED_MOTION_DIAGNOSTIC" if objects == "away"
                       else "IN_LIMIT_TASK_RECORDED_MOTION_DIAGNOSTIC"),
        "selection": "Longest contiguous recorded native-joint in-limit window; earliest tie",
        "objects": objects,
        "common_ball_offset_mm": [0, 0] if objects == "away" else [5, 0],
        "initialization": "Recorded in-limit joint pose; omitted prefix is not replayed",
        "source_sha256": hashes, "safety_thresholds_changed": False,
        "learned_policy_evaluated": False, "physical_grasp_gate_passed": False,
        "panda_allowed": False, "runs": []}
    atomic_json(out / "MANIFEST.json", report)
    (out / "executed_script.py").write_text(Path(__file__).read_text())
    windows = {}
    with So101PickPlaceEnv(config, render_images=False) as env:
        for ep in (1, 2, 3):
            indices = np.flatnonzero(joint["episode_index"] == ep)
            qpos, _, _, _ = reference_for_scene(env, joint["state"][indices])
            start, stop = longest_valid_window(qpos, env.limits)
            windows[ep] = indices[start:stop]
    for ep, indices in windows.items():
        for mode in MODES:
            atomic_json(out / "RUNNING.json", {"at": now(), "episode": ep, "mode": mode})
            summary, rows = run_trial(config, joint, cart, indices,
                ({"ball_xy": [.5, .3], "bowl_xy": [.5, -.3]} if objects == "away"
                 else references["episodes"][str(ep)]),
                references["review"][str(ep)], mode, [0, 0] if objects == "away" else [5, 0])
            summary.update(episode=ep, source_start_s=float(joint["timestamp"][indices[0]]),
                           source_end_s=float(joint["timestamp"][indices[-1]]))
            assert summary["reference_outside_joint_limits_ticks"] == 0
            assert summary["initial_reference_pose_error_m"] < 1e-9
            report["runs"].append(summary)
            atomic_json(out / f"episode{ep}_{mode}_trace.json", {"rows": rows})
            atomic_json(out / "report.json", report)
            print(json.dumps({k: summary[k] for k in ("episode", "mode", "ticks",
                "source_start_s", "source_end_s", "initial_physical_valid", "ik_reasons")}), flush=True)
    for path, expected in hashes.items():
        assert hash_file(Path(path)) == expected, f"Input changed during run: {path}"
    atomic_json(out / "COMPLETE.json", {"at": now(), "runs": len(report["runs"]),
        "status": "diagnostic_complete", "grasp_success": (
            "not_evaluated" if objects == "away" else "reference_replay_only_not_policy"),
        "panda_allowed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/so101_persistent_target_20261019/in_limit_windows"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--objects", choices=("away", "task"), default="away")
    args = parser.parse_args()
    with queue_lock(Path("outputs/experiments_20261016_stage1")), queue_lock(args.out):
        if (args.out / "COMPLETE.json").exists():
            print("In-limit motion diagnostic already complete; no duplicate work.")
            return
        if (args.out / "MANIFEST.json").exists():
            raise FileExistsError("Existing partial run retained; choose a new --out path")
        try:
            run(args.out, args.config, args.objects)
        except BaseException as error:
            atomic_json(args.out / "FAILED.json", {"at": now(), "error": repr(error),
                "traceback": traceback.format_exc()})
            raise
        finally:
            (args.out / "RUNNING.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
