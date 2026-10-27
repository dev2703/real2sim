"""Isolate zero-action drift from vision, labels, reference initialization and ball contact."""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import yaml
from run_so101_experiments import atomic_json, hash_file, now, queue_lock

from real2sim.evaluation.reference_control import STATIONARY_MODES as MODES
from real2sim.evaluation.reference_control import probe as probe

OUT = Path("outputs/so101_root_cause_20261017/stationary_hold")
CONFIG = Path("results/audit/geometry_control_20261016/g2_fitted_xy_local/sim_f0.6_d1.0_b12.0.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    out, config = args.out, args.config
    with queue_lock(Path("outputs/experiments_20261016_stage1")), queue_lock(out):
        if (out / "COMPLETE.json").exists():
            print("Stationary probe already complete; no duplicate work.")
            return
        if (out / "report.json").exists() or (out / "FAILED.json").exists():
            raise FileExistsError("Existing partial probe retained; choose a new --out path")
        atomic_json(out / "RUNNING.json", {"at": now()})
        try:
            joint = dict(np.load(".cache/derived_joint/arrays.npz"))
            source_files = [Path(__file__), config, Path(".cache/derived_joint/arrays.npz"),
                Path("src/real2sim/evaluation/reference_control.py"),
                Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/base_env.py"),
                Path(yaml.safe_load(config.read_text())["task_config"]),
                Path("src/real2sim/policy/adapters.py"), Path("src/real2sim/sim/kinematics.py")]
            hashes = {str(p): hash_file(p) for p in source_files}
            report = {"kind": "NO_POLICY_STATIONARY_HOLD_CONTROL_DIAGNOSTIC",
                "source_sha256": hashes, "safety_thresholds_changed": False,
                "panda_allowed": False, "runs": []}
            (out / "executed_script.py").write_text(Path(__file__).read_text())
            for ep, timestamp in ((1, 18.), (2, 18.), (3, 20.)):
                selected = np.flatnonzero(joint["episode_index"] == ep)
                index = selected[np.argmin(abs(joint["timestamp"][selected] - timestamp))]
                for mode in args.modes:
                    run = probe(config, joint["state"][index], mode)
                    run.update(episode=ep, reference_timestamp=float(joint["timestamp"][index]))
                    report["runs"].append(run)
                    atomic_json(out / "report.json", report)
                    print(json.dumps({k: v for k, v in run.items() if k != "rows"}), flush=True)
            for path, expected in hashes.items():
                assert hash_file(Path(path)) == expected
            atomic_json(out / "COMPLETE.json", {"at": now(), "status": "diagnostic_complete",
                "runs": len(report["runs"]), "grasp_success": "not_evaluated", "panda_allowed": False})
        except BaseException as error:
            atomic_json(out / "FAILED.json", {"at": now(), "error": repr(error),
                                               "traceback": traceback.format_exc()})
            raise
        finally:
            (out / "RUNNING.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
