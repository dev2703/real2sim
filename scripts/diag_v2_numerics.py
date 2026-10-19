"""R3 provisional unloaded holds at matched 12.5 Hz control / 500 and 1000 Hz physics.

Uses explicitly selected recorded poses; no camera calibration or retention claim is made.
"""

import argparse
import copy
import json
from pathlib import Path

import yaml
from diag_so101_stationary_hold import MODES, probe

from real2sim.config import load_yaml
from real2sim.data.raw import RawDataset, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v2",
    )
    p.add_argument(
        "--poses", nargs=3, required=True, help="episode:source_frame, all valid and distinct"
    )
    p.add_argument("--config", type=Path, default=Path("configs/sim_so101.yaml"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    raw = RawDataset(a.root)
    base = load_yaml(a.config)
    data = load_yaml(base["data_config"])
    data.update(policy_hz=12.5)
    # No rendered observations are used. This diagnostic keeps the old camera declaration
    # and explicitly must not supply data or observations to learned v2 policies.
    dp = a.out / "diagnostic_clock.yaml"
    dp.write_text(yaml.safe_dump(data))
    states = []
    for spec in a.poses:
        ep, frame = map(int, spec.split(":"))
        states.append((spec, raw.stack(raw.episode_rows(ep), "observation.state")[frame]))
    if len(set(a.poses)) != 3:
        raise ValueError("three distinct recorded poses required")
    rows = []
    for hz in (500, 1000):
        cfg = copy.deepcopy(base)
        cfg.update(control_hz=12.5, physics_hz=hz, data_config=str(dp.resolve()))
        path = a.out / f"provisional_{hz}hz.yaml"
        path.write_text(yaml.safe_dump(cfg))
        for spec, state in states:
            for mode in MODES:
                result = probe(path, state, mode, ticks=250)
                result.update(physics_hz=hz, policy_hz=12.5, source_pose=spec)
                rows.append(result)
                (a.out / "progress.json").write_text(
                    json.dumps({"completed": len(rows), "total": 24})
                )
    report = {
        "diagnostic": "R3 provisional unloaded controller/numerical sensitivity",
        "source_config_sha256": sha256_file(a.config),
        "source_task_sha256": sha256_file(Path(base["task_config"])),
        "valid_for_learning_gate": False,
        "reason": "mapping, camera and task-start calibration remain unresolved",
        "rows": rows,
    }
    (a.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in ("physics_hz", "source_pose", "mode", "max_drift_mm", "ik_rejections")
                }
                for r in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
