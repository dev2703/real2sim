"""Merge every results/metrics/*/trials.csv into results/metrics/trials.csv and print a summary table.

    uv run python scripts/aggregate_trials.py

Every attempt stays a row (plan rule 4); the summary counts successes/N per experiment and
controller_source and the phase-failure breakdown. No filtering.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"


def main() -> None:
    rows: list[dict] = []
    for f in sorted(METRICS.glob("*/trials.csv")):
        if f.parent.name.startswith("debug_"):
            continue
        with f.open() as fh:
            for r in csv.DictReader(fh):
                r["_run_dir"] = f.parent.name
                rows.append(r)
    if not rows:
        print("no trials.csv found")
        return
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    out = METRICS / "trials.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["_run_dir"], r.get("experiment_id", ""), r.get("controller_source", ""), r.get("robot", ""))].append(r)
    summary = []
    truthy = {"True", "true", "1"}
    print(f"{'run':24s} {'experiment':22s} {'controller':28s} {'robot':6s} {'succ':>7s} {'lift':>5s} {'deliv':>5s} {'rel':>4s} phase_failed")
    for (run, exp, src, robot), rs in sorted(groups.items()):
        n = len(rs)
        succ = sum(r.get("success") in truthy for r in rs)
        lift = sum(r.get("lifted") in truthy for r in rs)
        deliv = sum(r.get("delivered") in truthy for r in rs)
        rel = sum(r.get("released") in truthy for r in rs)
        phases = Counter(r.get("phase_failed", "") for r in rs)
        print(f"{run:24s} {exp:22s} {src:28s} {robot:6s} {succ:>3d}/{n:<3d} {lift:>5d} {deliv:>5d} {rel:>4d} {dict(phases)}")
        summary.append({"run": run, "experiment_id": exp, "controller_source": src, "robot": robot, "n": n, "successes": succ,
                        "lifted": lift, "delivered": deliv, "released": rel, "phase_failed": dict(phases),
                        "checkpoint_sha256": sorted({r.get("checkpoint_sha256", "") for r in rs}),
                        "mean_latency_ms": round(sum(float(r.get("mean_latency_ms") or 0) for r in rs) / n, 1)})
    (METRICS / "summary_all.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out} ({len(rows)} rows) and summary_all.json")


if __name__ == "__main__":
    main()
