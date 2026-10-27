"""Validate and combine the nominal and isolated continuation evaluation phases.

The continuation uses an isolated copy of the original frozen source. Every
dependency and setting must match; nominal trials are never rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from real2sim.artifacts import sha256_file
from real2sim.config import REPO_ROOT
from real2sim.evaluation.metrics import TRIAL_COLUMNS
from real2sim.serialization import write_json

if __package__:
    from scripts import evaluate_v2_frozen_generalization_20261027 as evaluator
else:
    import evaluate_v2_frozen_generalization_20261027 as evaluator


def canonical_hashes(protocol, runtime_root):
    result = {}
    for value, digest in protocol["source_sha256_before"].items():
        path = Path(value)
        if path.is_relative_to(Path(protocol["source"])):
            key = "checkpoint_source/" + str(path.relative_to(Path(protocol["source"])))
        elif path.is_relative_to(runtime_root):
            key = str(path.relative_to(runtime_root))
        elif path.is_relative_to(REPO_ROOT):
            key = str(path.relative_to(REPO_ROOT))
        else:
            raise ValueError(f"Unrecognized frozen dependency: {path}")
        if key in result:
            raise ValueError(f"Duplicate dependency: {key}")
        result[key] = digest
    return result


def run(out, continuation, runtime_root):
    original = evaluator.read_protocol(out)
    continued = evaluator.read_protocol(continuation)
    invariant_keys = (
        "checkpoint_sha256", "robots", "layouts", "conditions", "frozen_settings",
        "task", "size_protocol", "friction_protocol", "lighting_protocol", "scoring",
        "static_collision_map_assumption",
    )
    if any(original[key] != continued[key] for key in invariant_keys):
        raise ValueError("Continuation changed scientific settings or trial schedule")
    before = canonical_hashes(original, REPO_ROOT)
    after = canonical_hashes(continued, runtime_root)
    if before.keys() != after.keys():
        raise ValueError("Continuation changed the frozen dependency inventory")
    changed = [key for key in before if before[key] != after[key]]
    if changed:
        raise ValueError(f"Isolated continuation changed frozen dependencies: {changed}")
    # Nominal reports retain their own original before/after provenance. Only the
    # continuation sources are expected to remain present at their frozen paths.
    evaluator.verify_sources(continued)
    reports, rows, artifacts = {}, [], {}
    for condition in original["conditions"]:
        phase = out if condition == "nominal" else continuation
        protocol = original if condition == "nominal" else continued
        reports[condition] = {}
        for robot in original["robots"]:
            path = phase / condition / robot / "report.json"
            report = json.loads(path.read_text())
            if report["status"] != "complete":
                raise ValueError(f"Incomplete report: {path}")
            evaluator.verify_completed_report(phase, protocol, report, robot, condition)
            artifacts[str(path)] = sha256_file(path)
            rows.extend({"condition": condition, **row} for row in report["trials"])
            item = {"report": str(path), "summary": report["summary"]}
            if condition != "nominal":
                nominal = json.loads((out / "nominal" / robot / "report.json").read_text())
                pairs = [(a["pick_place_success"], b["pick_place_success"])
                         for a, b in zip(nominal["trials"], report["trials"], strict=True)]
                item["paired_success_transitions"] = {
                    "both_succeed": sum(a and b for a, b in pairs),
                    "nominal_only": sum(a and not b for a, b in pairs),
                    "variation_only": sum(not a and b for a, b in pairs),
                    "both_fail": sum(not a and not b for a, b in pairs),
                }
            reports[condition][robot] = item
    if len(rows) != 280:
        raise ValueError("Expected all 280 scheduled trials")
    with (out / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["condition", *TRIAL_COLUMNS], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    isolation = {
        "schema": "frozen_generalization_source_isolation_20261027_v1",
        "reason": "Concurrent workspace edit to the JSON report writer after all 40 nominal trials completed; variation launch refused before inference. Remaining runs use an isolated exact copy of all original frozen dependencies, including the recovered original JSON writer.",
        "nominal_protocol": str(out / "protocol.json"),
        "continuation_protocol": str(continuation / "protocol.json"),
        "isolated_runtime_root": str(runtime_root),
        "nominal_trials_rerun": False,
        "scientific_settings_and_checkpoint_changed": False,
        "changed_dependencies": changed,
        "invariant_fields_checked": list(invariant_keys),
        "unchanged_dependency_count": len(before),
        "concurrent_workspace_edit_preserved": True,
    }
    write_json(out / "source_isolation.json", isolation)
    for path in (out / "protocol.json", continuation / "protocol.json",
                 out / "source_isolation.json", out / "validation/report.json",
                 Path(original["layout_novelty_audit"]), out / "trials.csv"):
        artifacts[str(path)] = sha256_file(path)
    result = {
        "schema": "frozen_shared_semantic_act_generalization_20261027_combined_v1",
        "status": "complete", "checkpoint_sha256": original["checkpoint_sha256"],
        "planned_trials": 280, "completed_trials": len(rows), "unique_layouts": 20,
        "ACT_training_performed": False, "ACT_weights_changed": False,
        "same_semantic_checkpoint_both_arms": True,
        "image_encoder_ACT_or_uploaded_HF_checkpoint_evaluated": False,
        "scientific_dependencies_and_settings_unchanged": True,
        "source_isolation": str(out / "source_isolation.json"),
        "physical_calibration_accepted": False, "conditions": reports,
        "all_failures_and_aborts_in_denominators": True,
        "aborted_trials": sum(bool(row["abort"]) for row in rows),
        "artifact_sha256": artifacts,
    }
    write_json(out / "report.json", result)
    write_json(REPO_ROOT / "results/v2/frozen_generalization_20261027.json", result)
    print(json.dumps({c: {r: v["summary"]["pick_place_successes"] for r, v in arms.items()}
                      for c, arms in reports.items()}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/v2/frozen_generalization_20261027"))
    parser.add_argument("--continuation", type=Path, default=Path("outputs/v2/frozen_generalization_20261027_variations"))
    parser.add_argument("--runtime-root", type=Path, default=Path("/private/tmp/real2sim_frozen_generalization_runtime_20261027"))
    args = parser.parse_args()
    run(args.out, args.continuation, args.runtime_root)
