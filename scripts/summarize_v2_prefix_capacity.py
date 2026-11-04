"""Compare frozen three-episode runs at the same optimizer/example budget.

This report consumes executed-prefix audits. It never scores the unsupervised
chunk tail, refits a gate, or grants physical geometry acceptance.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
from pathlib import Path

import yaml

from real2sim.data.raw import sha256_file

EPISODES = [11, 8, 23]


def read_json(path):
    return json.loads(path.read_text())


def verify_plans(baseline, prefix):
    """Whitelist the declared objective/name/metadata change, not arbitrary drift."""
    steps = prefix.get("reconstruction_loss", {}).get("supervised_prefix_steps")
    if (type(steps) is not int or steps != 2 or prefix["model"]["n_action_steps"] != steps
            or prefix["model"]["chunk_size"] < steps):
        raise ValueError("prefix objective must supervise the two executed action offsets")
    if baseline.get("reconstruction_loss", {}).get("supervised_prefix_steps") is not None:
        raise ValueError("baseline already uses prefix supervision")
    if prefix["model"]["use_vae"] or prefix["model"]["kl_weight"]:
        raise ValueError("comparison requires the deterministic non-VAE training plan")
    expected_name = baseline["name"] + "_prefix2"
    if prefix["name"] != expected_name or prefix["experiment"].get("supervised_prefix_steps") != 2:
        raise ValueError("prefix experiment identity differs from its declaration")
    if not prefix["experiment"].get("comparison"):
        raise ValueError("prefix comparison metadata is missing")
    reduced = copy.deepcopy(prefix)
    reduced["name"] = baseline["name"]
    del reduced["reconstruction_loss"]["supervised_prefix_steps"]
    if not reduced["reconstruction_loss"] and "reconstruction_loss" not in baseline:
        del reduced["reconstruction_loss"]
    for field in ("comparison", "supervised_prefix_steps"):
        del reduced["experiment"][field]
    if reduced != baseline:
        raise ValueError("plans differ beyond the declared prefix objective/name/metadata")
    if prefix["train"].get("diagnostic_episodes") != EPISODES:
        raise ValueError("the declared representative training episodes changed")
    return {"only_intended_changes": True, "supervised_action_offsets": [0, 1],
            "output_chunk_size": prefix["model"]["chunk_size"],
            "matched_seed": prefix["train"]["seed"],
            "matched_optimizer_horizon": prefix["train"]["schedule_steps"],
            "matched_effective_batch": prefix["train"]["batch_size"] * prefix["train"].get("grad_accum", 1),
            "full_chunk_metrics_compared": False}


def frozen_gate(run, ledger):
    """Resolve a retained runner whose hash matches the run's source ledger."""
    digests = [v for k, v in ledger["source_sha256"].items()
               if Path(k).name == "run_v2_three_episode_fit.py"]
    candidates = [run / "executed_runner.py", *sorted(run.glob("resume_runner_*.py"))]
    matched = [p for p in candidates if p.exists() and sha256_file(p) in digests]
    if len(digests) != 1 or not matched:
        raise ValueError("no archived runner matches the source ledger")
    path = matched[0]
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gate_from_validation")
    namespace = {"math": math, "EPISODES": EPISODES, "GATE_VERSION": ledger["gate_version"]}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["gate_from_validation"], ast.dump(node, include_attributes=False), path


def verify_payload(data, manifest):
    checks = {}
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name:
            raise ValueError("derived payload paths must be local basenames")
        checks[name] = sha256_file(data / name) == expected
    if not checks or not all(checks.values()):
        raise ValueError("derived data payload changed")
    return checks


def verify_checkpoint(stage, data, manifest, plan, prefix_steps):
    checkpoint = Path(stage["checkpoint"])
    if sha256_file(checkpoint / "model.safetensors") != stage["checkpoint_sha256"]:
        raise ValueError("scored checkpoint weights changed")
    contract = read_json(checkpoint / "real2sim_contract.json")
    stats_equal = read_json(checkpoint / "stats.json") == read_json(data / "stats.json")
    if contract["derived_files_sha256"] != manifest["files_sha256"] or not stats_equal:
        raise ValueError("checkpoint data contract or normalization differs")
    for field in ("source", "split", "camera_keys", "state_names", "action_names", "policy_hz",
                  "action_representation", "controller_semantics", "calibration_sha256", "camera_transforms"):
        if contract.get(field) != manifest.get(field):
            raise ValueError(f"checkpoint semantic contract differs: {field}")
    saved = read_json(checkpoint / "config.json")
    for field in ("chunk_size", "n_action_steps"):
        if saved[field] != plan["model"][field]:
            raise ValueError(f"saved execution/output configuration differs: {field}")
    info = read_json(checkpoint / "run_info.json")
    effective_batch = plan["train"]["batch_size"] * plan["train"].get("grad_accum", 1)
    if info["step"] * effective_batch != stage["examples_seen"]:
        raise ValueError("checkpoint optimizer step differs from the scored example budget")
    if info.get("reconstruction_loss", {}).get("supervised_prefix_steps") != prefix_steps:
        raise ValueError("checkpoint supervision differs from its declared objective")
    return {"checkpoint": str(checkpoint), "weights_sha256": stage["checkpoint_sha256"],
            "normalization_matches": True, "data_contract_matches": True,
            "executed_action_steps": saved["n_action_steps"], "output_chunk_size": saved["chunk_size"]}


def verify_motion_contract(baseline, prefix):
    for field in ("version", "action_representation", "episodes", "valid_actions",
                  "excluded_extrapolated_actions", "excluded_invalid_actions",
                  "excluded_invalid_observation_actions", "executed_actions_per_call", "thresholds",
                  "protocol", "baseline_contract"):
        if baseline.get(field) != prefix.get(field):
            raise ValueError(f"executed-prefix scoring contract differs: {field}")
    for name in ("hold", "persistence"):
        if baseline["scores"][name] != prefix["scores"][name]:
            raise ValueError(f"causal baseline values differ: {name}")


def verify_object_probe(path, selected_stage, data):
    probe = read_json(path)
    plan = probe["plan"]
    checkpoint = Path(selected_stage["checkpoint_integrity"]["checkpoint"])
    if (probe.get("schema") != "canonical_matched_object_probe_v1"
            or probe.get("reserved_episodes_used")
            or Path(plan["checkpoint"]).resolve() != checkpoint.resolve()
            or Path(plan["data"]).resolve() != data.resolve()):
        raise ValueError("object probe is not from the selected prefix checkpoint/data")
    weights = [v for k, v in plan["source_sha256"].items() if Path(k).name == "model.safetensors"]
    if weights != [selected_stage["checkpoint_integrity"]["weights_sha256"]]:
        raise ValueError("object probe checkpoint hash differs")
    if sha256_file(path.with_suffix(".plan.json")) != probe["plan_sha256"]:
        raise ValueError("object probe plan hash differs")
    for source, expected in plan["source_sha256"].items():
        if sha256_file(Path(source)) != expected:
            raise ValueError(f"object probe source changed: {source}")
    return {"path": str(path), "sha256": sha256_file(path),
            "selected_pairs": probe["selected_pairs"], "supported_pairs": probe["supported_pairs"],
            "groups": probe["groups"], "checkpoint_matches": True}


def summarize(baseline_run, prefix_run, object_report=None):
    plans = [yaml.safe_load((p / "plan.yaml").read_text()) for p in (baseline_run, prefix_run)]
    ledgers = [read_json(p / "report.json") for p in (baseline_run, prefix_run)]
    plan_check = verify_plans(*plans)
    prefix_ledger = ledgers[1]
    if (not prefix_ledger.get("stages") or (not prefix_ledger.get("gate_passed")
            and prefix_ledger.get("total_examples_seen") != 32000)):
        raise ValueError("the declared prefix run has not finished")
    gates = [frozen_gate(p, d) for p, d in zip((baseline_run, prefix_run), ledgers, strict=True)]
    if ledgers[0]["gate_version"] != ledgers[1]["gate_version"] or gates[0][1] != gates[1][1]:
        raise ValueError("the capacity gate changed between experiments")
    # Model construction, dataset sampling and scoring are unchanged; the loss source
    # is added only in the prefix ledger, and runner changes are explicitly audited.
    for name in ("train.py", "act.py", "dataset.py", "validation.py"):
        values = [[v for k, v in ledger["source_sha256"].items() if Path(k).name == name] for ledger in ledgers]
        if len(values[0]) != 1 or values[0] != values[1]:
            raise ValueError(f"matched experiment dependency differs: {name}")
    data = Path(plans[1]["data"])
    manifest = read_json(data / "manifest.json")
    for ledger in ledgers:
        for name in ("manifest.json", "stats.json"):
            expected = [v for k, v in ledger["source_sha256"].items() if Path(k).name == name]
            if len(expected) != 1 or sha256_file(data / name) != expected[0]:
                raise ValueError(f"run-ledger source data changed: {name}")
    payload_check = verify_payload(data, manifest)
    paths = [Path(__file__), data / "manifest.json", data / "stats.json", gates[0][2], gates[1][2]]
    checked_runs = []
    for run, plan, ledger, (gate, _, _) in zip((baseline_run, prefix_run), plans, ledgers, gates, strict=True):
        if ledger["training_episodes"] != EPISODES or ledger.get("reserved_episodes_used"):
            raise ValueError("training episode selection or reserved isolation changed")
        stages = []
        for stage in ledger["stages"]:
            path = Path(stage["training_motion_audit"])
            motion = read_json(path)
            if gate(motion) != stage["gate"]:
                raise ValueError("recorded capacity gate differs from the frozen function")
            checkpoint = verify_checkpoint(stage, data, manifest, plan,
                                           None if run == baseline_run else 2)
            stages.append({"examples_seen": stage["examples_seen"], "gate": stage["gate"],
                           "motion": motion, "checkpoint_integrity": checkpoint})
            paths.extend([path, Path(checkpoint["checkpoint"]) / "run_info.json",
                          Path(checkpoint["checkpoint"]) / "real2sim_contract.json",
                          Path(checkpoint["checkpoint"]) / "stats.json", Path(checkpoint["checkpoint"]) / "config.json"])
        checked_runs.append(stages)
        paths.extend([run / "plan.yaml", run / "report.json"])
    selected_prefix = checked_runs[1][-1]
    if selected_prefix["examples_seen"] != prefix_ledger["total_examples_seen"]:
        raise ValueError("final ledger example count differs from its final scored stage")
    if selected_prefix["gate"]["pass"] != prefix_ledger["gate_passed"]:
        raise ValueError("final ledger gate differs from its final scored stage")
    matching = [s for s in checked_runs[0] if s["examples_seen"] == selected_prefix["examples_seen"]]
    if len(matching) != 1:
        raise ValueError("no unique baseline stage at the prefix run's final example budget")
    selected_baseline = matching[0]
    verify_motion_contract(selected_baseline["motion"], selected_prefix["motion"])
    result = {"schema": "matched_prefix_capacity_comparison_v1", "physical_calibration_accepted": False,
              "checkpoint_promoted": False, "reserved_episodes_used": [], "development_rollouts_run": False,
              "scheduled_examples_compared": selected_prefix["examples_seen"], "plan_check": plan_check,
              "gate_unchanged": True, "derived_payload_integrity": payload_check,
              "baseline": selected_baseline, "prefix2": selected_prefix,
              "all_stages": {"baseline": checked_runs[0], "prefix2": checked_runs[1]},
              "limitations": ["Training-fit diagnostic under provisional geometry; no calibration or task-success acceptance.",
                  "Stationary metrics are teacher-forced summed-error proxies, not closed-loop drift.",
                  "Only action offsets 0/1 receive direct reconstruction supervision; full-chunk L1 is not compared.",
              "One matched seed cannot establish general architecture or generalization claims.",
                  "Prefix training does not resolve latent teleoperator timing or verify object dependence by itself."]}
    if object_report is not None:
        paths.append(object_report)
        paths.append(object_report.with_suffix(".plan.json"))
        result["object_probe_report"] = verify_object_probe(object_report, selected_prefix, data)
    result["source_sha256"] = {str(p): sha256_file(p) for p in paths}
    return result


def markdown(report, doc_path, summary_path=Path("results/v2/prefix_capacity_20261025.json")):
    baseline, prefix = report["baseline"], report["prefix2"]
    passed = prefix["gate"]["pass"]
    original_score, prefix_score = (stage["motion"]["scores"]["model"] for stage in (baseline, prefix))
    old_moving, new_moving = (score["movement"]["vector_error_mean"] for score in (original_score, prefix_score))
    old_stationary, new_stationary = (score["stationary_drift"]["summed_error_per_run_mean"]
                                    for score in (original_score, prefix_score))
    lines = ["# Two-action supervision on three episodes — 25 October 2026", "",
             f"The prefix-supervised ACT **{'passes' if passed else 'fails'}** the unchanged three-episode capacity gate after **{report['scheduled_examples_compared']:,} scheduled examples**. Its result is compared with the original full-chunk objective at the same optimizer/example budget. Both use episodes 11 (baseline), 8 (ball shift), and 23 (bowl shift).", "",
             f"Moving-vector error changes **{old_moving:.3f} → {new_moving:.3f} mm**, and mean stationary-run summed error changes **{old_stationary:.3f} → {new_stationary:.3f} mm**. The per-condition fitting and baseline gates below determine acceptance; improvement in these means alone does not establish success.", "",
             f"The [machine-readable comparison]({os.path.relpath(summary_path, doc_path.parent)}) contains every per-condition gate, both stage histories and checkpoint/data integrity checks. No reserved episode was selected or scored.", "",
             "## Controlled change", "",
             "The model still predicts 25 actions and executes two. The new objective supervises only action offsets 0/1, applying terminal padding and normalizing by real supervised components. The backbone, model initialization seed, episode/data contracts, labels, source normalization, sampler, batch 8×2, learning-rate horizon and update cap match the original run. No target filtering or relabeling was applied.", "",
             "VAE is disabled, so the experiment has no future-action latent-encoder input. Direct gradients to output-tail positions are zero; shared decoder parameters may still affect those outputs. Full-chunk validation L1 is descriptive and is not used for this comparison. Selection and the capacity gate use the unchanged executed-prefix audit.", "",
             "## Matched-budget result", "",
             "| Metric on the same training actions | Original objective | Two-action objective | Hold | Persistence |",
             "|---|---:|---:|---:|---:|"]
    for label, group, metric in (("Moving-vector error (mm)", "movement", "vector_error_mean"),
                                 ("Mean stationary-run summed error (mm)", "stationary_drift", "summed_error_per_run_mean"),
                                 ("Maximum stationary-run summed error (mm)", "stationary_drift", "summed_error_per_run_max"),
                                 ("Mean predicted stationary step (mm)", "stationary_drift", "predicted_step_mean")):
        values = [baseline["motion"]["scores"]["model"][group][metric],
                  prefix["motion"]["scores"]["model"][group][metric],
                  prefix["motion"]["scores"]["hold"][group][metric],
                  prefix["motion"]["scores"]["persistence"][group][metric]]
        lines.append("| " + label + " | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", "Stationary-run values are teacher-forced target-error sums, not measured rollout drift. Hold/persistence values and all scoring exclusions are identical between experiments.", "",
              "| Episode / objective | Translation error (mm) | Gripper MAE | Moving error (mm) | Mean / max stationary-run error (mm) | Gate |",
              "|---|---:|---:|---:|---:|---|"]
    for episode in ("11", "8", "23"):
        for label, source in (("Original", baseline), ("Prefix2", prefix)):
            score = source["motion"]["scores"]["model"]["per_episode"][episode]
            drift = score["stationary_drift"]
            gate = source["gate"]["per_episode"][episode]
            lines.append(f"| {episode} / {label} | {score['translation_mean_mm']:.3f} | {score['gripper_mae']:.4f} | {score['movement']['vector_error_mean']:.3f} | {drift['summed_error_per_run_mean']:.3f} / {drift['summed_error_per_run_max']:.3f} | {'Pass' if gate['pass'] else 'Fail'} |")
    lines += ["", "Every condition must have translation error below 1 mm and gripper MAE below 0.03, and improve both causal baselines on moving-vector error and mean/maximum stationary-run error. Missing strata fail; exact zero-error floor ties are allowed. The archived gate function is identical between runs and recomputes every recorded gate result.", "",
              "## Integrity and remaining checks", "",
              "The summary verifies the allowed plan differences, matching model/data/scoring dependency hashes, all derived payload hashes, every scored checkpoint weight hash, checkpoint normalization, saved execution counts and matched scoring support. It consumes frozen audits without fitting new thresholds or decoding reserved images.", ""]
    if passed:
        lines.append("The small fitting gate passes. Common development rollouts remain the next gate before reserved evaluation.")
    else:
        lines.append("The small training success remains unestablished. Common development rollouts and reserved evaluation were not run for this candidate; the unchanged gate is retained.")
    if "object_probe_report" in report:
        relative = os.path.relpath(report["object_probe_report"]["path"], doc_path.parent)
        probe = report["object_probe_report"]
        lines += ["", "## Matched-object action probes", "",
                  f"The separate [object-dependence report]({relative}) uses the same scored prefix checkpoint and the previously frozen pair/mask selection. **{probe['supported_pairs']} / {probe['selected_pairs']} pairs** meet the original object-support rule. The following development active-state rows use object-region swaps with recipient arm-state input fixed.", "",
                  "| Object | Informative pairs | Mean swap effect (two-action norm) | Effect / observed contrast | Mean direction cosine |",
                  "|---|---:|---:|---:|---:|"]
        for group in probe["groups"]:
            if (group["partition"], group["context"], group["mode"]) != ("val", "active", "paired_object_region"):
                continue
            ratio, cosine = group["mean_informative_amplitude_ratio"], group["mean_informative_cosine"]
            ratio_text = "Unavailable" if ratio is None else f"{100 * ratio:.2f}%"
            cosine_text = "Unavailable" if cosine is None else f"{cosine:.3f}"
            lines.append(f"| {group['object']} | {group['informative_pairs']} | {group['mean_swap_effect_prefix_norm_mm']:.3f} mm | {ratio_text} | {cosine_text} |")
        lines += ["", "Ratios/cosines use only predeclared informative target contrasts. Reciprocal directions are correlated; donor actions are observational comparisons rather than counterfactual ground truth. Mask unions can transfer background/occlusion pixels. Full-image swaps, stationary starts and training-pair endpoint overlap remain in the linked report. Input sensitivity alone does not establish correct object-conditioned actions or rollout success.", "",
                  "The paired-checkpoint object comparison (see `docs/PROJECT_SUMMARY.md`) compares this candidate with the original objective and retains stationary development-start checks separately."]
    else:
        lines += ["", "Object dependence requires a separate matched-layout image/object-swap check. The earlier matched-object pilot (see `docs/PROJECT_SUMMARY.md`) motivates that test; it does not measure this checkpoint."]
    lines += ["", "The separate motion-onset audit (see `docs/PROJECT_SUMMARY.md`) finds stationary immediate prefixes with movement in the future tail, and nearly identical observations with differing immediate labels. Prefix supervision changes the trained horizon; it does not remove that observed timing ambiguity. No labels or sampling weights were changed.", "",
              "Physical camera/base registration and joint/gripper verification remain unresolved. This experiment uses provisional FK labels and cannot justify a geometric correction. The fixed-base evidence (see `docs/PROJECT_SUMMARY.md`) and multi-height landmark review (see `docs/PROJECT_SUMMARY.md`) retain those limits. The calibration capture sheet (see `docs/PROJECT_SUMMARY.md`) specifies the next physical evidence.", "",
              "Reproduce in fresh paths:", "", "```sh",
              ".venv/bin/python scripts/run_v2_three_episode_fit.py \\",
              "  --out outputs/v2/three_episode_prefix2_NEXT --device cpu --supervised-prefix-steps 2",
              ".venv/bin/python scripts/summarize_v2_prefix_capacity.py \\",
              "  --baseline outputs/v2/three_episode_motion_20261024 \\",
              "  --prefix outputs/v2/three_episode_prefix2_NEXT \\",
              "  --out outputs/v2/prefix_capacity_NEXT/report.json --doc outputs/v2/prefix_capacity_NEXT/review.md", "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/v2/three_episode_motion_20261024"))
    parser.add_argument("--prefix", type=Path, default=Path("outputs/v2/three_episode_prefix2_20261025"))
    parser.add_argument("--out", type=Path, default=Path("results/v2/prefix_capacity_20261025.json"))
    parser.add_argument("--doc", type=Path, default=Path("outputs/v2/prefix_capacity_report.md"))
    parser.add_argument("--object-report", type=Path)
    args = parser.parse_args()
    result = summarize(args.baseline, args.prefix, args.object_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.doc.write_text(markdown(result, args.doc, args.out))
    print(json.dumps({"examples_compared": result["scheduled_examples_compared"],
                      "prefix_gate_passed": result["prefix2"]["gate"]["pass"]}, indent=2))
