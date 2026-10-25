"""Compare canonical full-chunk and prefix ACTs on the same frozen object probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from real2sim.data.raw import sha256_file


def verify_paired_probe_reports(baseline, prefix):
    if baseline["schema"] != "canonical_matched_object_probe_v1" or prefix["schema"] != baseline["schema"]:
        raise ValueError("canonical matched-object reports required")
    for key in ("rules", "pairs", "actual_training_episodes", "data"):
        if baseline["plan"][key] != prefix["plan"][key]:
            raise ValueError(f"paired probe inputs differ: {key}")
    if baseline["selected_pairs"] != prefix["selected_pairs"] or baseline["supported_pairs"] != prefix["supported_pairs"]:
        raise ValueError("paired probe support differs")
    def identity(group):
        return tuple(group[k] for k in ("partition", "object", "context", "mode"))
    left, right = ({identity(g): g for g in report["groups"]} for report in (baseline, prefix))
    if set(left) != set(right) or len(left) != len(baseline["groups"]) or len(right) != len(prefix["groups"]):
        raise ValueError("paired probe group identities differ")
    for key in left:
        for field in ("pairs", "reciprocal_directions", "informative_pairs", "mean_target_contrast_prefix_norm_mm",
                      "mean_original_target_step_mm"):
            if left[key][field] != right[key][field]:
                raise ValueError(f"paired probe targets/support differ: {key}/{field}")
    return left, right


def compare(baseline_path, prefix_path):
    baseline, prefix = (json.loads(p.read_text()) for p in (baseline_path, prefix_path))
    left, right = verify_paired_probe_reports(baseline, prefix)
    infos, configs = [], []
    dependencies = [Path(__file__), baseline_path, prefix_path]
    for report, path in ((baseline, baseline_path), (prefix, prefix_path)):
        checkpoint = Path(report["plan"]["checkpoint"])
        plan_path = path.with_suffix(".plan.json")
        if sha256_file(plan_path) != report["plan_sha256"]:
            raise ValueError("completed probe plan changed")
        for name in ("model.safetensors", "config.json", "run_info.json", "real2sim_contract.json"):
            source = (checkpoint / name).resolve()
            if sha256_file(source) != report["plan"]["source_sha256"][str(source)]:
                raise ValueError(f"scored checkpoint changed: {source}")
            dependencies.append(source)
        dependencies.append(plan_path)
        infos.append(json.loads((checkpoint / "run_info.json").read_text()))
        configs.append(json.loads((checkpoint / "config.json").read_text()))
    if configs[0] != configs[1]:
        raise ValueError("checkpoint model configurations differ")
    for field in ("step", "seed", "diagnostic_episodes", "examples_seen", "scheduler", "scheduler_total_steps", "initial_checkpoint"):
        if infos[0].get(field) != infos[1].get(field):
            raise ValueError(f"training comparison is unmatched: {field}")
    full_loss, prefix_loss = (dict(info["reconstruction_loss"]) for info in infos)
    if full_loss.pop("supervised_prefix_steps", None) is not None or prefix_loss.pop("supervised_prefix_steps", None) != 2:
        raise ValueError("comparison requires full-chunk versus two-action prefix supervision")
    if full_loss != prefix_loss:
        raise ValueError("other reconstruction settings differ")
    rows = []
    for identity, old in left.items():
        new = right[identity]
        rows.append({"partition": identity[0], "object": identity[1], "context": identity[2], "mode": identity[3],
                     "full_chunk": old, "prefix_two": new})
    return {"schema": "prefix_object_action_comparison_v1", "physical_calibration_accepted": False,
            "checkpoint_promoted": False, "reserved_episodes_used": [],
            "selected_pairs": baseline["selected_pairs"], "supported_pairs": baseline["supported_pairs"],
            "training_episodes": infos[0]["diagnostic_episodes"], "examples_per_model": infos[0]["examples_seen"],
            "updates_per_model": infos[0]["step"], "seed": infos[0]["seed"],
            "groups": rows, "frozen_rules": baseline["plan"]["rules"],
            "training_endpoint_overlap_counts": {str(n): sum(p["actual_training_endpoint_count"] == n
                for p in baseline["plan"]["pairs"]) for n in (0, 1, 2)},
            "limitations": [baseline["plan"]["interpretation"],
                "These results compare different training objectives at the same schedule/seed with fixed model, labels and normalization. They do not isolate policy task performance.",
                "The original 256-update explicit-ball-feature pilots are separate experiments; their action effects are not this full-chunk capacity baseline.",
                "Object-position displacement is measured in uncalibrated policy pixels; camera/base geometry remains provisional.",
                "Stationary matched starts have zero immediate target contrast. Their original predicted motion and swap sensitivity cannot establish correct directional object use."],
            "source_sha256": {str(p): sha256_file(p) for p in dependencies}}


def markdown(report):
    examples, updates = report["examples_per_model"], report["updates_per_model"]
    development = {(row["object"], row["context"], row["mode"]): row for row in report["groups"]
                   if row["partition"] == "val"}
    ball = development["ball", "active", "paired_object_region"]
    bowl = development["bowl", "active", "paired_object_region"]
    lines = ["# Object-action probes after prefix supervision — 25 October 2026", "",
        f"The canonical full-chunk and two-action-prefix ACTs are compared at **{updates:,} updates / {examples:,} scheduled examples each**, using the same model, seed, source labels, normalization and 24 frozen matched-state object pairs. Only reconstruction supervision changes. This measures image dependence alongside natural observed action contrast; it does not establish physical counterfactual correctness or task success.", "",
        f"For active development ball pairs, the object-region effect stays near **{100 * ball['prefix_two']['mean_informative_amplitude_ratio']:.1f}%** of observed target contrast while its mean direction cosine improves **{ball['full_chunk']['mean_informative_cosine']:.3f} → {ball['prefix_two']['mean_informative_cosine']:.3f}**. Bowl-region sensitivity grows **{100 * bowl['full_chunk']['mean_informative_amplitude_ratio']:.1f}% → {100 * bowl['prefix_two']['mean_informative_amplitude_ratio']:.1f}%**, but direction remains poorly aligned (**{bowl['prefix_two']['mean_informative_cosine']:.3f}**). Increased sensitivity alone therefore does not establish useful bowl-conditioned actions.", "",
        "The [paired report](../results/v2/prefix_object_actions_20261025.json) retains all training/development group metrics and links the complete per-direction predictions and checkpoint hashes. Source reports remain under `outputs/v2/prefix_object_probe_20261025`.", "",
        "## Reused observations and input contract", "",
        f"All **{report['supported_pairs']}/{report['selected_pairs']}** existing pairs meet the original object-mask area and center-separation thresholds. The same masks, pairs, object support, two-action labels and reciprocal directions are reused; no detector, segmentation or sample selection is rerun. Canonical arrays/images are verified byte-identical to the source of those masks. Each checkpoint has exactly 13 canonical state features and a single iPhone image, with matching checkpoint/source normalization.", "",
        "The state is fixed during image swaps. A full-image swap replaces all image pixels; an object-region swap replaces donor pixels in the dilated union of the two measured object masks. No explicit ball-feature slots exist in either model. Natural donor predictions retain their own approximately matched recorded arm states and are scored separately.", "",
        "The original train/validation labels describe the full source split, while these capacity policies train only on episodes 11, 8 and 23. **No frozen pair has both endpoints in those three episodes.** Training-endpoint overlap is explicit in the machine report, so the source-training probes must not be interpreted as three-episode fitting checks. Reserved episodes are excluded.", "",
        "## Active development object dependence", "",
        "The target contrast is the difference between observed two-action translation prefixes, represented as one six-dimensional vector. A frozen 1 mm norm threshold determines whether amplitude/direction are informative. Three ball pairs and two bowl pairs meet that threshold. Both reciprocal directions enter averages and are correlated.", "",
        "| Object / replacement | Informative pairs | Full-chunk effect / contrast | Prefix effect / contrast | Full-chunk cosine | Prefix cosine |",
        "|---|---:|---:|---:|---:|---:|"]
    for group in report["groups"]:
        if group["partition"] != "val" or group["context"] != "active":
            continue
        old, new = group["full_chunk"], group["prefix_two"]
        mode = "full image" if group["mode"] == "paired_full_image" else "object region"
        lines.append(f"| {group['object']} / {mode} | {old['informative_pairs']} | {100 * old['mean_informative_amplitude_ratio']:.2f}% | {100 * new['mean_informative_amplitude_ratio']:.2f}% | {old['mean_informative_cosine']:.3f} | {new['mean_informative_cosine']:.3f} |")
    lines += ["", "These amplitudes are averages of per-direction ratios on informative pairs. Object-region effects can include transferred background, shadows or occluders; a bowl after release can also contain the ball. Full-image effects include arm/background differences. Donor labels are observational comparators, not known correct actions after synthetic swaps. The full-chunk capacity baseline is distinct from the earlier 256-update explicit-ball pilots.", "",
        "## Stationary development starts", "",
        "All six stationary development pairs have zero recorded translation over their two scored actions. Their useful checks are false motion and input sensitivity.", "",
        "| Layout group | Full-chunk mean predicted step | Prefix mean predicted step | Target |", "|---|---:|---:|---:|"]
    for group in report["groups"]:
        if group["partition"] == "val" and group["context"] == "stationary_start" and group["mode"] == "paired_object_region":
            old, new = group["full_chunk"], group["prefix_two"]
            lines.append(f"| {group['object']} | {old['mean_original_prediction_step_mm']:.4f} mm | {new['mean_original_prediction_step_mm']:.4f} mm | 0 mm |")
    lines += ["", "These are sparse offline start observations, not closed-loop drift or a summary of every stationary hold within a trajectory. They do not replace the unchanged three-episode movement/stationary-error gate, followed by common development rollouts only if it passes. No checkpoint is promoted by object sensitivity alone.", "",
        "## Reproduction", "", "```sh",
        ".venv/bin/python scripts/diag_v2_prefix_object_actions.py \\",
        "  --checkpoint outputs/v2/three_episode_motion_20261024/training/checkpoints/002000 \\",
        "  --name full_chunk_2000 --out outputs/v2/prefix_object_probe_NEXT/full_chunk_2000.json",
        ".venv/bin/python scripts/diag_v2_prefix_object_actions.py \\",
        "  --checkpoint outputs/v2/three_episode_prefix2_20261025/training/checkpoints/002000 \\",
        "  --name prefix_2000 --out outputs/v2/prefix_object_probe_NEXT/prefix_2000.json",
        ".venv/bin/python scripts/summarize_v2_prefix_object_actions.py \\",
        "  --baseline outputs/v2/prefix_object_probe_NEXT/full_chunk_2000.json \\",
        "  --prefix outputs/v2/prefix_object_probe_NEXT/prefix_2000.json \\",
        "  --out results/v2/prefix_object_actions_NEXT.json \\",
        "  --doc docs/v2_prefix_object_actions_NEXT.md", "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.prefix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.doc.write_text(markdown(report))
    print(json.dumps({"examples_per_model": report["examples_per_model"],
                      "selected_pairs": report["selected_pairs"], "supported_pairs": report["supported_pairs"]}, indent=2))
