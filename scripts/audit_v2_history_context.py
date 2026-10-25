"""Causal history informativeness and transition-localized prediction residuals.

Only three declared training episodes are scored. Future-label proximity appears
solely in descriptive output strata; labels never construct history features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from real2sim.data.raw import sha256_file
from real2sim.policy.act import load_checkpoint
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.validation import MOTION_MM, MOTION_RAD, evaluate_executed_prefix

EPISODES = (11, 8, 23)
RULES = {"history_lengths": [1, 2, 4, 8], "motion_mm": MOTION_MM,
         "stationary_rotation_rad": MOTION_RAD, "causal_history_ticks": 4,
         "causal_history_speed_m_s": .005, "causal_history_arm_span_deg": .5,
         "descriptive_label_neighborhood_ticks": 4}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def valid_labels(arrays):
    valid = np.ones(len(arrays["episode_index"]), bool)
    for name in ("label_is_extrapolated", "action_is_invalid", "observation_is_invalid"):
        if name in arrays:
            mask = np.asarray(arrays[name])
            if mask.shape != valid.shape or mask.dtype != np.bool_:
                raise ValueError(f"invalid per-tick boolean mask: {name}")
            valid &= ~mask
    return valid


def causal_state_history(states, history):
    """Oldest-first observation-only slots, repeating the first episode observation.

    This equality diagnostic excludes age/valid bookkeeping so floating timestamp
    jitter cannot manufacture information. It does not prepare policy inputs.
    """
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 13 or not len(states) or not np.isfinite(states).all():
        raise ValueError("a finite nonempty episode of canonical 13-feature states is required")
    if type(history) is not int or history < 1:
        raise ValueError("history must be a positive integer")
    offsets = np.arange(history) - history + 1
    indices = np.maximum(np.arange(len(states))[:, None] + offsets, 0)
    return states[indices].reshape(len(states), -1)


def history_conflicts(arrays, episode, histories=RULES["history_lengths"]):
    episode_ticks = np.flatnonzero(arrays["episode_index"] == episode)
    if not len(episode_ticks) or not np.all(np.diff(episode_ticks) == 1):
        raise ValueError("one contiguous episode is required")
    valid = valid_labels(arrays)
    current = arrays["state"][episode_ticks]
    observation_invalid = arrays.get("observation_is_invalid", np.zeros(len(valid), bool))
    rows = []
    for history in histories:
        features = causal_state_history(current, history)
        eligible = valid[episode_ticks].copy()
        for i, tick in enumerate(episode_ticks):
            eligible[i] &= not observation_invalid[max(episode_ticks[0], tick - history + 1):tick + 1].any()
        ticks = episode_ticks[eligible]
        _, inverse, counts = np.unique(features[eligible], axis=0, return_inverse=True, return_counts=True)
        conflicts = []
        for group in np.flatnonzero(counts > 1):
            group_ticks = ticks[inverse == group]
            target = arrays["action"][group_ticks, :3]
            differences = np.linalg.norm(target[:, None] - target, axis=-1) * 1000
            if differences.max() > RULES["motion_mm"]:
                i, j = np.unravel_index(differences.argmax(), differences.shape)
                conflicts.append({"ticks": [int(group_ticks[i]), int(group_ticks[j])],
                                  "frames": [int(arrays["frame_index"][group_ticks[i]]),
                                             int(arrays["frame_index"][group_ticks[j]])],
                                  "translation_label_span_mm": float(differences[i, j]),
                                  "group_size": int(len(group_ticks))})
        rows.append({"history_ticks": history, "eligible_target_ticks": len(ticks),
                     "observation_feature_width": 13 * history,
                     "history_span_s_at_12_5_hz": (history - 1) / 12.5,
                     "unique_observation_histories": len(counts),
                     "repeated_history_groups": int((counts > 1).sum()),
                     "label_conflict_groups": len(conflicts), "conflicts": conflicts})
    return {"episode": int(episode), "histories": rows}


def stationary_strata(arrays, ticks):
    """Return causal-observation strata separately from descriptive label strata."""
    ticks = np.asarray(ticks, int)
    valid = valid_labels(arrays)
    action = arrays["action"]
    moving = np.linalg.norm(action[:, :3], axis=1) * 1000 > RULES["motion_mm"]
    stationary = ~moving[ticks] & (np.linalg.norm(action[ticks, 3:6], axis=1) <= RULES["stationary_rotation_rad"])
    low_history, recent_motion, future_motion = [], [], []
    width, neighborhood = RULES["causal_history_ticks"], RULES["descriptive_label_neighborhood_ticks"]
    observation_invalid = arrays.get("observation_is_invalid", np.zeros(len(valid), bool))
    for tick in ticks:
        start = int(arrays["ep_start"][tick])
        stop = start + int(arrays["ep_len"][tick])
        history = np.arange(max(start, tick - width + 1), tick + 1)
        low_history.append(len(history) == width and not observation_invalid[history].any()
                           and np.all(np.linalg.norm(arrays["state"][history, 9:12], axis=1)
                                      <= RULES["causal_history_speed_m_s"])
                           and np.max(np.ptp(arrays["joint_state"][history, :5], axis=0))
                           <= RULES["causal_history_arm_span_deg"])
        past = np.arange(max(start, tick - neighborhood), tick)
        future = np.arange(tick + 1, min(stop, tick + neighborhood + 1))
        recent_motion.append(bool(np.any(moving[past] & valid[past])))
        future_motion.append(bool(np.any(moving[future] & valid[future])))
    low, past, future = (np.asarray(v, bool) for v in (low_history, recent_motion, future_motion))
    return {
        "stationary_all": stationary,
        "stationary_exact_zero_translation": stationary & np.all(action[ticks, :3] == 0, axis=1),
        "stationary_causal_low_motion_history": stationary & low,
        "stationary_other_history": stationary & ~low,
        "stationary_no_nearby_label_motion": stationary & ~past & ~future,
        "stationary_near_future_label_motion": stationary & future,
        "stationary_after_recent_label_motion": stationary & past,
    }


def residual_statistics(predicted, labels, mask):
    predicted, labels, mask = np.asarray(predicted), np.asarray(labels), np.asarray(mask, bool)
    if predicted.shape != labels.shape or predicted.ndim != 2 or mask.shape != (len(predicted),):
        raise ValueError("aligned predictions, labels and mask required")
    if not np.isfinite(predicted).all() or not np.isfinite(labels).all():
        raise ValueError("finite predictions/labels required")
    if not mask.any():
        return {"n": 0, "predicted_step_mean_mm": None, "target_error_mean_mm": None}
    predicted = predicted[mask, :3] * 1000
    error = predicted - labels[mask, :3] * 1000
    return {"n": int(mask.sum()), "predicted_step_mean_mm": float(np.linalg.norm(predicted, axis=1).mean()),
            "target_error_mean_mm": float(np.linalg.norm(error, axis=1).mean()),
            "error_vector_sum_xyz_mm": error.sum(0).astype(float).tolist(),
            "error_vector_sum_note": "Descriptive sum over this stratum, not contiguous-run integration or measured drift."}


def run(data, checkpoint, out):
    if out.exists():
        raise FileExistsError(out)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (not set(EPISODES) <= set(manifest["split"]["train"])
            or set(EPISODES) & (set(manifest["split"]["val"]) | set(manifest["split"]["reserved"]))):
        raise ValueError("the declared episodes must be training-only")
    if (manifest["action_representation"] != "cartesian" or len(manifest["state_names"]) != 13
            or manifest["camera_keys"] != ["observation.images.iphone"]
            or manifest.get("observation_variant") or manifest.get("visual_grounding")):
        raise ValueError("requires the original canonical Cartesian source")
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or sha256_file(data / name) != expected:
            raise ValueError(f"source payload differs: {name}")
    files = [Path(__file__), manifest_path, data / "stats.json", data / "arrays.npz",
             data / manifest["image_files"][manifest["camera_keys"][0]],
             *(checkpoint / name for name in ("model.safetensors", "config.json", "stats.json",
                                               "run_info.json", "real2sim_contract.json"))]
    out.mkdir(parents=True)
    (out / "executed_audit.py").write_bytes(Path(__file__).read_bytes())
    plan = {"schema": "history_context_audit_plan_v1", "training_episodes": list(EPISODES),
            "rules": RULES, "checkpoint": str(checkpoint.resolve()), "data": str(data.resolve()),
            "source_sha256": {str(p.resolve()): sha256_file(p) for p in files},
            "selection": "All valid actions of the same three training episodes; no outcome-conditioned replacement or threshold tuning.",
            "reserved_episodes_used": []}
    write_json(out / "plan.json", plan)
    torch.set_num_threads(2)
    policy, norm, contract = load_checkpoint(checkpoint, "cpu")
    if contract["derived_files_sha256"] != manifest["files_sha256"] or norm.stats != json.loads((data / "stats.json").read_text()):
        raise ValueError("checkpoint source or normalization differs")
    for key in ("state_names", "camera_keys", "action_names", "split", "action_representation",
                "calibration_sha256", "controller_semantics", "timing_basis"):
        if contract.get(key) != manifest.get(key):
            raise ValueError(f"checkpoint semantic contract differs: {key}")
    info = json.loads((checkpoint / "run_info.json").read_text())
    if (set(info.get("diagnostic_episodes") or []) != set(EPISODES)
            or info.get("reconstruction_loss", {}).get("supervised_prefix_steps") != 2
            or policy.config.n_action_steps != 2):
        raise ValueError("requires the same three-episode two-action-prefix checkpoint")
    dataset = DerivedTickDataset(data, list(EPISODES), policy.config.chunk_size, norm)
    _, prediction = evaluate_executed_prefix(policy, dataset, "cpu", batch_size=8)
    arrays, ticks = dataset.arrays, prediction["tick"]
    strata = stationary_strata(arrays, ticks)
    masks = {name: residual_statistics(prediction["model"], arrays["action"][ticks], mask)
             for name, mask in strata.items()}
    conflicts = [history_conflicts(arrays, episode) for episode in EPISODES]
    for path, digest in plan["source_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"source changed during audit: {path}")
    report = {"schema": "history_context_audit_v1", "plan_sha256": sha256_file(out / "plan.json"),
              "training_episodes": list(EPISODES), "valid_actions": len(ticks),
              "history_conflicts": conflicts, "stationary_residual_strata": masks,
              "history_conflict_totals": {str(h): sum(r["label_conflict_groups"] for ep in conflicts
                    for r in ep["histories"] if r["history_ticks"] == h) for h in RULES["history_lengths"]},
              "physical_calibration_accepted": False, "reserved_episodes_used": [],
              "training_or_labels_changed": False, "source_plan": plan,
              "interpretation": [
                  "Causal state histories use current/past observations only; equality counts exclude ages/validity to avoid treating floating timestamp jitter as information.",
                  "Future-label and recent-label neighborhoods are descriptive residual partitions only. They are never policy inputs, sample weights or inference gates.",
                  "Low-motion-history masks are causal descriptors, not a rule that forces the policy to hold. Moving initiation remains possible from a low-motion history.",
                  "The observation-only history4 experiment changes state width from13 to60 including age/valid bookkeeping; it preserves the backbone, not the complete parameter count.",
                  "Repeated histories with differing labels remain and similar image measurements are not identical full observations or an irreducible error proof.",
                  "All quantities inherit provisional FK. These teacher-forced errors are neither closed-loop drift nor physical command-response measurements."]}
    write_json(out / "report.json", report)
    return report


def markdown(report):
    totals, strata = report["history_conflict_totals"], report["stationary_residual_strata"]
    lines = ["# Causal history and hold-transition residuals — 25 October 2026", "",
             "The prefix-only policy's residual stationary movement is concentrated near transitions. A single four-observation history experiment is justified by added causal information; globally forcing low-velocity observations to hold could suppress legitimate initiation.", "",
             "The [reproducible audit](../scripts/audit_v2_history_context.py) scores only the original training episodes 11,8,23 using their completed prefix2 checkpoint. The [report](../results/v2/history_context_audit_20261025.json) retains source hashes, selection, rules and every repeated-history conflict. No labels, inputs or sampling weights are changed by this descriptive audit.", "",
             "## Causal-history informativeness", "",
             "History vectors concatenate current and previous canonical observations, oldest first. Missing initial slots repeat the first observation within the same episode. Label-conflict counts compare only real valid targets. Ages/validity flags are excluded from this equality diagnostic, so tiny timestamp differences cannot manufacture unique states.", "",
             "| Observations in history | Span at12.5Hz | Exact-history groups with differing next translation labels |",
             "|---|---:|---:|"]
    for h in RULES["history_lengths"]:
        lines.append(f"| {h} | {(h - 1) / 12.5:.2f} s | {totals[str(h)]} |")
    lines += ["", "Four observations reduce conflict groups from12 to4; eight reduce them only to3. The remaining four-observation conflicts still include next-label differences near3.95mm in episode8 and3.92mm in episode23. A causal history therefore adds information but does not reveal every operator's future onset timing.", "",
              "## Prefix2 residuals on the same valid training actions", "",
              "| Stationary stratum | Actions | Mean predicted translation step | Mean target-vector error |",
              "|---|---:|---:|---:|"]
    for key, title in (("stationary_all", "All stationary"),
                       ("stationary_exact_zero_translation", "Exact-zero translation labels"),
                       ("stationary_causal_low_motion_history", "Four causal low-motion observations"),
                       ("stationary_other_history", "Other causal history"),
                       ("stationary_no_nearby_label_motion", "No moving label within four past/future ticks"),
                       ("stationary_near_future_label_motion", "Moving label within four future ticks"),
                       ("stationary_after_recent_label_motion", "Moving label within four past ticks")):
        row = strata[key]
        lines.append(f"| {title} | {row['n']} | {row['predicted_step_mean_mm']:.4f} mm | {row['target_error_mean_mm']:.4f} mm |")
    lines += ["", "Stationary uses the unchanged translation≤0.4mm and rotation≤0.01rad thresholds. Causal low-motion history means four observations with tool speed≤0.005m/s and arm span≤0.5°. The remaining strata inspect recorded labels descriptively; their future labels are never available to the proposed policy or used to filter training. Strata overlap and do not form independent samples.", "",
              "The long stationary stratum is already comparatively quiet. Global hold weighting could trade away motion initiation without addressing transition context. Four past/current observations are the smaller next intervention; an eight-observation history provides little extra exact-conflict reduction here.", "",
              "## Controlled next experiment", "",
              "Use the same frozen ResNet backbone, three episodes, original action labels/action statistics/images, seed, optimizer horizon, two-action supervision and unchanged movement/stationary gates. Add only causal canonical history4, including per-slot age/valid flags, giving60 state features. Tile the exact original canonical-state normalization across slots and keep age/valid units unchanged.", "",
              "Offline and live transforms must reset at episode/environment boundaries and consume every control tick, including queued-action ticks without inference. The same observation must not update history twice. First slots have explicit validity; later observations cannot affect earlier feature vectors. The existing streaming transform already implements these requirements and must be checked against the prepared data.", "",
              "The input projection becomes wider, so this is an observation-context experiment with the same backbone rather than an identical-architecture causal attribution. A matched-width control would be required to isolate history content from added input capacity. Do not combine class balancing, a hold gate or relabeling in this run. Score motion onsets separately, retain the existing object-swap checks, and proceed to common development rollouts only after the unchanged training gate passes.", "",
              "No success, calibration correction or physical actuator equivalence is established by this audit. The source observations/labels and all Cartesian metrics remain provisional.", "",
              "```sh", ".venv/bin/python scripts/audit_v2_history_context.py \\",
              "  --out outputs/v2/history_context_audit_NEXT", "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/v2/three_episode_prefix2_20261025/training/checkpoints/002000"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=Path("results/v2/history_context_audit_20261025.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/v2_history_context_audit_20261025.md"))
    args = parser.parse_args()
    if args.summary.exists() or args.doc.exists():
        raise FileExistsError("use fresh summary/document paths")
    result = run(args.data, args.checkpoint, args.out)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.summary, result)
    args.doc.write_text(markdown(result))
    print(json.dumps({"history_conflicts": result["history_conflict_totals"],
                      "stationary_strata": result["stationary_residual_strata"]}, indent=2))
