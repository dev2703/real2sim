"""Record layout novelty against available pre-evaluation JSON evidence.

This is an artifact audit, not a guarantee about unrecorded runs or complete
physical scene registration. It does not open a simulator or run a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from real2sim.artifacts import sha256_file
from real2sim.serialization import write_json

ROOT = Path(__file__).resolve().parents[1]
TASK = Path("configs/v2/task.scalar_measured_20261026.yaml")
DEFAULT_OUT = Path("results/v2/generalization_layout_novelty_20261027.json")
SEEDS = tuple(range(10000, 10020))
SEED_KEYS = {
    "seed",
    "seeds",
    "train_seeds_declared_before_collection",
    "validation_seeds_declared_before_collection",
    "development_seeds_not_collected",
    "development_rollout_seeds",
    "train",
    "val",
    "reserved",
}


def disc(rng, radius):
    """Match environment reset draw order: radial uniform, then angle."""
    distance = radius * np.sqrt(rng.uniform())
    angle = rng.uniform(0, 2 * np.pi)
    return distance * np.array([np.cos(angle), np.sin(angle)])


def planned_layouts(task):
    region = task["ball"]["start_region"]
    ball_center = np.asarray(region["center_task_xy"], float)
    bowl_center = np.asarray(task["bowl"]["center_task_xy"], float)
    result = []
    for seed in SEEDS:
        ball = ball_center + disc(np.random.default_rng(seed), float(region["radius"]))
        bowl_offset = disc(np.random.default_rng(np.random.SeedSequence([seed, 20261027])), 0.03)
        result.append(
            {
                "seed": seed,
                "ball_xy": ball.tolist(),
                "bowl_xy": (bowl_center + bowl_offset).tolist(),
                "bowl_offset_xy_m": bowl_offset.tolist(),
            }
        )
    return result


def coordinates(value, first, second):
    try:
        ball, bowl = np.asarray(value[first], float), np.asarray(value[second], float)
        if ball.ndim == bowl.ndim == 1 and len(ball) >= 2 and len(bowl) >= 2:
            pair = np.r_[ball[:2], bowl[:2]]
            if np.isfinite(pair).all():
                return pair
    except (KeyError, ValueError, TypeError):
        pass
    return None


def inspect(value, path, seeds, explicit, location="$", initial_context=False):
    if isinstance(value, dict):
        if initial_context:
            pair = coordinates(value, "ball_xyz_task", "bowl_xyz_task")
            if pair is not None:
                explicit.append(
                    {
                        "source": path,
                        "json_path": location,
                        "ball_xy": pair[:2].tolist(),
                        "bowl_xy": pair[2:].tolist(),
                    }
                )
        pair = coordinates(value, "ball_xy", "bowl_xy")
        if pair is not None:
            explicit.append(
                {
                    "source": path,
                    "json_path": location,
                    "ball_xy": pair[:2].tolist(),
                    "bowl_xy": pair[2:].tolist(),
                }
            )
        for key, item in value.items():
            if key in SEED_KEYS:
                candidates = item if isinstance(item, list) else [item]
                for candidate in candidates:
                    if isinstance(candidate, int) and not isinstance(candidate, bool):
                        seeds.setdefault(candidate, set()).add(path)
            inspect(
                item,
                path,
                seeds,
                explicit,
                f"{location}.{key}",
                initial_context
                or key.lower().startswith("initial")
                or key.lower() in {"start_metrics", "start_scoring_only"},
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            inspect(item, path, seeds, explicit, f"{location}[{index}]", initial_context)


def excluded(path, output):
    relative = path.relative_to(ROOT)
    if path.resolve() == output.resolve():
        return "audit output itself"
    # Include prior 26 October geometric generalization evidence. Exclude only
    # new 27 October generalization products to preserve a pre-evaluation audit.
    if any("generalization" in part and "20261027" in part for part in relative.parts):
        return "new 27 October generalization artifact"
    return None


def run(output):
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise FileExistsError(output)
    task_path = ROOT / TASK
    task = yaml.safe_load(task_path.read_text())
    planned = planned_layouts(task)
    seen, explicit, inventory, exclusions, errors = {}, [], [], [], []
    for directory in (ROOT / "outputs/v2", ROOT / "results/v2"):
        for path in sorted(directory.rglob("*.json")):
            relative = str(path.relative_to(ROOT))
            reason = excluded(path, output)
            if reason:
                exclusions.append({"path": relative, "reason": reason})
                continue
            if path.stat().st_size > 50_000_000:
                exclusions.append({"path": relative, "reason": "JSON exceeds 50 MB audit budget"})
                continue
            entry = {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            inventory.append(entry)
            try:
                inspect(json.loads(path.read_text()), relative, seen, explicit)
            except (ValueError, OSError) as exc:
                errors.append({"path": relative, "error": f"{type(exc).__name__}: {exc}"})
    inventory.sort(key=lambda item: item["path"])
    inventory_hash = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    overlap = sorted(set(SEEDS) & seen.keys())
    historical_seeds = sorted(seed for seed in seen if 0 <= seed <= 1_000_000)
    region = task["ball"]["start_region"]
    center, bowl = (
        np.asarray(region["center_task_xy"], float),
        np.asarray(task["bowl"]["center_task_xy"], float),
    )
    reconstructed = [
        {
            "seed": seed,
            "ball_xy": (
                center + disc(np.random.default_rng(seed), float(region["radius"]))
            ).tolist(),
            "bowl_xy": bowl.tolist(),
        }
        for seed in historical_seeds
    ]
    references = reconstructed + explicit
    planned_array = np.array([row["ball_xy"] + row["bowl_xy"] for row in planned])
    comparisons = []
    if references:
        reference_array = np.array([row["ball_xy"] + row["bowl_xy"] for row in references])
        for row, point in zip(planned, planned_array, strict=True):
            differences = point - reference_array
            pair_distances = np.linalg.norm(differences, axis=1)
            closest = int(np.argmin(pair_distances))
            comparisons.append(
                {
                    "seed": row["seed"],
                    "minimum_joint_ball_bowl_xy_l2_m": float(pair_distances[closest]),
                    "nearest_reference": references[closest],
                    "exact_pair_matches_at_1e_9_m": int(
                        np.sum(np.max(np.abs(differences), axis=1) <= 1e-9)
                    ),
                    "minimum_ball_xy_distance_m": float(
                        np.linalg.norm(differences[:, :2], axis=1).min()
                    ),
                }
            )
    duplicates = sum(row["exact_pair_matches_at_1e_9_m"] for row in comparisons)
    report = {
        "schema": "generalization_layout_novelty_20261027_v1",
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "audit_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "audit_script_sha256": sha256_file(__file__),
        "source_roots": ["outputs/v2", "results/v2"],
        "source_inventory": inventory,
        "source_inventory_sha256": inventory_hash,
        "json_files_scanned": len(inventory),
        "parse_errors": errors,
        "exclusions": exclusions,
        "seed_keys_inspected": sorted(SEED_KEYS),
        "known_seed_or_episode_values": sorted(seen),
        "known_seed_or_episode_sources": {
            str(seed): sorted(paths) for seed, paths in sorted(seen.items())
        },
        "planned_seed_overlap_with_prior_evidence": overlap,
        "planned_seeds_absent_from_scanned_prior_evidence": not overlap,
        "task": str(TASK),
        "task_sha256": sha256_file(task_path),
        "layout_protocol": {
            "ball": "Unchanged environment uniform-area start disc; default_rng(seed), radial draw then angle.",
            "bowl": "Independent uniform-area 0.03 m disc about nominal bowl XY; default_rng(SeedSequence([seed, 20261027])), radial draw then angle.",
            "robots": "Identical explicit ball/bowl XY schedule supplied to both arms.",
            "selection": "Fixed seeds 10000 through 10019; no outcomes used to choose or replace layouts.",
        },
        "planned_layouts": planned,
        "historical_default_layouts_reconstructed": reconstructed,
        "explicit_initial_or_scenario_layouts_found": len(explicit),
        "explicit_initial_or_scenario_layouts": explicit,
        "coordinate_comparisons": comparisons,
        "exact_pair_matches_in_available_references": duplicates,
        "claim_supported": not overlap and duplicates == 0 and not errors,
        "limitations": [
            "Novelty is established only against scanned local JSON evidence, not unrecorded or external runs.",
            "Seed/episode fields are collected conservatively; they include training and audit random seeds that may not identify simulator layouts.",
            "Historical default-layout reconstruction applies the current task distribution to known seed/episode values; past configs and explicit overrides can differ.",
            "Explicit coordinates are extracted only from initial-context ball_xyz_task/bowl_xyz_task pairs or paired ball_xy/bowl_xy fields. Missing coordinates, bowl-only schedules, traces, images and videos are not exhaustively reconstructed.",
            "No exact coordinate match is a bookkeeping result, not evidence of physical scene registration or large separation from training positions.",
            "The schedule uses existing ball jitter and an independently declared bowl-shift stress test. It is a simulation distribution, not a newly measured physical distribution.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "json_files_scanned",
                    "planned_seed_overlap_with_prior_evidence",
                    "explicit_initial_or_scenario_layouts_found",
                    "exact_pair_matches_in_available_references",
                    "claim_supported",
                    "source_inventory_sha256",
                )
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run(parser.parse_args().out)
