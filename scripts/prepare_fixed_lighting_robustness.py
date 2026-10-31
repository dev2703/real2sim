"""Predeclare paired layout, size and friction tests with unchanged lighting.

Nominal source scenes stay untouched. Radius variants rebuild object geometry,
inertia, spawn heights and scoring using the declared size, at constant mass.
Friction variants use an explicit model hook: scaling just the ball material
would not consistently change contacts under MuJoCo's maximum-coefficient rule.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml

from real2sim.artifacts import sha256_file
from real2sim.config import load_yaml, project_path
from real2sim.serialization import write_json

SOURCE = Path("configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml")
CONDITIONS = {
    "nominal": {"ball_disc_m": .03, "bowl_disc_m": 0., "radius_scale": 1., "sliding_scale": 1.},
    "bowl_jitter": {"ball_disc_m": .03, "bowl_disc_m": .03, "radius_scale": 1., "sliding_scale": 1.},
    "layout_extended": {"ball_disc_m": .05, "bowl_disc_m": .03, "radius_scale": 1., "sliding_scale": 1.},
    "ball_small": {"ball_disc_m": .03, "bowl_disc_m": .03, "radius_scale": .95, "sliding_scale": 1.},
    "ball_large": {"ball_disc_m": .03, "bowl_disc_m": .03, "radius_scale": 1.05, "sliding_scale": 1.},
    "friction_low": {"ball_disc_m": .03, "bowl_disc_m": .03, "radius_scale": 1., "sliding_scale": 2 / 3},
    "friction_high": {"ball_disc_m": .03, "bowl_disc_m": .03, "radius_scale": 1., "sliding_scale": 1.5},
}


def disc_draw(rng, radius):
    radial, angle = radius * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    return radial * np.array([np.cos(angle), np.sin(angle)])


def scenarios(task, seeds, factors):
    """Independent deterministic ball/bowl draws, shared across both arms."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply nonempty unique layout seeds")
    ball = np.asarray(task["ball"]["start_region"]["center_task_xy"], float)
    bowl = np.asarray(task["bowl"]["center_task_xy"], float)
    return {str(seed): {
        "ball_xy": (ball + disc_draw(np.random.default_rng(seed), factors["ball_disc_m"])).tolist(),
        "bowl_xy": (bowl + disc_draw(np.random.default_rng(np.random.SeedSequence([seed, 20261028])), factors["bowl_disc_m"])).tolist(),
    } for seed in seeds}


def apply_model_condition(env, model_effects):
    """Scale collision sliding friction once, never any lighting coefficients."""
    effects = dict(model_effects or {})
    if unknown := effects.keys() - {"collision_sliding_scale"}:
        raise ValueError(f"Unknown model effects: {sorted(unknown)}")
    scale = float(effects.get("collision_sliding_scale", 1.))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Sliding friction scale must be positive and finite")
    model = env.model
    enabled = (model.geom_contype | model.geom_conaffinity) != 0
    before = model.geom_friction.copy()
    model.geom_friction[enabled, 0] *= scale
    return {"collision_sliding_scale": scale,
            "collision_geom_ids": np.flatnonzero(enabled).tolist(),
            "geom_friction_before": before.tolist(),
            "geom_friction_after": model.geom_friction.copy().tolist(),
            "scope": "All collision-enabled geoms, including robot/static contacts; torsional/rolling coefficients unchanged",
            "lighting_changed": False}


def prepare(out, *, source_scene=SOURCE, seeds=tuple(range(1000, 1100)), conditions=None):
    if out.exists():
        raise FileExistsError(out)
    requested = tuple(CONDITIONS) if conditions is None else tuple(conditions)
    if not requested or len(set(requested)) != len(requested) or set(requested) - CONDITIONS.keys():
        raise ValueError("Supply unique known robustness conditions")
    sim = load_yaml(source_scene)
    task_path = project_path(sim["task_config"])
    task = load_yaml(task_path)
    protocol = {"schema": "fixed_lighting_paired_robustness_v1",
        "prepared_before_rollouts": True, "lighting_conditions_changed": False,
        "nominal_source_scene": str(source_scene), "seeds": list(seeds),
        "conditions": {}, "source_sha256": {str(p): sha256_file(p) for p in
            (Path(__file__), source_scene, task_path, Path("physical_measurements.md"))},
        "size_protocol": "Rebuild true geometry/inertia/spawn/scorer at constant mass. The runner must separately disclose whether perception knows the varied radius; this plan alone does not freeze perception's radius assumption.",
        "friction_protocol": "Apply model_effects via apply_model_condition once after model construction. Only sliding friction of all collision-enabled geometries changes; robot/static contacts also vary.",
        "layout_protocol": "Uniform-area discs; default_rng(seed) ball and default_rng(SeedSequence([seed,20261028])) bowl; identical explicit XY pairs for both arms and all controllers.",
        "limitations": "Extended layouts are stress tests, not guaranteed feasible starts. Every failure/abort must remain in the denominator. No new physical-registration claim or lighting perturbation.",
    }
    out.mkdir(parents=True)
    for name in requested:
        factors = CONDITIONS[name]
        variant = copy.deepcopy(task)
        variant["ball"]["radius"] *= factors["radius_scale"]
        variant["ball"]["start_region"]["radius"] = factors["ball_disc_m"]
        if "radius_interval_m" in variant["ball"]:
            variant["ball"]["radius_interval_m"] = [v * factors["radius_scale"] for v in variant["ball"]["radius_interval_m"]]
        variant["robustness_condition"] = {"name": name, **factors,
            "lighting_changed": False, "nominal_measurements_replaced": False}
        directory = out / name
        directory.mkdir()
        config = copy.deepcopy(sim)
        generated_task = directory / "task.yaml"
        generated_task.write_text(yaml.safe_dump(variant, sort_keys=False))
        config["task_config"] = str(generated_task.resolve())
        generated_scene = directory / "sim_so101.yaml"
        generated_scene.write_text(yaml.safe_dump(config, sort_keys=False))
        protocol["conditions"][name] = {"source_scene": str(generated_scene.resolve()),
            "seeds": list(seeds), "scenarios": scenarios(task, seeds, factors),
            "model_effects": {"collision_sliding_scale": factors["sliding_scale"]},
            "factors": factors, "generated_sha256": {str(p): sha256_file(p) for p in (generated_task, generated_scene)}}
    write_json(out / "protocol.json", protocol, atomic=True)
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-scene", type=Path, default=SOURCE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1100)))
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS))
    args = parser.parse_args()
    prepare(args.out, source_scene=args.source_scene, seeds=args.seeds, conditions=args.conditions)
