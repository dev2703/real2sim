"""SO-101 v2 experiment definitions, evidence gates and clustered evaluation.

No physical calibration values or experimental outcomes are inferred here.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from real2sim.data.conditions import CONDITIONS, manifest_hash

CAMERAS = ["observation.images.iphone", "observation.images.top"]
SCREEN_ANCHORS = 64_000
FINAL_ANCHORS = 320_000
VALIDATE_ANCHORS = 8_000


def experiment_matrix(
    base: dict,
    *,
    backbone: str = "resnet18",
    head_lr: float = 1e-4,
    effective_batch: int = 16,
    physical_batch: int = 8,
    clip_model: dict | None = None,
) -> list[dict]:
    """Resolve adaptive arms explicitly; never select a winner from training loss."""
    if backbone not in ("resnet18", "dinov2") or effective_batch not in (16, 64):
        raise ValueError("choose a measured ResNet/DINO baseline and batch 16 or 64")
    if not math.isfinite(head_lr) or head_lr <= 0 or physical_batch < 1:
        raise ValueError("positive finite head LR and physical batch required")
    arms = []

    def add(
        name,
        stage,
        cameras=CAMERAS,
        vision=backbone,
        batch=effective_batch,
        lr=head_lr,
        action="cartesian",
        blocks=0,
        backbone_lr=1e-5,
        family="act",
    ):
        if batch % physical_batch or SCREEN_ANCHORS % batch or VALIDATE_ANCHORS % batch:
            raise ValueError("physical batch must divide effective batch; exposures must be exact")
        cfg = copy.deepcopy(base)
        cfg.update(
            name=name, data=f".cache/v2/{action}_{'_'.join(k.split('.')[-1] for k in cameras)}"
        )
        cfg["model"].update(
            vision_backbone=vision, optimizer_lr=lr, optimizer_lr_backbone=backbone_lr
        )
        if vision == "resnet18":
            cfg["model"] = {k: v for k, v in cfg["model"].items() if not k.startswith(("dinov2_", "clip_"))}
            cfg["model"].update(
                pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1", freeze_backbone=True
            )
        elif vision == "clip":
            if clip_model is None:
                raise ValueError("CLIP experiments require a pinned architecture/configuration")
            cfg["model"] = {k: v for k, v in cfg["model"].items() if not k.startswith("dinov2_")}
            cfg["model"].pop("freeze_backbone", None)
            cfg["model"].update(copy.deepcopy(clip_model))
            cfg["model"].update(pretrained_backbone_weights=None, clip_trainable_blocks=blocks)
        else:
            cfg["model"].pop("freeze_backbone", None)
            cfg["model"].update(pretrained_backbone_weights=None, dinov2_trainable_blocks=blocks)
        cfg["train"].update(
            batch_size=physical_batch,
            grad_accum=batch // physical_batch,
            steps=SCREEN_ANCHORS // batch,
            schedule_steps=FINAL_ANCHORS // batch,
            val_every=VALIDATE_ANCHORS // batch,
            save_every=VALIDATE_ANCHORS // batch,
        )
        cfg["scheduler"]["warmup_steps"] = int(0.05 * FINAL_ANCHORS) // batch
        arms.append(
            {
                "id": name,
                "stage": stage,
                "family": family,
                "camera_keys": list(cameras),
                "action_representation": action,
                "controller_semantics": "accepted_target"
                if action in ("cartesian", "cartesian_tool")
                else "current_pose",
                "examples": SCREEN_ANCHORS,
                "seed": 0,
                "effective_batch": batch,
                "config": cfg,
                "status": "requires_evidence",
            }
        )

    for camera in ([CAMERAS[0]], [CAMERAS[1]], CAMERAS):
        add(
            "L1_resnet_" + "_".join(k.split(".")[-1] for k in camera),
            "L1",
            camera,
            "resnet18",
            16,
            1e-4,
        )
    add("L2_dino_frozen", "L2", vision="dinov2", batch=16, lr=1e-4)
    for batch in (16, 64):
        for lr in (3e-5, 1e-4, 3e-4):
            # The identical L1 dual ResNet / L2 DINO control is reused.
            if batch == 16 and lr == 1e-4:
                continue
            add(f"L3_{backbone}_b{batch}_lr{lr:g}", "L3", batch=batch, lr=lr)
    add("L4_dino_frozen_control", "L4", vision="dinov2")
    for lr in (3e-6, 1e-5):
        add(f"L4_dino_last2_lr{lr:g}", "L4", vision="dinov2", blocks=2, backbone_lr=lr)
    for action in ("canonical_joint", "cartesian", "cartesian_absolute"):
        add(f"L5_{action}", "L5", action=action)
    for objective in ("regression", "flow"):
        add(f"L6_dino_{objective}", "L6", vision="dinov2", family=objective)
    if clip_model is not None:
        add("L7_clip_frozen", "L7", vision="clip")
        for lr in (3e-6, 1e-5):
            add(f"L7_clip_last2_lr{lr:g}", "L7", vision="clip", blocks=2, backbone_lr=lr)
        for objective in ("regression", "flow"):
            add(f"L8_clip_{objective}", "L8", vision="clip", family=objective)
        # Compare coordinates with the selected baseline; retain the existing task-frame control.
        add("K1_tool_delta", "K1", action="cartesian_tool")
    # Record aliases where an adaptive control exactly matches an existing ACT arm.
    identities = {}
    for arm in arms:
        identity = manifest_hash(
            {
                "family": arm["family"],
                "model": arm["config"]["model"],
                "train": arm["config"]["train"],
                "scheduler": arm["config"]["scheduler"],
                "data": arm["config"]["data"],
            }
        )
        if identity in identities:
            arm["reuse"] = identities[identity]
        else:
            identities[identity] = arm["id"]
    return arms


def gate_failures(evidence: dict, *, include_tiny_fit=True) -> list[str]:
    """Evaluate numeric evidence, rather than accepting a human-written `passed: true`."""
    failures = []

    def check(ok, reason):
        if not ok:
            failures.append(reason)

    def finite(value):
        return (
            isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
        )

    def bounded(value, bound):
        return finite(value) and 0 <= value <= bound

    starts = evidence.get("starts", {})
    check(
        starts.get("all_valid") is True
        and starts.get("reference_clipped") is False
        and bool(starts.get("task_start_version")),
        "valid, unclipped, versioned task starts",
    )
    holds = evidence.get("unloaded_holds", [])
    check(
        len(holds) == 3
        and len({r.get("pose_id") for r in holds}) == 3
        and all(
            finite(r.get("duration_s"))
            and r["duration_s"] >= 20
            and bounded(r.get("max_drift_mm"), 0.5)
            and r.get("ik_rejections") == 0
            for r in holds
        ),
        "three distinct 20-second unloaded holds: drift <=0.5 mm, zero IK rejections",
    )
    calibration = evidence.get("calibration", {})
    check(
        calibration.get("independent_verification") is True
        and calibration.get("jaw_verified") is True
        and all(bounded(calibration.get("landmark_p95_source_px", {}).get(k), 10) for k in CAMERAS)
        and bounded(calibration.get("object_xy_error_mm"), 5),
        "independent two-camera, jaw and tabletop verification",
    )
    references = evidence.get("references", [])
    check(
        len(references) == 10
        and len({r.get("episode_index") for r in references}) == 10
        and set(r.get("condition") for r in references) == set(CONDITIONS)
        and sum(r.get("pick_place_success") is True for r in references) >= 9
        and all(bounded(r.get("closure_transport_error_p95_mm"), 3) for r in references),
        "ten reference episodes across conditions: >=9 placements, <=3 mm closure/transport p95",
    )
    retention = evidence.get("retention", {})
    check(
        finite(retention.get("demonstrated_loaded_s"))
        and retention["demonstrated_loaded_s"] > 0
        and finite(retention.get("sim_retained_s"))
        and retention["sim_retained_s"] >= retention["demonstrated_loaded_s"],
        "retention through demonstrated loaded duration",
    )
    if include_tiny_fit:
        fits = evidence.get("tiny_fit", [])
        check(
            len(fits) == 3
            and set(r.get("condition") for r in fits) == set(CONDITIONS)
            and all(
                bounded(r.get("translation_error_mm"), 0.999999)
                and bounded(r.get("gripper_mae"), 0.02999999)
                and finite(r.get("examples_seen"))
                and 0 < r["examples_seen"] <= 32000
                for r in fits
            ),
            "three training-condition tiny fits: <1 mm translation, <0.03 gripper MAE, <=32000 anchors",
        )
    return failures


def freeze_bundle(paths: dict[str, str | Path]) -> dict:
    from real2sim.data.raw import sha256_file

    files = {
        name: {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}
        for name, path in sorted(paths.items())
    }
    result = {"version": "so101_v2_12p5_v1", "files": files}
    result["sha256"] = manifest_hash(result)
    return result


def verify_bundle(bundle: dict) -> None:
    from real2sim.data.raw import sha256_file

    if bundle.get("sha256") != manifest_hash(bundle):
        raise ValueError("frozen bundle checksum mismatch")
    for name, entry in bundle["files"].items():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"frozen {name} changed; create a new experiment version")


def final_summary(rows: list[dict], *, bootstrap_seed=0) -> dict:
    """90 scenario clusters shared across seeds; no 270-independent-trial interval."""
    by_case = {}
    for row in rows:
        seed = row["training_seed"]
        if seed not in (0, 1, 2) or row["condition"] not in CONDITIONS:
            raise ValueError(
                "final evaluation requires training seeds 0,1,2 and declared conditions"
            )
        key = (row["condition"], row["scenario_id"])
        group = by_case.setdefault(key, {})
        if seed in group or not isinstance(row["pick_place_success"], bool):
            raise ValueError("duplicate scenario/seed or non-boolean success")
        if row.get("timeout_s") != 45:
            raise ValueError("v2 confirmation uses the declared 45-second timeout")
        group[seed] = row["pick_place_success"] and not row.get("abort")
    if len(rows) != 270 or any(set(v) != {0, 1, 2} for v in by_case.values()):
        raise ValueError("expected exactly 90 common scenarios x 3 training seeds")
    matrices = {}
    for condition in CONDITIONS:
        values = [list(v.values()) for (c, _), v in sorted(by_case.items()) if c == condition]
        if len(values) != 30:
            raise ValueError("expected 30 scenarios per condition")
        # Use explicit seed indexing (input row order is irrelevant).
        matrices[condition] = np.array(
            [[v[s] for s in (0, 1, 2)] for (c, _), v in sorted(by_case.items()) if c == condition],
            float,
        )
    all_values = np.concatenate(list(matrices.values()))
    rng = np.random.default_rng(bootstrap_seed)
    samples = np.stack(
        [v[rng.integers(0, 30, size=(10000, 30))].mean(axis=(1, 2)) for v in matrices.values()]
    ).mean(axis=0)
    return {
        "n_scenarios": 90,
        "n_rollouts": 270,
        "mean_success": float(all_values.mean()),
        "milestone_60_percent": bool(all_values.mean() >= 0.6),
        "per_seed": {str(s): float(all_values[:, s].mean()) for s in range(3)},
        "per_condition_seed": {
            c: {str(s): float(v[:, s].mean()) for s in range(3)} for c, v in matrices.items()
        },
        "scenario_cluster_bootstrap_95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "uncertainty_scope": "condition-stratified scenario clusters, shared across the three fixed training seeds; not uncertainty over new training seeds",
    }


def read_json(path):
    return json.loads(Path(path).read_text())


def scenario_suite(registration: dict, *, reserved=False, seed=0) -> dict:
    """Draw common object layouts only from an explicitly measured task distribution."""

    def xy(name):
        value = np.asarray(registration[name], dtype=float)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain two finite task-frame metres")
        return value

    ball, bowl = xy("ball_xy"), xy("bowl_xy")
    ball_shift, bowl_shift = xy("ball_shift_xy"), xy("bowl_shift_xy")
    if not all(np.isclose(np.linalg.norm(v), 0.05, atol=1e-6) for v in (ball_shift, bowl_shift)):
        raise ValueError("confirmed object shift vectors must each be 5 cm")
    jitter = float(registration["ball_jitter_radius_m"])
    if not np.isfinite(jitter) or not 0 < jitter < 0.05:
        raise ValueError(
            "declare a positive, measured development/evaluation ball jitter radius below 5 cm"
        )
    if not registration.get("task_start_version") or registration.get("status") != "measured":
        raise ValueError("scenario registration and task start must be measured/versioned")
    rng = np.random.default_rng(seed + (1_000_000 if reserved else 0))
    n = 30 if reserved else 10
    rows = []
    for condition in CONDITIONS:
        for i in range(n):
            radius, theta = jitter * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
            displacement = radius * np.array([np.cos(theta), np.sin(theta)])
            rows.append(
                {
                    "scenario_id": f"{'reserved' if reserved else 'dev'}_{condition}_{i:03d}",
                    "condition": condition,
                    "seed": int(rng.integers(0, 2**31)),
                    "reset": {
                        "ball_xy": (
                            ball + displacement + (ball_shift if condition == "ball_shift" else 0)
                        ).tolist(),
                        "bowl_xy": (
                            bowl + (bowl_shift if condition == "bowl_shift" else 0)
                        ).tolist(),
                    },
                }
            )
    result = {
        "version": "v2_reserved_v1" if reserved else "v2_development_v1",
        "seed": seed,
        "timeout_s": 45,
        "registration_sha256": manifest_hash(registration),
        "task_start_version": registration["task_start_version"],
        "scenarios": rows,
    }
    result["sha256"] = manifest_hash(result)
    return result


def promote_finalists(matrix: dict, reports: list[dict]) -> dict:
    """Select one ACT and the strongest distinct challenger from common development cases."""
    by_id = {a["id"]: a for a in matrix["arms"]}
    ranked = []
    common = None
    for report in reports:
        arm = by_id[report["arm_id"]]
        if arm.get("reuse"):
            raise ValueError("use the original control report, not an alias as a second candidate")
        rows = report["rows"]
        identities = {(r["condition"], r["scenario_id"]) for r in rows}
        if (
            len(rows) != 30
            or len(identities) != 30
            or any(sum(r["condition"] == c for r in rows) != 10 for c in CONDITIONS)
        ):
            raise ValueError("selection requires 30 unique development cases, ten per condition")
        if any(
            not r["scenario_id"].startswith("dev_") or not isinstance(r["pick_place_success"], bool)
            for r in rows
        ):
            raise ValueError(
                "selection requires development cases with boolean complete-task outcomes"
            )
        signature = (report["scenario_suite_sha256"], identities)
        if common is not None and signature != common:
            raise ValueError("all selection reports must use the identical development suite")
        common = signature
        closure, latency = float(report["closure_error_mm"]), float(report["p95_latency_ms"])
        if not np.isfinite([closure, latency]).all() or closure < 0 or latency <= 0:
            raise ValueError(
                "finite nonnegative closure error and positive measured latency required"
            )
        if latency >= 120:
            continue
        score = sum(r["pick_place_success"] and not r.get("abort") for r in rows) / 30
        ranked.append(((-score, closure, latency), arm, report))
    ranked.sort(key=lambda entry: entry[0])
    best_act = next((r for r in ranked if r[1]["family"] == "act"), None)
    if best_act is None:
        raise ValueError("no ACT candidate meets the measured inference budget")
    challenger = next((r for r in ranked if r[1]["id"] != best_act[1]["id"]), None)
    if challenger is None:
        raise ValueError("no distinct challenger meets the measured inference budget")
    finalists = []
    for _, source, report in (best_act, challenger):
        for seed in (0, 1, 2):
            arm = copy.deepcopy(source)
            arm["id"] = f"finalist_{source['id']}_seed{seed}"
            arm.update(
                seed=seed, stage="finalist", examples=FINAL_ANCHORS, status="requires_evidence"
            )
            train = arm["config"]["train"]
            if train["schedule_steps"] * train["batch_size"] * train["grad_accum"] != FINAL_ANCHORS:
                raise ValueError("screen did not declare the 320000-anchor continuation horizon")
            train.update(seed=seed, steps=FINAL_ANCHORS // arm["effective_batch"])
            arm["config"]["name"] = arm["id"]
            if seed == 0:
                arm["resume_checkpoint"] = report["screen_checkpoint"]
            finalists.append(arm)
    return {
        "version": "v2_finalists_v1",
        "selection_reports_sha256": manifest_hash({"reports": reports}),
        "selection": "complete development task success, then closure error, then inference latency",
        "arms": finalists,
    }
