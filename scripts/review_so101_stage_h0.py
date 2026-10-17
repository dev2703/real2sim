"""Read-only data/checkpoint audit and phase comparison after the H0 screen."""

import hashlib
import json
from pathlib import Path

import numpy as np

from real2sim.data.prepare import compute_stats
from real2sim.policy.observation_variants import (
    ObservationVariant,
    fit_state_statistics,
    transform_observations,
)

ROOT = Path("outputs/experiments_20261016_stage1")
DATA = Path(".cache/experiments_20261016")
ARMS = {
    "A1R": "canonical", "P1": "no_velocity", "P2a": "ema_velocity_05",
    "P2b": "ema_velocity_075", "P3": "joints", "P4": "canonical_joints", "H0": "history3",
}
PHASES = {
    "active": "arm_active", "stationary": "arm_stationary",
    "approach_proxy": "reviewed_stationary_ball_approach_proxy",
    "closure_proxy": "observed_gripper_closure_proxy",
    "upward_closed_proxy": "upward_with_gripper_at_most_45pct_proxy",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    source = dict(np.load(DATA / "canonical/arrays.npz"))
    canonical_manifest = read(DATA / "canonical/manifest.json")
    split = canonical_manifest["split"]
    sets = [set(split[k]) for k in ("train", "val", "historical_val")]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    verified_hashes = {}
    for name, expected in read(ROOT / "QUEUE.json")["identity"]["frozen_files"].items():
        assert sha(Path(name)) == expected, name
        verified_hashes[name] = expected
    contracts = {}
    for arm, variant in ARMS.items():
        data = DATA / variant
        manifest = read(data / "manifest.json")
        stats = read(data / "stats.json")
        arrays = dict(np.load(data / "arrays.npz"))
        checkpoint = ROOT / "runs" / arm / "checkpoints/best"
        contract = read(checkpoint / "real2sim_contract.json")
        assert read(checkpoint / "stats.json") == stats, arm
        for key in ("split", "observation_variant", "state_names", "action_names",
                    "camera_keys", "letterbox", "label_semantics", "policy_hz"):
            assert contract[key] == manifest[key], (arm, key)
        assert contract["derived_files_sha256"] == manifest["files_sha256"], arm
        for name, expected in manifest["files_sha256"].items():
            resolved = str((data / name).resolve())
            if resolved not in verified_hashes:
                assert sha(Path(resolved)) == expected, (arm, name)
                verified_hashes[resolved] = expected
            assert verified_hashes[resolved] == expected
        assert manifest["split"] == split
        for key in source:
            if key != "state":
                np.testing.assert_array_equal(arrays[key], source[key], err_msg=f"{arm}:{key}")
        cfg = ObservationVariant.from_dict(manifest["observation_variant"])
        recomputed = transform_observations(source, cfg)
        np.testing.assert_array_equal(arrays["state"], recomputed, err_msg=arm)
        assert fit_state_statistics(recomputed, arrays["episode_index"], split["train"], cfg) == stats["observation.state"]
        valid_train = np.isin(arrays["episode_index"], split["train"]) & ~arrays["label_is_extrapolated"]
        assert compute_stats(arrays["action"][valid_train]) == stats["action"], arm
        result = read(ROOT / "runs" / arm / "training_result.json")
        assert result["steps"] == 2000, arm
        contracts[arm] = {"variant": variant, "state_dim": len(manifest["state_names"]),
                          "status": "PASS", "steps": result["steps"], "sampled_anchors": 32000,
                          "checkpoint_sha256": sha(checkpoint / "model.safetensors")}

    comparisons = {}
    for section in ("val", "historical_val"):
        baseline = read(ROOT / "results" / f"A1R.{section}.json")
        rows = {}
        for arm in ARMS:
            audit = read(ROOT / "results" / f"{arm}.{section}.json")
            assert audit["checkpoint_sha256"] == contracts[arm]["checkpoint_sha256"]
            assert audit["episodes"] == split[section] == baseline["episodes"]
            assert audit["valid_actions"] == baseline["valid_actions"]
            scores = audit["scores"]["policy"]
            for phase in PHASES.values():
                assert scores[phase]["n"] == baseline["scores"]["policy"][phase]["n"]
            row = {name + "_mm": scores[phase]["translation_vector_error_m"]["mean"] * 1000
                   for name, phase in PHASES.items()}
            row["closure_gripper_mae"] = scores[PHASES["closure_proxy"]]["mae_raw_per_dim"][6]
            counts = np.array([audit["per_episode"][str(ep)]["active"]["n"] for ep in audit["episodes"]])
            difference = np.array([
                (audit["per_episode"][str(ep)]["active"]["translation_vector_error_m"]["mean"]
                 - baseline["per_episode"][str(ep)]["active"]["translation_vector_error_m"]["mean"]) * 1000
                for ep in audit["episodes"]])
            draws = np.random.default_rng(0).integers(0, len(counts), (10000, len(counts)))
            boot = (difference[draws] * counts[draws]).sum(1) / counts[draws].sum(1)
            row["active_delta_vs_A1R_mm"] = float(np.average(difference, weights=counts))
            row["paired_episode_bootstrap_95pct_mm"] = np.quantile(boot, [.025, .975]).tolist()
            rows[arm] = row
        for name in ("zero_motion_hold_gripper", "previous_motion_hold_gripper"):
            scores = baseline["scores"][name]
            rows[name] = {key + "_mm": scores[phase]["translation_vector_error_m"]["mean"] * 1000
                          for key, phase in PHASES.items()}
            rows[name]["closure_gripper_mae"] = scores[PHASES["closure_proxy"]]["mae_raw_per_dim"][6]
        comparisons[section] = rows

    gate = read(Path("results/audit/geometry_control_20261016/gate.json"))
    assert not gate["panda_allowed"] and not gate["recorded_complete_grasp_gate_passed"]
    report = {
        "terminal_stage": read(ROOT / "STAGE_COMPLETE.json"), "contract_checks": contracts,
        "verified_file_count": len(verified_hashes), "comparisons": comparisons,
        "geometry_gate": gate,
        "interpretation": "No observation variant improves active translation over A1R in either split. Retain canonical inputs for architecture screens; no physical success established.",
        "limitations": ["Single training seed and frozen vision; bootstrap measures episode variability only.",
                       "Historical validation has been reviewed repeatedly; phases are kinematic proxies.",
                       "FK mapping is provisional. A corrected geometry branch needs regenerated states, velocities, labels and normalizers together."],
        "next_scope": "A2/A3/A4 bounded architecture screen on the unchanged provisional dataset, stop after A4. No deployment or Panda.",
    }
    (ROOT / "results/stage_H0_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"contract_status": "PASS", "arms": len(contracts), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
