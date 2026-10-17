"""Verify frozen A1R/A2/A3/A4 contracts and report phase trade-offs."""

import json
from pathlib import Path

import numpy as np
import yaml
from review_so101_stage_h0 import DATA, PHASES, ROOT, read, sha

from real2sim.data.prepare import compute_stats
from real2sim.policy.observation_variants import ObservationVariant, fit_state_statistics


def paired_delta(candidate, reference):
    assert candidate["episodes"] == reference["episodes"]
    episodes = candidate["episodes"]
    counts = np.array([candidate["per_episode"][str(ep)]["active"]["n"] for ep in episodes])
    differences = np.array([
        (candidate["per_episode"][str(ep)]["active"]["translation_vector_error_m"]["mean"]
         - reference["per_episode"][str(ep)]["active"]["translation_vector_error_m"]["mean"]) * 1000
        for ep in episodes])
    draws = np.random.default_rng(0).integers(0, len(episodes), (10000, len(episodes)))
    means = (differences[draws] * counts[draws]).sum(1) / counts[draws].sum(1)
    return {"active_delta_mm": float(np.average(differences, weights=counts)),
            "paired_episode_bootstrap_95pct_mm": np.quantile(means, [.025, .975]).tolist()}


def main():
    terminal = read(ROOT / "STAGE_COMPLETE.json")
    assert terminal["through"] == "A4" and not terminal["panda_gate_passed"]
    frozen = read(ROOT / "QUEUE.json")["identity"]["frozen_files"]
    for name, expected in frozen.items():
        assert sha(Path(name)) == expected, name
    data = DATA / "canonical"
    manifest, stats = read(data / "manifest.json"), read(data / "stats.json")
    for name, expected in manifest["files_sha256"].items():
        assert sha(data / name) == expected, name
    arrays = dict(np.load(data / "arrays.npz"))
    np.testing.assert_array_equal(arrays["state"], arrays["canonical_state"])
    split = manifest["split"]
    sets = [set(split[k]) for k in ("train", "val", "historical_val")]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    assert fit_state_statistics(arrays["state"], arrays["episode_index"], split["train"],
                                ObservationVariant()) == stats["observation.state"]
    mask = np.isin(arrays["episode_index"], split["train"]) & ~arrays["label_is_extrapolated"]
    assert compute_stats(arrays["action"][mask]) == stats["action"]
    contracts = {}
    audits = {section: {} for section in ("val", "historical_val")}
    for arm, batch, width, heads in (("A1R", 16, 512, 8), ("A2", 64, 512, 8),
                                    ("A3", 64, 384, 8), ("A4", 64, 384, 12)):
        run = ROOT / "runs" / arm
        checkpoint = run / "checkpoints/best"
        contract = read(checkpoint / "real2sim_contract.json")
        assert read(checkpoint / "stats.json") == stats
        assert contract["derived_files_sha256"] == manifest["files_sha256"]
        for key in ("split", "observation_variant", "state_names", "action_names", "camera_keys",
                    "letterbox", "label_semantics", "policy_hz"):
            assert contract[key] == manifest[key], (arm, key)
        config = yaml.safe_load((ROOT / "configs" / f"{arm}.yaml").read_text())
        saved = read(checkpoint / "config.json")
        for key, value in {"dim_model": width, "n_heads": heads, "n_encoder_layers": 4,
                           "n_decoder_layers": 1, "chunk_size": 20, "n_action_steps": 2}.items():
            assert config["model"][key] == saved[key] == value, (arm, key)
        assert config["train"]["batch_size"] * config["train"]["grad_accum"] == batch
        result = read(run / "training_result.json")
        assert result["steps"] * batch == 32000 and not result["stopped_early"]
        digest = sha(checkpoint / "model.safetensors")
        contracts[arm] = {"status": "PASS", "effective_batch": batch, "width": width,
                          "heads": heads, "sampled_anchors": 32000, "seed": 0,
                          "checkpoint_sha256": digest, "training_result": result}
        for section in audits:
            audit = read(ROOT / "results" / f"{arm}.{section}.json")
            assert audit["checkpoint_sha256"] == digest
            assert audit["episodes"] == split[section]
            audits[section][arm] = audit
    comparisons = {}
    for section, models in audits.items():
        rows = {}
        for arm, audit in models.items():
            assert audit["valid_actions"] == models["A1R"]["valid_actions"]
            scores = audit["scores"]["policy"]
            rows[arm] = {name: {"n": scores[phase]["n"],
                                "translation_mean_mm": scores[phase]["translation_vector_error_m"]["mean"]*1000,
                                "translation_p95_mm": scores[phase]["translation_vector_error_m"]["p95"]*1000,
                                "rotation_mean_rad": scores[phase]["rotation_vector_error_rad"]["mean"],
                                "gripper_mae": scores[phase]["mae_raw_per_dim"][6]}
                         for name, phase in PHASES.items()}
        comparisons[section] = {"scores": rows, "paired_changes": {
            f"{a}_minus_{b}": paired_delta(models[a], models[b])
            for a, b in (("A2", "A1R"), ("A3", "A2"), ("A4", "A3"), ("A4", "A1R"))},
            "baselines": {k: v for k, v in models["A1R"]["scores"].items() if k != "policy"}}
    report = {"terminal_stage": terminal, "contracts": contracts, "comparisons": comparisons,
              "interpretation": "Batch64 slightly improves active translation but worsens inner-val stationary and closure gripper errors. Twelve heads have no consistent advantage. No architecture promoted or grasp success established.",
              "limitations": ["One training seed; episode bootstrap excludes training-seed variability.",
                              "Historical validation is repeatedly reviewed, not untouched.",
                              "Old nominal-time FK contract is internally consistent but physical timing and calibrated geometry remain unresolved."],
              "next_scope": "Bounded full-DINO flow-head hardware smoke/profile only. Comparative training awaits timing/provenance and geometry review.",
              "geometry_gate_passed": False, "panda_gate_passed": False}
    (ROOT / "results/stage_A4_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"contracts": "PASS", "paired_changes": {
        k: v["paired_changes"] for k, v in comparisons.items()}}, indent=2))


if __name__ == "__main__":
    main()
