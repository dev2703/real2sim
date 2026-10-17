"""Check the completed F0 screen against unchanged data and common ACT phase audits."""

import json
from pathlib import Path

import numpy as np
from review_so101_stage_h0 import DATA, PHASES, ROOT, read, sha
from safetensors import safe_open

from real2sim.data.prepare import compute_stats
from real2sim.policy.observation_variants import ObservationVariant, fit_state_statistics


def main():
    terminal = read(ROOT / "STAGE_COMPLETE.json")
    assert terminal["through"] == "F0" and not terminal["panda_gate_passed"]
    assert not (ROOT / "RUNNING.json").exists()
    for name, expected in read(ROOT / "QUEUE.json")["identity"]["frozen_files"].items():
        assert sha(Path(name)) == expected, name
    data = DATA / "canonical"
    manifest, stats = read(data / "manifest.json"), read(data / "stats.json")
    for name, expected in manifest["files_sha256"].items():
        assert sha(data / name) == expected, name
    arrays = dict(np.load(data / "arrays.npz"))
    split = manifest["split"]
    sets = [set(split[k]) for k in ("train", "val", "historical_val")]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    np.testing.assert_array_equal(arrays["state"], arrays["canonical_state"])
    assert fit_state_statistics(arrays["state"], arrays["episode_index"], split["train"],
                                ObservationVariant()) == stats["observation.state"]
    mask = np.isin(arrays["episode_index"], split["train"]) & ~arrays["label_is_extrapolated"]
    assert compute_stats(arrays["action"][mask]) == stats["action"]
    cache = DATA / "dino_features"
    cached = read(cache / "manifest.json")
    for name, expected in cached["sha256"].items():
        assert sha(cache / name) == expected, name
    np.testing.assert_array_equal(np.load(cache / "ticks.npy"), np.arange(len(arrays["state"])))
    assert cached["image_identity"]["images"]["observation.images.iphone"] == manifest["files_sha256"]["images_iphone.npy"]
    contracts, reports = {}, {}
    for objective in ("regression", "flow"):
        arm = f"F0_{objective}"
        run = ROOT / "runs" / arm
        checkpoint = run / "checkpoint"
        report = read(run / "report.json")
        reports[arm] = report
        assert read(checkpoint / "stats.json") == stats
        contract = read(checkpoint / "real2sim_contract.json")
        assert contract["derived_files_sha256"] == manifest["files_sha256"]
        for key in ("split", "observation_variant", "state_names", "action_names", "camera_keys",
                    "letterbox", "label_semantics", "policy_hz"):
            assert contract[key] == manifest[key], (arm, key)
        assert report["feature_cache_sha256"] == cached["sha256"]
        assert report["examples"] == report["updates"] * report["effective_batch"] == 32000
        assert report["effective_batch"] == report["microbatch"] * report["grad_accum"] == 64
        assert report["split"] == split and report["seed"] == 0
        config = read(checkpoint / "config.json")
        assert report["config"] == config
        for key, value in {"objective": objective, "state_dim": 13, "action_dim": 7,
                           "dim_model": 256, "n_heads": 8, "n_layers": 2,
                           "chunk_size": 20, "n_action_steps": 2}.items():
            assert config[key] == value, (arm, key)
        with safe_open(checkpoint / "model.safetensors", framework="numpy") as model:
            with safe_open(cache / "backbone.safetensors", framework="numpy") as backbone:
                for key in backbone.keys():
                    np.testing.assert_array_equal(model.get_tensor("backbone." + key),
                                                  backbone.get_tensor(key), err_msg=key)
        contracts[arm] = {"status": "PASS", "checkpoint_sha256": sha(checkpoint / "model.safetensors"),
                          "sampled_anchors": report["examples"], "updates": report["updates"],
                          "trainable_parameters": report["trainable_parameters"],
                          "cached_head_training_seconds": report["train_wall_s"]}
    for key in ("seed", "examples", "updates", "microbatch", "grad_accum", "effective_batch",
                "lr", "weight_decay", "grad_clip", "split", "trainable_parameters", "feature_cache_sha256"):
        assert reports["F0_regression"][key] == reports["F0_flow"][key], key
    comparisons, parity = {}, {}
    for section in ("val", "historical_val"):
        baseline = read(ROOT / "results" / f"A1R.{section}.json")
        scores = {}
        for arm in ("A1R", "A1D", "A2", "F0_regression", "F0_flow"):
            audit = read(ROOT / "results" / f"{arm}.{section}.json")
            assert audit["episodes"] == split[section]
            assert audit["valid_actions"] == baseline["valid_actions"]
            if arm in contracts:
                assert audit["checkpoint_sha256"] == contracts[arm]["checkpoint_sha256"]
            scores[arm] = {}
            for name, phase in PHASES.items():
                values = audit["scores"]["policy"][phase]
                assert values["n"] == baseline["scores"]["policy"][phase]["n"]
                scores[arm][name] = {"n": values["n"],
                    "translation_mean_mm": values["translation_vector_error_m"]["mean"] * 1000,
                    "rotation_mean_rad": values["rotation_vector_error_rad"]["mean"],
                    "gripper_mae": values["mae_raw_per_dim"][6]}
            for base in ("zero_motion_hold_gripper", "previous_motion_hold_gripper"):
                assert audit["scores"][base] == baseline["scores"][base]
            if arm in reports and section == "val":
                key = "inner_val" if arm == "F0_regression" else "inner_val_4_steps"
                parity[arm] = {}
                # Approach definitions differ in the trainer; use only shared strata.
                for phase, source in (("active", "active"), ("stationary", "stationary"),
                                      ("closure_proxy", "closure_proxy"),
                                      ("upward_closed_proxy", "upward_closed_lift_proxy")):
                    cached_score = reports[arm][key]["strata"][source]["model"]
                    assert cached_score["n"] == scores[arm][phase]["n"]
                    delta = scores[arm][phase]["translation_mean_mm"] - 1000 * cached_score["translation_mean_m"]
                    parity[arm][phase] = delta
                    assert abs(delta) < 0.01, (arm, phase, delta)
        comparisons[section] = {"scores": scores,
            "baselines": {k: v for k, v in baseline["scores"].items() if k != "policy"}}
    report = {"terminal_stage": terminal, "contracts": contracts, "comparisons": comparisons,
        "full_vision_minus_cached_feature_translation_mm": parity,
        "flow_8_step_inner_val": reports["F0_flow"]["inner_val_8_steps"],
        "interpretation": "No F0 promotion: flow underperforms its matched regression head and zero-motion active baseline in both splits. Eight steps do not improve the cached inner-val comparison. Prioritize trace-verified collection and independent geometry validation before expanding training.",
        "limitations": ["One seed and 500 updates: not a general verdict on flow matching.",
            "F0 regression versus flow matches head/data/budget; ACT has different capacity/loss and best-validation checkpoint selection versus F0 final checkpoint.",
            "Phase proxies overlap and are not contact annotations; historical validation is repeatedly reviewed.",
            "Internal nominal-time contracts pass; physical timing, gripper provenance and calibrated geometry remain unresolved."],
        "geometry_gate_passed": False, "panda_gate_passed": False}
    (ROOT / "results/stage_F0_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"contracts": "PASS", "feature_parity_delta_mm": parity,
                      "decision": report["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
