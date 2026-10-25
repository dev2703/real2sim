# Two-action supervision on three episodes — 25 October 2026

The prefix-supervised ACT **fails** the unchanged three-episode capacity gate after **32,000 scheduled examples**. Its result is compared with the original full-chunk objective at the same optimizer/example budget. Both use episodes 11 (baseline), 8 (ball shift), and 23 (bowl shift).

Moving-vector error changes **3.292 → 2.316 mm**, and mean stationary-run summed error changes **1.800 → 1.272 mm**. The per-condition fitting and baseline gates below determine acceptance; improvement in these means alone does not establish success.

The [machine-readable comparison](../results/v2/prefix_capacity_20261025.json) contains every per-condition gate, both stage histories and checkpoint/data integrity checks. No reserved episode was selected or scored.

## Controlled change

The model still predicts 25 actions and executes two. The new objective supervises only action offsets 0/1, applying terminal padding and normalizing by real supervised components. The backbone, model initialization seed, episode/data contracts, labels, source normalization, sampler, batch 8×2, learning-rate horizon and update cap match the original run. No target filtering or relabeling was applied.

VAE is disabled, so the experiment has no future-action latent-encoder input. Direct gradients to output-tail positions are zero; shared decoder parameters may still affect those outputs. Full-chunk validation L1 is descriptive and is not used for this comparison. Selection and the capacity gate use the unchanged executed-prefix audit.

## Matched-budget result

| Metric on the same training actions | Original objective | Two-action objective | Hold | Persistence |
|---|---:|---:|---:|---:|
| Moving-vector error (mm) | 3.292 | 2.316 | 7.183 | 3.541 |
| Mean stationary-run summed error (mm) | 1.800 | 1.272 | 0.104 | 2.708 |
| Maximum stationary-run summed error (mm) | 9.044 | 8.539 | 0.490 | 15.639 |
| Mean predicted stationary step (mm) | 0.397 | 0.299 | 0.000 | 0.568 |

Stationary-run values are teacher-forced target-error sums, not measured rollout drift. Hold/persistence values and all scoring exclusions are identical between experiments.

| Episode / objective | Translation error (mm) | Gripper MAE | Moving error (mm) | Mean / max stationary-run error (mm) | Gate |
|---|---:|---:|---:|---:|---|
| 11 / Original | 2.789 | 0.0091 | 3.360 | 1.710 / 7.046 | Fail |
| 11 / Prefix2 | 1.958 | 0.0064 | 2.337 | 1.554 / 3.823 | Fail |
| 8 / Original | 2.495 | 0.0102 | 3.063 | 2.121 / 9.044 | Fail |
| 8 / Prefix2 | 1.730 | 0.0087 | 2.124 | 1.431 / 8.539 | Fail |
| 23 / Original | 2.728 | 0.0076 | 3.411 | 1.520 / 8.157 | Fail |
| 23 / Prefix2 | 1.971 | 0.0067 | 2.455 | 0.807 / 2.523 | Fail |

Every condition must have translation error below 1 mm and gripper MAE below 0.03, and improve both causal baselines on moving-vector error and mean/maximum stationary-run error. Missing strata fail; exact zero-error floor ties are allowed. The archived gate function is identical between runs and recomputes every recorded gate result.

## Integrity and remaining checks

The summary verifies the allowed plan differences, matching model/data/scoring dependency hashes, all derived payload hashes, every scored checkpoint weight hash, checkpoint normalization, saved execution counts and matched scoring support. It consumes frozen audits without fitting new thresholds or decoding reserved images.

The small training success remains unestablished. Common development rollouts and reserved evaluation were not run for this candidate; the unchanged gate is retained.

## Matched-object action probes

The separate [object-dependence report](../outputs/v2/prefix_object_probe_20261025/prefix_2000.json) uses the same scored prefix checkpoint and the previously frozen pair/mask selection. **24 / 24 pairs** meet the original object-support rule. The following development active-state rows use object-region swaps with recipient arm-state input fixed.

| Object | Informative pairs | Mean swap effect (two-action norm) | Effect / observed contrast | Mean direction cosine |
|---|---:|---:|---:|---:|
| ball | 3 | 1.042 mm | 29.61% | 0.955 |
| bowl | 2 | 1.508 mm | 34.16% | -0.023 |

Ratios/cosines use only predeclared informative target contrasts. Reciprocal directions are correlated; donor actions are observational comparisons rather than counterfactual ground truth. Mask unions can transfer background/occlusion pixels. Full-image swaps, stationary starts and training-pair endpoint overlap remain in the linked report. Input sensitivity alone does not establish correct object-conditioned actions or rollout success.

The [paired-checkpoint object comparison](v2_prefix_object_actions_20261025.md) compares this candidate with the original objective and retains stationary development-start checks separately.

The separate [motion-onset audit](v2_motion_onset_audit_20261025.md) finds stationary immediate prefixes with movement in the future tail, and nearly identical observations with differing immediate labels. Prefix supervision changes the trained horizon; it does not remove that observed timing ambiguity. No labels or sampling weights were changed.

Physical camera/base registration and joint/gripper verification remain unresolved. This experiment uses provisional FK labels and cannot justify a geometric correction. The [fixed-base evidence](v2_fixed_landmark_tracking_20261025.md) and [multi-height landmark review](v2_fixed_base_landmark_review_20261024.md) retain those limits. The [calibration capture sheet](v2_calibration_capture_20261025.md) specifies the next physical evidence.

Reproduce in fresh paths:

```sh
.venv/bin/python scripts/run_v2_three_episode_fit.py \
  --out outputs/v2/three_episode_prefix2_NEXT --device cpu --supervised-prefix-steps 2
.venv/bin/python scripts/summarize_v2_prefix_capacity.py \
  --baseline outputs/v2/three_episode_motion_20261024 \
  --prefix outputs/v2/three_episode_prefix2_NEXT \
  --out results/v2/prefix_capacity_NEXT.json --doc docs/prefix_capacity_NEXT.md
```
