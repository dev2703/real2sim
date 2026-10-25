# Three-episode movement and drift gate — 25 October 2026

The deterministic ACT fails the declared three-training-episode gate after **32,000 scheduled examples**. The episodes are 11 (baseline), 8 (ball shift), and 23 (bowl shift). This is a capacity diagnostic under the existing provisional FK labels.

The [machine-readable report](../results/v2/three_episode_motion_20261025.json) preserves both stage audits, every per-condition gate result, checkpoint hashes, exact-resume records and the final integrity check. Full artifacts remain in `outputs/v2/three_episode_motion_20261024`.

## What the gate requires

Every episode must have mean translation error below 1 mm and gripper MAE below 0.03. It must also improve both causal hold and persistence baselines on moving-vector error and mean/maximum stationary-run summed target error. Missing movement/stationary strata fail; an exact zero-error tie is permitted. A strong aggregate score cannot hide a failed condition.

Stationary-run error is a teacher-forced summed-error proxy, not measured closed-loop drift. Predicted stationary step magnitude is descriptive because hold predicts exactly zero. Scoring uses the two-action execution prefix, excludes terminal extrapolation, and covers only the three training episodes.

## Final training fit

| Episode | Translation error | Gripper MAE | Gate |
|---|---:|---:|---|
| 11 | 2.789 mm | 0.0091 | Fail |
| 8 | 2.495 mm | 0.0102 | Fail |
| 23 | 2.728 mm | 0.0076 | Fail |

| Metric across the three training episodes | Model | Hold | Persistence |
|---|---:|---:|---:|
| Moving-vector error (mm) | 3.292 | 7.183 | 3.541 |
| Mean stationary-run summed error (mm) | 1.800 | 0.104 | 2.708 |
| Maximum stationary-run summed error (mm) | 9.044 | 0.490 | 15.639 |
| Mean predicted stationary step (mm) | 0.397 | 0.000 | 0.568 |

The per-episode checks and the 16,000-example intermediate result remain in the JSON. Baselines use causal recorded motion history under the same scoring contract; their results are not task-success estimates.

## Training and interruption lineage

The existing ResNet ACT configuration uses deterministic decoding without VAE/dropout, batch 8 with two accumulation steps, and a single 2,000-update learning-rate horizon. It fits 986 ticks. Labels, train-only source statistics, geometry, limits and contact settings retain their original contracts. This was one capacity run, with no backbone sweep.

Training began on 24 October and resumed on 25 October from saved checkpoints 500 and 1,000. Each resume restored optimizer, sampler and RNG state; unsaved updates were recomputed. The example count describes the final optimizer schedule, excluding discarded replay work. The runner now persists progress before training and supports exact restart; the original plan and executed/resume snapshots are retained.

All final derived payload hashes match the manifest, checkpoint normalization equals source normalization, and the final checkpoint hash matches the scored checkpoint. A separate retained integrity audit verifies source-file lineage. Hashing complete files does not score or decode reserved examples.

## Causal-input audit

Twelve exact 13-feature proprioception groups in these training episodes have differing next translation labels above 0.4 mm. For example, episode 8 frames 58/60 have identical proprioception and image MAE only 0.307 uint8 levels, but the next-action labels differ by 3.954 mm. Episode 23 frames 66/68 has a similar 3.920 mm difference.

These observations identify possible hold-to-motion timing ambiguity. Images are similar rather than identical, so they do not prove irreducible visual-policy error. The deterministic loss also predicts 25 actions (2 seconds), while execution uses two (0.16 seconds). These are possible contributors to investigate, not an established cause or a reason to relax the gate.

## Next gate

The small training success has not been established. Common development rollouts and reserved evaluation were not run for this checkpoint. The separate matched-object probes still show weak action dependence, and physical camera/base registration remains unresolved; no joint-zero/gripper correction or corrected-label retraining is adopted.

Reproduce training in a fresh output directory, or resume the same saved plan:

```sh
.venv/bin/python scripts/run_v2_three_episode_fit.py \
  --out outputs/v2/three_episode_motion_NEXT --device cpu
.venv/bin/python scripts/run_v2_three_episode_fit.py \
  --out outputs/v2/three_episode_motion_NEXT --device cpu --resume
```

Focused geometry, action-intervention, registration, label-lineage and capacity-gate checks pass (45 tests). The restart test also runs without the ignored historical experiment configuration.
