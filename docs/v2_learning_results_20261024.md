# Completed v2 learning screen — 24 October 2026

All **22 unique training runs completed without failure**. The 24 matrix entries include two reused controls. Each run consumed 64,000 examples with seed 0, using 29 training and six validation episodes. The six reserved episodes were not evaluated. These are exploratory learning results on provisional geometry; they do not establish successful pick-and-place.

The best two-view Cartesian ACT validation result is CLIP with its final two blocks trained at backbone learning rate `3e-6`: **0.43749**, 4.51% below the matched ResNet result. The only iPhone-only checkpoint remains the practical full-simulator candidate because the recorded wrist view, named `top` in the dataset, has no measured simulator camera mount.

Evidence and complete checkpoint hashes are in [completed_screen_review_20261024.json](../results/v2/completed_screen_review_20261024.json). The source training reports remain unchanged under `outputs/v2/learning_screen_20261022`.

## How to read the numbers

Lower normalized L1 is better **within the same target coordinates, normalization, and validation protocol**. Camera and backbone ACT comparisons below share the same base-frame Cartesian labels and verified-identical action normalization. ACT validation covers nonpadding offsets of overlapping 25-action chunks anchored at all 2,077 validation ticks. It does not specifically measure only the two actions executed before replanning.

The smaller regression/flow heads instead report 2,071 unique executed actions, excluding extrapolation and padding. Their values cannot be ranked directly against the ACT values. Joint, absolute-pose, base-frame delta and tool-frame delta normalized scores also cannot rank those representations against each other.

This was the first 64k examples of a 320k-example learning-rate schedule, about 6.65 passes through the 9,617 training ticks. There is one seed, no confidence interval, and checkpoint selection uses the same six validation episodes. These small differences need further validation, not a claim of a statistically established winner.

## 1. iPhone, wrist, or both

| ResNet18 ACT input | Best normalized L1 | Best examples seen |
| --- | ---: | ---: |
| iPhone | 0.44761 | 64,000 |
| Wrist (`top`) | 0.46357 | 48,000 |
| Both | 0.45815 | 64,000 |

iPhone alone is best in this screen. Adding the wrist image did not improve the validation result at this budget. This does not show that the wrist view is intrinsically unhelpful; optimization and image use remain possible causes.

For a full simulator diagnostic, use `runs/L1_resnet_iphone/attempt-001/checkpoints/004000` beneath the screen directory. Its single image key is supported, but its geometry and joint calibration remain provisional.

## 2. ResNet18 versus frozen DINOv2

| Both-camera ACT backbone | Best normalized L1 |
| --- | ---: |
| ResNet18 | 0.45815 |
| Frozen DINOv2-small | 0.44951 |

Frozen DINO reduces this matched validation error by 1.89%. No corresponding closed-loop success improvement has been established.

## 3. Effective batch size and head learning rate

| Effective batch | Head learning rate | Best normalized L1 | Optimizer updates completed |
| ---: | ---: | ---: | ---: |
| 16 | 3e-5 | 0.45779 | 4,000 |
| 16 | 1e-4 (reused baseline) | 0.45815 | 4,000 |
| 16 | 3e-4 | 0.47617 | 4,000 |
| 64 | 3e-5 | 0.46887 | 1,000 |
| 64 | 1e-4 | 0.45784 | 1,000 |
| 64 | 3e-4 | 0.45855 | 1,000 |

All runs use physical batch eight and gradient accumulation. Larger batches receive fewer optimizer updates at the fixed example budget. Batch 64 at `1e-4` essentially matches the batch-16 baseline; it is not a clear performance gain. Batch 16 at `3e-4` is worse.

## 4. DINOv2 fine-tuning

| DINOv2 ACT variant | Best normalized L1 |
| --- | ---: |
| Frozen control | 0.44951 |
| Last two blocks, backbone LR 3e-6 | 0.45201 |
| Last two blocks, backbone LR 1e-5 | 0.45070 |

Neither fine-tuned variant improves the frozen control in this screen.

## 5. CLIP frozen and fine-tuned

| CLIP ViT ACT variant | Best normalized L1 | Best examples seen |
| --- | ---: | ---: |
| Frozen | 0.44195 | 56,000 |
| Last two blocks, backbone LR 3e-6 | **0.43749** | 48,000 |
| Last two blocks, backbone LR 1e-5 | 0.44160 | 48,000 |

The `3e-6` variant is the best matched two-view ACT checkpoint. Its final checkpoint worsens to 0.45921, so evaluation must use its best checkpoint, `checkpoints/003000`, rather than the last checkpoint. The improvement over frozen CLIP is 1.01%.

## 6. Flow matching versus matched regression heads

The table uses each run's best four-step validation checkpoint. Regression is deterministic; flow uses one deterministic noise draw per anchor, not best-of-many sampling. Both heads within each backbone pair have matching trainable parameter counts, input features, data, seed, batch size, head learning rate and example budget.

| Frozen features | Head | Best normalized L1 | Mean translation error | Best examples seen |
| --- | --- | ---: | ---: | ---: |
| DINOv2 | Regression | 0.48524 | 3.601 mm | 48,000 |
| DINOv2 | Flow, four steps | 0.59838 | 4.522 mm | 64,000 |
| CLIP | Regression | **0.40285** | **2.844 mm** | 64,000 |
| CLIP | Flow, four steps | 0.61981 | 4.664 mm | 64,000 |
| Same validation actions | Persistence baseline | 0.32632 | 2.466 mm | — |

Regression outperforms flow for both backbones at this budget. Increasing flow sampling to eight steps worsens the final DINO score to 0.63000 and CLIP to 0.66883. None of these four learned heads beats the persistence baseline in normalized L1. Persistence uses observed motion and the current gripper observation; this teacher-forced baseline is not an autonomous task controller.

## 7. Joint targets, persistent deltas, and absolute poses

| Representation | Best normalized L1, within its own coordinates | Raw per-axis MAE at its best checkpoint |
| --- | ---: | --- |
| Recorded canonical joint commands | 0.20835 | Arm: 2.62°, 6.62°, 6.23°, 7.20°, 2.19°; gripper: 5.67 percentage points |
| Persistent base-frame deltas | 0.45815 | Translation: 1.6, 1.4, 2.2 mm; gripper: 0.0537 fraction |
| Next absolute pose | 0.23023 | Position: 12.9, 10.4, 14.4 mm; gripper: 0.0508 fraction |

These scores use different units, dimensions, normalization and target semantics. **The smaller joint or absolute normalized score does not show that either controller is better.** Joint targets reproduce recorded leader commands, while the Cartesian labels describe achieved follower motion using provisional FK. A common physical-space and closed-loop comparison is still required.

## 8. Tool-frame deltas and trajectory-consistency loss

| Tool-frame ResNet ACT | Best normalized L1 | Best examples seen |
| --- | ---: | ---: |
| No trajectory loss | 0.43348 | 48,000 |
| Trajectory loss weight 0.1 | 0.46220 | 56,000 |

Adding the trajectory loss worsens the matched tool-frame L1 by 6.63%. This evaluates prediction error, not a trajectory-drift or success-rate benefit. Tool-frame and base-frame normalized scores have different coordinate-dependent standard deviations and cannot establish which representation is better.

## Current inference and task status

A synchronized local **MPS** check used the same 24 distinct recorded validation anchors, three warmup calls, and 30 measured batch-one calls per model:

| Two-view model | Median policy-call latency | p95 |
| --- | ---: | ---: |
| ResNet18 ACT | 24.26 ms | 26.97 ms |
| Best fine-tuned CLIP ACT | 78.07 ms | 94.72 ms |

Both pass the 120 ms policy-call target in this bounded check. These times include cached image access, normalization, transfers, all vision/head computation, and action denormalization. They exclude rendering and sensor acquisition. They are not an end-to-end control deadline guarantee. Earlier CLIP CPU latency does not describe its MPS performance.

Reproduce with `.venv/bin/python scripts/diag_v2_mps_inference.py --out <new-report.json>`. See [mps_inference_profile_20261024.json](../results/v2/mps_inference_profile_20261024.json) for every timing, selected tick, device, synchronization, and checkpoint hash.

The full learned-policy geometry diagnostics still fail during approach. The camera and wrist fits are unaccepted, and missing wrist-camera placement blocks faithful two-view simulator input. Better offline error or scripted retention is not learned pick-and-place success. Continue with the single-camera full rollout, executed-prefix motion/bias analysis, controller feasibility, and measured geometry before spending another full training budget.
