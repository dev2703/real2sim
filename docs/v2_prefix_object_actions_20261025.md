# Object-action probes after prefix supervision — 25 October 2026

The canonical full-chunk and two-action-prefix ACTs are compared at **2,000 updates / 32,000 scheduled examples each**, using the same model, seed, source labels, normalization and 24 frozen matched-state object pairs. Only reconstruction supervision changes. This measures image dependence alongside natural observed action contrast; it does not establish physical counterfactual correctness or task success.

For active development ball pairs, the object-region effect stays near **29.6%** of observed target contrast while its mean direction cosine improves **0.798 → 0.955**. Bowl-region sensitivity grows **13.0% → 34.2%**, but direction remains poorly aligned (**-0.023**). Increased sensitivity alone therefore does not establish useful bowl-conditioned actions.

The [paired report](../results/v2/prefix_object_actions_20261025.json) retains all training/development group metrics and links the complete per-direction predictions and checkpoint hashes. Source reports remain under `outputs/v2/prefix_object_probe_20261025`.

## Reused observations and input contract

All **24/24** existing pairs meet the original object-mask area and center-separation thresholds. The same masks, pairs, object support, two-action labels and reciprocal directions are reused; no detector, segmentation or sample selection is rerun. Canonical arrays/images are verified byte-identical to the source of those masks. Each checkpoint has exactly 13 canonical state features and a single iPhone image, with matching checkpoint/source normalization.

The state is fixed during image swaps. A full-image swap replaces all image pixels; an object-region swap replaces donor pixels in the dilated union of the two measured object masks. No explicit ball-feature slots exist in either model. Natural donor predictions retain their own approximately matched recorded arm states and are scored separately.

The original train/validation labels describe the full source split, while these capacity policies train only on episodes 11, 8 and 23. **No frozen pair has both endpoints in those three episodes.** Training-endpoint overlap is explicit in the machine report, so the source-training probes must not be interpreted as three-episode fitting checks. Reserved episodes are excluded.

## Active development object dependence

The target contrast is the difference between observed two-action translation prefixes, represented as one six-dimensional vector. A frozen 1 mm norm threshold determines whether amplitude/direction are informative. Three ball pairs and two bowl pairs meet that threshold. Both reciprocal directions enter averages and are correlated.

| Object / replacement | Informative pairs | Full-chunk effect / contrast | Prefix effect / contrast | Full-chunk cosine | Prefix cosine |
|---|---:|---:|---:|---:|---:|
| ball / full image | 3 | 37.73% | 38.14% | 0.649 | 0.962 |
| ball / object region | 3 | 29.75% | 29.61% | 0.798 | 0.955 |
| bowl / full image | 2 | 11.41% | 48.04% | 0.061 | -0.250 |
| bowl / object region | 2 | 13.00% | 34.16% | 0.116 | -0.023 |

These amplitudes are averages of per-direction ratios on informative pairs. Object-region effects can include transferred background, shadows or occluders; a bowl after release can also contain the ball. Full-image effects include arm/background differences. Donor labels are observational comparators, not known correct actions after synthetic swaps. The full-chunk capacity baseline is distinct from the earlier 256-update explicit-ball pilots.

## Stationary development starts

All six stationary development pairs have zero recorded translation over their two scored actions. Their useful checks are false motion and input sensitivity.

| Layout group | Full-chunk mean predicted step | Prefix mean predicted step | Target |
|---|---:|---:|---:|
| ball | 0.0415 mm | 0.0996 mm | 0 mm |
| bowl | 0.0477 mm | 0.0982 mm | 0 mm |

These are sparse offline start observations, not closed-loop drift or a summary of every stationary hold within a trajectory. They do not replace the unchanged three-episode movement/stationary-error gate, followed by common development rollouts only if it passes. No checkpoint is promoted by object sensitivity alone.

## Reproduction

```sh
.venv/bin/python scripts/diag_v2_prefix_object_actions.py \
  --checkpoint outputs/v2/three_episode_motion_20261024/training/checkpoints/002000 \
  --name full_chunk_2000 --out outputs/v2/prefix_object_probe_NEXT/full_chunk_2000.json
.venv/bin/python scripts/diag_v2_prefix_object_actions.py \
  --checkpoint outputs/v2/three_episode_prefix2_20261025/training/checkpoints/002000 \
  --name prefix_2000 --out outputs/v2/prefix_object_probe_NEXT/prefix_2000.json
.venv/bin/python scripts/summarize_v2_prefix_object_actions.py \
  --baseline outputs/v2/prefix_object_probe_NEXT/full_chunk_2000.json \
  --prefix outputs/v2/prefix_object_probe_NEXT/prefix_2000.json \
  --out results/v2/prefix_object_actions_NEXT.json \
  --doc docs/v2_prefix_object_actions_NEXT.md
```
