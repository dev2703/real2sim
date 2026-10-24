# Matched-arm object/action probes — 25 October 2026

The existing ball-input ACT pilot responds weakly to changed object locations in
matched arm states. On three active development ball pairs, replacing the ball
region changes its two-action translation prediction by **0.151 mm** on average;
the effect is **4.20%** of the observed target contrast, with mean direction cosine
**−0.630**. The same model's full-image swap points in the observed contrast's
direction but has only **5.79%** of its amplitude. Image sensitivity alone therefore
does not establish useful object-conditioned actions. These are two existing
256-update pilots, not a conclusion about every policy or architecture.

The [grouped report](../results/v2/matched_object_actions_20261025.json) keeps
training and development results separate and links the complete per-pair traces.
The [diagnostic](../scripts/diag_v2_matched_object_actions.py) adds matched-state
selection, localized image interventions and target-contrast metrics. No checkpoint,
training labels, normalizer, geometry or simulator setting was changed.

## Selection and localization support

Selection was frozen on 24 October before image-model measurements or policy
inference. It uses recorded joint readbacks, gripper readbacks and causal velocity;
recorded target actions never select the pairs. All comparisons stay within the
inherited late image regime, episodes 7 onward. This is an image-regime assumption,
not accepted camera registration; the separate fixed-base investigation shows that
mat placement can vary. Pixel displacements below are not metric object motion.

The bounds are at most **5° per arm joint**, **3° arm RMS**, **5 percentage points
of gripper opening**, and **0.03 m/s** velocity difference. Each episode pair supplies
its closest eligible state pair, with deterministic ties and no result-conditioned
replacement. Actual arm RMS differences are **0.35–1.80°**. Each partition contains
three ball and three bowl pairs for each of two contexts:

- Stationary starts: first 25 policy ticks, four real history ticks, arm history
  span at most 0.5°, current tool-speed proxy at most 0.005 m/s.
- Active states: current tool-speed proxy at least 0.02 m/s. Some closest bowl
  pairs are after release during the return motion; phase/timing remains a confound.

This gives **24 pairs / 37 unique frames**, including 12 training and 12 development
pairs. No reserved episode enters selection or inference. Grounding DINO and SAM2
use the existing pinned local weights and frozen prompts/thresholds. All **24/24**
pairs meet the declared minimum mask area and **8 policy-pixel** object-center
separation. Ball separation is **17.58–26.82 px**, bowl separation
**54.09–58.04 px**, at the actual **256×192 letterboxed policy resolution**.
Model scores are not calibrated localization errors. The complete mask montage
visually places outlines on the intended objects; no pair was removed after review.

The initial measurement process stopped after 24 completed frames. On 25 October,
the remaining 13 used the unchanged measurement function. The recovery validated
completed mask classes, dimensions, areas and centroids against retained numeric
logs, and verified all 24 completed masks remained byte-identical. The original
selection, resume script, log hashes, mask hashes and resumption plan are retained
under `outputs/v2/matched_object_actions_20261024`.

## Direct input interventions

Both the disabled-feature control and ball-input checkpoint receive exactly the
same pairs in both directions. Each intervention keeps the recipient's canonical
arm-state input fixed. The natural donor prediction is scored separately because
its recorded arm state is close, not identical.

1. Full-image replacement swaps the iPhone image and recomputes its ball features.
2. Object-region replacement takes donor pixels within the union of the recipient
   and donor object masks, dilated by two pixels; all other pixels stay unchanged.
   Ball features are recomputed from the resulting image.
3. Feature-only replacement holds the image fixed and replaces the four ball
   feature values. This deliberately inconsistent input isolates use of those
   slots. The disabled-feature model returns exactly zero change in this test.

Object-region replacement can also transfer background, shadow and occluder pixels.
Bowl swaps after release may affect a ball in the bowl. Full-image replacement
also replaces arm and background appearance. These interventions are diagnostic
inputs, not physically valid counterfactual scenes. The recorded donor action is
an observational comparator, not a known correct target for the synthetic image.

## Active development results

The primary contrast is the difference between the two recorded **two-action
translation prefixes**, treated as a concatenated six-dimensional vector. A
predeclared norm of at least **1 mm** makes direction/amplitude informative. Three
of three active ball pairs and two of three active bowl pairs meet that threshold.
Both reciprocal directions are included; they are not independent samples.

| Object / replacement | Informative pairs | Control effect / target contrast | Control cosine | Ball-input effect / target contrast | Ball-input cosine |
|---|---:|---:|---:|---:|---:|
| Ball / full image | 3 | 6.90% | 0.932 | 5.79% | 0.910 |
| Ball / object region | 3 | 3.87% | −0.238 | 4.20% | −0.630 |
| Bowl / full image | 2 | 11.84% | 0.075 | 11.82% | 0.094 |
| Bowl / object region | 2 | 8.75% | 0.050 | 8.56% | 0.102 |

Across all three active pairs for each object, the ball-input object-region effect
norm averages **0.151 mm for ball** and **0.408 mm for bowl**. The mean observed
target-contrast norms are 3.660 and 2.643 mm respectively. Ratios/cosines in the
table exclude the one below-threshold bowl pair. Ball-feature-only replacement
has a mean effect of **0.0819 mm**, 2.35% of target contrast, with cosine −0.289.

The policies also underpredict motion itself in these sampled active observations:
ball-input mean step magnitudes are **0.823 mm** against **4.252 mm** labels in the
ball group and **0.907 mm** against **5.230 mm** labels in the bowl group. Thus these
probes expose both weak object dependence and a broader movement-amplitude deficit.
The small observational donor-error changes do not establish correct grounding
or task improvement. Training-pair scores remain separately available in the report.

## Stationary development starts

All six development stationary pairs have zero recorded translation for the scored
prefixes. They cannot establish a correct directional response to a moved object;
their informative checks are false motion and input sensitivity.

| Stationary pair group | Target mean step | Control mean predicted step | Ball-input mean predicted step |
|---|---:|---:|---:|
| Ball layouts | 0 mm | 0.348 mm | 0.389 mm |
| Bowl layouts | 0 mm | 0.276 mm | 0.324 mm |

These are offline start observations, not closed-loop drift or simulator success.
No stationary rollout success, physical correction or checkpoint promotion follows
from them. The separate three-episode movement/drift gate and common development
rollouts remain necessary before reserved evaluation.

## Reproduction and verification

Use a fresh output directory; each phase verifies the frozen source/model/data
hashes. The script preserves missing or insufficiently separated pairs as unsupported
instead of selecting replacements:

```sh
.venv/bin/python scripts/diag_v2_matched_object_actions.py prepare --out outputs/v2/matched_object_actions_NEXT
.venv/bin/python scripts/diag_v2_matched_object_actions.py measure --out outputs/v2/matched_object_actions_NEXT
.venv/bin/python scripts/diag_v2_matched_object_actions.py score --out outputs/v2/matched_object_actions_NEXT
```

Six focused tests pass, covering partition isolation, target-independent selection,
causal stationary history, real successor requirements, individual joint/velocity
bounds, preservation of pixels outside the object union, informative-contrast
direction/amplitude, and unsupported object measurements. Ruff passes. The executed
37-frame measurement and 96 reciprocal model/pair observations completed successfully.
