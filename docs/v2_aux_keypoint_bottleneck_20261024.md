# Auxiliary image-keypoint bottleneck — 24 October 2026

Auxiliary supervision makes a frozen-DINO head predict the ball detector's image
center, but the action head still uses its location weakly. Neither matched
candidate completes approach in three bounded simulator trials. This separates
learning image localization from learning to act on object location.

The [curated report](../results/v2/aux_keypoint_bottleneck_20261024.json) contains
all validation metrics, checkpoint hashes, dependence interventions and rollout
traces. The full training artifact is
`outputs/v2/aux_keypoint_pilot_20261024/report.json`.

## Matched experiment and contract

The new policy computes a ball heatmap and detector-event probability from frozen
DINOv2-small spatial patches. Softmax heatmap coordinates and the predicted
probability form three features for a small Cartesian action head. Canonical
13-component proprioception is the other input. Both models contain exactly
**57,552 trainable parameters** and predict two actions before replanning.

The action-only control and auxiliary-supervised candidate have identical initial
weights, seed 0 minibatch draws, architecture, source labels, action normalization
and training budget: **256 updates ×16 examples = 4,096 draws**. The cache contains
1,024 episode-balanced training observations selected before fitting and all 1,037
validation inference anchors. Sampling uses replacement. No reserved image is
cached or evaluated. This is a different architecture and training subset from
the earlier ACT screen; their scores do not establish a matched model ranking.

Auxiliary targets come from the existing fixed HSV/shape detector on the current
letterboxed image. Visible pseudo targets supervise a spatial heatmap and image
center; all rows supervise the detector-event probability. Missing detections do
not supply center targets. These are **pseudo labels**, not independent object
annotations or physical measurements. The experiment is ball-only. Unverified
open-vocabulary bowl detections are not substituted as supervision.

Pseudo centers never enter the action head directly during training or inference.
Both always use the model's predicted center and event probability. The checkpoint
contains its frozen backbone, normalization, feature order and source geometry
contract. Raw-image inference and cached-feature training share the same ImageNet
normalization, bottom/right replicate padding and explicit float16 feature
round-trip. Portable checkpoint reload matches cached normalized actions within
**2.38×10⁻⁷** on the six checked anchors. The normal checkpoint factory and simulator
controller serve these weights without editing their saved contracts.

## Validation and image dependence

Every nonpadding, nonextrapolated validation action is scored once at the declared
two-action cadence: **2,071 actions** from all six development episodes.

| Metric | Action-only control | Auxiliary ball supervision |
|---|---:|---:|
| Executed-prefix normalized L1 |0.33874|0.33899|
| Mean translation error |2.377 mm|2.382 mm|
| Positive movement direction |93.94%|93.88%|
| Median moving-step/label magnitude |0.806|0.797|
| Mean predicted stationary step |0.503 mm|0.508 mm|
| Maximum stationary-run summed-error proxy |7.880 mm|7.908 mm|
| Offline shortlist score |0.54429|0.54779|
| Mean center error against visible pseudo targets |120.16 px|3.90 px|
| Center-error p95 against visible pseudo targets |148.93 px|9.24 px|

The auxiliary head learns the detector's center on 962 visible validation anchors.
Its detector-event classification accuracy is 92.77%, exactly the always-visible
majority baseline. Occlusion/missing-detection handling is therefore not established.
The action-only head is free to encode a nonphysical location; its 120 px pseudo-center
error shows why an unsupervised internal coordinate cannot be assumed to track an
object.

On the eight preselected source frames, the auxiliary center is 3.42 letterboxed
pixels from the separately pretrained Grounding DINO/SAM2 visible-mask centroid
on average. This is cross-model agreement, not independent annotation accuracy;
visible-mask centroids and enclosing-ball centers differ under occlusion. The
source hashes and individual comparisons are preserved in the report.

Swapping predicted centers within fixed validation batches changes translation
predictions by only **0.0044 mm** for the auxiliary candidate. Deliberately shifting
its horizontal center by 20% of the image width changes translation by only
**0.080/0.081 mm** on average, versus a 5.67 mm mean moving label. Removing the predicted
event signal changes translation by 0.543 mm. These single-input interventions break
image/feature consistency and have no measured counterfactual action labels; they
measure dependence, not accuracy on a changed scene.

Improved pseudo localization does not improve action error or force the policy to
use object position. The recorded trajectories allow the action head to rely on
proprioception. Correct action labels for varied object locations at matched arm
states, stronger object-relative action objectives, or recovery demonstrations
remain a separate learning bottleneck.

## Bounded development approach screen

Both saved checkpoints run through the normal observation adapter, Cartesian
controller and native physics in the unchanged provisional scene. The strict
recorded episode 9/frame 100 reset and seeds 0/1/2 are shared exactly. Each trial has
125 ticks / 10 seconds. Ground-truth positions enter scoring only; policy inputs
remain the observed image and proprioception.

| Seed | Closest tool/ball distance, action-only | Auxiliary |
|---|---:|---:|
|0|146.08 mm|166.32 mm|
|1|125.57 mm|145.01 mm|
|2|137.08 mm|156.92 mm|

Both candidates fail approach **0/3**, with no lift or release. The auxiliary
candidate is farther from the ball in these short matched traces. All 750 ticks
execute with zero IK rejections, zero robot/table penetration and no abort. This
failure does not depend on numerical IK rejection in these cases. Default NoSlip
iterations remain zero; collision and joint limits are unchanged. CPU policy-call
p95 is 76.7–86.8 ms across these trials; rendering and the initial localization probe
are excluded. Relevant source/config hashes remain unchanged through execution.

These bounded trials do not establish full-task success or physical calibration.
Neither checkpoint is promoted and reserved evaluation remains unused. Full
artifacts are under `outputs/v2/aux_keypoint_rollouts_20261024`.

## Reproduction and checks

Use fresh output paths:

```sh
.venv/bin/python scripts/run_v2_keypoint_bottleneck.py cache \
  --cache .cache/v2_keypoint_NEXT
.venv/bin/python scripts/run_v2_keypoint_bottleneck.py train \
  --cache .cache/v2_keypoint_NEXT --out outputs/v2/aux_keypoint_NEXT
.venv/bin/python scripts/diag_v2_keypoint_rollouts.py \
  --pilot outputs/v2/aux_keypoint_NEXT --out outputs/v2/aux_keypoint_rollouts_NEXT
```

Five focused unit checks pass: shared raw/cache precision, predicted-feature action
inputs, missing-center masking, padding, contract rejection and portable backbone
reload. Ruff and whitespace checks pass. Existing provisional FK geometry remains
the source of all Cartesian labels; no metric calibration or contact-parameter
adoption follows from these results.
