# Corrective experiments after approach failure — 24 October 2026

The explicit ball-input pilot does **not** establish an improvement or task success.
Both matched candidates fail all three bounded development approach trials. Offline
motion errors worsen with ball features at this small budget. The implementation
now supports a matching image-feature contract, motion-sensitive checkpoint
shortlisting, an explicit nominal wrist camera, and audited FK regeneration.
Collision gates, native joint limits and default contact physics remain unchanged.
No checkpoint is promoted and no reserved evaluation is run.

## What was implemented and tested

The dataset and live observation adapter call the same fixed ball detector on the
current letterboxed RGB image. Four inputs extend canonical proprioception:
image u/width, v/height, enclosing radius/width, and a visibility indicator.
Missing detections produce four zeros; there is no future-frame latch, metric
backprojection or simulator-object fallback. The saved checkpoint contains the
complete detector specification and feature order. Runtime and offline checks
reject a mismatched transform. Paired-image replacement with stale features is
explicitly refused. Photometric augmentation, when requested, precedes both the
visual input and its feature extraction.

This is one explicit-grounding experiment: a fixed colour/shape ball proposal.
It does not implement a bowl tracker, annotated keypoint head or auxiliary target
loss. The detector matches a contour in 94.62% of training images; that coverage
is not localization accuracy or calibrated confidence. Image coordinates remain
image coordinates until independently registered camera geometry is accepted.

The matched control has the same 17-input ACT architecture and four constant-zero
feature slots. Both use the existing frozen pretrained ResNet18 backbone, identical
parameter counts, seed 0, episode split, label arrays, action statistics and execution
contract. Each receives 256 updates at effective batch 16: **4,096 examples**, with
32 warmup steps and a 256-step learning-rate horizon. Feature statistics use only
9,617 training images. Existing source images/labels remain immutable. This differs
from the old 64,000-example screen and cannot rank against it as a training ablation.

All six development episodes are scored at the two-action execution cadence:
**2,071 unique valid actions**, excluding six terminal extrapolation placeholders.
The [pilot report](../results/v2/visual_grounding_pilot_20261024.json) records
checkpoint/config/data hashes, matched-contract checks and full validation metrics.

| Metric | Disabled-feature control | Ball-image input |
|---|---:|---:|
| Executed-prefix normalized L1 | 0.47080 | 0.48516 |
| Translation error | 3.673 mm | 3.815 mm |
| Positive movement direction | 89.09% | 88.39% |
| Median moving step/label magnitude | 0.537 | 0.509 |
| Mean predicted stationary step | 1.014 mm | 1.093 mm |
| Maximum stationary-run summed-error proxy | 28.161 mm | 31.676 mm |
| Fixed offline shortlist score, lower is better | 0.97621 | 1.03783 |

The [parity and sensitivity report](../results/v2/ball_input_sensitivity_20261024.json)
checks 24 validation anchors. Added features match the live adapter **exactly**.
Swapping only ball features changes translation predictions by 0.0968 mm on average
for the ball-input model and exactly zero for the disabled control. Joint image/feature
swaps change the ball-input model by 0.8572 mm, but improve this sparse sample's
translation error from 2.628 to 2.543 mm. Sensitivity alone does not show correct
grounding. The small feature effect and absence of a worsening swap result do not
support a useful image-grounding claim. One short seed also cannot rule out better
detectors, supervised keypoints or auxiliary target learning.

## Bounded development rollouts and promotion

Both checkpoints then run through the normal inference adapters on the unchanged
provisional scene, strict recorded episode-9/frame-100 start, and common seeds 0/1/2.
Each trial has **125 ticks / 10 seconds**, retaining the existing control, joint,
collision and solver gates. The saved action contract uses accepted-target Cartesian
integration. These are bounded approach screens, not full-timeout task evaluations.
Ground-truth object poses are used only for scoring.

| Seed | Closest tool/ball distance, control | Ball input |
|---|---:|---:|
| 0 | 178.98 mm | 178.12 mm |
| 1 | 213.61 mm | 197.18 mm |
| 2 | 165.28 mm | 157.69 mm |

Both candidates are **0/3**, with no lift, release or abort. Ball features modestly
improve these short approach traces but remain more than 15 cm from the ball. The
control has 55 collision rejections on seed 1; the ball-input candidate has none.
Maximum robot/table penetration is 0.553 mm versus zero, below the unchanged 2 mm
gate. Source hashes remain unchanged through execution, and initial native arm,
tool and ball states match exactly between candidates. The ball contour is visible
at all 125 observed ticks in each ball-input trial.

The [rollout report](../results/v2/ball_grounding_rollouts_20261024.json) links every
per-tick trace and preserves the initial sandbox graphics setup failure separately
from the six executed trials. No outcome justifies promotion. The established
30-case common development gate and then frozen reserved suite remain required.
The six reserved dataset episodes were not evaluated or used for feature fitting.

## Camera, labels, controllers and contact evidence

The [calibration follow-up](v2_calibration_followup_20261024.md) verifies the existing
pinned CAD wrist mount and optical camera, exposes an explicit `top` binding and
renders both camera streams. Independent recorded optical registration remains
unaccepted. Old checkpoint/task hashes remain enforced; nominal-camera experiments
receive their own task/data contract.

The broad local search found the existing operator scalar measurements and encoder
calibrations. Independent intrinsics, verified static correspondences, physical
sign/zero checks, metric jaw gaps and loaded hold/release traces did not surface in
the searched locations. The [search record](../results/v2/calibration_measurement_search_20261024.json)
classifies the fitted and nominal files rather than treating filenames as metrology.

The new FK-regeneration tool rebuilds every supported Cartesian representation and
training normalization into a new immutable contract. Regeneration with only the
nominal optical binding changes training/validation states and targets by **exactly
zero**. It also exposes 3,219/9,617 training and 642/2,077 validation ticks outside
native arm limits, without clipping. This is a lineage check, not a physical label
repair. A justified joint or TCP correction must precede corrected-label retraining;
the ball pilot deliberately uses the unchanged provisional mapping.

The [control comparison](v2_control_contract_comparison_20261024.md) scores 90 common
native-compatible validation actions. Original Cartesian IK and the optional
position-priority fallback both accept 80/90, with 4.632 mm command/FK error to the
next follower pose. Soft orientation projection gives 4.666/4.661 mm. Learned
leader-joint commands accept 76/90, with 12.991 mm error; their target semantics differ
from achieved follower motion. Next-follower joints provide a feasible-label reference
(79/90 accepted, 0.298 mm after gate/rate handling), not a trained policy. All figures
are bounded command/FK proxies with exclusions and endpoint limitations disclosed.

The [hold/release evidence review](../results/v2/loaded_hold_release_evidence_20261024.json)
retains the simulator's strict-hold and scripted-release sensitivities, with no claim
of a matching physical measurement. NoSlip remains an explicit simulator option;
its physical relevance is unestablished and default physics stays unchanged.

## Reports, checks and reproduction

New ACT and frozen-head reports include executed-prefix, direction/amplitude,
stationary-drift and physical-unit metrics, with causal hold/persistence baselines.
The [selection protocol](v2_motion_selection_20261024.md) defines a fixed offline
shortlist score; full-chunk L1 remains descriptive. Resume rejects a changed metric
protocol or image-feature transform. Offline shortlisting does not authorize
reserved evaluation.

The complete unit suite passes **679 tests**, with two CUDA-only skips, including
the actual camera projection render tests with CoreGraphics access. Thirteen
targeted controller/collision integration checks pass. Focused ACT/head resume and
full-coverage validation integration checks also pass. Ruff and whitespace checks pass.

Use fresh output paths:

```sh
.venv/bin/python scripts/run_v2_visual_grounding.py \
  --out outputs/v2/ball_grounding_NEXT \
  --data-out .cache/v2_ball_grounding_NEXT --steps 256 --device cpu
.venv/bin/python scripts/diag_v2_ball_input.py \
  --pilot outputs/v2/ball_grounding_NEXT --data .cache/v2_ball_grounding_NEXT \
  --out results/v2/ball_sensitivity_NEXT.json
.venv/bin/python scripts/diag_v2_ball_grounding_rollouts.py \
  --disabled-checkpoint outputs/v2/ball_grounding_NEXT/disabled_control/checkpoints/best \
  --ball-checkpoint outputs/v2/ball_grounding_NEXT/ball_input/checkpoints/best \
  --out outputs/v2/ball_grounding_rollouts_NEXT
```

The next justified training step depends on accepted geometry or stronger independent
visual supervision. Repeating appearance changes or promoting this pilot would not
resolve the demonstrated approach failure or provisional FK labels.
