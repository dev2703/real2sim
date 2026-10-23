# Geometry and control investigation — 23 October 2026

The historical learned-policy results remain 0/10. The current offline learning screen
has not demonstrated task completion. This investigation tests separate causes while
preserving all frozen training inputs and original evaluation artifacts.

## Reachable carry targets

The SO-101 has five arm joints. Independently specifying a ball position and a complete
gripper orientation can create an unreachable target. A new privileged diagnostic plans
the held ball's position in joint space, treats carry orientation as a soft preference,
and checks the exact FK target with the existing IK workspace, collision and joint gates.
It uses simulator object positions and the actual grasp offset; it is not a learned policy.

| Scene and seeds | Matched full-pose control | Feasible carry |
|---|---:|---:|
| Historical 12 mm base lift, seeds 0–9 | 0/10 | 10/10 |
| Same scene, fresh seeds 10–19 | 0/10 | 8/10 |
| Nominal no-spacer 2.4 mm base height, seeds 0–9 | 0/10 | 6/10 |

These counts use the existing pick/lift/release/rest criterion, not just ball-in-bowl.
They use the original 10 Hz/30-second diagnostic clock and 500 Hz physics, not the current
12.5 Hz learning screen's later evaluation protocol. Other scene registration and gripper
parameters remain provisional even when the base height is nominal.
Native 3 mm position, 0.35 rad orientation acceptance, 2 mm collision tolerance, joint
limits, actuators and friction are unchanged within each comparison. The no-spacer
comparison raises the workspace floor to the actual tabletop and versions that change.
The no-spacer controller places every ball it successfully lifts; four trials fail grasp.
Trying a vertical approach instead of the 15-degree approach failed all ten fresh-seed
trials, so that alternative was rejected rather than silently replacing failed attempts.

A subsequent, explicit privileged mode searches a fixed set of grasp tilts and IK seeds
before the first action. It ranks only paths satisfying the unchanged geometry/IK gates
by joint travel, then executes the checked joint segment. Jaw commands, grasp depth,
close FSM, friction, physics and scoring stay unchanged. A discovered implementation bug
used separate clipping per joint and departed from the validated segment; execution now
uses one common progress fraction bounded by every joint's velocity cap.

| Nominal no-spacer scene | Carry planner only | Reachable grasp plus carry |
|---|---:|---:|
| Development seeds 0–9 | 6/10 | 10/10 |
| Fresh seeds 20–29, algorithm frozen before evaluation | 5/10 | 10/10 |

All new successes include a held lift, release and resting placement. The fresh comparison
used no subsequent parameter changes. This establishes a scripted feasibility baseline in
the nominal simulator; simulator object positions remain privileged inputs, and the learned
policy's 0/10 has not changed. It is not a calibration acceptance result.
The [reachable-grasp report](../results/v2/reachable_grasp_diagnostic_20261023.json)
preserves the initial failed attempt, the path correction, fresh results and provenance.

This diagnostic uses the oracle's physically settled reset, not the original learned
benchmark's folded start. The original oracle also used weaker carry orientation settings.
Consequently oracle success never established camera/joint correspondence or ruled out
contact mismatch. The new strict comparison isolates a feasible carry strategy; it does
not explain away the historical learned policy's earlier approach failures.

Reproduce without rendering or training:

```sh
.venv/bin/python scripts/diag_so101_feasible_carry.py \
  --modes matched_pose feasible_carry --n-trials 10 --seed-start 0 \
  --out outputs/v2/feasible_carry_NEW
```

The [combined report](../results/v2/feasible_carry_diagnostic_20261023.json) retains every
trial and its failure stage. Individual reports are in `outputs/v2/feasible_carry_20261023`,
`outputs/v2/feasible_carry_20261023_fresh` and
`outputs/v2/feasible_carry_20261023_no_spacer`. Configuration and retained failures
matter when comparing them. Dependency hashes recorded after earlier runs are explicitly
retrospective audits, not claimed original-run freezes.

## Joint-coordinate alternatives

Six mapping hypotheses were compared on 9,617 training and 2,077 validation ticks.
Reserved episodes were excluded. All 35 training/validation episode starts exceed native
limits under the current provisional mapping. Clipping the six validation starts changes
the tool position by a median **12.51 mm** before control begins.

Train-derived offsets remove validation limit violations but increase validation table
penetrations beyond 2 mm from **691 to 910 ticks**. Homing-as-zero and endpoint-rescaling
alternatives do not establish physical correspondence either. There is no basis for
choosing zeros by the number of collision-free poses or rescaling calibrated degrees to
make the native bounds fit. No alternative mapping was adopted.

See [the joint comparison](../results/v2/joint_hypotheses_20261023.json) and
`scripts/diag_v2_joint_hypotheses.py`. Static counts include poses outside native limits;
they describe reference inconsistency and are not physical task-failure counts.

With the arm-independent camera below held fixed, a further conditional image fit is more
promising. An initial axis-line fit improves residuals but can assign different locations
along a hinge axis to the same visible horn in different frames. Broadening its pan bound
changes that solution, exposing the depth ambiguity rather than establishing a motor zero.

A stronger model fits pan/lift/elbow zeros plus one fixed axial horn coordinate per hinge
across every capture. It uses training images from episodes 0, 6 and 23; validation images
from episodes 31 and 36 do not enter optimization. Wrist flex/roll zeros and all signs remain
fixed. It estimates offsets **+1.78°, −2.55°, −4.73°**, with no active parameter bounds.
Validation point RMS decreases **12.59 → 6.65 px**. After fitting, separate static validation
checks reduce table penetrations beyond 2 mm from **691 to 66** and self-penetrations from
**695 to 307**; those contact counts never select the fit.

This remains a hypothesis: **620 validation ticks still exceed native joint limits**.
The six-parameter Jacobian has full local rank, but uncertainty assumes the camera/base/CAD
are exact and omits their systematic error. The shared-point and axis-only results should
not be mixed or used to widen native limits. The validation camera frame was previously
inspected, so this is exploratory verification rather than an untouched acceptance test.

See `outputs/v2/conditional_joint_zeros_20261023/report.json`. A separate
[verification record](../results/v2/conditional_joint_verification_20261023.json) checks
synthetic parameter recovery and confirms that changing validation pixels cannot change
the training objective. Independent raw-video decoding also confirms one training and one
validation JPEG/state frame identity. It does not establish original exposure/state timing.

## Camera and base registration

A training-only image similarity model describes much of the episode-6/7 view change:
scale is approximately 0.973 around the image center. On held-out validation feature
matches, median displacement drops from approximately 11.9 px to 0.7–0.8 px. Full-match
p95 remains approximately 27 px because repeated features generate large association
errors. The smaller RANSAC-subset p95 is descriptive, not independent verification.

A second approach registers nominal holes in the fixed base foot against the mat-plane
camera, without arm joint readings. It supplies a provisional mat-to-base bridge and
suggests roughly 92 mm disagreement in mat center relative to the legacy scene. Repeated
hole patterns, unverified physical identities, mat thickness and the assumed pinhole
model still prevent adoption. The candidate and alternative alignment basins are saved
in `outputs/v2/static_base_camera_20261023/report.json`.
The [camera-only descriptors](../results/v2/camera_render_candidate_20261023.json) export
early/late optical parameters in the renderer's explicit pinhole format. They use the
native base frame and mark zero distortion as assumed; they are not a registered task
configuration. The late-view similarity is represented by focal/principal parameters
and a small optical rotation, without warping the recorded policy images.

An arm-image check, performed without fitting joint offsets, is promising. On validation
episode 31, source frame 400, distances from three manually identified servo-horn centers
to their projected joint-axis lines change as follows:

| Visible hinge | Legacy camera | Static-foot candidate |
|---|---:|---:|
| Shoulder lift | 22.1 px | 1.7 px |
| Elbow flex | 66.2 px | 9.9 px |
| Wrist flex | 152.2 px | 14.9 px |

Combined RMS falls from 96.65 to 10.38 px; the competing foot alignment gives 299.69 px.
These are distances to axis lines, not verified physical point reprojection errors.
Manual identification uncertainty is approximately ±8 px. The training frame's shoulder
axis check worsens, so this is not universal agreement or a passed independent-calibration
gate. Inspect [the three-way overlay](../outputs/v2/camera_joint_overlay_20261023/ep31_f0400.jpg)
and [the measurement record](../outputs/v2/camera_joint_overlay_20261023/report.json).

The renderer previously represented optics only through a centered field of view.
An explicit `projection_model: opencv_pinhole` now supports source-resolution `fx`, `fy`,
`cx`, and `cy` for both robot scenes and link cameras. It requires `width`, `height`,
`intrinsics_px` and explicit zero `distortion_coefficients`; nonzero distortion requires
rectified images and their corresponding intrinsics. Source/render aspect mismatches
are rejected. Legacy FOV behavior remains the default and no active camera config changes.

The implementation uses MuJoCo's focal/principal pixel parameters, whose principal value
is an image-plane offset rather than an OpenCV principal point. See the
[MuJoCo camera reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-camera).
Compiled frusta and rendered sphere pixels were checked against the requested intrinsics
at two render resolutions, including unequal focal lengths and an off-center principal point.
The paired-view backprojection helper now reads those same compiled intrinsics, avoiding a
silent fallback to a centered, square-pixel camera in later camera audits.

## Why a local projection is insufficient

On 123 selected in-limit validation actions from the completed both-camera ResNet,
bounded local Jacobian projection changes mean translation error only from **2.7588 to
2.7584 mm**. Its mean translation adjustment is 0.066 mm. This does not support local
projection as a repair for learned closed-loop failure. Sixty-six sampled actions outside
native limits are explicitly excluded rather than clipped; three episode-end actions are
padded. Second-prefix predictions use recorded successor states, not simulated rollouts.

See [the manifold diagnostic](../results/v2/action_manifold_resnet_20261023.json).
The full-pose carry planning problem above is a different, longer-horizon constraint.

## Camera uncertainty and wrist zeros

The [follow-up analysis](../results/v2/wrist_camera_followup_20261023.md) tests 36 explicit
camera assumptions and additional wrist landmarks. Camera height spans 0.499–0.560 m
across these assumptions; this is a sensitivity envelope, not a confidence interval.
Nonrepeating fixed-base contours reject the 180-degree registration alias, but another
alias fits them better than the selected registration. All candidates fail the roughly
3 px annotation scale. The camera/base transform is therefore still unaccepted.

With a fixed nominal lens mount, a full-cycle wrist search gives additional offsets of
flex +2.74 degrees and roll −59.40 degrees. Validation RMS improves from 51.03 to 16.75 px,
which is still inadequate. Illustrative one-degree camera rotations move the fitted roll
between −81 and −38 degrees. Freeing the lens mount exposes an exact roll/mount ambiguity
(Jacobian rank 4 of 5). More optimization of those same unknowns cannot identify physical
wrist zeros. Neither these landmarks nor the previous hinge-axis landmarks identify the
gripper closure zero. No offsets or native limits have been changed.

## Ball-support geometry

Both robot environments now support an optional `ball.support` box with explicit height,
footprint, task-frame center and yaw. It participates in actual contacts, resets separately
from the ball and reports support contacts. Sphere spawn height clears box sides and
corners as well as the top. Lift and one-jaw height thresholds use the ball's initial rest
height, preventing a tall support from receiving pickup credit. Absent support preserves
the previous scene and metrics. A rigid box does not model folded material compliance.

The [silhouette diagnostic](../results/v2/ball_support_sensitivity_20261023.json) estimates
ball center by fitting its exact optical tangent cone, independently of arm FK. Under the
nominal camera and 33.5 mm radius, six captures suggest support heights of 10.6–23.2 mm
above the tabletop. Including the disclosed camera assumptions and a ±1 mm ball-radius
sensitivity expands their combined range to approximately −28–56 mm. Sphere radius and
optical distance have an exact scale ambiguity. These images do not identify support
thickness; no height has been accepted. Native tabletop Z stays −2.4 mm even when a
hypothesis includes mat thickness, which is subtracted separately for height above the mat.

Four [image-based scene hypotheses](../results/v2/ball_support_scene_candidates_20261023.json)
use the observed off-center paper footprint and predeclared heights of 0/5/10/20 mm.
Ball, bowl, mat and camera positions change coherently. These are whole scene hypotheses,
not an isolated height sweep or accepted calibration. The executed configurations retain
their exact bytes and hashes; the manifest flags inherited legacy provenance strings that
do not describe the new coordinates. Fixed image placement has zero jitter, so different
random seeds would not supply independent placement trials.

## Loaded grasp retention

The [long-hold follow-up](../results/v2/grasp_retention_followup_20261023.json) requires
positive force on both jaws, lift above the initial rest height and no more than 10 mm
ball/tool displacement throughout a 30-second hold. This catches slip earlier than the
existing permissive `ball_held` indicator. With the original contact regularization, all 45
exploratory and eight coherent image-scene cases fail the strict 30-second criterion.
They are diagnostic sensitivities, not task trials.

| Comparison | Result |
|---|---|
| Image-scene default grasps, 0/5/10/20 mm support | Bilateral loss at 11.3/12.2/12.3/12.5 s; strict slip failure at 6.2–6.8 s |
| Encoder-width latch at three preloads | Worse retention than fixed closure |
| Deeper grasps and 45/60-degree approach bank | No retained 30-second case; only one high-angle scene had feasible paths and it failed acquisition |
| Newton iteration caps 10/50/100, same loaded state | Bit-identical contact trajectories; actual iterations never exceed seven |
| Physics rates 500/1000/2000 Hz, same loaded state | Bilateral loss at 9.1/9.1/9.2 s; strict slip failure at 5.1 s in every case |

Image-scene arm tracking remains within 0.3 mm while the ball slides toward the jaw tips.
This implicates contact/grasp modeling independently of arm tracking. Increasing numerical
resolution does not fix it; it does not by itself exclude contact regularization. Failed
high-angle searches that fall back to 15 degrees are explicitly labeled in the reports.
A successful brief carry cannot establish stable retention or physical geometry agreement.

## A contact-model cause and a working retention candidate

The subsequent [exact-state comparison](../results/v2/contact_regularization_20261023.json)
isolates friction regularization. It acquires each grasp with the original physics and
restores the complete, bitwise-checked integration state for three predeclared options.
Geometry, friction coefficients, mass, actuator gains, joint targets and collision gates
are identical. Four loaded states cover three original-scene placements and the fixed
10 mm image-support scene.

| Contact option | Strict 30-second holds | Maximum ball/tool drift |
|---|---:|---:|
| Original elliptic solver, `impratio=10`, NoSlip disabled | 0/4 | Bilateral loss after 3.9–12.3 s |
| `impratio=100`, NoSlip disabled | 3/4 | 3.63–10.27 mm; one exceeds the 10 mm limit at 29.5 s |
| `impratio=10`, `noslip_iterations=3` | 4/4 | 0.023–0.163 mm |

All four NoSlip comparisons have zero robot/table or robot/support penetration and
less than 0.50 mm arm target error. In the image-scene baseline, early sliding friction
utilization weighted by normal force is only 0.221; only 0.61% of normal load is on nearly
saturated contacts. Effective jaw/ball sliding friction is 1.0 because upstream jaw
priority overrides the scene's nominal 0.6. The diagnostic records compiled contact
parameters rather than assuming the task coefficient applies to every pair.

A separate [acquisition-first check](../results/v2/contact_regularization_preacquisition_20261023.json)
applies each option before settling, reset and grasp acquisition. Both original-scene and
image-scene NoSlip runs acquire and hold for 30 seconds, with 0.154 and 0.022 mm drift.
The higher-impratio alternatives also hold in these two cases, with 8.16 and 3.49 mm drift.
No physical model arrays change, and there are no robot/table or robot/support penetrations
during acquisition or hold. These two deterministic scenes are not a statistical success rate.

This identifies regularized friction creep as a sufficient cause of the tested long-hold
failures. It agrees with [MuJoCo's slip guidance](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip):
soft friction can creep even inside its cone; higher `impratio` or the NoSlip postprocessor
reduces that effect. NoSlip can destabilize more complex contacts and makes inverse
dynamics ill-defined, so these bounded hold checks do not establish general stability.
The correction requires explicit options; benchmark defaults and frozen training inputs
stay unchanged. This is a useful simulator retention candidate,
not measured contact calibration, learned-policy improvement or complete task evaluation.

Reproduce a complete acquisition and hold with the explicit diagnostic correction:

```sh
.venv/bin/python scripts/diag_so101_long_hold.py \
  --config outputs/v2/image_support_scenes_20261023/image_support_10mm_sim.yaml \
  --seeds 0 --seconds 30 --noslip-iterations 3 \
  --out outputs/v2/retention_noslip_NEW
```

The standard SO-101 and Panda scene loaders now also accept explicit simulator settings:

```yaml
contact_solver:
  impratio: 10.0
  noslip_iterations: 3
```

Omitting this block preserves each pinned model's original settings. Only these two options
are supported; invalid values and unknown fields fail before simulation. This permits
versioned comparisons through normal environment/evaluation entry points without runtime
monkeypatching. Evaluation summaries record the actual compiled options, including defaults.
Tests compare compiled geometry, friction, mass, inertia, gains and force limits bitwise.
No active task or simulator configuration has been changed.

The [full transport check](../results/v2/contact_transport_20261023.json) also exercises
grasp, lift, carry, release and one-second settled placement: original solver **4/4**,
NoSlip **4/4**, across three original-scene placements and one fixed image-support case.
There are no aborts, numerical warnings or IK rejections. Maximum table penetration is
0.0432 mm versus 0.0421 mm; support penetration is zero. This small matched test finds no
release regression. It uses privileged object positions and remains separate from learned
task performance and the stricter stationary retention test.

## Small learned inference check

The [inference report](../results/v2/inference_smoke_20261023.json) checks completed
both-camera ResNet and fine-tuned CLIP checkpoints on 24 recorded observations spanning
all six validation episodes. Each outputs a finite `[1, 25, 7]` chunk; 45 nonpadded
executed-prefix actions per model are scored. No reserved episodes are used.

| Checkpoint | Sample translation error | CPU median / p95 |
|---|---:|---:|
| Both-camera ResNet | 1.82 mm | 73 / 111 ms |
| CLIP last-two-block fine-tuning | 1.91 mm | 472 / 754 ms |
| Zero-translation reference on the same sample | 2.62 mm | — |

These sparse teacher-forced errors are not a model ranking or rollout score. Swapping
paired images changes predictions, but does not consistently worsen every error; visual
grounding is not established. CPU timings use two threads during the active training run
and exclude live capture/rendering. CLIP exceeds the 160 ms interval between two-action
chunks at 12.5 Hz on this CPU configuration; GPU timing remains to be measured once the
training device is free.

An actual single-camera ResNet simulator check executes **25 ticks / 2 seconds / 13 policy
calls**, with no abort, IK rejection or action saturation. Mean CPU inference is 55.94 ms,
maximum 131.9 ms. It uses the exact frozen provisional task and a strict, collision-free,
in-range recorded validation start (episode 9, frame 100). All 47 source/configuration
hashes remain stable during that run. The ball is not approached or lifted in two seconds;
this is an execution smoke test, not a full task attempt or success-rate estimate.

Two-camera simulator inference still lacks a declared wrist-camera mount. The code
correctly refuses to invent one. The default v2 calibrated task is also absent, and its
configured start exceeds native joint bounds. The single-camera diagnostic supplies an
explicit valid recorded start without weakening strict reset checks or formal calibration
gates. Its separate command and source snapshots are linked in the report.

Evaluation now closes environments exactly once after setup or rollout failures, including
checkpoint, tracker and video errors. Physics aborts remain failed rows. Regression checks
cover both resource cleanup and existing target-reference behavior. Actual robot/support
penetration tests also confirm both IK solvers reject moved support obstacles under their
existing tolerance; no collision-gate change was needed.

## Remaining evidence

Independent static CAD correspondences and intrinsics are needed to accept camera/base
registration. A verified wrist landmark or mount location must break the roll/mount
ambiguity. Ball radius and folded-support thickness/shape need independent measurements,
and gripper closure/force response and jaw surfaces remain unverified. Refit joint zeros
only against the accepted camera, then replay recorded tasks before evaluating learned
rollouts. Test learned-controller changes on the unchanged benchmark and report them
separately from privileged simulator diagnostics.

Verification: the latest CPU unit and selected control/contact integration suite passed **665
tests**; three MPS/CUDA cases were skipped and three graphics-dependent camera cases
were excluded from this run (they passed separately in the earlier camera investigation).
All 59 frozen training metadata hashes match. The queue remains active at 19/22 completed
jobs, with no reported failures. This includes the explicit contact settings, evaluation
cleanup, target-reference behavior and support collision regressions. Ruff and whitespace
checks pass for the changed files.
