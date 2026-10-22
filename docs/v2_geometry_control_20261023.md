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
Native 3 mm position, 0.35 rad orientation acceptance, 2 mm collision tolerance, joint
limits, actuators and friction are unchanged within each comparison. The no-spacer
comparison raises the workspace floor to the actual tabletop and versions that change.
The no-spacer controller places every ball it successfully lifts; four trials fail grasp.
Trying a vertical approach instead of the 15-degree approach failed all ten fresh-seed
trials, so that alternative was rejected rather than silently replacing failed attempts.

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

## Remaining obstacles

The source images show the ball on folded white material, while the scene starts it directly
on the table. Support height/contact geometry and the background appearance therefore do
not match yet. Existing fixed-target loaded-hold diagnostics also lose grasp after roughly
11–15 seconds, despite good unloaded arm tracking. A successful brief carry does not settle
retention geometry.

Resolve the camera/base correspondences against independent visible features, then assess
joint zeros using those fixed transforms. Version the actual object support and grasp
geometry before replaying complete recorded tasks. Test any learned controller change on
the unchanged benchmark and report it separately from these privileged diagnostics.
