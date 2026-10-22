# V2 real/simulation geometry alignment — 21 October 2026

**Execution update, 22 October:** the operator subsequently requested all learning comparisons
run before geometry acceptance. The [offline exploratory queue](v2_learning_screen_20261022.md)
is running against explicitly provisional labels and recorded images. The physical
verification requirements in this report remain unmet; the queue does not claim otherwise.

Geometry work continues alongside the authorized provisional learning screen. Independently
verified registration and reference replay remain physical acceptance requirements.
**Real/sim convergence is not yet established.** The
[23 October investigation](v2_geometry_control_20261023.md) records new camera/base,
joint-mapping and controller comparisons.
The operator confirmed motor calibration, robot base and cameras unchanged since v2 recording;
see the [setup record](../configs/v2/setup_provenance_20261021.json). Current measurements can
therefore check the recordings. Historical timing and command provenance remain unverified.

## Follow-up — 22 October 2026

Joint-zero and camera agreement remain the first execution priority. The current input
inventory includes useful measurements in [`configs/task.yaml`](../configs/task.yaml): table
height **0.737 m**, mat size **0.305 × 0.406 m**, camera distance **24 in from the nearest
robot edge** and height **21 in above the table**, with stated ±3% camera-position uncertainty.
The bowl dimensions are also operator-supplied. These measurements are preserved in the
[config inventory](../results/v2/config_measurements_inventory_20261022.json); they should
be reused rather than requested again.

The saved iPhone `fx=fy=1786`, centred principal point and pose are a v1 fit using those
position constraints, mat pixels and provisional robot FK. Zero distortion was assumed.
The generating historical script clipped recorded joint poses before fitting and estimated
the nearest base edge using bounding spheres. Consequently its fitted task-frame camera pose
is not an independent reference for verifying joint zeros; the raw operator distance/height
constraints remain useful. Its report has joint RMS 77.57 px and mat RMS 40.78 px. Historical
joint-offset candidates also exist, conditional on fitted camera/landmark geometry; they are
not independently verified v2 offsets.

There is still no completed independent v2 camera/joint landmark file: both camera
intrinsics and distortion are null in the v2 measurement template, and there are no measured
v2 joint landmark rows. The local static-CAD inventory has 12 nominal anchors, zero verified
physical identities and only two tentative image associations. Earlier camera fits used
unverified robot FK or jointly fitted nuisance geometry, so they cannot independently fix
the camera for joint-zero verification.

A new [mat-only camera initialization](../results/v2/mat_camera_initialization_20261022.json)
uses the stored 0.305 × 0.406 m dimensions and annotated straight outer mat edges in training
episode 0, source frame 200. It uses no robot FK. The fitted source-resolution focal length
is **1,371.15 px** under centred-principal-point, square-pixel and zero-distortion assumptions.
The absolute camera-to-mat-plane distance is **529.47 mm**; the historical 21-inch height
constraint is 533.4 mm and was excluded from fitting/solution selection. Its applicability to
v2 and the mat thickness remain unverified. Mat-axis assignment follows the existing config
and is a hypothesis to check, not a new physical measurement.

Moving one corner coordinate at a time by ±3 pixels gives focal lengths **1,339.79–1,403.49 px**
and plane distances **520.58–538.59 mm**, with no focal-bound hits. This is deterministic
sensitivity, not a confidence interval; it does not cover joint corner errors, lens model
error or mat nonplanarity. The 0.074 px fit RMS uses the same four corners as the seven-parameter
fit and is not held-out verification. The saved transform is optical-from-**mat**, not a
registered task/base transform, so it cannot yet fix the camera for joint-zero acceptance.

[Annotations](../configs/v2/mat_camera.annotations_20261022.json) and the
[edge review overlay](../outputs/v2/mat_camera_20261022/edges.png) preserve the selected
pixels, source hashes and boundary ambiguity. Reproduce with new output paths:

```sh
.venv/bin/python scripts/diag_v2_mat_camera.py \
  --output outputs/v2/mat_camera_NEXT.json --overlay outputs/v2/mat_camera_NEXT.png
```

The existing [iPhone stability diagnostic](../outputs/v2/iphone_stability_20261021/README.md)
adds a concrete correspondence issue to resolve. Relative to episode 0, static-feature
displacement has a median of **0.65 source pixels** in the five sampled training episodes
before episode 7 and **11.62 pixels** in the 23 later training episodes; later apparent
similarity scale is 0.97288. The six validation episodes give 11.85 pixels. The effect appears
in the fixed foot, lower clamp and cabinet handle; it is not confined to moving arm links.
All 75 saved source-image hashes were checked, and rerunning episode 0 against episodes
6, 7 and 31 reproduced the saved matches exactly. Reserved episodes were excluded.

These are descriptive image displacements, **not landmark reprojection errors or a failed
calibration acceptance test**. The cause could include capture cropping/focus/intrinsics or
camera motion; this does not overturn the operator's unchanged-hardware attestation. Check
the applicable camera model on both recording periods before treating newly measured
intrinsics/extrinsics as valid for the whole dataset. No corrective image warp follows from
this diagnostic alone.

Registration code now rejects reuse of a source frame under different capture-group names
and duplicate frame/landmark observations. Joint landmarks from the same source frame must
use the same joint/gripper readings, and each named physical landmark must keep its native
body and local coordinates. This closes a route for fitting and verification to share data
under different labels. Source identity remains operator-supplied; renamed files or an
incorrect provenance statement cannot be independently detected by these checks.

Candidate reports now include complete data-only Jacobian null directions and the weakest
parameter direction, including fits with fewer observations than unknown offsets. Inspect
these against `arm_joint_order`; a low residual cannot identify an unobserved joint. Optical
depth and behind-camera counts are also reported per fit/verification partition. Synthetic
tests demonstrate that an impossible behind-camera fit can have zero pixel error. These
diagnostics do not establish physical calibration or adopt a candidate automatically.

The next physical step is calibrated source-resolution iPhone intrinsics/distortion and
independent metric landmarks with verified source-frame identities. Fit/check the fixed
camera first, then joint zeros with base/signs/landmark geometry fixed, then the wrist mount
with verified link poses. Keep whole captures separate for fitting and verification and
retain reserved policy episodes outside both. Use the existing commands below once those
inputs are available. The provisional offline learning screen is running separately.

The previously blank experiment row is already implemented: **K1 tool-frame deltas and K3
trajectory-consistency loss**. A newly reproduced mixed-precision NaN in the trajectory
loss is fixed by denormalizing and composing poses in at least float32 with autocast
disabled for those operations; gradients and coordinate semantics are preserved. Their
learning comparison is included in the provisional screen; physical acceptance still requires
geometry and reference replay. See the
[follow-up verification record](../results/v2/geometry_followup_20261022.json).

## Standard hardware and no-spacer update

The operator subsequently confirmed a standard SO-101 with **no spacer** and requested use
of public hardware sources. The official [SO-101 simulation documentation](https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/608122e9ac330a753735f2e18aee73338e9ac407/Simulation/SO101/README.md)
distinguishes midpoint-range zeros in the new model from horizontal-extension zeros in the
old model, and explicitly notes that LeRobot's 0–100 gripper mapping is absent from its
URDF/MJCF. Menagerie derives from the new-calibration model, so switching to the old zero
convention would not be justified by the current evidence.

The [official assembly guide](https://huggingface.co/docs/lerobot/main/en/assemble_so101#gripper--handle)
attaches the moving claw directly to the servo horns. This supports a 1:1 output-angle-to-jaw
rotation for the stock hardware. The listed 1/345 follower gearing is inside the servo, not
an extra factor to apply to calibrated output-encoder angles. This supports using the saved
encoder span for the nominal slope; it does not establish the installed claw's exact closed
zero. The assembly page's historical software instructions are not used to change our recorder.

The base mesh has a broad, downward-facing support plane **2.4 mm below its native origin**:
182 coplanar triangles, approximately 2,051 mm². Thus no spacer implies an origin 2.4 mm
above the tabletop, not zero. The old 12 mm offset leaves a 9.6 mm air gap. This conclusion
comes from the standard CAD and the operator's setup description, not a collision-minimizing
fit. See the [base support audit](../results/v2/no_spacer_base_geometry_20261021_v1.json).

A [versioned nominal candidate](../configs/v2/geometry.nominal_so101_v1.json) now specifies
the 2.4 mm support offset and encoder-derived gripper slope, with closure still provisional.
It is executable in the static contact diagnostic; it is not a calibrated task configuration.

| Static geometry | Table samples beyond 2 mm | Self-collision samples beyond 2 mm | Maximum table penetration |
|---|---:|---:|---:|
| Old 12 mm base offset, old gripper conversion | 948 | 712 | 31.33 mm |
| No spacer: 2.4 mm base offset, old conversion | 1,593 | 712 | 40.93 mm |
| No spacer plus encoder-derived gripper slope | 1,593 | 712 | 40.93 mm |

All arms use the same 9,392 in-limit source poses; 4,486 out-of-limit poses remain excluded.
These are descriptive full-recording geometry checks, not task-success counts. Correcting
support height exposes more mismatch. Arm self-collision is invariant to base translation;
gripper scale does not repair the fixed-jaw/table path. Arm reference correspondence remains
the next cause to resolve, with camera registration needed to compare it to recorded images.

Reproduce the combined candidate without hardware or manual jaw measurements:

```sh
.venv/bin/python scripts/diag_v2_geometry_collision.py \
  --geometry-candidate configs/v2/geometry.nominal_so101_v1.json \
  --out results/v2/geometry_standard_no_spacer_NEXT.json
```

The [combined result](../results/v2/geometry_standard_no_spacer_20261021_v1.json) hashes the
candidate, source code, model and dataset. This isolated test keeps the camera/table/object
world positions fixed while changing robot height. Register all task-frame coordinates
coherently before using the candidate for rendering, dynamics or learned rollouts. Native
limits and collision checks stay active; the legacy configuration remains reproducible.

## What the new diagnosis establishes

The earlier 0.30–0.33 mm tracking error compares a simulated tool with a reference generated
by the same provisional mapping. It establishes local controller tracking before rejection,
not correspondence with the real tool. The remaining differences have distinct causes:

| Question | Evidence | Consequence |
|---|---|---|
| Are arm values degrees or normalized ±100? | All 27,736 rows fit the degree-to-encoder lattice, p95 residual at most 0.000043 tick; the alternative is about 0.4–0.49 tick. | Keep degree units. Physical joint signs/zeros still need measured landmarks. |
| Are table failures just oversized colliders? | All 948 in-limit sampled poses over the 2 mm collision tolerance also put the same body's visual mesh below the table plane; 932 exceed 2 mm in the visual mesh too. Collider-minus-visual depth median 0.327 mm, maximum 0.361 mm. | Removing/shrinking colliders alone cannot repair the reconstructed path. |
| Is the joint transform implementation inconsistent? | Native FK agrees with independent composition of the pinned XML transforms. | This checks implementation, not physical zero angles or assembly geometry. |
| Does gripper percentage represent the same angle? | Calibration endpoints 2026–3498 imply 129.407° servo motion per 100%, while the old simulation mapping uses 110.002°. | Verify a separate affine conversion; do not derive calibration from native limits. |
| Is base/table height measured adequately? | The provisional 12 mm base-origin lift leaves the lowest base CAD vertex 9.6 mm above the table. | Measure support/mount thickness and register the native base frame. Raising the robot to remove collisions is not calibration. |

Sources: [joint normalization audit](../results/v2/joint_normalization_provenance_20261021_v1.md),
[CAD/collision audit](../results/v2/geometry_collision_cad_20261021_v1.json),
[mapping and gripper audit](../results/v2/mapping_gripper_20261021_v1.json).
The full descriptive scans include previously inspected reserved recordings; they are not
calibration fitting or an untouched evaluation. Phase boundaries remain unannotated, so the
counts include return motion and do not imply 948 failures during grasp/transport.

Examples, with original source frames:

| Episode:frame | Collider below table | Lowest robot visual vertex below table |
|---|---:|---:|
| 0:326 | 8.284 mm | 8.060 mm |
| 21:228 | 3.207 mm | 2.970 mm |
| 36:280 | 2.619 mm | 2.268 mm |
| 13:682 | 31.330 mm | 30.984 mm |

The first three correspond to early rejected targets in the earlier motion audit. A global
base/table translation also cannot resolve the shoulder/lower-arm self-collision hypotheses.
Visual mesh convex-hull overlap does not prove overlap of concave physical parts; verify
assembly and joint landmarks before changing any self-collision exclusions.

## Implemented corrections and diagnostic tools

`JointMapping` now accepts an explicit `robot_so101.gripper.percent_to_joint` with
`offset_rad` and `scale_rad_per_percent`. Its inverse is used by observed state and canonical
proprioception. Legacy endpoint mappings retain their previous behavior when absent.
The candidate slope is **0.0225857112873464 rad/%**, conditional on positive direct drive;
the old closure offset −0.1745 rad is retained only for the diagnostic. No production task
configuration has adopted it. Updating a mapping also requires a consistent versioned start
and regenerated labels/normalizers/checkpoint contracts.

Candidate comparison with the same provisional closure:

| Readback | Angle change from old mapping | Old tip-sphere separation | Candidate separation |
|---|---:|---:|---:|
| 0% | 0° | 2.627 mm | 2.627 mm |
| 15% | 2.911° | 23.745 mm | 27.750 mm |
| 30% | 5.821° | 46.249 mm | 54.003 mm |
| 45% | 8.732° | 67.853 mm | 78.743 mm |
| 60% | 11.643° | 88.045 mm | 101.208 mm |

These are distances between the specific `fixed_jaw_sph_tip1` and
`moving_jaw_sph_tip1` surfaces, **not measured real jaw gaps** or minimum distances over
entire jaws. All recorded measured openings (0–66.304%) fit the candidate's native joint
interval; 100% would exceed it. There is no reason to widen the limits. Physical closure,
transmission sign/ratio and contact surfaces remain to be verified.

Cartesian adapters now expose native gripper clipping after calibration. Reference replay
and kinematic-control traces retain requested/bounded angles, adapter saturation, native
joint clipping and velocity limits. A previously silent pre-clamp no longer appears as an
unconstrained command in these v2 reports. Legacy replay/oracle reporting is unchanged.

The registration fitter estimates camera pose from independent metric landmarks with fixed
intrinsics, then estimates joint-zero offsets with camera/base/landmark geometry fixed.
Fit and verification captures are disjoint, reserved groups are rejected, reference states
are never clipped, and reports expose rank, conditional uncertainty and limit violations.
Full input and dependency hashes make candidates reproducible. No candidate is auto-adopted.
The wrist mode fits only a rigid camera-to-link transform from independent landmarks and
separately verified link poses. Using FK for these poses requires prior arm/base verification;
the wrist fit cannot validate the same unverified kinematics used to construct its inputs.

## Physical inputs and execution order

1. Use the standard-CAD/no-spacer candidate above for nominal diagnosis. A versioned copy of
   [the measurement template](../configs/v2/measurements.template.json) remains available for
   independent verification. If jaw checks are needed, use unloaded gaps near 0/15/30/45/60%,
   actual readbacks, consistent contact locations and repeatability; reserve 15% and 45%
   before fitting. No spacer is already confirmed; do not request its thickness again. Measure
   ball/bowl dimensions and independently register the base/table frame.
2. Obtain calibrated source-resolution intrinsics/distortion and independent task-frame
   landmarks, including off-plane observations or independent camera-pose verification.
   Do not fit a camera to unverified robot FK and use that same camera as proof of the FK.
3. Fill [the landmark template](../configs/v2/geometry_landmarks.template.json) with whole
   calibration captures assigned to fit/verification before optimization. Use the fixed
   iPhone to verify known link landmarks across varied poses and determine arm zeros/signs.
   A sign hypothesis must be explicit; the zero-offset fitter does not silently flip signs.
4. Verify the rigid wrist camera-to-link transform after arm/base correspondence is checked.
   Preserve `observation.images.top`. Different camera intrinsics, distortion or crop cannot
   be corrected solely by moving the camera. Version rectification if needed by the renderer.
5. Repeat the unchanged-limit geometry audit and the phase-reviewed trajectory comparisons.
   Separately test loaded retention before any friction/contact tuning.

Reproduce the current read-only diagnostics with new output filenames:

```sh
.venv/bin/python scripts/diag_v2_geometry_collision.py \
  --out results/v2/geometry_collision_NEXT.json
.venv/bin/python scripts/diag_v2_mapping.py \
  --gripper-calibration ~/.cache/huggingface/lerobot/calibration/robots/so_follower/aligned_follower.json \
  --out results/v2/mapping_gripper_NEXT.json

# Only after filling independent measured inputs; the empty template is rejected.
.venv/bin/python scripts/fit_v2_geometry.py camera \
  --input configs/v2/geometry_landmarks.MEASURED.json \
  --output outputs/v2/camera_candidate_NEXT.json
.venv/bin/python scripts/fit_v2_geometry.py joint-zero \
  --input configs/v2/geometry_landmarks.MEASURED.json \
  --output outputs/v2/joint_candidate_NEXT.json
.venv/bin/python scripts/fit_v2_geometry.py wrist \
  --input configs/v2/geometry_landmarks.MEASURED.json \
  --output outputs/v2/wrist_candidate_NEXT.json
```

Full collection details are in the [operator session](v2_operator_session.md). The new
registration tests use explicitly synthetic measurements and establish code behavior only.

## Gate for resuming the learning matrix

- No concealed clipping or invalid overlap at the declared, phase-reviewed task start.
- Independent landmark p95 ≤10 source pixels and tabletop object XY error ≤5 mm; jaw
  measurements agree on held-out openings with declared measurement uncertainty.
- Three unloaded 20-second holds: maximum tool drift ≤0.5 mm, no hold-stage IK rejection.
- Ten development reference episodes across conditions: ≥9 full placements, closure/transport
  tool-reference error p95 ≤3 mm, and loaded retention through the demonstrated duration.

Only then run tiny-set learnability and the existing bounded ResNet/DINO/CLIP/flow/action
matrix. Original folded-start evaluation stays separate if a measured pre-approach start is
needed. The 60% SO-101 milestone and frozen Panda transfer remain unachieved.
