# V2 real/simulation geometry alignment — 21 October 2026

Geometry is the current priority. Learning screens remain deferred until independently
verified registration and reference replay pass. **Real/sim convergence is not yet established.**
The operator confirmed motor calibration, robot base and cameras unchanged since v2 recording;
see the [setup record](../configs/v2/setup_provenance_20261021.json). Current measurements can
therefore check the recordings. Historical timing and command provenance remain unverified.

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

1. Fill a versioned copy of [the measurement template](../configs/v2/measurements.template.json).
   Measure unloaded gaps near 0/15/30/45/60%, actual readbacks, consistent contact locations,
   repeatability and base support thickness. Reserve 15% and 45% before fitting. Measure
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
