# Vision alignment and remaining learning bottlenecks — 24 October 2026

Pretrained vision models now measure source-image object boundaries and support
conditional camera/object estimates on randomly selected frames. The geometry
candidates fail fresh confirmation, so no physical mapping is adopted and no
corrected-FK Cartesian retraining is claimed. Two additional matched learning
experiments address object localization and follower-target semantics. All four
new policies still fail their bounded development approach screens. Reserved
evaluation, collision gates, native joint limits and default physics stay unchanged.

## Random source frames and actual vision models

Seed 20261024 selects eight source captures before inspection: four training
captures for fitting, two training captures for verification and two validation
captures for verification. Each has original 1280×720 iPhone and wrist images,
giving 16 model inputs. Selection and role assignment are immutable, with exact
PNG/video hashes in `outputs/v2/random_vision_geometry_20261024/selection.json`.

The local pipeline runs [Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
at revision `a2bb814dd30d776dcf7e30523b00659f4f141c71` and
[SAM2.1 tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny)
at `de431c4043854a71d8101e17995dfe596bf101a5`. Text prompts propose tennis ball,
bowl, black mat and white paper boxes; SAM2 estimates visible masks. A separately
recorded visual foot-box prompt supports the CAD comparison. All inference is
local. Confidence scores remain model scores, not physical measurement uncertainty.

Model overlays and source masks are saved under `outputs/v2/grounded_geometry_20261024`.
The [camera report](../results/v2/model_camera_geometry_20261024.json) and
[object report](../results/v2/object_geometry_measurements_20261024.json) retain
pixels, assumptions, complete report references and model/source hashes.

## What the images constrain

Mat edges are refined from model masks against source grayscale gradients;
straight-line intersections define virtual corners. The baseline uses the supplied
0.305×0.406 m mat, centered principal point, square pixels and zero distortion.
It does not use robot FK. Metric camera poses are in mat coordinates.

| Quantity | Model-assisted estimate | Interpretation |
|---|---|---|
| Early fixed-camera focal length | 1399.17 px | Conditional on optical and mat assumptions |
| Late focal length | 1354.48 px | Same restrictions |
| Camera distance normal to mat | 0.53837 / 0.53573 m | Close to the existing approximate 21-inch scalar, not full registration |
| Vertical FOV | 28.86° / 29.77° | Derived from the conditional focal lengths |
| Bowl rim diameter candidates at declared 30 mm elevation | About 116–121 mm | Inner/outer rim ambiguity; supplied dimensions imply 104 mm outer diameter |
| Bowl diameter across declared optical/elevation scenarios | About 111–125 mm | Scenario envelope, not a confidence interval |
| Ball-support height under nominal camera/radius | −2.9 to −3.9 mm | Physically inconsistent; do not assign this to paper thickness |
| Ball-support assumption envelope | About −36 to +28 mm | Camera, mat plane, segmentation and radius uncertainty dominate |

The bowl stage fits observed near-complete rim arcs, rather than treating the whole
bowl silhouette as its rim. It requires at least 80% angular coverage and 1.5 px
radial RMS. One occluded capture has no eligible rim and stays excluded. Sphere
depth uses exact ray tangency at a declared nominal 33.5 mm radius; only three
unobscured balls on paper qualify, all fitting captures. Jaw/bowl-occluded or held
balls cannot become full-silhouette support measurements. Mat-frame +z points away
from the camera, so height above mat is −z. Source-scale conventions remain explicit.

The negative nominal support heights and configured/estimated bowl discrepancy
are checks against accepting a visually plausible fit. Supplied physical dimensions
stay unchanged. No simulator outcome selects a camera or object dimension.

## Confirmation and quantities that remain ambiguous

The original development verification includes up to 30.74 px corner RMS. Some
verification scores were inspected during extraction debugging, so they are not
presented as untouched acceptance evidence. After freezing the exact extraction
rules and original K/T hashes, seed 2026102402 selects two fresh captures from
unused training episodes. Their roles are confirmation before decoding or inspection.
The same local models and frozen extraction run without adaptation, replacement or
refitting:

| Fresh capture | Original camera candidate | Corner RMS | Result at fixed 3 px scale |
|---|---|---:|---|
| Episode 2, frame 82 | Early | 14.54 px | Fail |
| Episode 15, frame 19 | Late | 30.57 px | Fail |

The [fresh confirmation report](../results/v2/fresh_model_camera_confirmation_20261024.json)
records the plan written before image access, immutable hashes, every extracted
measurement and failure. This confirms that shared camera/mat pose is not accepted.
Mat movement, camera movement, segmentation and optical assumptions remain possible
causes; these images do not uniquely distinguish them.

Repeated observations of one plane do not identify unrestricted intrinsics: the
data-only rank is 8/10 for free fx/fy/cx/cy plus pose and 8/12 with two radial terms.
The fixed-foot silhouette screen preserves all six historical base yaw candidates.
Changing from a top-plane to solid-foot raster changes the preferred basin, while
contour disagreement remains large. Mask semantics, clamp/sidewall inclusion and
occlusion make this an unstable registration criterion. No base yaw is adopted.

Wrist views also expose model errors: a ball can receive a confident bowl label,
and many useful boundaries are cropped or occluded. The existing CAD wrist optical
pose remains available for explicitly nominal experiments, but those detections do
not independently calibrate its intrinsics or separate mount rotation from wrist
joint zero. Physical joint signs/zeros, exact TCP and jaw-gap registration therefore
remain unresolved. RGB snapshots do not supply mass, friction, compliance or loaded
force/release response. Existing encoder and scalar measurements retain their
original provenance; nominal values are not relabeled as vision measurements.

## Learning bottleneck: localization without action dependence

A new portable frozen-DINO bottleneck predicts a ball heatmap center and visibility,
then uses the predicted center and canonical proprioception to predict two actions.
The matched control has identical 57,552 trainable parameters and architecture.
Only the auxiliary image-derived localization loss differs. Both train for 256×16
examples on the same 1,024 cached training observations; validation covers all 2,071
unique nonextrapolated actions. No true object coordinates enter action inference.

| Metric | Action-only control | Auxiliary ball supervision |
|---|---:|---:|
| Visible pseudo-center error | 120.16 px | 3.90 px |
| Executed-prefix translation error | 2.3773 mm | 2.3822 mm |
| Bounded approach success | 0/3 | 0/3 |

These pixel errors compare against fixed image-derived pseudo labels, not physical
ground truth. The learned center is also 3.42 letterboxed pixels from SAM2 mask
centroids on the original eight random global images, with center-definition and
occlusion caveats. An always-visible classifier already matches the measured 92.77%
visibility-event accuracy; robust disappearance handling is not demonstrated.

Moving the supplied bottleneck center by 20% of image width changes translation
outputs by only about 0.08 mm, compared with typical moving labels near 5.67 mm.
Accurate localization does not force the action head to use location. In matched
125-tick rollouts, auxiliary supervision is worse in closest approach and neither
model lifts or releases. Both have zero IK rejections, aborts and table penetration.
Portable raw-image/cached inference agrees within 2.38×10⁻⁷ normalized action units.

The [auxiliary experiment](v2_aux_keypoint_bottleneck_20261024.md) documents the new
checkpoint type, common loader, losses, complete metrics and all six traces. The
remaining learning problem is causal action dependence on object location. Stronger
same-arm/different-object training pairs or an explicit image-relative action objective
are justified directions; an additional localization loss alone has not solved it.

## Control bottleneck: recorded leader commands versus achieved follower targets

The new joint-target experiment retains original ticks and compares current recorded
leader targets against next observed follower arm positions, preserving the current
leader gripper command. Both share an explicit native-feasibility mask and pooled
training-only joint normalization. Current/target invalid observations never anchor
training, and chunks stop at gaps rather than compressing time or bridging exclusions.
Mask reasons and terminal extrapolation remain separate.

There are 6,362 usable training rows and 1,427 validation actions: 66.15% and 68.70%
of the source partitions, across 38 contiguous segments. Coverage by condition and
every exclusion are reported; this is not a full-dataset score. Checkpoint semantics
are saved and checked on resume. A new explicit joint controller checks original
workspace, native limits and collision tolerance on endpoints and 17 path samples.
It rejects invalid commands; it does not relax gates or certify continuous collision
avoidance between samples.

Both actual ACT policies train for 256×16 examples. Common next-follower FK error
improves from 32.114 to 24.579 mm with follower supervision, but both are worse than
the recorded-current hold reference at 5.434 mm. Both are 0/3 in bounded approach
trials; the follower policy is slightly worse in closest distance. This is a short
underfit pilot, not evidence of a solved controller or a corrected physical mapping.
Next achieved positions are desired setpoints, not measured actuator commands; this
does not identify servo lag or loaded response.

The label feasibility audit is independently useful: 274/1,427 native-compatible
next-follower targets fail the provisional scene's existing gates—214 collision
(15.00%) and 60 workspace (4.20%). These are model/scene constraints, not proof of
real robot collision. Recorded target inconsistency reinforces the geometry gate.
The training mask is not silently changed to match this provisional scene.

The [target experiment](v2_follower_targets_20261024.md) includes data/segment
contracts, actual training, physical-unit validation, guarded execution and all six
rollouts. Joint/collision limits and NoSlip=0 remain unchanged. No checkpoint is
promoted and no reserved evaluation is consumed.

## Validation and reproduction

The complete unit suite passes **709 tests**, with two CUDA-only skips. Seventeen
selected training-resume, head-lifecycle and controller/collision integration checks
pass, plus the new joint adapter's actual-physics regression. Ruff and whitespace
checks pass. Full model outputs, source PNGs, masks, plans and trace rows stay local.

Use fresh output paths:

```sh
.venv/bin/python scripts/sample_v2_vision_geometry.py --out outputs/v2/random_geometry_NEXT
.venv/bin/python scripts/measure_v2_vision_geometry.py \
  --selection outputs/v2/random_geometry_NEXT/selection.json --out outputs/v2/masks_NEXT
.venv/bin/python scripts/extract_v2_model_mat_edges.py --help
.venv/bin/python scripts/diag_v2_multiframe_camera.py --help
.venv/bin/python scripts/measure_v2_object_geometry.py --help
.venv/bin/python scripts/run_v2_keypoint_bottleneck.py --help
.venv/bin/python scripts/run_v2_follower_joint_pilot.py --help
```

Physical alignment and corrected-FK Cartesian retraining remain gated by independent
registration and confirmation. The new image measurements make the contradictions
and uncertainty concrete; changing simulator appearance or accepting model confidence
as metrology would not resolve them.
