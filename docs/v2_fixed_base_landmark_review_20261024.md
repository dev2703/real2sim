# Fixed-base landmark evidence — 24 October 2026

The footage contains useful stationary features at multiple nominal CAD heights,
but the current annotations do not establish a unique physical camera/base
registration. The existing inventory contains twelve candidate landmarks and
**zero verified physical identities**. A new [evidence inventory](../results/v2/fixed_base_multilevel_inventory_20261024.json)
records the model attachment check, usable image candidates, earlier tracking
coverage and missing measurements. No geometry or policy correction is adopted.

## Fixed parts and usable source regions

The pinned model attaches `base_so101_v2`, `base_motor_holder_so101_v1`, the pan
servo and Waveshare plate to the fixed `base` body. The similarly named
`motor_holder_so101_base_v1` is attached to `shoulder`, after `shoulder_pan`, and
must not supply stationary anchors. Sweeping pan through −1, −0.4, 0.4 and 1 rad
leaves every fixed mesh centroid unchanged; the shoulder meshes move. This tests
the nominal model hierarchy, not the physical printed-part revision.

[Episode 0, frame 200](../outputs/v2/iphone_stability_20261021/source_frames/ep00_f0200.png)
is a useful tracking reference because the arm clears the upper base. The original
frame-98 reference partly hides that region. The candidate identities below come
from the [existing CAD inventory](../outputs/v2/cad_landmarks_20261021/static_landmarks.json)
and were visually checked against frame 200; their approximate pixels remain
unverified annotations.

| Candidate | Native base XYZ (mm) | Approximate source pixel | Main ambiguity |
|---|---|---|---|
| E: large upper base opening | −6.365, 0, 69.6 | 198, 248 | Use the projected rim, not an ellipse centroid; arm occlusion changes visibility. |
| I: holder side hole | −10.86275, −22.449, 45.7 | 190, 300 | Confirm the visible side and printed holder revision. |
| Foot mounting holes / diamond apertures | z = 15.1 | No new named pixel assignments | Repeated apertures and occlusion still permit different XY/yaw associations. |

For image tracking, tight initial rectangles around E are x=177–220,
y=237–259, and around I are x=179–203, y=290–311. These are image regions,
not measured CAD correspondences. Rejected or occluded tracks should remain
missing. Enlarging the upper region into the moving shoulder can turn arm motion
into apparent camera motion. The foot remains a separate low-height region;
the lower table clamp and cabinet handle provide useful independent scene checks.

E, I and the foot span three heights over 54.5 mm, but the base occupies a small
part of the image. Even correctly named points there may poorly constrain full
image intrinsics and distortion. Under the previously selected static-foot camera,
E and I project 20.58 px and 17.60 px from their tentative annotations. All six
prior basins remain in the JSON. This is a discrepancy check against approximate
pixels, not an independent calibration score or grounds to select a new basin.

## What the prior tracking actually established

The [existing relative-motion run](../outputs/v2/relative_scene_motion_20261024/report.json)
compares nine frames to episode 0/frame 98. Every comparison has only one supported
region in its global static fit. Six have no accepted held-out housing matches;
the other three have only two or three. Mat-pattern matching produces zero to two
matches per comparison and never fits its required similarity. Thus isolated
reported mat residuals cannot establish mat motion independently of edge or
segmentation error. A temporal track with an arm-clear reference, explicit
occlusion handling, and direct raw-image mat-edge checks addresses a real gap.

The earlier broader first-frame study supports an early/late change in capture
coordinates: a similarity fitted without foot features predicts validation foot
matches well in the median, while repeated-hole outliers leave p95 near 27 px.
Similarity scale alone does not distinguish camera movement, crop/zoom or focus.
Physical camera motion versus optical changes still requires independent optics
or spatially separated landmarks with sufficient depth support.

## Registration decision and next capture

Existing footage can test whether base coordinates and mat boundaries move
together. It currently cannot supply an accepted registration from the available
unverified identities. Do not fit joint zeros or gripper offsets against a camera
chosen to make those same joints agree. A conditional fit could be reported once
named correspondences are defensible, but cannot remove the remaining identity
and optical uncertainty merely by lowering its residual.

For a short new calibration session:

1. Record a rigid board of known dimensions at several tilts and distances using
   original-resolution frames and fixed capture settings. Keep an independent
   set of captures untouched until scoring.
2. Register the board to named robot-base landmarks or an independently measured
   rigid target. A board placed on the mat alone gives camera-to-board geometry;
   it does not identify the robot-base transform.
3. Include arm-clear views of the foot, upper opening and side hole. Confirm the
   printed-part revision and base support/spacer thickness. Use at least six
   named correspondences with multiple heights and useful spatial spread, and
   reserve whole captures and identifiable points for verification.
4. Record unloaded and measured loaded stationary holds followed by release,
   with synchronized follower states/actions, object dimensions and jaw-gap
   measurements. These observations support the separate gripper/contact check.

Only after independent camera/base verification should the joint-zero and gripper
checks drive FK-label regeneration, training-only normalization and retraining.
