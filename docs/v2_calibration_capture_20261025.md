# Calibration capture sheet — 25 October 2026

**27 October update:** the robot and cameras are available and unchanged, but no
printed target or printer is available. The active
[hand-marked target session](v2_physical_capture_20261027.md) uses the operator's
ruler and paper/cardboard. The checkerboard design below remains an optional
historical route, rather than a requirement to proceed.

The next physical session should resolve camera/base registration and provide a
measured gripper hold/release reference. The existing footage supplies useful image
tracks but no accepted metric registration. The operator reports **no calibration
target yet**. The proposed next target is the printable checkerboard below, mounted
on a flat rigid backing. Its physical dimensions remain unmeasured.

The [printed-board session plan](../configs/v2/calibration_capture_20261025.printed_board_plan.json)
records that availability and the proposed design separately from actual measurements.
Print the board, mount it flat, then measure its pitch before recording. This sheet
also supports another rigid target if one becomes available.

Copy the [blank session log](../configs/v2/calibration_capture_20261025.template.json)
and fill actual observations. Its nulls are missing measurements. It deliberately
contains no assumed optical transform, verified CAD identity, jaw clearance or load.

The [print-ready A4 checkerboard design](../assets/calibration/checkerboard_a4_25mm_design.svg)
has 9×7 squares, 8×6 inner corners, 25 mm **design** pitch. Print A4
landscape at 100% with fit/shrink disabled, mount it flat on a rigid surface, and
measure the printed pitch in both directions and the 100 mm reference bar. Record
the measured values and uncertainty. The design dimensions do not populate any
measured calibration field automatically.

## Before recording

- Record the target definition and **measured** square/marker/outer dimensions,
  material/rigidity, dimensional uncertainty and target coordinate axes. Record
  actual print dimensions rather than relying on a print-dialog scale setting.
- Mark a unique target origin and **+x / +y** directions on the rigid backing,
  outside the detected checkerboard area. Photograph these marks with the target
  and robot base during registration. Keep this orientation assignment fixed across
  captures; a symmetric checkerboard alone can leave its origin/orientation ambiguous.
- Record the robot/printed-part revision, base support/spacer thickness, applicable
  leader/follower calibration JSON and controller settings. Keep original files and
  their hashes. A new session does not establish the original demonstrations'
  recording-time provenance.
- Keep each camera at its intended capture setting. Record actual source size,
  lens, zoom, focus and stabilization settings and any setting the device cannot
  report. Retain original videos and timestamps; resizing comes after measurement.
- Start the synchronized robot trace with follower readbacks, sent commands and
  applied commands if the logger exposes both. Record joint order/units, clock and
  the camera-to-trace timestamp alignment method and uncertainty.
- Declare whole **fit** and **verification** capture groups now. Keep the
  verification groups out of parameter fitting, threshold tuning and replacement
  selection. Reserved policy-evaluation episodes remain separate from this session.

## Short capture sequence

These are proposed recording blocks, not completed measurements or prescribed joint
motions. Use familiar operator-controlled poses within the existing recording setup.

| Block | Approximate capture time | Required evidence |
|---|---:|---|
| Optical calibration | 45–60 s | Several clear target views at different tilts, image positions and distances. Hold each briefly for sharp frames. Include the image corners and centre. Assign the last independent view groups to verification before fitting. |
| Target-to-base registration | 20–40 s | Keep target and base rigid relative to one another. Record an arm-clear view showing named base features and the target, plus independent dimensional measurements that establish their full relative transform. |
| Joint landmarks | 30–60 s | Several familiar arm poses with timestamped follower states and clearly named link landmarks. Include separate verification poses. Record these now; fit joint zeros only after camera/base verification. |
| Unloaded/loaded hold and release | 30–60 s | Record an unloaded stationary reference, two loaded stationary segments followed by release, and one independent loaded verification repeat. Retain several seconds of steady observation before each release. |

For base registration, include identifiable fixed features at multiple heights and
useful spatial spread: the foot, upper opening and side feature are candidates, but
their IDs must be confirmed against the actual hardware. The existing repeated foot
apertures must not be assigned by whichever correspondence gives the lowest residual.

**A board placed on the mat alone gives camera-to-board geometry; it does not
register the robot base.** Either independently measure `T_base_target` with a rigid
fixture/target or independently measure sufficient named points in both frames.
Record the method and uncertainties. Do not obtain this transform from the FK whose
joint zeros are being checked. If the target moves between optical views, each
target-to-base registration capture needs its own applicable rigid relation.

For loaded holds, record object mass and its measurement method, object dimensions,
the actual held joint readbacks, jaw gap and its definition/uncertainty, source-frame
hold intervals and release timestamp. Record gripper openings/gaps at additional
familiar settings using the existing [measurement template](../configs/v2/measurements.template.json).
The purpose is an applicable physical trace, not an estimate from the rendered ball.

## What to retain

Keep the original videos, timestamped robot trace, target definition, dimension and
fixture measurements, source-resolution identity photos, calibration/controller
files, and the completed session log. Each capture group needs source-frame/trace
intervals and an unchanged fit/verification assignment. Record a moved mat or camera
explicitly rather than absorbing it into one shared scene pose.

Use the actual measured source image dimensions, even if the previous dataset was
1280×720. If either camera's crop/lens/settings changes, its earlier intrinsics and
registration do not automatically apply to the new capture.

## Acceptance and implementation order

1. Fit intrinsics/distortion from independent target captures. Register the fixed
   camera to the independently measured base/task geometry, then score untouched
   captures and identifiable off-plane points. Retain every declared verification
   error and inspect metric point/object checks as well as pixel residuals.
2. With camera and base fixed, verify arm signs/zeros using named body-local points
   across the recorded poses. Check gripper clearance/TCP registration separately
   with the measured unloaded/loaded traces. A low camera residual cannot validate
   a joint or jaw correction fitted against the same unverified geometry.
3. Populate copies of [geometry landmarks](../configs/v2/geometry_landmarks.template.json)
   and [physical measurements](../configs/v2/measurements.template.json). The session
   log carries collection evidence; it is not itself an accepted registration input.
4. Use [the geometry fitter](../scripts/fit_v2_geometry.py) for staged camera,
   joint-zero and wrist candidates and [the readiness audit](../scripts/audit_v2_calibration_readiness.py)
   to inspect evidence completeness. Acceptance still requires the independent
   checks in the [geometry protocol](v2_geometry_alignment_20261021.md).
5. Once a physical correction is justified, use [FK regeneration](../scripts/regenerate_v2_fk_labels.py)
   to rebuild affected labels and training-only statistics under a new contract;
   retrain affected policies with matching inference geometry. Keep the three-episode
   movement/stationary and object-dependence gates before common development rollouts.

The physical capture is pending operator access to the robot and target. Existing
nominal values remain labelled as such; this checklist does not grant calibration
acceptance.
