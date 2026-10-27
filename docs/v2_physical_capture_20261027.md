# Physical registration session — 27 October 2026

The operator confirms: **“Available, with robot and cameras unchanged.”**
This permits fresh checks against the recorded setup. It is a setup report, not
an independently verified camera transform or joint-zero measurement.

The [prepared session folder](../outputs/v2/physical_calibration_20261027/)
contains a session log, separate camera fitting inputs, a measurement worksheet,
copies of the current calibration files, and target reference layouts. Its status is
`prepared_not_captured_or_measured`; measured fields remain empty. No physical
robot or camera was opened while preparing it.

The operator has also confirmed that no calibration board or printer is available,
but a ruler/measuring tape and plain paper or cardboard are available. The active
route below therefore uses measured hand-marked points. Earlier checkerboard-only
inputs remain as unused preparation artifacts.

## First: make a target using the ruler

Use stiff cardboard, or secure paper flat on a hard book cover. Draw twelve small,
clear crosses in four columns and three rows, about 40 mm apart. Label them 01–12
using this [reference layout](../assets/calibration/manual_cross_target_reference.svg):

```text
09  10  11  12
05  06  07  08
01  02  03  04   → +x
↑ +y
```

Cross 01 is the origin. Measure each cross centre's horizontal and vertical
position relative to it, rather than assuming the suggested spacing is exact.
Record those actual millimetre coordinates, measurement method and uncertainty
in [manual_target_points.csv](../outputs/v2/physical_calibration_20261027/manual_target_points.csv).
Numbered identities stay fixed across every view. The target's coordinate plane
is separate from the robot-base frame.

Known metric points seen at different target orientations support planar camera
calibration without a checkerboard. The method follows
[Zhang's planar calibration](https://www.microsoft.com/en-us/research/?p=145334)
and [OpenCV's general calibration interface](https://docs.opencv.org/5.0/main_modules/calib.html).
The hand-marked target uses manual source-pixel correspondences; it is not passed
to the existing automatic checkerboard detector. Measurement and annotation error
remain explicit, and results receive independent verification.

The [manual target fitter](../scripts/calibrate_v2_manual_target.py) is implemented.
Seven focused tests pass, including known-camera recovery, verification isolation,
duplicate-source rejection and degenerate-view rejection. An explicitly
[synthetic CLI check](../outputs/v2/manual_target_synthetic_verification_20261027/fitted/report.json)
recovers its known focal lengths; it is software verification, not a physical
calibration result. Both prepared real-camera inputs remain incomplete until the
operator supplies actual measurements and source captures.

## Capture without changing the existing registration

Keep the robot base and camera mounts at their recording positions. Move the
board to obtain different views, and retain the camera resolution, lens, zoom and
capture settings. Record any settings the device cannot report as unknown.

1. **Camera optics:** For each camera, capture six sharp fitting views across the
   three declared fit groups and two independent verification views across the
   two declared verification groups. Vary board tilt, distance and position in
   the image. Keep verification groups out of fitting and selection.
2. **Board to base:** Record an arm-clear view showing the fixed board and named
   base features together. Independently measure the board origin, axes and
   height relative to those features. A board on the mat alone identifies a
   camera-to-board transform, not the robot-base transform.
3. **Joint and jaw registration:** Use familiar operator-controlled static poses.
   Record follower readbacks together with an independently measured physical
   upper-arm/link pose and identifiable reference features. Keep additional
   verification poses separate. At several held jaw openings, measure the actual
   pad gap and record the simultaneous gripper readback, including a physical
   closed reference. A reported angle of zero is not itself a known physical pose.

The [session log](../outputs/v2/physical_calibration_20261027/session.json)
holds source-file references and the independent base registration. The prepared
[manual global-camera input](../outputs/v2/physical_calibration_20261027/iphone_manual_optics.json)
and [manual wrist-camera input](../outputs/v2/physical_calibration_20261027/wrist_manual_optics.json)
already declare six fit and two verification view slots each. Their source-frame
paths, individual measured cross coordinates and source-pixel annotations must be
populated from the actual capture. Four declared corner IDs (01, 04, 09 and 12)
estimate each verification view's pose; the remaining eight points check that view.

The existing mat provides useful planar image-to-table checks, but its single
fixed view does not independently determine unrestricted camera intrinsics,
distortion and robot-base registration. The previously tested mat-camera candidate
failed confirmation. The new target is moved through varied tilts while the
camera mounts and mat remain fixed.

## Local collection and current connection state

At preparation time, no USB robot serial controller was visible in this session.
Only the debug console, Bluetooth incoming port and a Bluetooth headphone port
were enumerated. This does not contradict physical availability; the controller
may need connecting to the machine running capture.

The normal LeRobot robot connection configures the motors and its recording loop
sends actions. It is not a passive position logger. The dedicated passive capture
helper reads the existing calibration and joint positions without requesting
motor motion or changing torque/configuration; its dry-run does not open devices.
Use the capture command documented by that helper after confirming current device
identities. Historical camera indices and USB port names are hints, not discovered
or verified current devices. No model-driven hardware rollout is part of this
measurement session.

The [passive helper](../scripts/capture_v2_passive_calibration.py) has been
checked with five fake-hardware tests. A lossless FFV1 video roundtrip preserves
RGB pixels exactly in the existing LeRobot environment. Device calibration is
read and compared with the supplied snapshot; disagreement stops collection
without rewriting it. Cleanup closes the serial port with `disable_torque=False`.
Exposure timestamps and original applied-command history remain unavailable.

From this project's directory, the following is a **dry-run example**. Replace
the port and camera indices with their confirmed current identities. Indices
0 and 2 are historical hints only:

```sh
../lerobot/.venv/bin/python -B scripts/capture_v2_passive_calibration.py \
  --port 'REPLACE_WITH_CURRENT_FOLLOWER_PORT' \
  --iphone-camera-index 0 --top-camera-index 2 \
  --calibration outputs/v2/physical_calibration_20261027/configuration_snapshot/follower_calibration.json \
  --out outputs/v2/physical_calibration_20261027/raw/capture_01 \
  --duration-s 30 --dry-run
```

After filling the actual device identities, use `--capture` in place of
`--dry-run` to collect. Each run requires a new output path. The helper defaults
to dry run and never overwrites an existing capture directory. Its lossless
videos retain native 1280×720 images; JSONL host timestamps and frame associations,
not constant-rate video playback, describe the observed timing.

## Process the actual measurements

Once actual source frames, measured cross coordinates, point annotations and
camera settings are recorded, inspect the prepared manual input:

```sh
.venv/bin/python scripts/calibrate_v2_manual_target.py inspect \
  --input outputs/v2/physical_calibration_20261027/iphone_manual_optics.json \
  --out outputs/v2/physical_calibration_20261027/iphone_manual_readiness_AFTER_CAPTURE.json
```

Use the wrist input for the second camera. A fitting run requires completed
measured inputs, not just the current template. Physical camera/base, joint-zero
and jaw verification follow optical fitting; new parameters are checked on the
declared verification captures before the frozen real-data ACT is retested.

The [full capture protocol](v2_calibration_capture_20261025.md) describes the
remaining source/timing and loaded hold/release records. Current measurements
can be compared against old footage because the operator reports unchanged
setup; that applicability still receives independent image and pose checks.
