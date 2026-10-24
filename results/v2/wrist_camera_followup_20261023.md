# Wrist zeros and camera uncertainty — 23 October 2026

The existing recordings reject the simple small wrist-offset model, but they do not
yet identify physical wrist zeros. No calibration, joint limits, scene geometry,
recorded images, or frozen learning inputs were changed. No reserved episodes enter
any fit or reported coordinate check.

## Conditional wrist tests

[The diagnostic](../../scripts/diag_v2_wrist_zeros.py) fits manually marked lens-front
and gripper-horn pixels in seven training captures from episodes 0, 6 and 23. Four
validation captures from episodes 31 and 36 are scored after fitting. The nominal
CAD lens point, static-foot camera and earlier proximal-joint fit are held fixed.
Manual point uncertainty is approximately 8 px per coordinate; the lens is partly
occluded and the horn's physical identity remains tentative.

| Conditional model | Train point RMS | Validation point RMS | Interpretation |
| --- | ---: | ---: | --- |
| Zero wrist offsets | 49.00 px | 51.03 px | Baseline under the provisional camera/proximal fit |
| Fixed nominal lens and horn, offsets within ±30° | 28.14 px | 29.73 px | Roll reaches −30° bound; small-offset model rejected |
| Same bounds, shared axial horn location | 22.87 px | 23.47 px | Roll still reaches −30° bound |
| Whole roll cycle, shared axial horn location | 17.45 px | 16.75 px | Exploratory hypothesis; residuals remain too large |

The whole-cycle model searches six initial roll angles on training residuals only.
Five converge to the same solution: wrist flex **+2.74°**, wrist roll **−59.40°**, and
horn axial location **+0.22 mm**. A sixth reaches a worse boundary solution. These
are *additional* offsets to the existing mapping, not measured actuator zeros.
Diagnostic search bounds are not native joint limits and do not alter them.

The apparent local rank is insufficient evidence for adoption. With the whole-cycle
model, changing one proximal zero by ±3° moves the fitted roll between **−72.93° and
−44.20°**. Illustrative ±1° rotations of the optical frame move it between
**−81.06° and −37.72°**. These are deterministic systematic sensitivity cases, not
confidence intervals or proposed camera updates. Some sensitivity cases improve
validation error, but validation never chooses a camera or fit.

Allowing the lens mount point to move freely exposes an exact gauge: rotating the
wrist zero and rotating that point around the same axis can cancel. The numerical
Jacobian has rank **4 of 5**, with smallest weighted singular value approximately
**1.6e−11**. No finite marginal uncertainty for roll is reported. A verified rigid
landmark/shape, measured lens mount, or independent roll reference is needed to
break this ambiguity. These landmarks also cannot determine gripper closure zero:
the lens is fixed to the gripper body and its hinge-axis point does not move when
the jaw opens.

A separate check on every recorded training/validation arm coordinate leaves
**620 of 2,077 validation ticks** outside native limits; **613** violate lift limits.
No clipping or contact-based selection hides this inconsistency. Even a better
downstream image fit would not establish agreement for the complete arm.

[Full report](../../outputs/v2/wrist_zeros_20261023_checked/report.json) includes
untrimmed per-landmark residuals, every initialization, local rank, source-image
hashes, raw coordinate ranges and sensitivity cases.
[Overlay](../../outputs/v2/wrist_zeros_20261023_checked/wrist_landmarks.jpg) shows
observed points in green, the zero-wrist baseline in orange, and the whole-cycle
conditional candidate in blue. It is a diagnostic image, not an acceptance result.

## Camera and fixed-base ambiguity

[Camera sensitivity evidence](camera_uncertainty_20261023.json) contains 36 explicit
scenarios: ±3 px individual mat corners, ±20 px principal point, ±1 px foot detections,
±3% measured mat sides, 3 mm mat thickness, ±1% pixel aspect, ±0.05 radial k1, and two
combined extremes. Every fit uses the original training image, all 21 detected foot
spots, the disclosed soft-L1 objective and the original selected yaw basin. No arm
states or validation/reserved images choose these scenarios.

Across those assumptions, camera position in native-base coordinates ranges from
**0.2288 to 0.2760 m** in X, **−0.7623 to −0.5659 m** in Y and **0.4986 to 0.5604 m**
in Z. This is a scenario envelope; unmodeled CAD, point identity, exposure timing,
mat planarity and lens effects remain outside it. It is not a confidence interval.

Fourteen manually reviewed, nonrepeating fixed-base contour points provide a
different spatial cross-check from the repeated holes. The approximately 180°
alias has **57.18 px RMS**, versus **16.87 px** for the original selected basin.
However, another basin approximately 66° from the selected one gives **14.88 px**
RMS. **Every basin fails the stated approximately 3 px annotation scale.** Thus the
contours reject some aliases but do not verify the leading registration. Repeated
hole agreement and a small mat-corner error cannot justify adopting the camera.
These contour pixels come from a previously inspected training image and are not
an untouched acceptance set. They do not refit or select the perturbation cases.

[Contour overlay](camera_uncertainty_20261023.jpg) shows all six alternatives.
Every recorded provenance hash in the camera sensitivity report was checked.

The next physical disambiguation must establish unique static CAD correspondences
and intrinsics independently before accepting arm zeros. Until then, carry the
camera/base alternatives as hypotheses when estimating ball support; avoid using
successful contacts or task completion to select a calibration.

## Verification

Eleven focused tests pass, including validation-pixel isolation, the free-point
roll gauge, preservation of out-of-range coordinates, mat/table thickness sign,
noncentral rectangular-pixel intrinsic recovery, and compiled camera projection.
Ruff passes for both diagnostics and the new tests.

```sh
.venv/bin/pytest -q tests/unit/test_v2_wrist_camera_diagnostics.py \
  tests/unit/test_v2_static_base_camera.py
.venv/bin/python scripts/diag_v2_wrist_zeros.py --out outputs/v2/wrist_zeros_NEW
```
