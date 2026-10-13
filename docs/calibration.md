# Calibration and physical parameters

**Status:** No full calibration has been done. On 2026-10-12 the user approved using standard values
until measurements exist (decision D-012), and supplied four bowl dimensions. Every value below
carries a provenance label. Results that depend on these values must cite this file.

| Label | Meaning |
|---|---|
| OPERATOR-SUPPLIED | Given by the user, 2026-10-12. Takes precedence over any other source. |
| PROVISIONAL-STANDARD | Comes from a published standard, a datasheet or a typical household size. Not measured on this setup. |
| FROM-CALIBRATION-FILE | Computed from the operator's own LeRobot calibration JSON. |
| DATA-DERIVED (planned) | Will be estimated at M2 from the recorded dataset, e.g. gripper FK at grasp and at release. |

`configs/task.yaml` holds the machine-readable copy. Any change goes into both files plus a
`docs/decisions.md` entry.

## 1. Objects

| Item | Value used | Label | Source / note |
|---|---|---|---|
| Tennis ball diameter | 0.067 m (radius 0.0335 m) | PROVISIONAL-STANDARD | ITF allows 6.54–6.86 cm |
| Tennis ball mass | 0.058 kg | PROVISIONAL-STANDARD | ITF allows 56.0–59.4 g |
| Ball inertia | thin spherical shell, I = 2/3·m·r² | PROVISIONAL-STANDARD | Tennis balls are hollow |
| Ball friction | 0.6, swept over 0.4–0.9 | PROVISIONAL-STANDARD | Felt on ceramic or rubber |
| Ball restitution | about 0.75, swept | PROVISIONAL-STANDARD | From ITF rebound: 135–147 cm off a 254 cm drop |
| Bowl inner radius at rim | 0.050 m | OPERATOR-SUPPLIED | |
| Bowl inner depth | 0.0254 m | OPERATOR-SUPPLIED | |
| Bowl rim (wall) thickness | 0.002 m | OPERATOR-SUPPLIED | |
| Bowl outer height | 0.030 m | OPERATOR-SUPPLIED | |
| Bowl base diameter | 0.060 m | PROVISIONAL-STANDARD | Not supplied |
| Bowl mass | 0.30 kg, modelled as static | PROVISIONAL-STANDARD | Not supplied. No bowl motion is seen in the data. |
| Bowl collision model | 20 tilted wall boxes plus a base cylinder; visual mesh kept separate | design | Plan §3.4 |

**Cross-check against the recorded images (please re-measure).** In the last iPhone frame of
episode 10 the ball is about 150 px wide and the bowl's outer rim about 267 px wide. Taking the
ball as 6.7 cm, the outer rim comes out near **12 cm**. The supplied values give 10.4 cm. The
visible outer wall also looks about 5–6 cm tall once the downward camera angle is accounted for,
against the supplied 3 cm.

With a 2.54 cm inner depth, a resting ball's centre (3.35 cm up) sits above the rim. The ball
would then be held only by the rim edge and could roll out more easily in simulation than in
reality.

The supplied numbers stay in use because operator input takes precedence. The simulation is
parameterised, so a corrected measurement is a one-line change. The bowl depth is also included
in the contact sensitivity sweep.


### Simulated bowl physics (measured in `tests/integration/test_so101_scene.py`, `results/audit/sim_bowl_physics.json`)

The simulated bowl is a straight-wall frustum: base radius 0.03 m (from the PROVISIONAL
base diameter) to rim radius 0.05 m, 20 wall boxes plus a floor slab. Ball-on-floor static
sink 0.05–0.18 mm, impact transient up to 1.93 mm (damping ratio 1.3), no tunnelling. A
floor-resting ball stays contained at every probed speed up to 0.5 m/s for all three depths
(0.0254, 0.045, 0.060 m); escape over the rim at the supplied depth needs more than about
0.7 m/s. The earlier "shallow bowl cannot contain a 0.12 m/s roll" statement was an artefact of
a vertical-wall model and is withdrawn; the result now depends on the unmeasured base diameter.

## 2. Workspace and frames

| Item | Value used | Label | Source / note |
|---|---|---|---|
| Table height | 0.75 m (world z of table top) | PROVISIONAL-STANDARD | Standard desk |
| Task frame | Origin on the table top directly below the SO-101 base origin. x away from the robot, y to its left, z up. | design | `docs/architecture.md` §4.3 |
| `T_task_so101base` | identity rotation; base origin 0.012 m above the table top (`base_z_offset`, D-014) so the table top is at z_task = −0.012 | PROVISIONAL (inferred) | |
| Black mat | 0.45 × 0.40 m, centre at (0.25, 0.0) m in task frame | PROVISIONAL-STANDARD | Typical large desk mat |
| Ball start region | disc of radius 0.03 m centred at (0.332, +0.101) m (paper marker) | DATA-DERIVED v3 (camera back-projection, §3) | Replaced at M2 by the mean gripper FK position at first close-on-ball over training episodes |
| Bowl centre | (0.330, −0.169) m | DATA-DERIVED v3 (camera back-projection, §3) | Replaced at M2 by the mean gripper FK position at release |

### Preliminary DATA-DERIVED estimates (2026-10-12, `real2sim.geometry.so101_fk`)

These come from the provisional joint mapping (§4) and gripper-frame FK on recorded follower
states. A "sustained hold" means the leader commands closed (< 8 %) while the follower is
blocked open by the ball (12–60 %) for at least 1 s. Only 12 of the 37 kept episodes contain
such a hold. In the others the leader gripper was not squeezed fully closed.

| Quantity | Mean (base frame, m) | Std (m) | n |
|---|---|---|---|
| Gripper position at grasp (ball centre) | x 0.340, y +0.042, z 0.034 | 0.030, 0.061, 0.042 | 12 |
| Gripper position at release (above bowl) | x 0.326, y −0.098, z 0.074 | 0.037, 0.076, 0.044 | 12 |

Shoulder pan is about −17° at grasp and +24° at release in 8 of the 12 holds. The mean grasp
height of 0.034 m matches the 0.0335 m ball radius. This supports the lift, elbow and wrist part
of the mapping, but does not verify it: pan sign against the camera is unchecked and the spread
is 3–7 cm.

Two further observations:

- 23 % of `shoulder_lift` readings, mostly the folded rest pose at about −105°, lie beyond the
  MJCF ±100° limit. Either the real range exceeds the model, or there is a small offset.
- Separation between the grasp and release clusters is 0.14 m by FK. A rough estimate from the
  image gives about 0.2 m.

These estimates replace the PROVISIONAL-STANDARD ball start and bowl centre in
`configs/task.yaml`.

## 3. Camera `observation.images.iphone`

| Item | Value used | Label | Source / note |
|---|---|---|---|
| Resolution and rate | 1280 × 720, 30 fps | dataset metadata | `meta/info.json` |
| Lens | iPhone main wide, 26 mm equivalent | PROVISIONAL-STANDARD | Phone model unknown |
| Horizontal FOV | 69.4° | PROVISIONAL-STANDARD | 2·atan(18/26) for a 16:9 video frame |
| Vertical FOV (MuJoCo `fovy`) | 42.6° | PROVISIONAL-STANDARD | 2·atan(10.125/26) |
| Intrinsics at 1280×720 | fx = fy ≈ 924, cx = 640, cy = 360 | PROVISIONAL-STANDARD | Video stabilisation may crop further and raise fx |
| Intrinsics after 192×256 letterbox | fx = fy ≈ 184.8, cx = 128, cy = 96 | derived | Scale 0.2 with 24 px top pad, via `adjust_intrinsics` |
| Extrinsics | about 0.55 m in front of and 0.45 m to the right of the base, 0.45 m above the table, pitched down about 40° | PROVISIONAL-STANDARD (eyeballed) | Replaced at M2 by a PnP fit on FK-projected gripper points and the bowl rim, with reprojection error reported |


### DATA-DERIVED camera fit (2026-10-13, `python -m real2sim.sim.camera_fit`)

PnP on 43 observable correspondences (held-ball centroids at gripper FK positions, ball at the
start marker, ball in the bowl) with the provisional intrinsics held fixed:

| Quantity | Value |
|---|---|
| Camera position (task frame) | (0.329, −0.444, 0.385) m after the D-014 refit (was 0.324, −0.419, 0.377) |
| Optical axis | pitched 42.1° down, rolled −10.9° |
| Inliers / RMS | 37 of 43 points after the D-014 refit (27 of 43 before; 17.2 px inlier RMS before) |
| Held-ball points | 23/27 inliers, 21.8 px |
| Start-marker points | 4/8 inliers, 36.6 px |
| In-bowl points | 0/8 inliers, 107 px: reprojection sits 104 px above the detections |

Focal refinement diverged (points span only a 0.15 × 0.25 × 0.15 m volume), so depth along the
optical axis is weakly constrained and the simulated robot renders slightly smaller than real.
The full axes are stored as `camera_iphone.xyaxes_task`; the scene uses them (the roll-free
look-at form costs 61.6 px). Label: DATA-DERIVED, approximate. Report and overlays:
`results/audit/camera_fit/`.

**Bowl and start region, DATA-DERIVED v3** (`python -m real2sim.sim.camera_fit backproject`,
after the 1.2 cm base lift of D-014): in-bowl ball detections (known-radius circle fit on the
ball outline) back-projected to the plane z = −0.012 + 0.0046 + 0.0335 give the bowl centre
**(0.330, −0.169) m**; start-marker detections at z = −0.012 + 0.0335 give the start region
**(0.332, 0.101) m**; both n = 8/8 with 1–2 cm spread. The FK-at-release value (0.33, −0.10) was
6–7 cm off because it marks where the gripper opened, not the bowl. Focal/depth uncertainty
adds about ±1 cm per axis. The refit with the lifted heights moved the camera by about 2.5 cm
and raised the inlier count from 27 to 37 of 43.

## 4. Robot joint conventions

LeRobot `DEGREES` mode for the five arm joints and `RANGE_0_100` for the gripper.
The 2026-10-13 numeric audit strongly supports these units: converting all included
recorded values back through the local follower/leader calibration yields integer
encoder ticks within 0.000043 tick at the 95th percentile for arm degrees, versus
0.40–0.49 tick for the alternative ±100 normalization. Gripper percent also matches
the encoder lattice. The original recording command and physical zero alignment
remain unconfirmed. Evidence: `results/audit/control_contract_20261013/normalization.json`.

```
arm joint [deg] = (ticks - (range_min + range_max)/2) * 360 / 4095
gripper  [%]    = (ticks - range_min) / (range_max - range_min) * 100
```

Follower calibration (`aligned_follower.json`), label FROM-CALIBRATION-FILE:

| Joint | Range (ticks) | Span |
|---|---|---|
| shoulder_pan | 789–3255 | 216.8° |
| shoulder_lift | 770–3172 | 211.2° |
| elbow_flex | 957–3168 | 194.4° |
| wrist_flex | 874–3133 | 198.6° |
| wrist_roll | 0–4095 | 360° |
| gripper | 2026–3498 | 129.4° over 0–100 % |

The real-to-sim mapping is PROVISIONAL-STANDARD. The Menagerie `robotstudio_so101` model derives
from TheRobotStudio `so101_new_calib`, which is designed to match LeRobot-calibrated degrees:

```
q_sim[rad]         = deg2rad(q_real[deg])                   # sign +1, offset 0, all five arm joints
q_sim_gripper[rad] = -0.1745 + (gripper% / 100) * 1.9199    # Menagerie range -10°..110°
```

Physical joint signs/offsets still require independent image/pose correspondence;
numeric encoder consistency does not verify the real-to-MJCF zero positions.
Do not infer a calibration offset from leader/follower tracking errors.

The local follower shoulder calibration spans ±105.5824°, whereas the MJCF is
limited to ±100°. An isolated diagnostic lowered only the shoulder lower joint
stop and actuator control bound to −105.5824°, retaining gains, geometry, weights,
and data. This improved shoulder tracking but did not improve leader-command tool
trajectory error (3.06 → 3.08 cm mean), and the frozen paired policy remained 0/10
with mean closest distance worsening 14.0 → 16.1 cm. **Not adopted** as a calibration
correction. Local calibration files are not proven to be recording-time files;
their endpoint spans alone do not establish physical joint alignment. See
`results/audit/control_contract_20261013/review.json`.

Gripper semantics: 0 % = closed. The follower reads about 1.2 % when closed on nothing and
30–40 % when closed on the ball. In the 7-D action, `gripper_target = gripper% / 100`.

## 5. Recording controller (do not change)

The follower runs P = 12, I = 0, D = 2 with safety limit 5, as reported by the operator. These
values apply to recording only. Simulated actuator gains are tuned separately at M2.

## 6. Still useful from the operator (optional)

- A re-check of the bowl outer diameter and height (see the cross-check in §1), plus bowl mass.
- Bowl and start-marker positions relative to the robot base.
- Phone model, height and distance.
- Jaw gap in mm at 0, 50 and 100 %.
- Confirmation that recording used `use_degrees=true`.
