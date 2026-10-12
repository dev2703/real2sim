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

## 2. Workspace and frames

| Item | Value used | Label | Source / note |
|---|---|---|---|
| Table height | 0.75 m (world z of table top) | PROVISIONAL-STANDARD | Standard desk |
| Task frame | Origin on the table top directly below the SO-101 base origin. x away from the robot, y to its left, z up. | design | `docs/architecture.md` §4.3 |
| `T_task_so101base` | identity | PROVISIONAL-STANDARD | |
| Black mat | 0.45 × 0.40 m, centre at (0.25, 0.0) m in task frame | PROVISIONAL-STANDARD | Typical large desk mat |
| Ball start region | disc of radius 0.03 m centred at (0.25, +0.12) m (paper marker) | PROVISIONAL-STANDARD → DATA-DERIVED | Replaced at M2 by the mean gripper FK position at first close-on-ball over training episodes |
| Bowl centre | (0.22, −0.12) m | PROVISIONAL-STANDARD → DATA-DERIVED | Replaced at M2 by the mean gripper FK position at release |

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

## 4. Robot joint conventions

LeRobot `DEGREES` mode for the five arm joints and `RANGE_0_100` for the gripper. These are
inferred from the data ranges; the operator's recording command is not yet confirmed.

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

M2 verifies the sign and offset of each joint by projecting simulated FK onto iPhone frames at
three distinctive poses.

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
