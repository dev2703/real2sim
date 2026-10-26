# Operator measurements available — 26 October 2026

The new `physical_measurements.md` is present and has now been ingested. Its
estimated ±5% bounds and original inch values are preserved in
[`measurements.operator_20261026.json`](../configs/v2/measurements.operator_20261026.json),
along with its SHA-256. This supersedes the earlier statement that no completed
measurement document had been found. The existing demonstrations are also usable
local inputs; a separate calibration video is not needed to begin inspecting them.

The new [`task.scalar_measured_20261026.yaml`](../configs/v2/task.scalar_measured_20261026.yaml)
is an executable task candidate that applies the unambiguous ball radius. An 8-inch
circumference gives **32.340 mm radius**, with estimated bounds **30.723–33.957 mm**.
Its camera, joint mapping, base height and gripper remain the prior provisional
values. A ball-size correction changes collision geometry, but does not justify
changing FK labels or retraining an unchanged Cartesian-label dataset.

The [`comparison report`](../results/v2/operator_measurements_20261026.json) exposes
larger discrepancies that changing network parameters cannot repair:

| Quantity | New report | Existing task | Meaning |
|---|---:|---:|---|
| Ball radius | 32.340 ± 1.617 mm | 33.500 mm | Existing value is within the reported bound; central value applied. |
| Ball–bowl distance | 304.8 ± 15.24 mm | 242.65 mm | Outside the bound if the measurement is centre-to-centre and applies to the same recorded layout. |
| Planar base–ball distance | 334.08 ± 16.70 mm | 410.40 mm | Outside the bound if the same base reference and recorded layout are intended. The planar norm avoids assuming an axis rotation. |
| Bowl diameter | 114.3 ± 5.715 mm | 100 mm inner; about 104 mm outer | Neither interpretation agrees with the new central value/bound; inner versus outer reference is unspecified. |
| Mat length × width | 406.4 × 304.8 mm | task x/y: 305 × 406 mm | Physical sizes agree after exchanging axes; the operator frame and task frame are not registered. |

The reported camera-from-ball components are **[774.7, 114.3, 406.4] mm**. A
translation triple alone cannot determine camera orientation, intrinsics, signed
axes or which camera/reference point was measured. These components are retained
as measurements in the operator's declared frame rather than inserted into a
camera-to-task rigid transform.

### Clarification received during this run

The operator confirmed that **16 inches is above the tabletop**, and described
the ball as lying along positive horizontal components from the base. The verbatim
reply is preserved in
[`measurements.clarification_20261026.json`](../configs/v2/measurements.clarification_20261026.json).
The [clarified measurement revision](../configs/v2/measurements.operator_clarified_20261026.json)
preserves the original record/hash; it does not invent two orthogonal task axes or
camera horizontal signs. Applying the height to the global iPhone camera remains
an explicitly labelled camera-identity assumption.

The separate [camera-height candidate](../configs/v2/task.camera_height_20261026.yaml)
sets camera task Z to **0.4064 − 0.012 = 0.3944 m**, accounting for the existing
base-origin lift. The previous scene camera was **0.5334 m above the table**, a
127 mm difference. In the [rendered comparison](../results/v2/camera_height_only_20261026.json),
changing height alone while retaining the provisional orientation changes valid
ball/bowl detections from 5/5 each to 0/5 each. On seed 0 the ball projects near
the image's top edge, clipping its silhouette. This is evidence that the old
pose cannot be corrected by changing height alone; it is not a fit to real images.
The successful waypoint rollouts retain their original, explicitly provisional
camera configuration and do not claim to validate this new camera candidate.

Two additional tests verify the table/task height conversion under both existing
base-offset hypotheses while preserving arm kinematics and uncertainty bounds.

The useful next clarifications are the signed length/width directions relative to
the robot, the base point used for `(2,13)`, whether bowl diameter means the inner
opening or outer rim, and whether ball–bowl spacing is centre-to-centre. Jaw gap at
several readbacks, ball support height, and loaded hold/release timing still require
their own observations. These omissions do not block software or training fixes
that use the existing recorded images and correctly identified action semantics.

## Joint-limit discrepancy is concentrated in shoulder lift

The separate `joint_limit_diagnostic` in the comparison report scores the **23,372
follower readbacks in train and validation**; reserved episodes are excluded. Under
the current identity arm mapping, **7,699 rows (32.94%)** exceed at least one native
joint bound. **7,009 of those rows (91.04%) involve shoulder lift**, including 4,971
with no other violating joint. This identifies a specific registration problem to
resolve before interpreting controller rejections or fitting more policies.

| Joint | Observed degree range | Native MJCF degree range | Violating rows |
|---|---:|---:|---:|
| Shoulder lift | −105.143 to 57.846 | −100 to 100 | 7,009 (29.99%) |
| Elbow flex | −94.725 to 96.923 | −96.830 to 96.830 | 2,639 (11.29%) |
| Wrist flex | −86.462 to 98.242 | −95 to 95 | 89 (0.38%) |

The evidence supports **degrees**, rather than accidentally treating normalized
±100 values as degrees. The local and installed follower configurations both
default `use_degrees=True`; their configuration and motor-normalization source
hashes agree. The five arm motors use `DEGREES`, while the gripper uses `RANGE_0_100`.
Inverting the degree formula reconstructs integer encoder ticks to within
0.000043 ticks at p95; the alternative ±100 formula leaves roughly 0.4–0.49 ticks
of residual. Dataset metadata itself does not record `use_degrees` or the historical
calibration hash, so those current-source defaults alone would not be sufficient.

LeRobot computes `degrees=(tick-(range_min+range_max)/2)*360/4095`. Its degree path
uses the original readback rather than clipping it to those endpoints. The current
calibration endpoint coordinates span ±105.582° at shoulder lift, while the model
allows ±100°. These endpoint coordinates are not independently measured physical
stops. More importantly, the midpoint of an operator-recorded servo range is not
established to equal the CAD joint's neutral pose. Thus the readback/model mismatch
can arise from zero or stop registration even though the numerical units are right.

The actionable next calibration is the shoulder-lift neutral and lower stop,
followed by wrist/elbow checks. A known physical pose relative to a registered base
or independently identified joint/link landmarks can determine the correction.
Adding the saved motor homing offset again, changing units, clamping source poses,
or enlarging the model limits merely to accept the demonstrations would conceal
the discrepancy. No arm mapping or native bound was changed by this diagnostic.

Exact local sources:

- [Follower default](/Users/deveshvyas/Downloads/lerobot/src/lerobot/robots/so_follower/config_so_follower.py:42)
  and [motor modes](/Users/deveshvyas/Downloads/lerobot/src/lerobot/robots/so_follower/so_follower.py:71).
- [Degree normalization](/Users/deveshvyas/Downloads/lerobot/src/lerobot/motors/motors_bus.py:874)
  and [Feetech homing convention](/Users/deveshvyas/Downloads/lerobot/src/lerobot/motors/feetech/feetech.py:278).
- [Native lift limit](../assets/menagerie/robotstudio_so101/so101.xml#L121)
  and [existing normalization/zero audit](../results/v2/joint_normalization_provenance_20261021_v1.md).

## Local recordings checked

| Folder | Metadata | Actual payload |
|---|---|---|
| `/Users/deveshvyas/.cache/huggingface/lerobot/local/so101_act_pickplace_v2` | 41 episodes, 27,736 frames, 25 Hz | 8 data parquet files and 16 camera videos |
| `/Users/deveshvyas/.cache/huggingface/lerobot/local/so101_act_pickplace_v2_25fps` | 1 episode, 467 frames, 25 Hz | 1 data parquet file and 2 videos |
| `/Users/deveshvyas/.cache/huggingface/lerobot/bottowork/rollout_so101_dagger_act03_20261023_223237` | 0 episodes and 0 frames | No data/video payload |

`/Users/deveshvyas/Downloads/lerobot/traces` is empty. The LeRobot source folder
contains example test videos, but no further task MP4/parquet files were found in
the targeted search outside installed-package and Git directories. This inventory
does not establish that other recordings cannot exist elsewhere.

## Reproduce ingestion

Use fresh output filenames so source lineage remains reviewable:

```sh
.venv/bin/python scripts/import_v2_operator_measurements.py \
  --source physical_measurements.md \
  --measurements-out configs/v2/measurements.operator_NEXT.json \
  --task-out configs/v2/task.scalar_measured_NEXT.yaml \
  --audit-out results/v2/operator_measurements_NEXT.json
```

Ten focused tests check circumference conversion, signed offsets, uncertainty propagation,
explicitly unknown transforms/references, contradictory spacing, duplicate source
fields and invalid measurement definitions. They pass; Ruff passes for the importer
and tests. No physical calibration acceptance is claimed.
