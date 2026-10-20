# V2 joint normalization and zero provenance, 21 October 2026

The five recorded arm coordinates are strongly consistent with **degrees**, not a normalized ±100 range. The unresolved part is their physical zero relative to the simulator. A separate, concrete discrepancy is the gripper scale: the current simulation maps a 100% opening change to 110.002°, while the matching local follower calibration implies 129.407° of servo rotation, approximately 15% more compression in simulation. No calibration or physics configuration was changed by this audit.

Machine-readable evidence, complete numeric checks, calibration comparisons and source/data hashes: [joint_normalization_provenance_20261021_v1.json](joint_normalization_provenance_20261021_v1.json). The audit used all 27,736 rows in 41 v2 episodes, independently for measured state and commanded action.

## What the source actually computes

Both the local recording checkout and the installed LeRobot package default `use_degrees=True`. Their motor normalization source files have the same SHA-256. The follower sets the five arm motors to `DEGREES` and the gripper to `RANGE_0_100`.

For each arm motor, with reported encoder value `t`, calibration endpoints `lo, hi` and resolution 4096:

```
m = (lo + hi) / 2
degrees = (t - m) * 360 / 4095
t = degrees * 4095 / 360 + m
```

The degree path uses the original reported value and does not clip it to the calibration endpoints. For this follower all stored drive modes are zero. Gripper normalization is:

```
percent = 100 * (clip(t, lo, hi) - lo) / (hi - lo)
```

Sources: `../lerobot/src/lerobot/robots/so_follower/config_so_follower.py:42`; `so_follower.py:71–80`; `../lerobot/src/lerobot/motors/motors_bus.py:854–907`.

| Measured v2 joint | Degree-inverse encoder residual, p95 ticks | Alternative ±100 inverse, p95 ticks |
|---|---:|---:|
| shoulder_pan | 0.0000093 | 0.474 |
| shoulder_lift | 0.0000429 | 0.487 |
| elbow_flex | 0.0000353 | 0.464 |
| wrist_flex | 0.0000324 | 0.466 |
| wrist_roll | 0.0000046 | 0.400 |
| gripper, percent inverse | 0.0000269 | Not applicable |

Residual means distance from the reconstructed encoder tick to the nearest integer. The tiny degree residual is consistent with float32 storage of integer encoder readings. This check does **not** uniquely identify the calibration file: shifting a midpoint by any integer number of ticks leaves its lattice residual unchanged. The current gripper span matches observed percentages much better than the available follower backups, which strengthens but does not prove recording-time provenance.

The dataset metadata contains feature order, nominal 25 FPS and `robot_type=so_follower`; it does not pin the recording-time use_degrees setting, collector commit, calibration hash or physical zero. The current recorder stores the returned sent command after its relative-target limiter (`../lerobot/src/lerobot/scripts/lerobot_record.py:389–399`, `so_follower.py:243–254`). Present source cannot retrospectively prove that historical recordings used identical semantics. Commanded actions are not readbacks and need not fit the follower readback lattice.

## Why the arm offset is not recoverable from this JSON alone

Feetech's implemented convention is `reported_tick = physical_encoder_tick - homing_offset`. The calibration routine sets the operator's approximate homing pose to tick `int(4095 / 2) = 2047`, then records range endpoints. Degree zero is subsequently the **range midpoint**, not necessarily that homing pose. Source: `../lerobot/src/lerobot/motors/feetech/feetech.py:278–287`; `motors_bus.py:775–797`; `so_follower.py:153–176`.

For a known physical zero encoder tick `t_zero`, a physical joint angle would have the form:

```
q = sign * (reported_tick + homing_offset - t_zero) * 2*pi/4095
```

`t_zero` relative to the MJCF joint frame is not stored in the calibration JSON. Adding the saved homing offset directly to normalized degrees is therefore not a justified correction: the offset has already affected the reported encoder position. Nor is subtracting the range midpoint again justified. The homing position is operator-aligned, and there are multiple historical calibration files with different endpoints and offsets.

The repository currently uses sign +1 and zero offset for all five arm joints (`src/real2sim/geometry/so101_fk.py:52–54`, `configs/task.yaml:57`). These remain provisional. Joint range feasibility can bound candidate offsets, but choosing offsets to eliminate collisions alone would not establish physical agreement. Independently registered image landmarks or physical pose measurements are required for selecting and verifying them.

## Gripper scale candidate worth testing first

The current local follower file is `~/.cache/huggingface/lerobot/calibration/robots/so_follower/aligned_follower.json`, SHA-256 `f20e3a4990d13f2e6e0670df6625a5ca199d2823380be353b70bfff729e419d3`. Its gripper endpoints are 2026 and 3498:

```
129.406593° per 100% = (3498 - 2026) * 360 / 4095
1.29406593° servo rotation per percent
```

Current simulation mapping (`configs/task.yaml:58`, `so101_fk.py:56–58`):

```
q_sim = -0.1745 + (percent / 100) * 1.9199 radians
1.10002167° per percent
```

The MJCF gripper limit is approximately **−10° to +100°**, not −10° to +110° (`assets/menagerie/robotstudio_so101/so101.xml:222`). The latter endpoint is a typo in the older prose calibration document; it is not the implemented upper limit.

At 30–40% opening, the two slopes differ by 5.82–7.76° if they share the same closed zero. That is large enough to justify a jaw geometry/scale diagnostic before friction tuning. It does not yet prove a real jaw-angle error: verify recording calibration identity, motor-to-jaw ratio/sign, physical closure and unloaded jaw separation at five openings, reserving openings for validation. Keep actual joint bounds; a slope correction is not permission to expand limits or claim that 100% is physically reachable in the MJCF.

The existing calibration report's full-span endpoint interpolation was a provisional assumption. This audit gives a better-supported **relative motor scale candidate**, while closure and physical jaw geometry remain unmeasured. It has not been promoted to training, replay or evaluation configuration.
