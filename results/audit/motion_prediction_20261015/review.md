# Frozen ACT motion audit

Held-out real images; every valid action scored once using the executed two-action prefix.

| Recorded-motion stratum | Actions | Parent L1 | DINO L1 | Zero-arm/hold-gripper L1 | Parent / DINO / zero translation error (mm) |
|---|---:|---:|---:|---:|---:|
| all | 4055 | 0.4251 | 0.4305 | 0.5631 | 3.15 / 3.15 / 4.66 |
| arm_active | 2736 | 0.5742 | 0.5880 | 0.8052 | 4.31 / 4.41 / 6.86 |
| arm_stationary | 1319 | 0.1159 | 0.1037 | 0.0609 | 0.75 / 0.54 / 0.10 |
| translation_above_3mm | 2206 | 0.6252 | 0.6394 | 0.9114 | 4.83 / 4.95 / 8.05 |
| near_zero_arm | 1037 | 0.1011 | 0.0866 | 0.0396 | 0.65 / 0.42 / 0.00 |
| reviewed_stationary_ball_approach_proxy | 1266 | 0.3874 | 0.3857 | 0.4414 | 3.03 / 2.98 / 3.89 |
| observed_gripper_closure_proxy | 174 | 0.5101 | 0.5158 | 0.8331 | 3.79 / 3.84 / 7.23 |
| upward_with_gripper_at_most_45pct_proxy | 1079 | 0.5574 | 0.5544 | 0.9158 | 4.14 / 4.10 / 7.94 |

L1 is normalized across seven action dimensions; translation error is the mean Euclidean error in raw metres, shown above in mm. These executed-prefix metrics differ from the existing overlapping full-20-action validation.

The zero baseline holds the gripper observed at inference for both actions; it does not access future states. Active means >1 mm translation or >0.01 rad rotation per 100 ms. Closure and upward-motion proxies do not prove contact or a successful lift. Full definitions, per-episode results, raw per-dimension errors and checkpoint hashes are in report.json; predictions.npz allows independent reanalysis.
