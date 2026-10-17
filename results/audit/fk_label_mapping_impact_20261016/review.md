# FK label impact of diagnostic joint mapping

All recorded episodes; unchanged follower joint samples, leader gripper targets and tool site. No candidate is adopted by this audit.

| Variant | Mean tool shift (mm) | Mean / p95 translation label change (mm) | Mean / p95 rotation label change (rad) | Active partition changes |
|---|---:|---:|---:|---:|
| wrist_roll_exact_minus90 | 16.9734 | 0.2719 / 0.8731 | 0.000000 / 0.000000 | 13 |
| wrist_roll_fitted | 17.1631 | 0.2765 / 0.8866 | 0.000000 / 0.000000 | 13 |
| gripper_only | 0.0000 | 0.0000 / 0.0000 | 0.000000 / 0.000000 | 0 |
| first_four_arm_zeros | 35.0477 | 0.5384 / 1.7054 | 0.001485 / 0.005025 | 20 |
| full_candidate | 20.0545 | 0.3449 / 1.0508 | 0.001485 / 0.005025 | 11 |

For a roll-only zero correction, T_new = T_old X with constant X. Thus R_new = R_old C and p_new = p_old + R_old r. The world-frame rotation increment is unchanged, but the new translation increment is dp_old + (R_next - R_current) r. The tool site is about12mm off the roll axis, so r is nonzero.

A gripper affine map changes native squeezing/opening commands and range clipping, but the current tool site is independent of gripper angle. Percent-valued gripper observations and targets retain their semantics; no arm FK labels change from that map alone.

Existing architecture/proprioception runs remain valid comparisons of the original provisional data contract, not evidence for the corrected physical contract. Before final training on an adopted arm/roll calibration, regenerate canonical pose, velocity, Cartesian targets, train-only normalizers and dependent observation variants together. Raw joints, images, splits, causal filter logic and frozen DINO image cache can be reused. The new task-specific normalizers must accompany regenerated states/actions.

If only wrist roll is adopted, an explicit fixed-tool-transform compatibility adapter can map calibrated current poses to the old virtual tool frame and transform requested deltas back exactly. That is an additional controller experiment requiring matching velocity/history transformations and collision checks; applying old Cartesian predictions unchanged is not equivalent. Full first-four-joint zero changes generally cannot use one constant tool transform.
