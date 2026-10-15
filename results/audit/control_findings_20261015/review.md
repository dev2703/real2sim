# IK, recorded-grasp and motion audit — 15 October 2026

IK convergence is fixed in isolated opt-in runs. Grasp execution is still unresolved: accurate follower tracking does not reproduce the real grasps. Calibration/contact remains a confound, so ACT failure is not yet isolated to policy/data coverage.

The final code passed 422 unit/integration tests covering IK, hard acceptance, collision/workspace rejection, joint limits, policy contracts and both robots. No checkpoint was trained or modified.

## Frozen-policy control

Three matched seeds per model/robot, 300 ticks each: 24 rollouts, 12 per convergence setting. Historical baseline trajectories reproduced within 1 micrometre; checkpoint hashes matched. Numerical convergence is 10 micrometres / 0.0001 rad. Position acceptance remains 3 mm, orientation acceptance 0.35 rad (SO-101) / 0.05 rad (Panda), penetration tolerance 2 mm; workspace, actuator, velocity and action limits are unchanged.

| Policy / robot | Accepted unchanged arm, old → polished | Median requested / achieved movement, old → polished (mm/tick) | Median movement error, old → polished (mm) | Closest ball, old → polished (mm) |
|---|---:|---:|---:|---:|
| parent_so101 | 885 → 0 / 900 | 1.691 / 0.011 → 0.773 / 0.751 | 1.698 → 0.227 | 202.1 → 102.2 |
| pilot_so101 | 891 → 0 / 900 | 0.111 / 0.011 → 1.075 / 0.734 | 0.115 → 0.397 | 202.4 → 97.4 |
| parent_panda | 0 → 0 / 900 | 1.262 / 0.617 → 1.221 / 0.602 | 0.662 → 0.633 | 112.0 → 111.6 |
| pilot_panda | 144 → 0 / 900 | 0.578 / 0.265 → 0.625 / 0.288 | 0.352 → 0.319 | 146.9 → 142.0 |

All 12 polished trials still failed before a grasp. Policy inputs diverge once motion changes, so cross-run error changes are not a fixed-command comparison. Shadow IK solves on identical seeds/targets independently establish that formerly accepted no-op commands now move. Panda still achieves roughly half the requested per-tick displacement; tighter IK alone does not remove servo lag.

[Full frozen results and hashes](../ik_polishing_20261015/verified/review.json). Each run has raw, clamped and achieved six-dimensional deltas, solver residuals, rejections and joint-control traces.

## Complete recorded grasps

Episodes 1, 2 and 3 visually show a real grasp, carried ball, release and ball remaining in the bowl. Durations are 59.7, 34.4 and 45.6 seconds. Every recorded 10 Hz tick is replayed, including the portion beyond the old 30-second cutoff. These are diagnostic selections; the original unknown outcome annotations are unchanged.

Five dynamic controls per episode: next follower state, recorded Cartesian deltas with legacy/polished IK, bounded feedback to the next recorded absolute pose, and polished deltas with the original leader-gripper labels. The first four use the next recorded follower gripper to isolate achieved jaw movement. The fifth retains the existing policy-label gripper semantics.

| Episode | Follower closure / transport error (mm) | Cartesian old → polished closure / transport (mm) | Bounded reference feedback closure / transport (mm) |
|---|---:|---:|---:|
| 1 | 0.31 / 0.37 | 67.14 / 34.75 → 12.42 / 25.39 | 0.31 / 0.37 |
| 2 | 0.18 / 0.27 | 33.86 / 44.27 → 24.87 / 25.79 | 0.18 / 0.27 |
| 3 | 5.71 / 0.81 | 36.81 / 18.04 → 19.14 / 11.74 | 13.97 / 5.14 |

All 15 dynamic replays had zero two-jaw contact substeps and zero held lifts. Ball contact was inspected at every 500 Hz physics substep, so this result is not a 10 Hz sampling artifact. Reference feedback passes the existing IK gates on the good tracking segments; it does not bypass rejected poses. Whole-episode errors are larger: start/return references violate native joint limits and some recorded poses violate the collision/workspace gates.

| Episode | First unilateral contact | Tracking error at contact (mm) | Sim ball moves >10 mm | Real centroid moves >8 px |
|---|---|---:|---:|---:|
| 1 | 8.700s, fixed_jaw_box3 | 0.37 | 8.9s | 14.8s |
| 2 | 8.182s, moving_jaw_box2 | 0.67 | 8.5s | 12.1s |
| 3 | 9.582s, fixed_jaw_box3 | 0.39 | 9.7s | 13.6s |

The centroid criterion requires three consecutive frames and is a 2-D image-motion proxy, affected by jaw occlusion. It is not a measured real contact time. Nevertheless, the images and substep traces agree that the simulated ball is displaced before the recorded closure. The unchanged camera/joint/gripper mapping and contact model therefore need verification. The physical real jaw opening is not measured; reported gaps are the distance between the model’s two tip spheres.

Using leader-gripper targets instead of follower openings creates roughly 48–50 mm mean jaw-gap error during transport in episodes 1 and 2. That is an execution/load/geometry diagnostic, not a reason to relabel the dataset automatically.

The full videos show exact source frames, projected jaw tips, the recorded kinematic pose, follower dynamics and polished Cartesian dynamics. The reference pane hides the ball because its real 3-D path is unknown; it must not be read as a physical replay.

- Episode 1: [complete video](../recorded_grasp_control_20261015/verified/episode1_complete.mp4), [comparison sheet](../recorded_grasp_control_20261015/verified/episode1_comparison.jpg), [first simulated contact](episode1_first_contact.jpg).
- Episode 2: [complete video](../recorded_grasp_control_20261015/verified/episode2_complete.mp4), [comparison sheet](../recorded_grasp_control_20261015/verified/episode2_comparison.jpg), [first simulated contact](episode2_first_contact.jpg).
- Episode 3: [complete video](../recorded_grasp_control_20261015/verified/episode3_complete.mp4), [comparison sheet](../recorded_grasp_control_20261015/verified/episode3_comparison.jpg), [first simulated contact](episode3_first_contact.jpg).

[Full replay metrics, phase definitions and provenance](../recorded_grasp_control_20261015/verified/report.json). Every mode also has joint states, per-tick tracking/jaw traces and substep contacts.

## Active versus stationary predictions

4,055 valid actions from nine held-out episodes are each scored once using the actual two-action execution prefix. Padding and terminal extrapolated labels are excluded. The zero baseline sends zero arm motion and holds the gripper observed at inference, without future-state information. Active means >1 mm translation or >0.01 rad rotation per 100 ms.

| Stratum | Actions | Parent / pilot / zero translation error (mm) | Parent / pilot / zero normalized L1 |
|---|---:|---:|---:|
| arm_active | 2736 | 4.31 / 4.41 / 6.86 | 0.5742 / 0.5880 / 0.8052 |
| arm_stationary | 1319 | 0.75 / 0.54 / 0.10 | 0.1159 / 0.1037 / 0.0609 |
| reviewed_stationary_ball_approach_proxy | 1266 | 3.03 / 2.98 / 3.89 | 0.3874 / 0.3857 / 0.4414 |
| observed_gripper_closure_proxy | 174 | 3.79 / 3.84 / 7.23 | 0.5101 / 0.5158 / 0.8331 |
| upward_with_gripper_at_most_45pct_proxy | 1079 | 4.14 / 4.10 / 7.94 | 0.5574 / 0.5544 / 0.9158 |

Approach, closure and upward-motion rows above are kinematic proxies and can overlap. Additional errors for the reviewed approach/closure/transport/release windows of episodes 1–3 are in report.json, including rotation and gripper errors. Phase boundaries are coarse visual review, not ground-truth contact labels.

[Motion audit](../motion_prediction_20261015/review.md) and [full predictions](../motion_prediction_20261015/predictions.npz).

Both ACT models beat zero motion on active real data. The pilot is slightly worse during active movement while improving stationary error. This does not establish policy/data coverage as the cause of the failed grasps. The next discriminating result is a physically successful recorded reference replay under a verified jaw/ball registration, followed by a frozen-policy comparison in that same setup.

## Reproduce

```sh
.venv/bin/python scripts/diag_ik_polishing.py --out <new-ik-directory>
.venv/bin/python scripts/diag_recorded_grasp_control.py --out <new-replay-directory>
.venv/bin/python scripts/diag_motion_prediction_audit.py --out <new-motion-directory>
.venv/bin/python scripts/summarize_control_audit.py --ik <ik-directory> --replay <replay-directory> --motion <motion-directory> --out <new-summary-directory>
```

Rendering requires macOS graphics access. Each run refuses to overwrite an existing output directory and records checksums. Executed script snapshots are stored as text alongside the reports. The verified directories contain the final iteration-budget fix; earlier audit directories retain the preliminary runs.
