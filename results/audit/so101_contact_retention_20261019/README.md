# Jaw retention diagnostic, 19 October 2026

**The candidate scene loses all three balls during fixed arm-and-gripper target holds. Transport motion and Cartesian drift are therefore not necessary for this simulated contact failure.** This narrows the unresolved mechanism to retention/contact creep under the scene's current geometry, mapping, actuation and contact model. It does not independently identify which of those physical parameters is wrong.

Runner: `scripts/diag_so101_contact_retention.py`. Compact evidence and source/trace hashes: [report.json](report.json). Full traces and executed source: `outputs/so101_contact_retention_20261019/`. Earlier report-only versions remain in sibling `_initial` and `_summary_v1` directories. The final run adds explicit command/opening and jaw-tip gap measurements; physics and reproduced trajectories are identical.

## Matched experiment

Replay the same recorded follower arm targets and original recorded gripper command for episodes1/2/3 with the existing common candidate mapping and +5mm ball X offset. Repeat each episode, freezing the last bounded native arm and gripper target once five consecutive10Hz samples show bilateral jaw contact and >30mm ball lift. Hold20s. The rule and duration are common to all episodes. Ball dynamics, geometry, contact parameters, actuators, velocity and joint bounds, collision checks and official success definitions remain unchanged. No policy runs or real hardware actions occur.

Each ordinary replay reproduces every prior recorded-reference ball position exactly (maximum difference0). Each frozen trial has an identical prefix to its corresponding ordinary replay (maximum difference0). All initial scenes pass the existing contact check. Samples occur at10Hz after the environment's existing `mj_forward`; loss times have0.1s resolution.

| Episode | Hold starts, source s | Loses hold after s | Ball drift before bilateral loss, mm | Maximum tool reference error during hold, mm | Hold clipping / rate limiting |
|---|---:|---:|---:|---:|---:|
|1|17.6|12.4|29.16|0.504|0 /0|
|2|18.2|11.0|26.93|0.336|0 /0|
|3|17.2|14.7|24.84|0.302|0 /0|

Episode2 loses bilateral contact at10.9s and loses the existing broader `ball_held` condition at11.0s. This distinction is preserved in the report. The ordinary episode2 and3 placements still succeed; both fail when asked to retain the ball longer. The hold experiment is not a placement-success test. After escape a ball can roll off the finite table, so reported diagnostic displacement stops at the last bilateral sample; the full trace retains the subsequent motion.

## Observed contact mechanism

In ordinary episode1, the gripper command remains closed at0.515% near the29.2s drop while recorded real opening stays36.07%. Simulated opening shrinks17.44→11.83% from27.5→29.1s, then collapses after the ball escapes. At17.6s the ball is at tool X0.74mm; by29.1s it is at28.38mm. The force-weighted angle between opposing jaw contact normals falls156.7°→90.2° (180° would be directly opposed). Immediately before escape both jaws still carry about24.6/24.8N normal load. Maximum sampled surface slip rises to11.9mm/s; some sliding-friction utilization values approach1. These quantities describe the simulated contact manifold, not calibrated real contact forces or a complete grasp-wrench certificate.

The frozen episode1 repeats the same progression: about29mm relative creep, progressively less opposed contacts, and escape while roughly25N per jaw remains just before loss. This rules out insufficient arm tracking and an opening command as necessary causes in this simulated case. It supports outward migration through a changing jaw contact configuration. Collision shape, gripper mapping/zero, actuator stiffness, friction and soft-contact regularization remain confounded. Increasing grip force or friction without measurement would obscure that distinction.

## Calibration discrepancy and next measurement

At the common hold anchors:

| Episode | Recorded real opening, % | Simulated loaded opening, % | Candidate tip gap from real state, mm | Simulated loaded tip gap, mm |
|---|---:|---:|---:|---:|
|1|36.07|27.14|78.31|65.78|
|2|35.33|26.62|77.28|65.03|
|3|36.89|28.13|79.42|67.19|

The model ball diameter is67mm. Tip gap is the distance between the two named tip-sphere centers minus their radii; it is **not** a measured physical inner-pad separation or a guarantee of contact at the sphere equator. The roughly12mm discrepancy motivates independent jaw calibration; it does not by itself prove an incorrect encoder zero.

1. Measure unloaded inner-jaw separation and tip/pad landmarks at several ordinary follower encoder openings, including empty closure. Fit encoder percent→hinge mapping/zero from those measurements, with no ball contact, success-selected offsets or episode-specific fit.
2. Measure actual ball diameter and jaw/pad geometry, including thickness. With that unloaded mapping frozen, compare loaded contact depth and pad normals in two calibrated views. A single held-ball image cannot distinguish zero, off-center contact depth, pad compliance and camera registration.
3. Collect the planned traced matched empty/loaded holds so requested, bounded/sent and measured gripper positions and actual timing are identifiable. Original recorded command provenance remains unverified.
4. Re-run this unchanged retention probe and held-out recorded grasps only after independent measurements justify a scene correction. Current exploratory geometry and these historical development episodes cannot validate real calibration or learned-policy success.

Validation: four integration tests cover contact-frame sign under geometry order, consecutive hold selection, table-contact exclusion and nonmutation, plus physical force direction and rolling-versus-sliding velocity. Ruff passes. The force check uses a simple ball-on-support scene independent of the production jaw geometry.
