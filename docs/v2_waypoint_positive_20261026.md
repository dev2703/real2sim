# Positive setup convention: actual rollouts — 26 October 2026

**Superseded bowl-direction hypothesis:** the operator subsequently clarified
that 12 inches is the total ball/bowl distance, not an x-only offset. The +x-only
layout below remains historical diagnostic evidence. Use the
[27 October total-distance candidate](v2_positive_total_distance_20261027.md)
for the revised geometry; these reach failures do not establish real bowl reach.

The controller was rerun unchanged on the newly assigned positive setup candidate,
with seeds 0–4 on both arms. **SO-101 picks up and lifts the ball in all five
trials, but places it in zero; Panda places it in two of five.** These results
must remain separate from the earlier 4/5 placements and the fresh-layout 8/10
results under the previous provisional scene.

The [curated report](../results/v2/waypoint_positive_setup_20261026.json) links
all requested/applied/readback traces, stage logs, initial/final images and videos.
Both full reports have unchanged source hashes, zero aborts, and unchanged native
joint, velocity, workspace, contact and scoring gates. No controller parameter
was tuned after inspecting these outcomes.

| Positive setup / robot | RGB targets acquired | Grasp planned | Held lift | Released resting placement |
| --- | ---: | ---: | ---: | ---: |
| SO-101 | 5/5 | 5/5 | 5/5 | 0/5 |
| Panda | 2/5 | 2/5 | 2/5 | 2/5 |

SO-101 fails in transport in every trial. Its existing carry optimizer leaves
**25.9 mm** ball-target error for the assumed bowl position, exceeding the
unchanged 3 mm waypoint tolerance. The bowl at `[0.3556, 0.3302]` m has planar
radius **485.3 mm** from the robot origin, compared with **334.1 mm** for the
nominal ball. This is a concrete reach/planning failure for this model and grasp
strategy; it does not prove that no alternative physical grasp/path exists.

The failure state currently opens the gripper. Consequently all five SO-101
trials release away from the bowl and eventually lose the ball beyond the table
during the retained timeout. That behavior is visible in the results and is not
counted as successful release. A future failure handler should preserve the last
closed gripper command after a carry-planning failure; that would prevent an
intentional drop but would not make the bowl reachable.

Panda succeeds on seeds 1 and 3. Seeds 0, 2 and 4 never acquire the ball contour
within the existing acquisition interval; the bowl contour is available. It
therefore fails before grasp planning. Maximum robot/table penetration is
0.477 mm for SO-101 and zero for Panda. No physics changes were used to improve
retention or redefine success.

## What was assigned versus measured

The [positive convention](../configs/v2/positive_setup_20261026/positive_convention.json)
sets camera-minus-ball horizontal offsets to `[+0.7747,+0.1143]` m, as requested,
and camera height to the confirmed 0.4064 m above the tabletop. The resulting
camera XY is `[0.8255,0.4445]` m under the working base/task-axis convention.
The reported ball/base components become `[+0.0508,+0.3302]` m.

The bowl's `+x` bearing at 12 inches from the ball follows the original file's
single-number/x-axis rule. Camera look-at, mat placement and task/operator axis
alignment are explicit working assumptions. Their values were not established
by independent camera/base measurements. Thus these trials test a reviewable
setup hypothesis, not an accepted physical calibration. The transport failure
particularly motivates checking the bowl bearing instead of changing learning
rates or relaxing collision/position thresholds.

## Mechanism and scope

The shared RGB waypoint state machine is the same on both robots. Each robot
plans its own native joint targets from the detected ball/bowl goals using its
kinematics, tool reference and existing grasp offset. No SO-101 joint vector or
ACT checkpoint was copied into Panda. This remains geometric strategy reuse
with robot adapters, not transfer of frozen learned weights.

```sh
.venv/bin/python scripts/run_visual_waypoint.py --robot so101 \
  --config configs/v2/positive_setup_20261026/sim_so101.yaml \
  --seeds 0 1 2 3 4 --video --out outputs/v2/so101_positive_NEW
.venv/bin/python scripts/run_visual_waypoint.py --robot panda \
  --config configs/v2/positive_setup_20261026/sim_panda.yaml \
  --seeds 0 1 2 3 4 --video --out outputs/v2/panda_positive_NEW
```

Example videos: [SO-101 held lift then transport failure](../outputs/v2/rgb_waypoint_so101_20261026_positive/seed_000/rollout.mp4),
[Panda successful placement](../outputs/v2/rgb_waypoint_panda_20261026_positive/seed_001/rollout.mp4).
