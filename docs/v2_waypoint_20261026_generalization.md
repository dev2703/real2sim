# Fresh layouts for the fixed RGB waypoint controller — 26 October 2026

The unchanged geometric controller completes **8/10 new SO-101 and 8/10 new
Panda pick-and-place trials** on seeds 10–19. Repeating those ball layouts with
declared **30 mm bowl shifts** again gives 8/10 on each arm. These are 40 new
development rollouts, separate from the earlier seeds 0–4. They do not train,
execute or transfer ACT weights.

The [machine-readable comparison](../results/v2/waypoint_20261026_generalization.json)
links all four full reports, source hashes, scenarios, per-tick requested/applied
joint commands and readbacks, and videos. No failed seed was replaced or excluded.

| Fixed controller / layout | RGB targets acquired | Grasp planned | Held lift | Released resting placement |
| --- | ---: | ---: | ---: | ---: |
| SO-101 / new ball seeds, original bowl | 10/10 | 8/10 | 8/10 | 8/10 |
| SO-101 / same balls, shifted bowl | 10/10 | 8/10 | 8/10 | 8/10 |
| Panda / new ball seeds, original bowl | 8/10 | 8/10 | 8/10 | 8/10 |
| Panda / same balls, shifted bowl | 8/10 | 8/10 | 8/10 | 8/10 |

SO-101 seeds 10 and 17 fail because its existing sampled branch search finds no
grasp satisfying native IK/path gates. This does not prove those physical grasps
are impossible. Panda seeds 14 and 16 fail because the initial RGB targets remain
unavailable. Every trial that plans a grasp in these runs subsequently holds,
lifts, transports, releases and rests the ball in the bowl. Failure trials retain
their full timeout, and all abort/rejection/saturation counts remain visible.

## What changed in the experiment

The first set changes only the seeded ball position within the existing 30 mm
start disc. The paired second set preserves those exact ball positions and
initial robot joint readbacks, and changes bowl position by the repeating
sequence `[-30,0]`, `[+30,0]`, `[0,-30]`, `[0,+30]` mm. This sequence was declared
before any shifted trial was run. The maintained controller, grasp offsets,
camera configurations, native limits, servo/contact settings and success scorer
are unchanged throughout all four runs.

The environment updates its static bowl collision map to each declared layout.
The controller receives that supplied map for collision checking; its bowl
**goal** still comes from RGB. In the paired rendered views, detected bowl shifts
match the declared displacements within **0.27–0.65 mm**. That number measures
consistency with exact simulated camera calibration, not real camera accuracy.

The scorer remains the existing held lift above 30 mm, release over the bowl,
then one second of unheld resting placement. Maximum robot/table penetration is
0.530 mm on SO-101 and zero on Panda, below the unchanged 2 mm limit. Both robot
models retain their configured start without clipping, settling or replacement.
SO-101 runs at 12.5 Hz and Panda at its existing 10 Hz; timeouts and phase durations
use each environment's clock. Source hashes remain unchanged during every run.

## What crossed between the embodiments

The shared part is the task strategy:

`RGB ball/bowl goals → approach → descend → close → lift → transport → release`.

Each robot independently converts task goals into its own feasible native joint
targets. SO-101 uses five arm joints, a fixed-jaw-tip tool reference and flexible
carry orientation. Panda uses seven arm joints, its between-pad TCP, a top-down
grasp and the previously fixed 8 mm seating offset. IK selects branch/phase
targets; bounded joint execution uses readback feedback and FK collision checks.
SO-101 joint targets are never copied into Panda's joint vector, and no learned
weights are used. This demonstrates reuse of a task strategy with explicit robot
adapters, rather than frozen learned-policy transfer.

The result establishes that the task is feasible in the supplied simulator and
that bowl-directed geometric behavior follows small new layouts. It leaves
camera/base physical registration, jaw loading, recovery, broader starts and
unassisted ACT behavior unresolved. The visible next geometric bottlenecks are
SO-101 branch search and Panda initial occlusion; changing friction is not
supported by these failures.

## Reproduce

Use fresh output directories:

```sh
.venv/bin/python scripts/run_visual_waypoint.py --robot so101 \
  --seeds 10 11 12 13 14 15 16 17 18 19 --video \
  --out outputs/v2/so101_fresh_NEW
.venv/bin/python scripts/run_visual_waypoint.py --robot panda \
  --seeds 10 11 12 13 14 15 16 17 18 19 --video \
  --out outputs/v2/panda_fresh_NEW
.venv/bin/python scripts/diag_v2_waypoint_generalization.py --robot so101 \
  --video --out outputs/v2/so101_bowl_shift_NEW
.venv/bin/python scripts/diag_v2_waypoint_generalization.py --robot panda \
  --video --out outputs/v2/panda_bowl_shift_NEW
```

The wrapper's scenario schedule, identical paired ball/readback assertions,
source/checkpoint absence and unchanged scoring were verified against the actual
saved trials. Ruff passes. Successful shifted seed-11 videos:
[SO-101](../outputs/v2/rgb_waypoint_so101_20261026_generalization_bowl/seed_011/rollout.mp4),
[Panda](../outputs/v2/rgb_waypoint_panda_20261026_generalization_bowl/seed_011/rollout.mp4).
