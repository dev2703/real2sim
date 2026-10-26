# Positive camera signs and total-distance setup — 27 October 2026

The operator's clarification, **“12 inches is total distance, not an x-only
offset,”** supersedes the earlier +x-only bowl hypothesis. Camera-minus-ball
horizontal signs remain positive as requested. The previous 485 mm bowl/base
radius and reach result apply only to that discarded hypothesis; they do not
establish that the real bowl is out of reach.

The revised candidate keeps the measured **304.8 mm** ball/bowl distance and
uses the previous image-derived relative layout as a **bearing prior**. The
old base/ball bearing is rotated by 51.334° to agree with the new (+2,+13)-inch
base/ball convention; the old bowl direction receives that same rotation and
is scaled to the measured total distance. This gives:

| Quantity | Candidate in task metres |
| --- | --- |
| Ball XY | [0.0508, 0.3302] |
| Bowl XY | [0.256311, 0.105104] |
| Camera XYZ | [0.8255, 0.4445, 0.3944] |
| Camera height above table | 0.4064 |
| Ball–bowl planar distance | 0.3048 |

The bowl/base radius is now 277 mm. The camera task Z accounts for the existing
12 mm base-origin lift. Base/task axis alignment, the source layout yaw, the
centre-to-centre interpretation, mat midpoint placement, old intrinsics and
roll-free camera look-at remain declared assumptions. This is a reproducible
setup candidate, not an independently accepted physical calibration.

## Actual unchanged-controller result

Five common seeds were executed on each arm using the existing RGB waypoint
controller, original contact/actuator settings and unchanged success scoring.
Every trial retains the full timeout and its failure.

| Robot | RGB ball acquired | RGB bowl acquired | Held lift | Pick-and-place |
| --- | ---: | ---: | ---: | ---: |
| SO-101 | 5/5 | 0/5 | 0/5 | 0/5 |
| Panda | 2/5 | 0/5 | 0/5 | 0/5 |

All trials fail during initial perception; no grasp or transport planning is
attempted. Both full reports have unchanged runtime source hashes, no aborts,
no executed-command IK rejections and no robot/table penetration. These results
are separate from the earlier 4/5 and 8/10 successes under the old provisional
view. They do not train, evaluate or transfer ACT weights.

The bowl is visible in the initial view. A scoring-only seed-0 contour
inspection identifies the specific fixed frontend gates:

* SO-101: the grey silhouette containing the projected bowl centre merges with
  nearby robot pixels. Its convexity is **0.825**, below the fixed 0.90 gate.
* Panda: the bowl silhouette backprojected to the rim plane has **5.27 mm**
  circle residual versus the fixed **4.16 mm** gate, and axis ratio **0.700**
  versus 0.75. The visible sidewall and changed perspective challenge the
  frontend's planar-circle assumption.

No contour thresholds or simulator physics were tuned after these failures.
The exact simulator pose used to score projection does not establish a matching
physical camera pose. A robust rim/occlusion frontend or a validated unobstructed
view is the next geometric intervention; the failed detector must not silently
fall back to a configured bowl centre.

The [curated evidence](../results/v2/positive_total_distance_20261027.json) links
full reports, every trace, videos and the [contour diagnostic](../results/v2/positive_total_distance_projection_20261027.json).
Initial views:
[SO-101](../outputs/v2/rgb_so101_positive_total_distance_20261027/seed_000/initial.png),
[Panda](../outputs/v2/rgb_panda_positive_total_distance_20261027/seed_000/initial.png).

## Loaded-failure handling remains unverified

The prior opt-in failure-hold experiment retained the last closed command after
a transport failure in the discarded +x-only bowl scene. **Zero of five** balls
remained held at the final 30-second tick. Avoiding an intentional open command
does not establish sustained loaded retention or recovery success. That variant
is not promoted by this calibration revision.

## Reproduction

```sh
.venv/bin/python scripts/prepare_v2_positive_total_distance_20261027.py \
  --out configs/v2/positive_total_distance_NEXT
.venv/bin/python scripts/run_visual_waypoint.py --robot so101 \
  --config configs/v2/positive_total_distance_NEXT/sim_so101.yaml \
  --seeds 0 1 2 3 4 --video --out outputs/v2/so101_total_distance_NEXT
.venv/bin/python scripts/run_visual_waypoint.py --robot panda \
  --config configs/v2/positive_total_distance_NEXT/sim_panda.yaml \
  --seeds 0 1 2 3 4 --video --out outputs/v2/panda_total_distance_NEXT
```

Four focused tests pass for exact total-distance geometry, bearing consistency,
positive offsets/table datum, preserved robot mapping/source data and degenerate
direction handling. Ruff passes. Historical source/config/report files remain
unchanged.
