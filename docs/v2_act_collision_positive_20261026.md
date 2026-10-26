# ACT collision and image-domain isolation — 26 October 2026

The successful joint commands come from the geometric waypoint controller, not
the learned ACT. A further bounded diagnostic now isolates the ACT failure using
the same frozen step-2,000 history-four checkpoint and original provisional scene.
The [curated evidence](../results/v2/act_collision_positive_20261026.json) links
the full report, images and a new complete feedback trace. No reserved episodes,
training, contact changes or checkpoint edits enter this experiment.

## What the rejection actually means

Every rejected target from the previous three execution settings was replayed
through the original native IK. All **326 / 319 / 329** collision rejections are
the **fixed gripper against the table**, with candidate penetrations around
2.0–2.6 mm, beyond the unchanged 2 mm gate. They are not base self-collisions.
The running arm stays below the gate because rejected targets are never applied.
These probes depend on the provisional scene and do not establish real collisions.

The policy descends near the base while making little progress in the lateral
direction of the ball. It starts near `[191, 6, 40]` mm; the ball is near
`[353, 228, 21]` mm. At the first rejection, more than 240 mm of horizontal error
remains. Maximum requested feedback over the full runs is only 3.0–3.8 mm, so
this is not large, unbounded accepted-target accumulation.

## Reference-mode ablation

A new full 375-tick rollout explicitly replaces `accepted_target` with
`current_pose` at runtime. The checkpoint, saved contract, initial joints,
seeded objects, action bounds and native contact/IK gates remain unchanged. This
is a diagnostic execution-semantics ablation, not a promoted checkpoint contract.

| Same ACT / seed 0 | Collision rejections | Closest tool–ball distance | Held / lifted / placed |
| --- | ---: | ---: | --- |
| Accepted target, queue two | 326 | 250.4 mm | No / no / no |
| Current pose, queue two | 0 | 227.6 mm | No / no / no |

The current-pose variant accepts all 375 commands but still approaches the table
near the base rather than the ball. The persistent reference aggravates the
rejection stall; removing the rejection alone does not solve the learned approach.
This does not support changing the default reference on all tasks: the existing
stationary-hold evidence for accepted-target integration remains a separate result.

## Image-only intervention

Episode 9 frames 100–198 supply 50 development observations. Their exact recorded
60-feature causal histories are held fixed while the camera image is replaced by
either its recorded image or the simulator render at the same native joint state.
No joint reference is clipped; no rows in this interval require exclusion.
Only the first supervised action is scored per observation.

| Input image, same recorded history | Mean translation error | p95 | Positive direction on 47 moving rows |
| --- | ---: | ---: | ---: |
| Recorded | 3.369 mm | 6.475 mm | 91.49% |
| Simulated | 2.838 mm | 6.494 mm | 91.49% |

Thus this particular paired sample does **not** support attributing the failure
to rendered appearance alone. It does not prove that the camera/domain gap is
small globally: the sample is one development episode, rows are correlated, and
the simulator's object layout/optics are still provisional. Teacher-forced action
error does not measure accumulated feedback error or complete task success.

The next useful learning experiment is to train or distil **object-relative
approach/grasp/transport/release goals** using the executable staged controller as
an explicit source of simulator demonstrations. It directly tests goal-directed
behavior and training/execution consistency. Any ACT plus geometric correction
must retain separate correction counts and success attribution; geometric success
must not be described as frozen ACT transfer.

Reproduce with fresh outputs:

```sh
.venv/bin/python scripts/diag_v2_act_collision_positive_20261026.py \
  --out outputs/v2/act_collision_positive_NEW
```

The script preserves full per-tick requested, accepted/applied and achieved
evidence. Source and checkpoint hashes are identical before and after the run.
Ruff passes. No controller fix is installed from this one-seed diagnostic.
