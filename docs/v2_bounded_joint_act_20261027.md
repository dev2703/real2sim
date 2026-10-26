# ACT command bounds and actual placement — 27 October 2026

The simulation-trained ACT now performs complete, unassisted SO-101 pick-and-place:
**3/5** original development layouts and **4/10** additional layouts. This uses
frozen learned commands, explicit physical gripper saturation and the original
arm/path/workspace/collision checks. No teacher goals or corrective IK generate
motion during these evaluations. This is a new simulation-teacher ACT checkpoint,
separate from the user's longer real-data Hugging Face checkpoint.

The [curated comparison](../results/v2/bounded_joint_act_20261027.json) links all
failed and successful traces, checkpoint hashes, and videos.

| Frozen checkpoint / execution | Held lift | Complete placement |
| --- | ---: | ---: |
| Original whole-command rejection, seeds 300–304 | 3/5 | 0/5 |
| Gripper actuator saturation, same seeds | 3/5 | 3/5 |
| Same saturation, additional seeds 310–319 | 4/10 | 4/10 |
| Training-seed diagnostic 100, corrected execution | 1/1 | 0/1 |

## The executable-command defect

The original guard rejects an entire six-joint request when any joint is out of
range. During closure ACT sometimes predicts a gripper value just below 0%,
such as −0.7%, despite otherwise valid arm targets. Rejecting that request holds
the arm and gripper, producing hundreds of rejected transport commands per trial.

`BoundedGripperJointActionAdapter` explicitly saturates only the gripper to the
intersection of its existing 0–100% mapping and native actuator interval. It then
checks the complete proposed command using the original arm, workspace, collision
and sampled path guards. It logs the raw prediction, bounded proposal, saturation,
accepted/applied command and achieved readbacks. Arm requests are never clipped
into validity and training labels/weights remain unchanged.

The three successful original-development trials require only 9, 9 and 10
gripper-saturated ticks. Once those commands execute, subsequent observations
allow ACT to continue transport and release. The rejected conditions originally
produced 312–317 native-limit failures each. This is a concrete feedback effect,
not a lower offline-loss claim.

## Training and limitations

The checkpoint was trained for 2,000 updates/32,000 examples on 15 successful
simulated SO-101 teacher episodes, with five successful validation episodes.
Failed teacher trials remain reported and are excluded from imitation labels.
Inputs are measured six-joint state plus RGB; targets are accepted absolute
servo commands at the same control interval. Labels are distinct from achieved
successor readbacks. The frozen ResNet/small ACT transformer runs at 12.5 Hz.

The ten additional seeds were evaluated without changing the correction. They
show incomplete generalization: four grasp/approach trials still hit configured
collision gates, and two miss the grasp without a rejection stall. The training
seed that formerly succeeded after a rejection-induced pause now fails release.
Thus saturation repairs a command-interface defect; it is not a general cure for
grasp timing, coverage or loaded retention. All failures remain in the denominator.

The successful runs use the original provisional scene with measured 32.34 mm
ball radius. They do not establish the newly assigned positive camera pose or
physical registration. The user-provided real-data HF model and Panda transfer
are scored separately. No reserved evaluation or physical robot success is claimed.

Four focused native-MuJoCo regressions verify that overclosed gripper predictions
can execute valid arm movement, impossible arm targets remain rejected, unchanged
limits are preserved, malformed predictions fail, and no task correction is used.
Dataset timing/normalization tests and Ruff pass.

Reproduce the original five-layout comparison with fresh output paths:

```sh
.venv/bin/python scripts/evaluate_v2_bounded_joint_act.py \
  --out outputs/v2/bounded_joint_ACT_NEW
```

Successful learned ACT video:
[SO-101 seed 300](../outputs/v2/bounded_gripper_act_20261027/development/seed_300/rollout.mp4).
