# Proposed next real-data recording session

Operator guidance for the proposed 30–40 additional episodes. This does not change the current
training schema, frozen split, simulator cameras, calibration or hardware settings.

## Check the camera view before collecting the batch

Record 2–3 pilot episodes, then inspect the **saved** videos at approach, jaw closure, lift and
release. Confirm both streams decode, timestamps align with state/action, and no stream is
frozen or missing its final frames. Check blur, exposure and whether the fingers hide the ball
at contact. A useful wrist view shows the fingertips and nearby object during the approach;
the external view should cover the ball, arm and bowl throughout the task.

Keep recording `iphone` from its existing fixed mount. Correct the wrist camera once, secure
its mount, then keep that mounting angle consistent across the batch. Wrist motion with the arm
is expected; remounting it between episodes adds another variable. Keep lighting and background
fairly consistent initially. Do not deliberately vary every condition at once.

The current dataset has `iphone` and an unusable `top` stream, not an audited usable wrist
stream. Record the corrected wrist view with a clearly documented camera identity in a new
dataset/version. Keep the old data intact. Existing and new episodes can potentially be combined
for an iPhone-only policy after audit; a two-camera policy needs an explicit missing-view/data
choice because the old episodes lack that view. Real-to-sim testing also needs a corresponding
simulated wrist camera. None of those schema/model changes are part of the current loss sweep.

## Suggested collection budget

These counts are a practical proposal, not a guarantee of sufficient data or an applied split.

| Purpose | 30 episodes | 40 episodes |
|---|---:|---:|
| Ordinary successful picks: six reachable ball regions, repeated examples per region | 18 | 24 |
| Small bowl-position changes, or more ball coverage if the bowl will always stay fixed | 3 | 4 |
| Successful corrections after a slightly missed approach or a small ball displacement | 3 | 4 |
| Whole episodes reserved for validation before training | 6 | 8 |

- Cover near/far and left/centre/right starts within the operator-verified reachable area.
  Add small offsets within each region; exact duplicates of one trajectory add less coverage.
- Keep the bowl fixed for most episodes. If deployment requires a moving target, use only
  two or three modest bowl positions initially, with the bowl clearly visible. Keep the same
  ball/bowl identities and task so the intended target is unambiguous from the recorded input.
- Use a consistent grasp style and pace, smooth motions and little idle time. Vary object
  position first; introduce modest initial-arm-pose variation only after the basic examples
  are reliable. Avoid making every episode a different grasp strategy.
- Capture corrections during operator-controlled teleoperation: re-approach after a miss,
  re-centre over the ball, then complete the task. Do not physically disturb an active robot
  or deliberately create collisions to manufacture recovery data.
- Choose held-out whole episodes across the planned conditions, including some new ball/bowl
  combinations. Never split adjacent frames from one episode between train and validation.
  Keep current held-out episodes held out when a future merged split is approved.

## Keep an episode log

Record episode ID, recording session, camera mounting version, ball/bowl start-region labels,
whether a recovery occurred, actual outcome and any recording problem. Retain failed episodes
with honest outcome labels for analysis; do not silently treat failed endings as successful
imitation targets. Review the saved videos every 5–10 episodes to catch camera drift or capture
failures before repeating them across a full session.

More coverage and an informative wrist view may help visual alignment and recovery. They do
not repair the existing simulation reset/contact or controller mismatch; those remain separate
diagnostic issues. The completed image ablation shows that the current policies do use images,
although it does not establish reliable ball localization.

Reference: LeRobot's [real-robot data collection guidance](https://huggingface.co/docs/lerobot/main/getting_started_real_world_robot#tips-for-gathering-data)
recommends visible objects, stable camera mounts, consistent grasping and gradual increases in
variation. The episode allocation above is a project-specific suggestion.
