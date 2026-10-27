# Frozen ACT, motion reconstruction and the real-data bridge — 27 October 2026

The subsequent [frozen generalization evaluation](v2_frozen_generalization_20261027.md)
now tests the shared semantic checkpoint on 20 new paired layouts and separate
lighting, ball-size and friction conditions. Its 280-trial evidence retains every
failure and leaves the checkpoints and controller settings unchanged.

Both requested software experiments have been executed. The
[combined evidence index](../results/v2/frozen_act_followup_20261027.json)
records the component reports, result denominators, artifact hashes and remaining
physical registration dependencies.

The operator has since confirmed that the physical robot and cameras are still
available and unchanged. The [physical capture preparation](v2_physical_capture_20261027.md)
records that confirmation and provides the next measurement session. No new
physical measurement has yet been captured.

## Meaning of frozen ACT

Frozen means that inference does not update the learned weights. The successful
simulation demos already use frozen ACT, but they do not use the same checkpoint
or demonstrate transfer of the user's uploaded real-demonstration checkpoint.

| Checkpoint and execution | Current placement evidence |
| --- | --- |
| Simulation-teacher RGB/joint ACT, native SO-101 execution | 3/5 development; 4/10 additional layouts |
| Same RGB/joint ACT's recorded learned motion, Panda adapter | 2/5 development layouts; trajectory transfer, not live Panda-image ACT |
| SO-only simulation-teacher semantic ACT, position preview of three ticks | Panda 4/5; SO-101 0/5 development layouts |
| Uploaded real-demonstration HF ACT, native saved processing | Root 0/3; latest 0/3 SO-101 trials |

The semantic ACT consumes initial RGB-derived object coordinates and live
grasp-point feedback; it has no image encoder. Its Panda evaluation uses a
declared mapped start and base translation. The RGB/joint checkpoint and semantic
checkpoint have different weights and action representations. The supplied HF
checkpoint retains its native two-camera image encoder, processors and joint
action representation. All results above use provisional physical registration.

Sources: [SO-101 ACT](v2_bounded_joint_act_20261027.md),
[learned Panda trajectory](v2_learned_panda_retarget_20261027.md),
[semantic ACT execution report](../outputs/v2/semantic_act_transfer_20261027_preview3_all5/report.json),
[uploaded ACT](v2_hf_act_h256_20261027.md).

## User-requested follow-up experiments

Replay controls deliberately bypass policy inference to test the action
interface; their success must not be reported as learned ACT task success.

1. Reconstruct five successful simulated SO-101 teacher demonstrations using
   native joint commands and the common grasp-point interface. Compare the
   current fixed-orientation/binary-gripper conversion against source orientation
   and continuous gripper commands, separately and together. Preserve original
   resets, timing, physics, native bounds and command/path gates.
2. Audit the recorded real demonstration's joint and camera registration, replay
   compatible source portions, compare frozen HF predictions under recorded and
   rendered images at the same joint state, and retest frozen root/latest weights
   with saved versus more frequent feedback. Preserve the native preprocessing
   and label incompatible source starts explicitly.

These diagnostics keep the existing ACT checkpoints unchanged.
No physical calibration is accepted merely because task success or pixel fit
improves. A physical correction needs independent geometric evidence and a
held-out check; camera, joint, source-object and clock uncertainties remain
separate findings.

## Motion-interface result

The [five-variant reconstruction experiment](v2_motion_interface_20261027.md)
completed **25/25 placements**: five known-successful teacher episodes in each
of native joint replay, current XYZ/binary-gripper conversion, source-orientation
reference/binary gripper, fixed orientation/continuous gripper, and
source-orientation reference/continuous gripper. The existing interface therefore
can express these five successful motions. It is not sufficient evidence that
the learned policy predicts the right motion or timing on new layouts.

Source-orientation references improve achieved trajectory fidelity: p95 grasp
error against the recorded source successor falls from **59.57 mm to 4.12 mm**
with binary gripper commands. Both variants still place 5/5. Orientation remains
a soft objective, and native controls are guarded replays rather than bit-exact
reproductions: 73 late ticks encounter slight wrist-limit overshoot. The report
retains every rejection and distinguishes static IK residual from dynamic tracking.

## Real-data registration and paired-image result

The [fresh registration audit](v2_registration_bridge_20261027.md) checks the
uploaded checkpoint's six validation episodes. **All six original starts** lie
outside the current native shoulder-lift range; **1,293/3,802 follower states**
violate at least one native limit. Degree units are supported, but those bounds
cannot identify a physical joint-neutral correction.

On the unchanged two-frame camera confirmation, the current scene/mat projection
has **53.71/55.94 px RMS**, and the existing conditional camera candidate has
**14.54/30.57 px RMS**. Neither passes the existing 3 px confirmation gate.
These errors combine camera and scene registration; they do not identify a
camera-only correction. No physically justified replacement mapping is adopted.

The recorded-demo bridge uses the first strict-compatible suffix of each
episode, disclosing the omitted 4.4–5.8 seconds and retaining full-start
incompatibilities. Recorded leader commands are accepted on **2,133/3,025 ticks**;
next-recorded-readback reference commands are accepted on **1,815/3,025 ticks**.
These are twelve controller-response replays with unmatched provisional objects,
not matched real-episode task-success evaluations. Native/scene guard failures
do not establish that the physical recordings were unsafe.

Frozen inference at **18 identical source joint states per checkpoint** compares
recorded and rendered RGB. Mean five-arm-joint prediction error against recorded
commands increases from **2.94° to 6.12°** for root weights and **2.86° to 6.13°**
for latest weights. Identical-input repeats differ by zero. The image swap varies
appearance, camera registration and unmatched objects together; it is evidence
of visual-input sensitivity, not an isolated camera-calibration estimate or a
reproduction of the model's historical validation score.

The [recorded replay and frozen-HF report](v2_real_replay_hf_bridge_20261027.md)
contains all twelve replay runs, 18 paired states per checkpoint, and the final
fresh feedback screen:

| Frozen uploaded weights / execution prefix | Placement | Held lift | Closest tool-to-ball distance | Accepted commands |
| --- | ---: | ---: | ---: | ---: |
| Root / saved 10 actions | 0/1 | 0/1 | 239.56 mm | 750/750 |
| Root / one action | 0/1 | 0/1 | 255.15 mm | 750/750 |
| Latest / saved 10 actions | 0/1 | 0/1 | 183.34 mm | 750/750 |
| Latest / one action | 0/1 | 0/1 | 169.86 mm | 750/750 |

These are four full 30-simulation-second runs on the same seed, not four unseen
layouts. Faster feedback changes the motion slightly but does not change the
failure phase: none approaches or lifts successfully. Every model request is
accepted, with no gripper saturation or hidden clipping. The observed failure is
therefore not a command-rejection stall in these runs. Synchronous host inference
pauses simulation; this experiment does not measure physical real-time control.

## Decision and remaining physical information

Both requested software experiments have been executed. The common action
interface passes the selected reconstruction controls. The recorded trajectories
do not reproduce cleanly under the current provisional real-to-simulation mapping,
and the uploaded frozen ACT is sensitive to the rendered visual input. These are
separate measured gaps; no single cause or corrective calibration is established.

No ACT training or checkpoint replacement occurred. The next physical correction
needs an independently known shoulder-lift pose paired with readback, a camera
target registered to the base with applicable source-resolution optics, and jaw
gap/closed-zero measurements. Existing recordings support the diagnostics above
but do not identify those missing physical transforms on their own.

Validation: seven focused integration checks passed across motion-reference
semantics, continuous jaw handling, strict unmodified source replay, and native
saved ACT processing. Ruff and whitespace checks pass. Component reports retain
the executed scripts, source hashes, complete traces, videos and bookkeeping
provenance. The physical registration itself remains unresolved.
