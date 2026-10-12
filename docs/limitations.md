# Limitations

Updated 2026-10-13 00:55 AEDT. Each item says what is limited, why, and what it means for the
claims this project can make.

## Data

- **One camera.** The `top` stream is aimed at the ceiling and was excluded. The policy sees a
  single phone view; depth along the camera axis is weakly observable.
- **37 of 40 episodes.** Episodes 25, 33 and 34 were excluded because their iphone video has
  fewer frames than data rows (2, 14 and 1). Train 28 / validation 9.
- **No per-episode outcome labels.** The operator states all retained episodes succeeded; the
  review manifest (`annotations/episode_outcomes.csv`) is still `unknown` for every episode.
- **Follower lag and offset.** The recorded `action` (leader) differs from achieved motion by
  up to 16 frames and a 13° shoulder offset. The joint baseline learns leader targets; the
  Cartesian policy learns achieved motion (D-003). Neither is "what the operator intended"
  exactly.
- **Image lags joints by about 67 ms** on the iphone stream. Not corrected.

## Physical model

- **No measured calibration.** Ball, mat, table and camera values are PROVISIONAL-STANDARD;
  bowl dimensions are operator-supplied but disagree with the video by roughly 15 % in
  diameter and 2× in height (see `docs/calibration.md` §1). Bowl centre and ball start are
  DATA-DERIVED from gripper FK with 3–8 cm spread.
- **Joint mapping is provisional.** Real degrees are assumed to equal MJCF radians with zero
  offset. One consistency check passed (grasp height equals ball radius); pan sign against
  the camera is unverified.
- **The dataset's folded rest pose (shoulder_lift ≈ −105°) is outside the Menagerie joint
  limit (±100°)** and is clamped in simulation.
- **Contact parameters are swept, not identified.** Friction 0.4–0.9, restitution via solref.
- **The supplied bowl depth (2.54 cm) is shallower than the ball radius (3.35 cm).** With the
  frustum model the ball is still contained up to 0.5 m/s, but that depends on the unmeasured
  base diameter; the depth sweep is kept.
- **Camera extrinsics are a PnP fit with 17 px RMS and weakly constrained depth**; the
  simulated robot renders slightly smaller than real. The bowl position disagrees with the
  images by 6–7 cm.
- **The simulated start pose penetrates the table by 2.3 cm** until the base-height decision
  D-014 is taken; every rollout currently begins with a push-out transient.

## Learned policy

- **Open-loop validation error, not task success**, is the only measured policy quality so
  far: joint ACT validation L1 ≈ 0.33 normalised (3° pan, 12–15° on the large joints).
- **Training is on Apple MPS**, 0.42 s/step, so runs are limited to 20k steps and a single
  seed each. No hyperparameter search.
- **Modal GPU is blocked** until a payment method is added to the account.

## Simulation and evaluation

- **E1 debug (joint ACT, step 2000, 3 seeds): 0/3**, all failing in the approach phase; the
  arm rises to about 0.17, 0.0, 0.32 m and holds. Not yet separated between an under-trained
  checkpoint, the camera gap (fit applied after this run) and the start-pose transient.
- **E0 replay of validation episode 1**: joint setpoints track to 0.05° median after the start
  transient (p95 elbow 5°, EE 3.2 cm); the FK→IK path accepts 96 % of ticks with 0.08 mm
  median residual but accumulates 5–13 cm EE error because deltas are relative to the drifted
  sim pose. The ball start of the real episode is unknown, so replay says nothing about grasps.
- **Primary closed-loop results are 0/10 for every learned policy** (E1 joint, E2 Cartesian,
  E3 Panda zero-shot), all failing in the approach phase; the scripted oracle is 10/10 and the
  null baselines 0/10. The dominant cause on the evidence is compounding error of a weak
  open-loop policy trained on 28 episodes, with a moderate rendering gap on top (×1.3–1.4
  open-loop error on sim renders). See `docs/experiment_protocol.md` §5.
- **The E2 run 1 contained a harness bug** (IK workspace floor above the start pose); it is kept
  as a labelled row set and the corrected run 2 is the reported result. Both are 0/10. When reported they carry the provisional-scene
  caveat above and are a sim-only evaluation of a real-data policy; nothing is claimed about
  sim-to-real.
- Placo is not importable on macOS; IK uses a MuJoCo damped-least-squares solver instead of
  the LeRobot Placo interface named in the plan.
- Recovery E5/E6 were not run: no correction data were collected, so the learned specialist
  (G6) is NOT COMPLETED; only the detector/FSM/MLP code with unit tests exists.

## Engineering limitations found in the 2026-10-13 cleanup

- **Two IK solvers.** The SO-101 (5-DoF, soft orientation) and Panda (7-DoF, full pose) each
  carry their own damped-least-squares IK with near-duplicate code paths.
- **`sim/camera_fit.py` is 1,100+ lines** with about 60 % test coverage; the PnP and
  back-projection core is tested, the rendering and CLI paths are not.
- **Recovery detector without phase hints** treats every tick as pre-grasp, so a successful
  placement would trigger a roll-away takeover. Nothing calls the recovery loop yet (E5/E6 not
  run); any future caller must pass phase hints.
- **Training resume is not bit-exact**: torch/numpy RNG states are not restored, so a resumed run
  replays a different batch order from the original.
- **`domain_gap` compares** a real frame with a sim render of the same robot state, but the sim
  ball/bowl come from seed 0, so the gap mixes rendering differences with scene-content ones.
- **The approach threshold (3 cm) is inside the ball radius (3.35 cm)**: a gripper that only
  touches the ball surface is scored `approach`, as the protocol states.
- **E3 run 1 was on a mis-registered Panda scene** (base lift and camera roll ignored); the
  corrected run 2 is reported and has the same 0/10 outcome.
