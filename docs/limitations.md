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
- Closed-loop results beyond the debug run are not yet measured. When reported they carry the provisional-scene
  caveat above and are a sim-only evaluation of a real-data policy; nothing is claimed about
  sim-to-real.
- Placo is not importable on macOS; IK uses a MuJoCo damped-least-squares solver instead of
  the LeRobot Placo interface named in the plan.
- Panda transfer (E3), recovery (E5/E6) and the oracle diagnostic (E4b) are not started.
