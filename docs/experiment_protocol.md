# Experiment protocol

**Status:** drafted 2026-10-13 00:50 AEDT before any closed-loop trial has been run. Nothing in
this file is a result. Results live in `results/metrics/**/trials.csv` and `summary.json` and are
reported in `docs/progress.md` and the README.

## 1. Experiments (plan §7)

| ID | Controller | Policy inputs | What it can support |
|---|---|---|---|
| E0 `raw_replay` | `SCRIPTED_REPLAY`: recorded joint states or FK→IK targets replayed in sim | none (no policy) | calibration and controller feasibility only |
| E1 `joint_ACT` | `LEARNED_ACT` from `outputs/act_joint_so101` (6-D joint targets, deg/%) | iphone image + 6 real-unit joints | real-data ACT integration, SO-101 only |
| E2 `cart_ACT_so101` | `LEARNED_ACT` from `outputs/act_cartesian` (7-D task-frame deltas) | iphone image + 13-float canonical proprio | main learned real→sim closed-loop result |
| E3 `cart_ACT_panda` | same frozen E2 weights on the Panda adapter | same schema | attempted zero-shot transfer |
| E4a | NOT APPLICABLE: no estimated-geometry slot exists in any trained schema | | |
| E4b `oracle_geometry_diagnostic` | scripted controller on simulator ground truth | GT (privileged, labelled) | feasibility of grasp/place in the scene; never an E2 upper bound |
| E5 `ACT+det_rec`, E6 `ACT+MLP_rec` | E2 frozen + recovery (M6, if time) | same as E2 | value of gating / learned specialist |
| null `HOLD_POSE` | holds the initial pose | none | sanity floor: success must be 0/N |

## 2. Fixed before final runs

- Checkpoint: the best-validation checkpoint of each run, named by step and SHA-256 of
  `model.safetensors`, recorded in every `trials.csv` row.
- Scene: `configs/task.yaml` (provenance labels inside) and `configs/sim_so101.yaml`.
- Seeds: `--seed-start 0`, `--n-trials N`, seed `s` gives the ball start inside the start
  region disc and the same initial arm pose. Matched seeds across E1/E2/E3/E5/E6.
- Episode: 30 s sim time (300 ticks at 10 Hz); success = ball inside the bowl and resting for
  1.0 s continuous. Timeout, STOP, ball off table, and IK rejection streaks all count as failures.
- Debug phase: 3 trials per checkpoint first (seeds 0–2, with video), then ≥10 trials
  (target 20) with seeds 0–19. Debug trials are reported separately and never merged into the
  final count.

## 3. Per-trial record (`trials.csv`)

`experiment_id, controller_source, robot, seed, checkpoint_sha256, checkpoint_step, success,
lifted, delivered, released, rolled_out, ball_escaped_table, phase_failed, min_ee_ball_distance_m,
ticks, ik_rejections, mean_latency_ms, max_action_saturation, video_path, abort`.

Definitions (tightened 2026-10-13 after a logic review; see `tests/unit/test_metrics_edge_cases.py`):

- `success`: ball inside the bowl, resting, and **not held** for 1 s of consecutive ticks. It does
  not require the ball to have been grasped; a ball pushed in counts, and `released` shows whether
  it was placed.
- `lifted`: the ball was above the lift height **while held**. A ball knocked into the air is not
  a lift.
- `rolled_out`: a released ball left the bowl and the trial did **not** end in success.
- `phase_failed` is the furthest phase reached: `none` (success), `release` (ball was in the bowl
  but no stable released success), `transport` (lifted but never in the bowl), `grasp` (held, or
  came within 3 cm, but never lifted), `approach` (neither), `aborted` (the trial stopped on an
  exception; the message is in `abort` and the trial is still counted).
- `mean_latency_ms`: steady-state policy inference latency, excluding the first (warm-up) call,
  which is in `summary.json` as `warmup_latency_ms`. For the oracle it is IK solve time.

## 4. Reporting rules

- Every attempt is a row; no filtering, no re-running a seed to get a better result.
- Report `successes/N` as a fraction, with the per-phase failure breakdown. No confidence
  intervals with N ≤ 20.
- Videos carry burned-in captions: `LEARNED ACT (sim)`, `SCRIPTED REPLAY`, `HOLD POSE`,
  `ORACLE (privileged)`, `PANDA FROZEN ACT`. A video without a caption is not evidence.
- Separate provenance: real data (training) versus simulator (evaluation) are never mixed in
  one metric.
- Anything derived from PROVISIONAL-STANDARD physical values (see `docs/calibration.md`) says
  so next to the number.

## 5. Results (2026-10-13)

| Experiment | Controller | Robot | Checkpoint | Success | Lifted | Phase failed | Notes |
|---|---|---|---|---|---|---|---|
| E0 `hold_pose_null` | HOLD_POSE (no policy) | SO-101 | — | **0/10** | 0 | approach ×10 | null floor |
| E0 `hold_pose_null` | HOLD_POSE (no policy) | Panda | — | **0/10** | 0 | approach ×10 | null floor |
| E1 `joint_ACT` | LEARNED_ACT | SO-101 | joint 020000 (`d27ff21f8698…`) | **0/10** | 0 | approach ×10 | min EE–ball 14–17 cm; arm rises and holds |
| E2 `cart_ACT_so101` (run 1) | LEARNED_ACT | SO-101 | cartesian 006000 (`0a3b22211c97…`) | **0/10** | 0 | approach ×10 | 4 trials had 227–238 IK rejections from a workspace-floor bug |
| E2 `cart_ACT_so101` (run 2, floor fixed) | LEARNED_ACT | SO-101 | cartesian 006000 | **0/10** | 0 | approach ×10 | 2–3 IK rejections per trial; min EE–ball 15–19 cm |
| E3 `cart_ACT_panda` (run 2, corrected scene) | LEARNED_ACT (same frozen E2 weights, no Panda training) | Panda | cartesian 006000 | **0/10** | 0 | approach ×8, grasp ×2 | min EE–ball 3.4–4.9 cm (run 1: 6–12 cm); seeds 4 and 8 held the ball 105 and 18 ticks without lifting; 0–85 IK rejections |
| E4b `oracle_geometry_diagnostic` | ORACLE (scripted on simulator GT, privileged) | SO-101 | — | **10/10** | 10 | none | scene feasibility only, not an E2 upper bound |
| E5/E6 recovery | — | — | — | not run | | | detector/FSM/MLP built; no correction data (G6 not completed) |

Every trial is a row in `results/metrics/trials.csv` (70 rows). Seeds 0–9, 300 ticks (30 s) at
10 Hz, success = ball resting inside the bowl and released for 1 s. Captioned videos per trial in
`results/media/` (not tracked in git; regenerate with the commands below).

**Failure isolation (plan M4.4).** The learned policies never reach the ball: minimum
end-effector–ball distance 14–19 cm on the SO-101 and 6–12 cm on the Panda, versus the oracle's
grasp at tick ~30. Evidence for each candidate cause:

- *Gripping / scene physics*: ruled out as the blocker. The scripted oracle grasps, carries and
  places the ball 10/10 with the same gripper, contacts and friction sweep.
- *IK / controller*: the first E2 run rejected 227–238 of 300 ticks in 4 trials because the IK
  workspace floor (z ≥ 0.015) sat above the lifted start pose; with the floor at the table top
  the re-run has 2–3 rejections per trial and the outcome is unchanged (0/10). Not the cause.
- *Vision domain gap*: measured directly by `real2sim.evaluation.domain_gap` on validation
  episode 1 (120 ticks): feeding the frozen policy a simulated render of the recorded state
  instead of the real frame raises the open-loop error from 10.4° to 13.5° (joint policy, ×1.29)
  and from 0.016 to 0.023 (Cartesian, ×1.44), with the gripper channel taking most of the hit
  (7.0→14.0 %, 0.082→0.127). A moderate gap, not a collapse.
- *Open-loop quality and compounding (distribution shift)*: the Cartesian policy's per-tick
  error on *real* frames is 1.5–2 mm against a 3.7 mm median tick motion (relative error ≈ 0.4);
  the joint policy's is 10–21° on the large joints. With 28 training episodes and no
  perturbation data, closed-loop execution drifts off the demonstrated manifold within the
  first second and never recovers. This is the dominant cause on the evidence above.

Latency: 30–70 ms per chunk on CPU (two runs sharing the machine); no timeouts from latency.

**Correction (2026-10-13 cleanup).** The first E3 run (`results/metrics/E3_cart_panda`) was
measured on a Panda scene that ignored the 12 mm base lift (D-014) and dropped the fitted camera
roll, so the frozen policy saw a shifted proprio and a rotated image unrelated to the
embodiment. The scene now shares the SO-101 task frame and camera
(`tests/integration/test_embodiment_consistency.py`); the re-run
`results/metrics/E3_cart_panda_v2` is the reported result and is also 0/10, but it behaves
differently: every trial gets within 3.4–4.9 cm of the ball (6–12 cm in run 1) and two trials
make contact and hold the ball without lifting it. Run 1 is kept, not deleted.
