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
ticks, ik_rejections, mean_latency_ms, max_action_saturation, video_path`.

`phase_failed` is one of `approach` (never came within 3 cm of the ball), `grasp` (reached but
never lifted), `transport` (lifted but never over the bowl), `release` (over the bowl but ball
not resting inside at the end), `none` (success).

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
