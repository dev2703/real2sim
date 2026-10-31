# Codebase audit — 1 November 2026

Scope: native frozen ACT, SO-101/Panda scene and control interfaces, camera
contracts, evaluation evidence, kinematics hot paths, training/data loading,
and regression checks. Priorities reflect effect on trustworthy task completion
before throughput. This is a simulation audit; no physical robot was commanded.

## Priority queue

| Priority | Bottleneck / evidence | Status and next action |
| --- | --- | --- |
| P0 | `panda/model.py` contained an indented orphan asset-check block and a duplicate `task_frame_offset` calling an undefined name. Panda imports failed; the initial unit run stopped with 12 collection errors. | **Fixed.** Restore the existing shared geometry imports, remove orphan definitions, stream asset hashing. Both-arm scene/control tests cover imports and frame consistency. |
| P1 | Frozen ACT has not demonstrated native pick-and-place on both simulated arms. Retained, independently checked Panda development evidence gives raw ACT 0/10; historical summaries report 0/100 per arm. | **Open.** Fresh matched results below distinguish ACT competence from recovery competence. No weights were trained or changed. |
| P1 | Input/physical registration gap. The retained 18-anchor `hf_alignment_factorial_20261029_render/report.json` reports five-arm-joint executed-prefix MAE of 2.858° with recorded views versus 11.752° with nominal renders; global-only substitution is 8.566°. Camera intrinsics/extrinsics, source joint registration and jaw mapping remain provisional. | **Open.** Validate the existing physical calibration capture/verification workflow. The image ablation supports a domain gap; it does not identify a valid camera correction. Do not compensate by widening joint limits or adjusting physics to fit failures. |
| P1 | Panda supplies a lossy virtual SO-101 observation: measured TCP position is projected into source joints, while source rotational priors and a declared top-down target supply missing degrees of freedom (`diag_v2_learned_panda_retarget_20261027.py`). Source training did not establish Panda visual or kinematic invariance. | **Open.** Verify source representability and both-arm task performance independently. A common action space and valid IK do not establish frozen-policy transfer. |
| P1 | Native scenes hard-coded v2 `top` camera names, dimensions and checkpoint identity. A v3 `wrist` checkpoint could not execute correctly through these runners. SO provenance also assumed one exact processor filename. | **Fixed software contract.** Derive camera roles, saved key order, dimensions and identity from the requested checkpoint; reject incompatible contracts. Both-arm scene tests use `wrist`, reversed camera order and different dimensions. Actual v3 weights/processors are **not yet loaded or task-tested**. |
| P1 | Historical summary completeness could be mistaken for locally verified evidence: each old per-arm 300-trial benchmark retains only three trial/trace pairs. | **Fixed audit tooling; historical gap remains.** `audit_frozen_act_results.py` verifies protocol/row/trace hashes, exact trial coverage, reset fingerprints, controller accounting, outcome stages and aggregates. Missing evidence produces a nonzero CLI exit. Fresh traces remain intact. |
| P2 | Native standalone latency scopes differed across arms, and success-hold trackers assumed one second instead of reading the task configuration. | **Fixed.** Time state acquisition/projection, rendering, saved processors and ACT together; use configured success dwell. Existing nominal scenes use one second, so the scoring correction does not change their criterion. |
| P2 | Synchronous inference creates long control ticks. A 25 Hz command period is 40 ms; fresh mean policy-call/render costs are 128 ms SO-101 / 153 ms Panda, while most ticks consume cached actions. | **Measured; open.** Profile render/backbone/processor costs separately before choosing an accelerator or an asynchronous queue. The simulation does not advance physics while inference is computing, so its success rate does not validate wall-clock robot timing. Preserve checkpoint cadence when comparing changes. |
| P2 | Independent IK/FK probes ran full velocity/force dynamics although their consumers need poses, Jacobians and contacts. SO-101's guarded command samples 35 FK probes. | **Fixed.** Use MuJoCo's complete position stage on independent solver data, retaining collision detection. Exact pose/contact/Jacobian and accepted/rejected IK parity tests pass on both arms. Measured SO guard median: 0.978 → 0.880 ms; this is a modest whole-call saving. |
| P2 | Learned Panda recovery nests five backtracking fractions with two sampled source-representation paths and repeated nonlinear source-projection solves (`common_recovery.py:200`, `:278`). | **Open, profiling first.** Add stage timings and inspect repeated solves before caching or changing search. This is a recovery cost, not an explanation for raw ACT approach failure. |
| P2 | Benchmark recomputes large-file fingerprints after each trial and repeats checkpoint hashing (`benchmark_frozen_act.py`). | **Open.** Measure provenance I/O, then reuse a digest within a single verification boundary while preserving before/after integrity checks. |
| P2 | Legacy diagnostics close environments only on normal completion (`evaluation/oracle.py`, `replay.py`, `domain_gap.py`); exceptions after renderer initialization can retain resources. Active native/matched evaluators have cleanup. | **Open.** Convert legacy paths to context-managed cleanup when touched; add exception-path checks. |
| P2 | Some active measured-scene YAMLs contain absolute `/Users/deveshvyas/...` task paths. | **Open.** Use repository-relative runtime config references while keeping original historical provenance unchanged. This affects portable reproduction. |
| P2 | Panda integration tests wrote diagnostic measurements into retained result paths, silently replacing old evidence during validation. | **Fixed.** Pytest diagnostics now use its temporary directory. The original retained measurement file was restored. |
| P3 | Benchmark builds scenes before checking whether every trial can resume; scenes are reconstructed per condition. | **Open.** Preflight completed artifacts and construct only missing workloads, preserving condition isolation and matched resets. |
| P3 | Training uses per-sample CPU image normalization and blocking transfers; exact-resume mode intentionally requires zero workers. Dataset loading already uses lazy image memmaps and persistent workers when enabled. | **Profile before changing.** Device/worker optimization must preserve deterministic sampler/RNG semantics; no measured training throughput claim is made in this audit. |
| P3 | Duplicate imports and shadowed hash/serialization helpers in data preparation and evaluation caused 12 Ruff errors including the Panda syntax error. | **Fixed.** Remove cleanup debris; preserve existing helper interfaces. |

## Frozen-weight verification

The main target is the checkpoint already cached and used by the repository:
`bottowork/so101-act-h256-b16`, revision
`e696c8a078935432e3501e2c28c1fb2598fbff7e`, `latest/`, SHA-256
`5dab91076d1e3b274674a888f522ce5fdddaf69ddd05c1adf3bce1742b99f65c`.
V3 support here means tested scene/schema plumbing; it does not mean tested v3
inference or success. The user did not supply a different cached checkpoint.

The fresh benchmark uses seeds 14000–14004, five per arm, 25 Hz native control,
ten saved actions per ACT call, 30-second horizons and identical per-seed reset
fingerprints across conditions. Task success requires a held lift, release over
the bowl and stable resting placement. Simulator truth is used for scoring only.
Every failure and abort stays in the denominator. RGB conditions are explicitly
geometric assistance; their successes are not raw ACT successes.

| Controller | SO-101 task successes | Panda task successes | Assistance |
| --- | ---: | ---: | --- |
| Frozen v2 ACT alone | 0/5 | 0/5 | None |
| RGB geometric controller | 5/5 | 5/5 | Every tick |
| Frozen v2 ACT → RGB recovery | 5/5 | 5/5 | Recovery in every trial |

All 30 trials completed with zero aborts. All ten raw ACT trials failed approach,
with zero held lifts and zero guard/IK rejections. Closest tool-to-ball distance
ranged from 0.329–0.363 m on SO-101 and 0.337–0.370 m on Panda. ACT→RGB switched
to recovery at tick 50 in every trial. The independently recomputed evidence
audit verifies **30/30** trial/trace pairs, matched resets and unchanged weights,
with no missing evidence or contradictions. These five-seed results describe
the tested nominal simulation distribution, not a reliability guarantee.

Mean raw policy-call latency, including rendering and Panda source projection,
was 128 ms SO-101 / 153 ms Panda on this machine. The ten-action queue amortizes
computation but synchronous inference still creates individual wall-clock stalls.
These timings overlapped regression execution and are audit observations, not
an isolated accelerator benchmark.

The executed Python sources are copied into `matched_v2/executed_sources/` with
the hashes recorded in its protocol. Final standalone artifact-discovery and
configurable-dwell fixes were applied afterward; neither changes the matched
benchmark's controller or scorer. Existing nominal dwell is one second.

Validation: **1,271 passed, three skipped** across the full unit suite and
selected native ACT, both-arm scene/control and kinematics integration tests.
The skips are two CUDA-only cases and an optional experiment artifact. After
the final standalone fixes, **35 focused tests passed**, including two new
headless regressions for configurable dwell and arbitrary processor filenames.
Ruff and `git diff --check` pass. Archived executed Python sources exactly match
all recorded protocol hashes; no archive file is missing.

```sh
.venv/bin/python scripts/benchmark_frozen_act.py \
  --robots so101 panda --conditions raw rgb hf_rgb \
  --seeds 14000 14001 14002 14003 14004 \
  --out outputs/v2/codebase_audit_20261101/matched_v2

.venv/bin/python scripts/audit_frozen_act_results.py \
  outputs/v2/codebase_audit_20261101/matched_v2/report.json \
  --out outputs/v2/codebase_audit_20261101/evidence_fresh.json
```

Use a new output directory for a different runtime protocol. On macOS, offscreen
rendering requires access to CoreGraphics; the workspace sandbox's connection
failed, and the authorized graphics-capable runs supplied the regression and
task evidence. No skip was substituted for these camera checks.

## V3 integration path

V2 remains the evaluation priority. The software changes remove the known v3
camera, render-size and fixed-artifact-name barriers. Next, obtain the pinned
v3 export from `bottowork/act_so101_pickplace_v3_varied_kaggle`, revision
`076452e6024c778a2bffabe99ecf33776de141e1`, including saved preprocessing,
postprocessing and all referenced processor tensors. No download was required
or performed for this audit.

Load it with the same native adapter and first verify finite `(50, 6)` actions,
saved normalization exactly once and unchanged file hashes. Its documented
training used LeRobot 0.6.2, while this environment pins 0.6.1: test the actual
export before changing dependencies, and isolate any required version change.
Then evaluate both arms through the existing `--snapshot` option with the same
five seeds and all three conditions in a new output directory. Preserve 25 Hz
source timing, ten executed actions, native guards, scene geometry and lighting.
Compare task outcomes and image/action sensitivity, rather than assuming the
newer model transfers. Physical camera and source-joint validation remain
necessary for both versions.

## Evidence files

Generated artifacts live in ignored `outputs/v2/codebase_audit_20261101/`:

- `matched_v2/protocol.json`, `matched_v2/report.json` and all per-trial
  `attempt_*/report.json` / `ticks.jsonl`: fresh task evidence.
- `evidence_fresh.json`: independent local integrity/coverage audit.
- `evidence_historical_so101.json`, `evidence_historical_panda.json`: each
  verifies 3/300 claimed trials and marks the remaining 297 missing.
- `evidence_retained_dev.json`: verifies all ten retained development trials.
- `evidence_retained_panda_dev10.json`: verifies all 60 latest Panda development
  trials, including raw ACT 0/10 versus RGB and ACT→RGB 10/10 each.
- `kinematics_profile.json`, `profile_kinematics.py`: reproducible seven-repeat
  microbenchmark, with 128 configurations per arm and zero geometry/Jacobian/
  contact mismatches. Position-stage forward speedups are 2.94× SO-101 and
  1.64× Panda; short Panda IK has no demonstrated whole-call speedup.
- `validation.json`: final regression counts, commands and fresh-run identity.

The next success-critical work is verified camera/joint/jaw registration and a
fresh checkpoint-specific matched evaluation. Faster IK and passing software
tests do not establish that a frozen real-demonstration policy transfers to
either simulated arm.
