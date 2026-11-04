# Codebase audit — 4 November 2026

Reviewed the supported evaluation/recovery CLIs, ACT training continuation,
runtime configuration, repository artifacts and existing project evidence.
Findings were reproduced against `1ec70c8`; fixes also retain the concurrent
changes from main through `3fb3d27`. The initial fixes landed on main in
[`d43d680`](https://github.com/dev2703/real2sim/commit/d43d680).
No physical robot was commanded.

## Blocking findings

| Priority | Finding and reproduction | Resolution |
| --- | --- | --- |
| P1 | Training can replace an existing run's checkpoints, configuration and history. Rejected exact resumes can change metadata, and resuming an old checkpoint can overwrite newer checkpoints. Reproduced with temporary synthetic data and real tiny CPU ACT training. | [#39](https://github.com/dev2703/real2sim/issues/39): reject accidental run reuse and same-directory rewinds; validate sampler/RNG continuation and existing history before writing. Tip continuation and older checkpoints continued into a new directory remain supported. |
| P1 | Recovery collection and evaluation ignore the requested checkpoint's camera interface. Six both-arm scene regressions fail with ordered iphone/wrist cameras at 180×320. | [#40](https://github.com/dev2703/real2sim/issues/40): propagate the saved scene contract into live, target and expert-shadow scenes, and use actual scene dimensions for video. |
| P1 | Three active scene configs contain machine-specific task references. Relocating the checkout loads another checkout's task or fails when it is absent. All three relocated-checkout regressions fail before the fix. | [#38](https://github.com/dev2703/real2sim/issues/38): use repository-relative runtime references. The task bytes and compiled geometry, limits and cameras remain identical. |
| P2 | RGB recovery always requires one second of success dwell, ignoring the task. A 0.12-second task at 25 Hz receives 25 ticks instead of three. | Included in [#40](https://github.com/dev2703/real2sim/issues/40): derive dwell from `success_hold_s`; both new scoring regressions pass. |
| P1, unresolved | The retained matched audit shows raw frozen ACT at 0/5 task successes per arm; RGB and ACT→RGB each score 5/5 per arm. The software fixes do not resolve unassisted transfer. | [#41](https://github.com/dev2703/real2sim/issues/41): validate physical input registration and repeat matched checkpoint-specific evaluation. |
| P1, fix proposed | Exact resume accepts some tensor-compatible model changes, including dropout, attention heads, KL weight and action cadence. | [PR #42](https://github.com/dev2703/real2sim/pull/42) addresses existing [#19](https://github.com/dev2703/real2sim/issues/19) by comparing effective saved/requested model semantics before constructing a policy or writing artifacts. |
| P2, fix proposed | Duplicate recovery seeds replace per-seed collection traces or interrupt evaluation after the first trial, leaving a running report. Collection can also accept its own reserved evaluation seeds. | [PR #44](https://github.com/dev2703/real2sim/pull/44) addresses [#43](https://github.com/dev2703/real2sim/issues/43): validate seeds and partition roles before model loading or output creation. |
| P1, fix proposed | The Linux CI unit job rejects a retained generalization layout audit that passes locally on macOS. It compares regenerated floating-point coordinates using exact equality. | [#45](https://github.com/dev2703/real2sim/issues/45): preserve exact seed/order/schema checks and allow only `1e-12 m` absolute XY roundoff, with zero relative tolerance. New protocols disclose that bound; historical evidence bytes remain unchanged. |

## Conservative cleanup

- Reuse the existing streaming fingerprint implementation while retaining
  `real2sim.config.sha256_file` as a compatible public import.
- Remove the stale Git whitespace exception for an already-deleted diagnostic XML.
- Remove four ignored Finder metadata files.
- Preserve referenced dated diagnostics, calibration inputs, manifest-pinned
  assets/licenses, datasets, checkpoints and retained trial evidence. Their age
  alone does not establish that they are unused.

Runtime scene-config hashes intentionally change when paths become portable.
Historical configs and recorded provenance remain unchanged; existing benchmark
protocol checks should continue to reject changed runtime inputs in an old run.

## Validation

The original sandboxed unit run produced 1,265 passes, four skips and three
CoreGraphics connection failures. Those rendering assertions remain enabled.
Final validation runs with macOS graphics access and includes the full unit
suite, training smoke/resume checks and both-arm scene/controller integrations:
**1,416 passed, four skipped**. Ruff, whitespace checks, locked dependency
resolution and the data audit CLI help check also pass. The skips are two
CUDA-only cases, an optional bounded experiment and the real paired warm-start
test whose parent checkpoint is absent. The latter now checks that optional
prerequisite explicitly, matching its existing optional pair-cache requirement.

The ACT continuation checks include bit-for-bit uninterrupted/resumed
training equality and byte-for-byte preservation after rejected operations.
The eight recovery regressions construct actual headless SO-101/Panda scenes
and were verified failing against the original scripts. These checks validate
scene/scoring interfaces; they do not establish a new checkpoint's task success.

Follow-up validation adds six exact-resume model-configuration cases (12 CPU
continuation checks total) and 34 recovery-seed regressions. The training PR
adds the CPU continuation checks to CI alongside the existing unit suite.
The Linux layout-audit correction passes 23 focused local checks, including
sub-tolerance roundoff, larger changes, wrong seed/order/schema and nonfinite
coordinate cases. Linux CI verifies the actual platform comparison separately.

The retained 1 November matched report was independently audited again:
**30/30 trial/trace pairs verified**, no missing evidence or contradictions.
This verifies historical evidence, without rerunning physics or asserting
equivalence with the current runtime. See the [original audit](CODEBASE_AUDIT_20261101.md)
for the protocol and benchmark commands.
