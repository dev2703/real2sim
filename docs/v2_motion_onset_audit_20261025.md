# Three-episode motion-onset supervision — 25 October 2026

The audit finds **122 stationary two-action prefixes whose 25-action training
chunk contains later movement**, across 980 prefixes with two real targets. It
also finds **11 moving next labels after four ticks of low-motion history**.
These are valid demonstration patterns with possible onset-timing ambiguity;
they are not automatically corrupt labels.

The [reproducible audit](../scripts/audit_v2_motion_onsets.py) consumes only training
episodes 11, 8 and 23 from the original single-iPhone Cartesian data. Its
[report and frozen plan](../results/v2/motion_onset_audit_20261025.json) retain every
selected pair, thresholds, source hashes and event frame. No labels, normalization,
policy inputs, episode splits or sampling weights change. No policy inference runs.

| Episode | Valid action ticks | Two-real-action prefixes | Stationary prefix, moving future tail | Low-motion histories | Moving next labels after low-motion history |
|---|---:|---:|---:|---:|---:|
| 11 / baseline | 351 | 350 | 40 | 36 | 3 |
| 8 / ball shift | 289 | 288 | 41 | 36 | 4 |
| 23 / bowl shift | 343 | 342 | 41 | 45 | 4 |

The inherited motion scale is translation above 0.4 mm per tick; a stationary
target also requires rotation at most 0.01 rad. The causal history consists of
four observations ending at the current tick, each with FK speed at most 0.005 m/s
and joint-history span at most 0.5°. The next-action label is inspected only as an
outcome. Future chunk labels never become policy inputs. All these Cartesian
quantities retain the source dataset's provisional FK.

## Repeated and nearby observations

The earlier repeated-state result reproduces exactly: **12 exact 13-feature
proprioception groups** have differing first-action translation labels above
0.4 mm (2/4/6 groups in episodes 11/8/23). For each group, the final tick is compared
with the closest earlier qualifying tick. Images are compared after the pair is
selected. Episode 8 frames 58/60 have identical proprioception and current-image
MAE 0.307 uint8 levels, while their next labels differ by 3.954 mm; episode 23
frames 66/68 gives 0.311 levels and 3.920 mm.

Across those 12 selected exact-state pairs, mean translation-label differences
are **2.926 mm over the first two actions** and **2.059 mm over the available
25-action chunk**. The shorter objective therefore does **not** remove the
observed onset ambiguity. It changes which horizon receives direct reconstruction
supervision, which the separate matched training experiment tests.

An additional adjacent-tick comparison uses explicit near-state bounds: at most
0.25° per arm joint, 0.5 gripper percentage points and 0.005 m/s velocity difference.
Of the pairs whose first translation labels differ by more than 0.4 mm, **21**
also have current-image MAE at most one uint8 level: 8/6/7 in episodes 11/8/23.
All candidates remain in the JSON, including those whose images exceed that bound.

Exact proprioception plus similar images does not establish identical complete
visual observations or an irreducible prediction error. The pair counts are
correlated within episodes. Near-state/image thresholds are descriptive bounds,
not calibrated perceptual equivalence. No observation is removed from training
because of these findings.

## Reading the prefix experiment

The 25-action target spans 2 seconds at 12.5 Hz, whereas two executed actions span
0.16 seconds. Stationary immediate targets with movement in the future tail show
that full-chunk reconstruction also supervises future onset timing. Prefix-only
reconstruction can test whether that horizon mismatch harms immediate movement
and stationary errors; success still requires the unchanged per-condition gates
and correct object-dependent action probes.

The physical [calibration capture sheet](v2_calibration_capture_20261025.md) proceeds
separately. This supervision audit cannot verify camera/base registration, joint
zeros, gripper geometry or physical timing for the original recordings.

```sh
.venv/bin/python scripts/audit_v2_motion_onsets.py \
  --out outputs/v2/motion_onset_audit_NEXT
```

Four focused tests pass, covering causal-history isolation, real tick-offset
comparison across invalid/end labels, episode boundaries and explicit near-state
bounds. Ruff and whitespace checks pass.
