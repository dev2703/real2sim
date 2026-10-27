# Real-data ACT registration bridge — 27 October 2026

The new audit reads the uploaded ACT's exact six-episode validation split and
recomputes source-state compatibility and camera confirmation errors. It verifies
degree units, but finds **no independently justified camera, joint-neutral or jaw
correction to adopt**. The appropriate next frozen-ACT retest therefore retains
the declared provisional scene and reports that limitation.

The [curated evidence](../results/v2/registration_bridge_20261027.json) links the
[complete fresh audit](../outputs/v2/registration_bridge_20261027/report.json),
unchanged input hashes, source snapshot, and
[source-image projection comparison](../outputs/v2/registration_bridge_20261027/camera_projection_comparison.jpg).
No checkpoint, original recording, calibration, physical limit, scene configuration
or command guard changed. No training, parameter optimization or source clipping
occurred. These checks do not establish successful real-to-simulation transfer.

## Exact uploaded-checkpoint validation split

The split comes from the pinned Hugging Face checkpoint's `experiment.json`, not
the project's historical 29/6/6 split. Some of these episodes were historically
reserved; this is the uploaded model's validation diagnostic and must not be
reported as an untouched project test set.

| Episode | Rows | Initial shoulder lift | State rows outside native limits | Action rows outside native limits | First native-compatible follower frame / time |
| --- | ---: | ---: | ---: | ---: | --- |
| 4 | 532 | −104.440° | 185 | 167 | 110 / 4.40 s |
| 7 | 628 | −104.703° | 181 | 163 | 127 / 5.08 s |
| 18 | 668 | −104.615° | 261 | 244 | 145 / 5.80 s |
| 32 | 629 | −104.703° | 219 | 202 | 110 / 4.40 s |
| 38 | 657 | −104.791° | 226 | 196 | 137 / 5.48 s |
| 40 | 688 | −104.703° | 221 | 203 | 142 / 5.68 s |

Of 3,802 follower rows, **1,293 (34.01%)** exceed at least one native limit.
Shoulder lift accounts for 1,176 violating rows, and elbow for 703, with overlap.
Of 3,802 recorded actions, **1,175 (30.90%)** violate a native bound, all at
shoulder lift. The native lift interval remains −100° to +100°.

These first compatible frames are numerical candidates, not approved episode
starts: they do not establish that the full grasp remains or that the reference
path is collision-free. Demonstration suffix replay must disclose its truncation.
All original frames remain in the audit; no start is clipped or omitted silently.

## Units established; physical neutral still unidentified

Using the actual current follower calibration, the inverse degree equation
reconstructs integer encoder ticks with per-joint p95 residual at most
**0.00004292 tick**. Treating the same readings as normalized ±100 produces
0.40–0.49 tick residuals. Degree units are strongly supported.

That arithmetic cannot establish physical signs or neutral. Current task signs
are all +1 and added offsets all 0°. The stored homing offset already acts in the
servo readback convention; applying it again would double-count it. Nor is the
range midpoint independently established to equal the CAD neutral. For example,
the stored homing-pose coordinate 2047 corresponds to a reported lift value of
6.681°, but there is no verified physical pose that makes −6.681° a justified
CAD correction.

As a feasibility diagnostic only, accepting all six validation episodes under
the same +1 lift sign requires an additive offset of at least +4.879°. Native
limits alone permit a broad range up to +64.308°; they therefore do not identify
the correct neutral. The full mathematical intervals are retained in the JSON
and are never promoted as calibration.

The current gripper mapping is 0.019199 rad per percent. The current recorded
encoder span would imply 0.0225857 rad per percent under a direct 1:1 output-horn
assumption; the current scale is about 15.0% below that encoder-derived scale. Physical closed zero
and clearance are unmeasured, so this audit does not adopt a scale or TCP change.
The currently available calibration file is hashed, but its presence does not
independently establish recording-time calibration provenance.

## Frozen camera confirmation rescore

The old confirmation frame selection, raw source pixels and candidate parameters
stay unchanged. The new script independently reproduces the exact prior candidate
errors and also projects the **current task's declared mat** through its compiled
camera at the original 1280×720 source resolution.

| Existing fit-independent confirmation frame | Current scene mat projection RMS | Frozen conditional mat-camera RMS | Candidate passes existing 3 px gate? |
| --- | ---: | ---: | --- |
| Episode 2 / frame 82, early regime | 53.71 px | 14.54 px | No |
| Episode 15 / frame 19, late regime | 55.94 px | 30.57 px | No |

The source overlays show measured corners in white, the current scene rectangle
in red, and the conditional candidate in yellow. The current-scene error combines
camera and mat/base placement errors. The conditional camera operates in its
own mat frame; inserting that optical transform alone into the current robot task
would mix coordinate registrations. Its lower error does not establish an
accepted physical correction. Both frozen confirmation scores fail their existing
3 px gate, so neither is adopted or refitted against these confirmation images.

The available inventory still contains **12 candidate fixed-base landmarks and
zero verified physical identities**. The geometry input has zero complete measured
camera-registration rows and zero complete measured joint-registration rows.
Neither camera has independently measured source-resolution intrinsics in that
input. Successful image tracking establishes image-coordinate consistency, not
metric camera/base/joint registration. The wrist transform remains nominal CAD.

## Specific evidence needed for a physical correction

1. A known physical shoulder-lift pose relative to the robot base, paired with
   follower readback, plus independent verification poses. This identifies the
   neutral that native bounds and servo endpoints cannot identify.
2. Intrinsics/distortion for each recording camera regime and a target-to-base
   transform independent of the FK being checked, with named fixed landmarks and
   held-out captures. The existing measured camera translation triple does not
   supply orientation, source optics or a registered base transform.
3. Jaw gaps at known readbacks, including a physical closed reference and its
   uncertainty. The encoder span alone cannot establish closure or grasp TCP.

These dependencies limit acceptance of physical corrections; they do not prevent
the controlled demonstration replay and frozen-HF experiments in the current
declared scene.

Reproduce in a fresh directory:

```sh
.venv/bin/python scripts/audit_v2_registration_bridge_20261027.py \
  --out outputs/v2/registration_bridge_NEW
.venv/bin/ruff check scripts/audit_v2_registration_bridge_20261027.py
```

The executed audit passes. It checks source image hashes, reproduces the frozen
candidate RMS values to 1e−8 px, and verifies every hashed input remains unchanged
through the run. The source-image overlay was inspected visually. Ruff passes.
