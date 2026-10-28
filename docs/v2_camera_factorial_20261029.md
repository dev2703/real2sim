# Camera-input interaction audit — 29 October 2026

The fixed-joint camera probe now preserves exact resized RGB inputs and complete
native ACT output chunks. Its full 2×2 image swaps show that the effect of one
camera depends strongly on the other camera. This improves attribution and
reproducibility; it does not establish a physical camera correction.

All 18 previous validation anchors were rerun without exclusions, using the same
training-only yaw hypothesis and frozen HF weights. Every previous prefix-error
metric reproduced exactly. Lighting, scene dimensions, camera parameters and
weights remain unchanged. The [curated audit](../results/v2/hf_alignment_factorial_20261029.json)
links the complete report and its SHA-256.

| Paired input effect; mean absolute change across five arm joints | Nominal render | Training-yaw render |
| --- | ---: | ---: |
| Swap wrist while global stays recorded | 2.880° | 1.661° |
| Swap wrist while global stays rendered | 8.667° | 5.101° |
| Swap global while wrist stays recorded | 8.813° | 8.364° |
| Swap global while wrist stays rendered | 11.829° | 10.561° |
| Two-camera interaction | 6.779° | 4.426° |

The interaction is computed on each signed action coordinate before absolute
aggregation: `both_rendered − wrist_only − global_only + both_recorded`.
These values measure changes from another model prediction, so they differ from
error against the recorded command. They retain the saved native execution prefix
and equal weight per anchor. Gripper percentages are reported separately.

The global image remains the stronger individual intervention in these anchors,
but the substantial interaction prevents treating the two image errors as
independent contributions. The probe changes optical pose, scene layout,
occlusion, background and rendered appearance together. Selecting a camera pose
by action error or task success would not identify which transform is physically
correct, and no such correction is adopted.

Every anchor's hashed NPZ retains the three unique RGB input pairs, all seven
prediction chunks, the real joint vector and the recorded action reference.
Mixed camera conditions reuse these saved images exactly. The audit recomputed
every prefix metric and factorial effect from the NPZ outputs, checked every
input-image hash, and checked all artifact hashes. Identical-input predictions
remain exactly equal, and no watched source changed during execution.

Container timestamps and source-row timestamps agree within 0.2 µs in these
anchors. This checks nominal indexing only: physical exposure-to-motor timing is
still unverified. The saved per-camera presentation timestamps expose that
distinction without fitting a lag from ACT error.

Independent measured camera optics and camera-to-base registration remain the
next calibration evidence required. The existing
[physical session](v2_physical_capture_20261027.md) supports a ruler-marked target
and independent verification captures; its prepared inputs still lack actual
captures and measurements. Repeated views of the static mat cannot resolve the
full intrinsic/base-registration ambiguity.

Reproduce the complete diagnostic with:

```sh
.venv/bin/python scripts/diagnose_hf_alignment_inputs.py \
  --out outputs/v2/hf_alignment_factorial_NEW
```

Nine focused tests pass, covering additive and interacting views, signed-effect
cancellation, scored-prefix isolation, invalid outputs and missing anchors. Ruff
passes. The initial sandboxed attempt could not open the macOS offscreen graphics
context and evaluated zero cases; its partial report is marked aborted. The
completed separate attempt obtained local graphics access and evaluated all 18
anchors.
