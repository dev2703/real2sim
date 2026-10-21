# Focused v2 measurement session

The recorded wrist view is much more useful than v1's ceiling view: it sees the gripper
and nearby ball. That does not yet establish its optical transform or placement visibility.
The iPhone pose also needs a new calibration. Preserve the stored wrist key
`observation.images.top` throughout this work.

Copy `configs/v2/measurements.template.json` to a new versioned measurement file; keep
unknown values null. Do not copy old camera positions into `task.calibrated.yaml` and label
them measured.

On 21 October the operator confirmed **motor calibration, base and cameras unchanged since
v2 recording**. The statement and current calibration hash are saved in
[`setup_provenance_20261021.json`](../configs/v2/setup_provenance_20261021.json).
This permits checking the old recordings with current measurements; it does not establish
their original command or exposure timestamps.

The operator also confirmed **standard SO-101 hardware and no spacer** and requested use of
the public hardware design. The [geometry report](v2_geometry_alignment_20261021.md#standard-hardware-and-no-spacer-update)
now documents the official gripper mapping caveat, direct horn attachment, nominal 2.4 mm
native-origin support offset, and the resulting static diagnostic. Manual jaw measurements
are not a prerequisite for that nominal diagnostic. Exact physical closure and image
correspondence remain verification work; do not substitute nominal values for measurements.

1. Use the reviewed conditions in `annotations.reviewed_v2.json`: 32 baseline, ball-shift
   episodes 6, 8, 9, 10, and bowl-shift episodes 21–25. The operator accepted that only four
   ball shifts may have been saved. Both shifts were toward the robot, without a sideways
   shift; the metric vector in the task frame still needs registration. The 29/6/6 split is
   saved in `results/v2/split.json`. Review approach, grasp, lift and release source-frame
   boundaries, corrections and bowl contacts, retaining episode 9's bowl-tipping note.
   Do not refit geometry using the reserved episodes.
2. Record calibration-board views for each camera covering the working field of view.
   Keep the iPhone fixed. Move the wrist through multiple ordinary arm poses while the
   board stays registered to the table. Estimate intrinsics/distortion and the rigid
   link-to-optical transform. Reserve independent landmark views for verification; report
   errors in original1280x720 pixels, not downsampled policy pixels. The simulator's present
   camera interface is a pinhole FOV plus rigid pose; if measured distortion/off-centre
   intrinsics prevent the10px p95 gate, implement/version rectification before preparing
   policy images. Do not compensate with guessed extrinsics.
3. Use the confirmed no-spacer support geometry; verify the remaining base/table registration,
   ball diameter, bowl inner/outer dimensions,
   depth and rim height. At five ordinary jaw openings including closure, record requested,
   sent and measured joint values and unloaded jaw separation at the same identified contact
   surfaces. Suggested commands are 0, 15, 30, 45 and 60%; reserve 15 and 45% for verification
   before fitting and repeat each reading. Record actual readback percent, not only requested
   percent. Use only ordinary reachable openings; do not force a mechanical stop. The encoder
   span implies 129.407 degrees per 100% under a direct-drive hypothesis, while the old mapping
   uses 110.002 degrees. Verify sign, transmission ratio and closure before adopting that scale.
   A full 100% test is unnecessary. Do not infer jaw shape from a single failed grasp.
4. Record exactly five short traced runs at nominal25Hz: two unloaded rehearsals, two
   matching loaded runs, and one normal task. Briefly hold two ordinary transport poses.
   Use the existing collector trace flag `--recording_trace_dir <new-session-root>` with
   the usual recording command. The integration is described in
   [the trace README](../integrations/lerobot_trace/README.md).
5. Preserve collector/calibration versions, units, requested/processed/sent commands,
   measured states and host observe/send timestamps. The current trace helper does **not**
   supply camera exposure times or frame IDs. Record camera acquisition/frame timestamps
   where the camera interface exposes them, and label host receipt times separately.
   No new trace can retroactively establish timing or command provenance of the41 old runs.

Read-only checks, after those traces exist:

```sh
.venv/bin/python scripts/audit_recording_trace.py PATH_TO_SESSION \
  --dataset PATH_TO_NEW_TRACED_DATASET --policy-hz 12.5 \
  --output results/v2/trace_audit_VERSION.json
```

For geometry, fill a versioned copy of `configs/v2/geometry_landmarks.template.json`.
`scripts/fit_v2_geometry.py` has `camera`, `joint-zero` and `wrist` modes. The wrist mode
requires separately verified link poses and keeps them fixed. Commands, evidence and
acceptance gates are in the [geometry alignment report](v2_geometry_alignment_20261021.md).

A valid later task start is a change in task definition. Verify it precedes the approach
and preserves grasp/lift/placement, then record its source episode/frame and version.
Evaluate the original folded start separately. Do not widen joint limits to force a pass.
