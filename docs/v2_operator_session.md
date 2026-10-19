# Focused v2 measurement session

The recorded wrist view is much more useful than v1's ceiling view: it sees the gripper
and nearby ball. That does not yet establish its optical transform or placement visibility.
The iPhone pose also needs a new calibration. Preserve the stored wrist key
`observation.images.top` throughout this work.

Copy `configs/v2/measurements.template.json` to a new versioned measurement file; keep
unknown values null. Do not copy old camera positions into `task.calibrated.yaml` and label
them measured.

1. Confirm the five ball-shift episode IDs, the bowl-shift IDs (21–25 are a visual candidate),
   and both shift vectors in a declared task coordinate frame. Intended totals are31/5/5.
   The extra episode was confirmed as baseline, but its index and the remaining ordering
   cannot be inferred from that answer. Copy `annotations.template.json`; review approach,
   grasp, lift and release source-frame boundaries, corrections and bowl contacts. Retain
   episode9's bowl-tipping note. Separate fitting, validation and reserved recordings only
   after those labels are confirmed.
2. Record calibration-board views for each camera covering the working field of view.
   Keep the iPhone fixed. Move the wrist through multiple ordinary arm poses while the
   board stays registered to the table. Estimate intrinsics/distortion and the rigid
   link-to-optical transform. Reserve independent landmark views for verification; report
   errors in original1280x720 pixels, not downsampled policy pixels. The simulator's present
   camera interface is a pinhole FOV plus rigid pose; if measured distortion/off-centre
   intrinsics prevent the10px p95 gate, implement/version rectification before preparing
   policy images. Do not compensate with guessed extrinsics.
3. Measure robot-base/table registration, ball diameter, bowl inner/outer dimensions,
   depth and rim height. At five ordinary jaw openings including closure, record requested,
   sent and measured joint values and unloaded jaw separation. Include repeated/held-out
   openings to verify the fitted mapping. Do not infer jaw shape from a single failed grasp.
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

A valid later task start is a change in task definition. Verify it precedes the approach
and preserves grasp/lift/placement, then record its source episode/frame and version.
Evaluate the original folded start separately. Do not widen joint limits to force a pass.
