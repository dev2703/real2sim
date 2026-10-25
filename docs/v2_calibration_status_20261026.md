# Calibration capture status — 26 October 2026

**Capture inputs are unavailable for this run; physical calibration and calibrated
retraining remain pending.** The operator initially reported local recordings, then
replied “not available ig” when asked for their folder. No new source path has been
linked. This does not establish that recordings are absent elsewhere.

The fresh [capture-readiness inspection](../results/v2/calibration_capture_readiness_20261026.json)
checks the existing printed-board plan. It finds zero source-backed fitting or
verification views. The [calibration-contract audit](../results/v2/calibration_contract_audit_20261026.json)
checks the current geometry/measurement templates against the cached dataset and
encoder calibration files. It finds zero complete static landmarks, joint landmarks,
jaw measurements or loaded-trace records. Neither audit performs optimization or
establishes physical acceptance.

The targeted search covered hidden/ignored files in this workspace, the sibling
`/Users/deveshvyas/Downloads/lerobot`, and
`/Users/deveshvyas/.cache/huggingface/lerobot`, excluding Git and installed-package
internals. No `SESSION.json` recording trace or completed new measurement document
surfaced. The existing dataset is
`/Users/deveshvyas/.cache/huggingface/lerobot/local/so101_act_pickplace_v2`:
41 episodes, 27,736 frames at nominal 25 Hz. Its 16 MP4 and 17 parquet files are
existing demonstrations, not newly identified calibration captures. Existing scalar
measurements and encoder calibrations remain useful evidence with their previously
documented limitations.

## Work completed and remaining evidence

| Bottleneck | Current status and next required evidence |
|---|---|
| Geometry and applicability | Capture plan exists; optics/base/joint/jaw correspondence is unverified. Record measured target pitch, both cameras' settings, every camera/board/mat move, independent base registration and evidence connecting each new setup to the old footage. Reserve whole verification groups before fitting. |
| Timing and action meaning | Trace auditing is available. The current collector supplies host observe/send timing but no exposure timestamps or camera frame IDs. Retain requested, accepted/sent and measured states separately; record acquisition timing, clock alignment and uncertainty. A send return does not by itself prove motor application. |
| Executable actions | Existing provisional-scene rejection results remain diagnostic. Compare requested action → accepted/applied command → achieved motion after registration, retaining every rejection and its reason. Provisional scene failures do not establish real collisions. |
| Loaded physics | No new measured loaded comparison is available. Record jaw gap/TCP, object mass/dimensions/support height, loaded and unloaded holds, relative ball/tool displacement, release onset/separation and independent verification repeats. Simulator retention sensitivity is not physical verification. |
| Feedback and latency | Recorded-observation errors and inference-only timing remain insufficient. Capture complete acquisition-to-action timing and genuine feedback rollouts, including rejections and accumulated error. Label host-only timing explicitly when exposure/application timing is unavailable. |
| Coverage and generalization | The existing three-episode capacity gate still fails. After justified calibration/label changes, repeat the small fit and movement/stationary/object-action checks, then broaden conditions, bowl-directed cases and seeds under the existing compute budget. Keep reserved episodes separate. |

The [new blank session log](../configs/v2/calibration_capture_20261026.template.json)
collects these inputs together. Nulls remain missing evidence. The
[capture sheet](v2_calibration_capture_20261025.md) provides the physical sequence;
use the new log to retain setup changes, timing, loaded response and version lineage.

After measurement verification, regenerate affected FK labels and training-only
statistics into a fresh data contract. Keep geometry, joint mappings, camera
transforms/preprocessing, normalizers and checkpoint hashes together. Train against
that contract and evaluate with matching inference geometry. Recovery and Panda
transfer remain subsequent stages after SO-101 policy and control verification.

No calibration fit, label replacement, training job or physical rollout was performed
in this run. The read-only audits completed successfully. Thirty focused tests passed
for checkerboard calibration, readiness arithmetic, recording-trace auditing and
integration, and FK regeneration. These checks verify software behavior, not physical
calibration or policy success.
