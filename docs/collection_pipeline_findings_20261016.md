# Collection findings and implications for SO-101 experiments

The user supplied collection-test evidence on16 October2026: requested/safe target discrepancies up to60–80degrees; measured rates28.15Hz for nominal30Hz and23.69Hz for nominal25Hz; software encoding at18.53Hz with queue drops; and a completed hardware-encoded30FPS test with563 data rows and563 frames per camera. Those test outputs have not yet been linked to the immutable dataset used by the current queue. Do not assign the reported original11 episodes to this dataset without provenance.

The user subsequently identified the new one-episode source as `~/.cache/huggingface/lerobot/local/so101_act_pickplace_v2_25fps`. Its read-only audit found467 rows and467 decoded frames per camera, nominal timing only, and a visibly successful grasp/lift/deposit sequence. Stable-target windows expose a persistent shoulder gap near5degrees during nominal8.60–11.60s and an elbow gap near3.16degrees during10.20–11.44s. These deserve a synchronized loaded-hold diagnostic; they do not identify a cause or justify changing limits. This new source is separate from the40-episode30FPS training dataset and cannot use its every-third-row10Hz conversion. Detailed review: `results/audit/collection_v2_25fps_20261016/review.md`.

## What the local code and data establish

- The current local LeRobot recorder saves `_sent_action`, returned by `robot.send_action()`, in `action`. The SO follower implementation applies its relative-target limiter before writing `Goal_Position` and returns that limited target. This is a command, not a readback or proof that the motor achieved it. Source: `../lerobot/src/lerobot/scripts/lerobot_record.py` and `../lerobot/src/lerobot/robots/so_follower/so_follower.py`.
- The dataset writer still assigns `timestamp = frame_index / self._meta.fps`. Existing datasets lack per-tick acquisition/send times. An optional sidecar trace was installed on17 October; it records host read/send timing for future captures when explicitly enabled, without changing the nominal dataset timestamp. Source: `../lerobot/src/lerobot/datasets/dataset_writer.py` and the recorder above.
- The queue uses a pinned40-episode,47,393-row, nominal30FPS source, `bottowork/so101_act_pickplace_v1`, revision`fd65a6600b0cd111a434ece1abfa3d44dc7b4d57`;37 episodes survive existing frame-integrity exclusions. Its local Parquet schemas contain one `action`, measured `observation.state`, nominal `timestamp`, and row/episode/task indices. No separate requested/sent targets or acquisition timestamps are present. No timing/trace/log/clock-named sidecars were found under the cached dataset root. External recording logs may still exist.
- The current recorder's patch does not establish which action semantics were used when those40 episodes were collected. Dataset revision, collector commit, configuration and per-episode recording session must establish that provenance.

## Impact on the current policy

The running Cartesian ACT comparisons do **not** train the first six outputs on leader arm targets. `src/real2sim/data/prepare.py` passes FK of measured follower states to `episode_labels`; `src/real2sim/data/labels.py` constructs the next-tick achieved Cartesian translation/rotation. The recorded arm actions are retained for diagnosis. Raw-joint-input ablations also keep these same Cartesian output labels.

Thus saving the wrong leader arm target does not directly contaminate those six outputs in the same way as the original joint-action baseline. It can still affect the demonstrated trajectory through clamping, lag or saturation. Matching a requested/safe warning does not diagnose physical motor tracking: compare the sent target to later measured positions, allowing for response delay, especially during sustained holds.

The seventh Cartesian output **does** use the source action's gripper value. Its manifest calls that value `leader_action`, but the underlying recording semantics need provenance verification. A pre-/post-patch mixture would therefore matter directly for the gripper. Replacing commands with measured gripper opening is not automatically a repair: contact can prevent closure even while a closed target is needed to maintain force. Keep command and observed opening separately identifiable.

The converter selects every third row and computes velocity as displacement times10Hz. This is exact in nominal row time only. The opening comment in `src/real2sim/data/resample.py` says the recording clock was audited as uniform; that wording overstates what the evidence establishes. **Only stored timestamp spacing was audited. Physical acquisition cadence remains unknown.** The frozen file is left unchanged during the running queue; correct its documentation in a subsequent source version.

If the reported28.15Hz applied uniformly to a nominal30Hz recording, three source intervals would span106.57ms, not100ms, and velocity inferred with the nominal interval would be6.57% too large. The25/23.69 ratio implies5.53% under the same assumption. These are illustrative consequences, not estimates for the current40-episode dataset. A mean-rate correction cannot reconstruct irregular stalls. Do not globally rescale, smooth or time-shift the current labels from these unrelated aggregate measurements.

Equal data/video counts and valid H.264 establish useful encoding integrity. They do not establish camera exposure age, frame freshness or camera/state synchronization. Existing nominal PTS checks remain useful but have the same limitation if the videos also use an artificial constant frame clock.

## Next diagnostic, ahead of longer training

1. Identify recording sessions and action semantics for the current dataset and the user's original11 episodes. Keep pre-/post-patch datasets and nominal25/30FPS sessions separate unless explicitly converted under a documented common contract.
2. For the next authorized hardware collection, record one trace row per control tick: episode/frame ID, raw leader request, postprocessor request, actual returned sent target, measured follower positions, monotonic observation read start/end and command send start/end, and each camera's frame ID/acquisition time when available. Record timestamp clock domain and capture-versus-retrieval semantics; do not invent capture times for APIs that only expose retrieval time. Buffer trace writing and measure its overhead. Preserve limiter settings and record collector commit, calibration and action-semantics version.
3. Compare requested-minus-sent separately from sent-minus-measured. Observation is read before sending that tick's target, so use later measurements and a response window; do not label their same-row difference a steady-state tracking failure. Summarize sustained holds, active approach, closure and lift separately. Existing limits and ordinary authorized task motions remain unchanged; no hardware test was started here.
4. With actual timestamp traces, report observation/send interval median, p95, p99, max; frame age/repeats; stalls; and tracking settling behavior. Regenerate a separate uniform10Hz dataset from supported timing, keep image selection causal to the observation time, and do not interpolate across long unobserved stalls or episode boundaries. Fit normalization on its training episodes only.
5. Run a matched canonical ResNet baseline on that trace-verified dataset, after checking the geometry contract. This controlled comparison is a higher priority than expanding training budgets or concluding that ACT architecture is the main cause.

Current A2/A3/A4 comparisons can finish as bounded comparisons under the same provisional contract. Their results cannot validate wall-clock alignment, physical tracking, calibrated contact or learned grasp success. No queue was duplicated, data changed, safety threshold relaxed, hardware action sent or Panda experiment launched in response to these findings.

Update17 October: all bounded training comparisons, including F0, have finished. No additional training is running. Flow did not beat its matched regression head or the zero-motion active baseline; see `docs/experiments_status_20261016.md`.

## Proposed next collection:20FPS pilot, then scale only after review

The user reports an approximately23Hz average achievable loop and is considering50 new episodes at nominal25FPS or20FPS. Recommend testing20FPS first: its50ms budget offers more headroom than25FPS's40ms budget. A23Hz average does not guarantee either a stable20Hz loop or bounded stalls. If25FPS data actually arrive uniformly at23Hz, nominal-time velocity is about8.7% too large; irregular gaps cannot be repaired with that aggregate ratio. This is an illustrative calculation, not a measured correction for any saved episode.

Before a50-episode session, add the per-tick timing/command trace described above, then collect five short pilot demonstrations at a fixed20FPS setting with hardware encoding and fixed cameras/calibration. Include brief holds at ordinary approach, grasp and transport poses; compare ordinary unloaded and ball-loaded behavior. Log the exact limiter configuration—the user reports having relaxed it—without increasing it further as a data-quality remedy. No code here changes hardware limits or collection settings.

Review the pilots for actual cadence tails and stalls, video freshness, correct sent-action labels, and requested→sent→subsequent-measured tracking. A persistent loaded residual may be a controller/load limitation; large requested→sent differences with good subsequent tracking instead point to operator speed/limiting. Neither observation alone establishes a defective motor. The successful25FPS episode shows that nonzero tracking error can coexist with useful real demonstrations.

Keep the real2sim policy at10Hz and chunk20/prefix2 for a controlled new-data comparison.20FPS permits nominal2:1 downsampling; acquisition timestamps must still support the conversion. Do not change policy cadence or reinterpret a20Hz action chunk as10Hz. Preserve measured follower states and sent targets separately so Cartesian achieved-motion labels and gripper command labels retain their intended semantics.

Once five pilots pass the timing/label audit and establish reproducible task behavior, collect the50-episode set with a predefined episode-level split (for example40 training/10 validation), controlled variation in object starts and approaches, unchanged camera placement, and explicit session provenance. A small offline memorization check can expose a broken training path, but is not a physical success gate. Preserve old30FPS and experimental25FPS sources separately for comparison; do not silently append mixed rates, limiter settings or action semantics. Geometry replay and learned SO-101 evaluation remain required before Panda.

## Installed opt-in trace and pilot commands

The actual collector in `/Users/deveshvyas/Downloads/lerobot` now accepts `--recording_trace_dir`. It remains disabled by default. Keep your working robot/teleoperator IDs, ports, calibration, cameras, hardware encoder and limiter configuration. For the five-episode pilot, replace the corresponding dataset options in your existing recording command with:

```sh
--dataset.repo_id=local/so101_act_pickplace_v3_20fps_trace \
--dataset.fps=20 \
--dataset.num_episodes=5 \
--dataset.push_to_hub=false \
--recording_trace_dir=/Users/deveshvyas/Downloads/real2sim/results/collection_traces
```

Use a fresh dataset root as well if your existing command sets `--dataset.root`; do not point it at the old25FPS dataset or resume that dataset. The fragment above is for insertion into your existing full command, not a standalone hardware command. No collection was started by the agent.

Each invocation logs a unique `session_UUID` directory. `SESSION.json` records nominal FPS, IDs, units, limiter, collector source hashes, calibration hashes and camera settings. Episode files distinguish saved, discarded and interrupted attempts. `COMPLETE.json` means trace-session closure, not task success or finalized-video validation; `FAILED.json` marks an interrupted capture. A hard process kill can lose the current in-memory attempt; `RUNNING.json` remains evidence of an incomplete trace. Existing completed episodes are preserved.

After collection, use the emitted session directory and the matching dataset path:

```sh
cd /Users/deveshvyas/Downloads/real2sim
.venv/bin/python scripts/audit_recording_trace.py \
  results/collection_traces/session_UUID \
  --dataset /Users/deveshvyas/.cache/huggingface/lerobot/local/so101_act_pickplace_v3_20fps_trace \
  --output results/audit/collection_v3_20fps_trace.json
```

The audit retains real stalls, excludes discarded attempts, checks row IDs and saved sent-action labels, and summarizes requested→sent differences separately from prior-sent→measured residuals. Stable-target windows use a configurable0.5 command-unit span,0.25s response window and0.5s minimum duration; these are analysis settings, not changes to hardware or acceptance thresholds. Interpret arm/gripper units from session metadata. Sustained commands do not prove the mechanism has settled; compare loaded and unloaded holds and inspect the raw trace before attributing residuals to a controller or motor.

Camera frame IDs/exposure times remain unavailable through the current observation interface. `frame_index` is a dataset row ID, and observation start/end bracket a host API call; neither establishes fresh image capture. Continue video frame-count/PTS/freshness review separately. The trace makes timing and command semantics testable; it does not by itself resolve geometry or physical synchronization.

Validation on17 October:11 trace/auditor tests passed, including a synthetic long stall, delayed feedback, discarded/interrupted captures and rejection of requested-action labels.40 existing LeRobot cycle-timer/temperature tests passed using the actual source; the collector CLI and lint passed. No hardware behavior or trace overhead has yet been measured. The local patch/snapshots are under `outputs/collection_trace_patch/`; maintained helper and patch builder are under `integrations/lerobot_trace/`.
