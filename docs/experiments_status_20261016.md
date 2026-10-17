# Five-track experiment status — updated 17 October 2026

SO-101 remains the first physical gate. Panda experiments have not been launched in this queue. Existing safety, acceptance and success thresholds are unchanged.

## Current state

The local queue has stopped after F0. A0 through A4, every proprioception/history arm, and the matched regression/flow screen are complete. The F0 completion callback was delivered; four supplemental full-vision phase audits also completed. The runner lock is free and no training is running. No duplicate queue was launched. These are offline comparisons under the unchanged provisional FK/nominal-time dataset; the new25FPS recording has not been added and no corrected-calibration or grasp success is claimed.

The17 October [targeted root-cause follow-up](so101_root_cause_experiment_20261017.md) completed nine matched recorded-motion replays and nine stationary holds. Under the same exploratory geometry, direct joint replay lifts3/3 and places2/3; Cartesian deltas lift/place0/3; privileged reference feedback lifts2/3 and places1/3. A no-policy, no-ball-contact hold test isolates accumulating control drift: repeated zero deltas move40.7–58.6mm in20s, whereas a fixed target stays within0.35mm. The current-pose command/reference behavior is now a concrete control investigation, separate from geometry and real recording quality. Both diagnostics have completion flags; no training or hardware run is active. The five-episode paired20FPS real tracking pilot is configured but awaits operator collection.

The manager maintains `RUNNING.json`, stage logs and per-command completion records. It writes `STAGE_COMPLETE.json` or `FAILED.json` and queues this conversation on termination. `STAGE_COMPLETE` does not mean all five experiments or SO-101 grasp success. Check the callback receipt under `notifications/` before claiming delivery. The Mac must remain available; a process killed without a catchable signal cannot deliver its callback.

No Hugging Face job was launched and no compute credit was spent. The user preferred local training, then allowed Hugging Face if needed. A private input archive is prepared locally. Its upload was rejected by automatic approval review over destination-specific data-egress authorization; nothing was uploaded. Local training proceeds without that upload. Do not silently move recordings off the machine as a fallback.

## Evidence so far

Collection findings supplied by the user add an unresolved timing/provenance limitation. The current Cartesian arm targets use achieved follower FK deltas, but the gripper still uses the recorded action column. The local recorder now saves sent targets; the cached40-episode dataset has no separate requested/sent traces or physical acquisition timestamps. Uniform stored timestamps and matching video counts do not prove physical cadence or image/state synchronization. See `docs/collection_pipeline_findings_20261016.md`. A2/A3/A4 remain bounded comparisons of the unchanged nominal-time contract; obtain trace/provenance evidence before treating a longer run as a remedy.

The user's new25FPS episode has been audited separately:467 rows and467 frames per camera, visible real grasp/lift/deposit, persistent stable-target shoulder gap near5degrees during transport, and no physical timestamp trace. It has not been added to training. See `results/audit/collection_v2_25fps_20261016/review.md`; prioritize synchronized loaded-hold tracking and new-scene registration before further budget expansion.

### Completed matched backbone screen

Both models used fresh nonvisual weights, frozen pretrained vision, the same22 inner-training episodes,32,000 sampled anchors, seed0, chunk20/prefix2 and training hyperparameters. Inner validation covered2,228 unique valid actions in six episodes.

| Inner-validation metric | ResNet18 | DINOv2 | Zero motion / hold grip | Previous motion / hold grip |
|---|---:|---:|---:|---:|
| Active translation error (mm) |3.505|3.744|6.203|3.405|
| Stationary translation error (mm) |0.739|0.675|0.091|0.675|
| Closure-proxy translation error (mm) |3.640|3.693|6.724|3.112|
| Upward/closed-proxy translation error (mm) |3.351|3.605|7.387|3.139|
| Closure-proxy gripper MAE (fraction of range) |0.0949|0.1019|0.1356|0.1356|
| Training time (minutes) |9.5|33.9|—|—|

ResNet was selected by the prespecified active-motion and closure guardrails. DINO's active error is0.239mm higher; paired episode bootstrap95% interval[0.180,0.299]mm, which does not include training-seed variability. Historical validation also favors ResNet for active translation:4.644 versus5.074mm. This is a single-seed frozen-feature screen, not a general ranking of fine-tuned backbones. Neither policy beats the causal previous-motion baseline on active/closure/upward translation; this motivates the history and proprioception tests. The learned gripper predictions do improve over hold-gripper during closure.

Reports: `outputs/experiments_20261016_stage1/results/backbone_selection.json`, `stage1_review.json`, and `A1R.val.json` / `A1D.val.json` (historical counterparts alongside).

### Completed proprioception and history screen

All six arms completed2,000 optimizer steps /32,000 sampled anchors with frozen ResNet. The following are unique executed-prefix prediction errors on inner validation; phase labels are kinematic proxies, not contact annotations. Translation values are millimetres; gripper MAE is a fraction of range.

| Inputs | Active | Stationary | Approach | Closure | Upward/closed | Closure gripper |
|---|---:|---:|---:|---:|---:|---:|
| Canonical baseline |3.505|0.739|2.148|3.640|3.351|0.0949|
| No velocity |4.105|0.840|2.575|4.222|3.962|0.0840|
| EMA velocity, alpha0.5 |3.731|0.729|2.336|3.705|3.788|0.0864|
| EMA velocity, alpha0.75 |3.576|0.802|2.273|3.760|3.479|0.0851|
| Raw joints |4.248|0.869|2.576|4.115|4.216|0.0822|
| Canonical + joints |3.548|0.713|2.275|3.613|3.491|0.0971|
| Three-state history |3.851|0.783|2.402|3.791|3.902|0.0869|
| Zero motion / hold gripper |6.203|0.091|2.670|6.724|7.387|0.1356|
| Previous motion / hold gripper |3.405|0.675|2.210|3.112|3.139|0.1356|

No variant improves active translation over the canonical baseline in either split. Canonical+joints is close: inner difference+0.043mm, paired episode-bootstrap95% interval[-0.031,+0.123]mm; historical difference+0.020mm, interval[-0.033,+0.072]mm. These intervals do not include training-seed variability. Three-state history worsens active error by0.346mm inner /0.425mm historical; broad smoothing is not supported as the current remedy. Several variants improve closure gripper error, so they are not uniformly worse on every output. Preserve canonical inputs for the architecture comparisons; do not claim a general failure of history models or joint proprioception from one frozen-feature seed.

`scripts/review_so101_stage_h0.py` independently verified all seven actual observation arrays against causal recomputation, unchanged non-state arrays and image hashes, disjoint splits, checkpoint contracts/stats/hashes, training-only normalization and exclusion of terminal extrapolated action labels. All checks passed, as did24 observation-adapter/variant/queue tests. The result and paired episode bootstrap comparisons are in `outputs/experiments_20261016_stage1/results/stage_H0_review.json`. This establishes internal consistency under the provisional FK contract, not correctness of the physical calibration. Calibration changes still require jointly regenerated observations, labels and normalization.

### Completed architecture screen

All arms retain frozen ResNet, canonical observations, encoder4/decoder1, chunk20/prefix2 and32,000 sampled anchors. Batch64 means physical batch16 with four-way gradient accumulation. These are single-seed comparisons; no architecture is promoted as a physical solution.

| Arm | Batch / width / heads | Active mm | Stationary mm | Closure mm | Upward/closed mm | Closure gripper MAE |
|---|---|---:|---:|---:|---:|---:|
| A1R |16 /512 /8|3.505|0.739|3.640|3.351|0.0949|
| A2 |64 /512 /8|3.429|0.782|3.623|3.322|0.1064|
| A3 |64 /384 /8|3.524|0.808|3.503|3.366|0.1089|
| A4 |64 /384 /12|3.545|0.767|3.662|3.413|0.1018|

A2 improves active translation by0.076mm, paired episode-bootstrap95% interval[-0.133,-0.006]mm, but worsens stationary error by5.8% and closure gripper MAE by12.1%. Historical active errors are4.644/4.430/4.642/4.598mm for A1R/A2/A3/A4; these repeatedly reviewed recordings are not an untouched test. A4 versus width-matched A3 changes active error by+0.021mm inner and−0.044mm historical, so the12-head setting has no consistent advantage. Neither A2 nor A4 beats previous-motion persistence on active translation. Bootstrap intervals exclude training-seed variability. Training took10.7/7.7/7.5min for A2/A3/A4.

`scripts/review_so101_architecture.py` verified frozen-source/data hashes, unchanged splits/action semantics, train-only normalizers, actual checkpoint dimensions and32,000-anchor budgets. Report: `outputs/experiments_20261016_stage1/results/stage_A4_review.json`.

### Full-vision flow feasibility

`outputs/experiments_20261016_preflight/flow_full_vision/COMPLETE.json` records a successful Mac MPS smoke/profile with actual frozen DINO and a2.42-million-parameter head. Each objective used64 training anchors and one batch64 optimizer update; this is not a learned-policy-quality result.

Across30 timed calls per mode, p95 latency was27.2ms for regression,44.3ms for four-step flow and55.9ms for eight-step flow. Timing includes CPU image/state normalization, transfer, frozen DINO, sampling, action denormalization and CPU output transfer; it excludes camera acquisition, robot reads and controller execution. All modes met the existing200ms inference cadence in this short profile. MPS driver allocation snapshot was1.41GB, not a measured peak. Warmup/compilation make the one-update training times unsuitable for throughput comparison.

The initial launch attempt was blocked because automatic approval review hit a usage limit; it did not execute. The retry after the user's17 October continuation succeeded. F0 has now completed the planned equal-budget objective comparison. Longer convergence training still requires timing/provenance and geometry work.

### Completed F0 objective screen

Both heads use the same frozen DINO spatial features,13-state/7-action contract, chunk20/prefix2,2.42 million trainable parameters, seed0 and32,000 sampled anchors:500 updates at effective batch64. Cached-feature head training took63.1s for regression and64.6s for flow; these durations exclude the187s feature-cache job and subsequent audits. This small head is feasible on the Mac. It is not an ACT architecture and does not remove the use of action chunks.

Common full-vision audits use identical executed-prefix coverage and phase definitions for every model. Translation is in millimetres; closure gripper MAE is a fraction of range. Phase names are kinematic proxies, not independently annotated contacts.

| Inner validation | Active | Stationary | Approach | Closure | Upward/closed | Closure gripper |
|---|---:|---:|---:|---:|---:|---:|
| ACT ResNet A1R |3.505|0.739|2.148|3.640|3.351|0.0949|
| ACT DINO A1D |3.744|0.675|2.342|3.693|3.605|0.1019|
| Small DINO regression |5.253|1.826|2.832|6.185|5.565|0.0970|
| Small DINO flow,4 steps |6.483|2.634|4.087|7.911|6.461|0.1855|
| Zero motion / hold grip |6.203|0.091|2.670|6.724|7.387|0.1356|
| Previous motion / hold grip |3.405|0.675|2.210|3.112|3.139|0.1356|

Flow also underperforms zero motion on historical active translation:7.441 versus6.859mm; regression scores6.487mm and A1R4.644mm. Eight sampling steps worsen cached inner-val active error to6.769mm and closure error to8.266mm. Do not promote either F0 head or expand its training budget from this screen. This is a one-seed,500-update result, not a general verdict on flow matching. Regression versus flow matches head/data/budget; comparisons with ACT additionally differ in model capacity, loss and checkpoint selection (F0 final versus ACT best validation checkpoint).

`scripts/review_so101_flow.py` verified frozen source/data hashes, train-only normalization, disjoint splits, unchanged observation/action contracts, equal sampled-anchor budgets, feature-cache hashes and every saved frozen-backbone tensor. Reloading full-vision checkpoints reproduces cached-feature translation means within0.000019mm on the shared active/stationary/closure/upward strata. The trainer's approach proxy differs, so only the common audit's approach scores are compared above. Report: `outputs/experiments_20261016_stage1/results/stage_F0_review.json`; four audit completion flag: `results/F0_AUDITS_COMPLETE.json`.

### Collector improvement completed after F0

An optional local trace is installed in `../lerobot`: `--recording_trace_dir=...` records raw leader requests, processed requests, returned sent targets, measured follower positions and host monotonic read/send times. It preserves the user's `_sent_action` label patch and existing robot commands/limits. Rows are buffered; trace files are written between episodes. No extra motor access, hardware run or data upload occurred.

Eleven trace/auditor tests and40 existing LeRobot timer/temperature tests passed. The installed recorder/helper pass LeRobot lint; the actual recording CLI accepts the new option. Hardware timing overhead remains to be measured. Camera exposure timestamps are unavailable and are explicitly marked as such. `scripts/audit_recording_trace.py` reports timing tails and stable-command residuals, compares targets to later measurements, and can verify trace-to-Parquet sent labels. Pilot instructions are in `docs/collection_pipeline_findings_20261016.md`.

| Track | Implemented / measured | Still outstanding |
|---|---|---|
| Architecture | Matched backbone, batch, width and head screens completed; small batch64 active gain has stationary/gripper trade-offs; no consistent12-head gain | Finalist confirmation only if justified; physical evaluation |
| Geometry | Jaw-image fit, arm/roll/gripper factorial, contact and XY sweeps; common exploratory+5mm X offset plus fitted mapping lifts3/3 and places2/3 | Episode1 placement, additional-recording confirmation and independently validated calibration |
| Proprioception | Seven immutable observation variants, train-only normalization, causal streaming parity; raw noise audit and all matched comparisons completed | No active-motion improvement found; seed confirmation if a variant becomes a finalist |
| Flow head | Feasibility, equal-budget objective comparison and common phase audits complete; flow worse than matched regression and zero-motion active baseline | No promotion; a larger budget needs new evidence |
| History | Three-state causal history trained and audited; active error worse than baseline; reset and skipped-inference tick behavior tested | A different temporal design would need its own controlled comparison; no closed-loop success established |

The MPS batch16 attention profile used about 2.80–2.86 GB of current driver allocation. Measured forward/backward samples were about0.57–0.60s: no demonstrated speed improvement from SDPA in this short run, and no FlashAttention kernel was observed. These are short profiles, not peak-memory or sustained-training forecasts. See `outputs/experiments_20261016_preflight/attention_mps.json`.

The noise audit used22 inner-training episodes (25,355 raw rows), found no isolated return spikes above1° and episode-median detrended low-motion RMS of0.008–0.014° per arm joint. This does not establish correct calibration or camera/motor synchronization. Broad joint smoothing is not supported as a known repair; no-velocity and causal-velocity filtering remain controlled experiments. See `results/audit/proprioception_noise_20261016/review.md`.

## Geometry interpretation

Six original direct-arm/follower-versus-leader gripper replays produced no bilateral lifts. An exploratory147-run XY sweep produced seven lifts, all episode1, and no placements. Those original setups also contained initial robot/table penetration and cannot qualify as physically valid pass cases. The36-run support-height sweep produced no lifts.

Fitting approximate visible jaw tips on original-training episodes5/12/20, with camera and meshes held fixed, suggested a wrist-roll zero near−91°, small arm-zero corrections, and a gripper map correction. This is a calibration hypothesis from manual image evidence, not ground-truth calibration. The resulting common mapping at the original12mm base offset yields valid initial overlap, bilateral lifts in episodes1/2 and placement in episode2. Episode3 still pushes the ball away before closure.

The24-run arm/roll/gripper factorial shows that the roll correction alone is insufficient. Combined arm and roll corrections permit episode2 placement; adding the fitted gripper map also permits episode1 lift. With the full fitted mapping, all nine tested friction/damping combinations preserve the same outcome: episodes1/2 lift, episode2 places, episode3 fails. Further blind friction tuning is therefore a lower priority than the remaining registration/contact-location error.

Primary evidence:

- `results/audit/geometry_control_20261016/g1_joint_jaw_fit/report.json`
- `results/audit/geometry_control_20261016/g1_fit_replay/report.json`
- `results/audit/geometry_control_20261016/g1_mapping_ablations/report.json`
- `results/audit/geometry_control_20261016/g3_fitted_contacts/report.json`

### Follow-up calibration checks

The read-only FK audit reproduced the existing dataset before comparing mappings over all37 recordings. The full image-fit candidate shifts the tool point by20.05mm on average and changes translation labels by0.345mm mean/1.051mm p95, and rotation labels by0.001485rad mean/0.005025rad p95. A roll-only correction leaves world-frame rotation increments unchanged but still changes tool positions and translation deltas. A gripper-only mapping leaves arm FK labels unchanged. See `results/audit/fk_label_mapping_impact_20261016/review.md`.

Adding the existing arm landmarks to the fitting objective changes the selected jaw identity from near=moving to near=fixed. Its complete three-episode replay yields zero lifts, and episodes1/3 fail the initial-overlap validity check. The calibration is therefore unresolved; the earlier−91° fit remains a hypothesis, not an adopted correction. Current training deliberately retains the old contract to isolate observation-representation effects. A corrected-data training branch must regenerate FK states, velocities, actions and normalizers together after calibration is supported.

A subsequent75-run local XY sensitivity test used the full jaw-only mapping with all25 common offsets from[-10,-5,0,5,10]mm per axis. A common[+5,0]mm offset permits valid bilateral lifts in all three recorded examples and complete placements in episodes2/3; episode1 still fails placement. Several neighboring common offsets also lift all three, but none places all three. This is an outcome-tested exploratory region, not independent validation of the ball location or calibration. Detailed outcomes and the still-false physical/Panda gate are saved in `g2_fitted_xy_local/summary.json` and `results/audit/geometry_control_20261016/gate.json`.

The complete episode1 trace for[+5,0]mm loses the held flag at29.2s during transport. At that tick tool tracking error is0.259mm, with no joint-limit clipping or rate limiting. Bilateral contact lasts14.176s continuously; longest bilateral lift is5.9s, maximum supported height165.3mm, and maximum measured ball-contact penetration1.91mm. Shoulder clipping occurs on167 other ticks, so it remains a separate trajectory-coverage concern, but it does not explain that loss-of-contact tick. Next inspect jaw contact location, orientation-dependent support/slip and transport alignment; do not relax the safety limits. Trace: `g2_common_offset_ep1_trace/report.json`.

## Resume after the training callback

1. Verify current process/lock state and terminal flags before starting any job. Review inner-val active, closure, upward-motion and stationary errors against zero-motion and persistence; keep historical validation explicitly separate.
2. Diagnose episode3 jaw/ball alignment and episode1 transport/release under the fitted mapping. Validate the mapping on additional recordings; retain raw annotations and do not choose each episode its own successful offset.
3. Quantify how the candidate mapping changes FK observations and Cartesian labels. The tool site is off the wrist-roll axis, so a constant roll-zero correction can change position and translation deltas even when world-frame rotation increments cancel. Small upstream arm-zero changes can affect both. Regenerate a separate, hashed dataset only after choosing a supported mapping; never mutate data under the running queue.
4. F0 quality review is complete and does not support promotion. A1 through A4, P1/P2/P3/P4/H0 and F0 should not be repeated. Use the installed optional collector trace for five20FPS pilots before collecting50 episodes. Audit timing/labels and loaded-versus-unloaded holds, then perform a matched baseline on a separately derived dataset after the geometry contract is supported. Confirm finalists with multiple seeds only if warranted.
5. Require common-parameter complete recorded grasps, controller tracking and learned SO-101 closed-loop success before Panda. Process completion and lower prediction loss do not open this gate.
