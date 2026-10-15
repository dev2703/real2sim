# SO-101 geometry, proprioception and architecture experiments

Status: proposed experiment plan, 15 October 2026. This document does not launch training or change the data/model contracts. Architecture pilots can proceed offline while the physical replay tests are repaired. Panda transfer is a later gate.

## Evidence and common protocol

The [control audit](../results/audit/control_findings_20261015/review.md) establishes an IK convergence defect that is now fixed in isolated configs. It also shows submillimetre simulated tracking at first contact, yet premature unilateral contact and no held lifts in all 15 complete diagnostic replays. The tracking reference and Cartesian labels share the same provisional FK: good simulated tracking does not independently establish real jaw alignment.

The [DINO pilot review](../results/metrics/dinov2_act_pilot_20261015/review.json) reports a 3.53% improvement in overlapping full-chunk validation L1 and lower paired real/sim prediction disagreement. The [executed-prefix audit](../results/audit/motion_prediction_20261015/review.md) instead gives active-motion translation error of 4.31 mm for ResNet and 4.41 mm for DINO, versus 6.86 mm for zero motion. Stationary error is 0.75/0.54 mm respectively. DINO is a candidate, not an established task-performance winner. Its previous run also changed fine-tuning, learning rates, regularization and paired-image exposure, so it is not a clean backbone-only comparison.

Common protocol for new comparisons:

- Retain immutable raw recordings and named derived variants with hashes. Fit normalizers and preprocessing parameters on training episodes only.
- Use an inner whole-episode split of the 28 training episodes for new selection; keep the existing nine validation episodes for historical comparison. They have been repeatedly inspected and cannot now be described as an untouched test set. Final confirmation needs fresh recordings or independently reserved cases, plus unseen simulated scenarios.
- Keep 10 Hz, chunk length 20, executed prefix 2, image processing and action semantics fixed except in experiments explicitly testing those factors.
- Use the verified IK convergence configuration consistently for new control comparisons. Preserve acceptance, workspace, penetration, velocity, action and task-success thresholds.
- Score each valid executed-prefix action once. Report active/stationary and approach/closure/lift errors, translation and rotation, gripper timing, amplitude and direction, p95 error and integrated drift. Include zero-motion/hold-gripper and persistence baselines.
- Screen with one seed, confirm finalists with three seeds and episode-level uncertainty. An initial learning screen can use 32,000 sampled anchors: 2,000 updates at effective batch 16 or 500 at effective batch 64. Match validation and LR-schedule progress by examples seen, then confirm finalists with a larger equal-example budget. Equal update counts alone give batch 64 four times the exposure.
- Closed-loop promotion requires repeatable bilateral contact and held lift under physically supported settings, followed by complete placement scoring. Add bilateral-contact duration, normal force, slip and penetration diagnostics; retain the official success definition. The current `ball_held` flag alone can also count one-jaw contact above the table.

## 1. Architecture: batch 64, four encoder layers, one decoder, twelve heads

The [existing ACT](../configs/act_cartesian.yaml) and [DINO ACT](../configs/act_cartesian_dinov2_pilot.yaml) already use four ACT encoder layers and one decoder layer. The DINO visual backbone is a different transformer: DINOv2-small has 12 layers, width 384 and six heads, with its last two layers trainable in the existing pilot.

The current ACT width is 512 with eight heads. Twelve heads require a divisible width. Propose ACT width **384 with 12 heads** (32 features per head), and a width-384/eight-head control. Width 768 is also compatible but is a larger laptop experiment, not the first candidate. Retain feed-forward width 3200 during this comparison so that another dimension is not changed silently. Keep DINO's pretrained internal dimensions/head count unchanged.

Changing ACT width prevents strict reuse of the existing width-512 nonvisual checkpoint. Train compared width variants with fresh nonvisual weights and matched pretrained vision initialization; do not silently partially load incompatible tensors. The current `n_heads` setting is shared by ACT's policy encoder, decoder and VAE encoder, so a global 12-head experiment changes all three. An encoder-only head-count experiment would need a separate setting.

Hardware inspected locally: Apple M3 MacBook Air, 16 GB unified memory, PyTorch 2.11.0. This installed build has MPS SDPA support and no MPS kernel for `aten::_scaled_dot_product_flash_attention`. The DINO wrapper already selects SDPA. ACT currently requests attention weights and discards them; using `need_weights=False` is the first attention optimization to evaluate. PyTorch documents that this enables its optimized SDPA path. This is an execution/memory experiment, not an assumed model-quality gain. [PyTorch MHA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.MultiheadAttention.html), [SDPA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html).

Apply the optimized path to all four ACT encoder layers; separately include decoder and VAE attention for consistency. Check outputs and gradients with dropout disabled, padding masks, checkpoint reload and real MPS forward/backward profiling. Do not label an MPS run “FlashAttention” merely because it calls SDPA. Actual CUDA/ROCm FlashAttention belongs to a compatible-GPU comparison. [Official FlashAttention support](https://github.com/Dao-AILab/flash-attention).

| ID | Change from control | Question |
|---|---|---|
| A0 | Existing attention → SDPA path, same weights/shapes | Is memory/time improved without changing semantics? |
| A1 | ResNet18 versus DINOv2-small, ACT 512/8, effective batch 16 | Does DINO improve active/phase errors under matched training? |
| A2 | Winning backbone, effective batch 16 → 64, ACT 512/8 | Does larger batch help at equal exposure? |
| A3 | Winning backbone, ACT 384/8, effective batch 64 | What is the width reduction's effect? |
| A4 | A3 with ACT 384/12 | Does the requested twelve-head variant help? |

Keep data and paired-image exposure identical in A1. Use fresh nonvisual initialization in both arms. Freeze both visual backbones for a cheap screening comparison if useful; identify that as a frozen-feature experiment and confirm a finalist with matched fine-tuning budgets. Do not compare a warmed-up DINO ACT head against a fresh ResNet head.

Profile physical batches 16, 32 and 64 before choosing one. The practical initial batch-64 configuration is microbatch 16 × accumulation 4, already supported by the trainer; 8 × 8 is a fallback. Accumulation gives an effective optimization batch of 64, not the activation footprint or necessarily the exact numerics of a physical batch of 64. Do not automatically multiply LR by four. Record actual memory, throughput and end-to-end inference latency; optimizer/scheduler steps must follow accumulated updates. The previous DINO pilot took about 36.7 minutes for 2,000 batch-16 updates, which is a historical reference, not a forecast for new variants.

Promotion: improvement in active/phase prediction without worse closure timing or excessive stationary drift; once geometry passes, improvement in matched physical rollouts. Use cost and confidence intervals to break small offline differences.

## 2. Make geometry and contact work

**G0 — complete the arm/gripper control comparison.** Use the same recorded follower arm trajectory and test (a) next follower opening, (b) recorded leader gripper command, and (c) a calibrated actuator command if independent calibration supports it. Existing `cartesian_leader_gripper` has Cartesian arm drift and does not isolate this factor. A follower jaw opening while grasping is an achieved, loaded state; targeting that opening may fail to recreate the squeezing force. The leader/follower gap during a grasp is not automatically bad data.

**G1 — independently verify registration.** Annotate visible robot joints and both jaw surfaces across several poses/openings, with measured base/mat/table dimensions where available. Estimate camera/base placement and joint zeros in constrained stages; test on other poses and episodes. Compare real and rendered jaw clearance at empty closure, open positions and loaded closure. Do not equate a held ball's centre with the tool origin. Correct image reprojection alone is insufficient if the camera can compensate for wrong joint zeros.

The numerical sim gripper map is approximately −10° to +100°, spanning 110°. Local follower calibration spans about 129.4° over 0–100%, but those motor endpoints do not establish the physical jaw zero or model range. Some existing prose has an incorrect +110° endpoint; use the numeric config and independent aperture evidence, not that comment, when designing calibration.

**G2 — bounded geometry sensitivity.** After fixing registration, vary ball XY within estimated image/calibration uncertainty. Then test collision geometry alternatives supported by the real jaw surfaces, one family at a time. A ±30 mm coarse XY sweep may be exploratory, but a success found there is not proof of the correct calibration. Refine a robust region and validate independently, rather than fitting each episode its own successful offset.

Do not sweep free ball Z independently: a ball placed above the support falls, while one below it begins in penetration. Height uncertainty must be represented through measured support/mat height, base placement or radius with a consistent resting ball position. Reject invalid initial overlaps. Do not arbitrarily move collision surfaces away from the visible jaws or attach the ball.

**G3 — contact physics after alignment.** Start with the existing diagnostic friction values 0.3/0.6/1.0 and damping ratios 0.7/1.0/1.3. Keep nominal mass/radius unless measurements support alternatives. Check force, slip and ball displacement before closure. Extreme softness/friction that rescues a miss is not an acceptable physical calibration.

Gate: reproduce complete known successful recorded grasps with bilateral closure, held lift and release, using common physically plausible parameters. Initial milestone: all three reviewed examples under the same model; then confirmation on additional recordings. Failure here still leaves registration, contact and command/load semantics in play, independently of ACT.

## 3. Proprioception and cleaning/smoothing

The [data audit](data_contract.md) and current manifests show exact raw/derived parity, uniform row timestamps and zero Cartesian jump flags. These checks do not exclude small encoder noise or image latency, but gross joint-data corruption is not demonstrated. Current processing takes every third 30 Hz sample, then computes FK and a backward 10 Hz linear velocity; it does not low-pass before decimation.

Test these with the same images, Cartesian action targets, backbone and training budget:

| ID | Observation change | Hypothesis |
|---|---|---|
| P0 | Existing 13-D FK pose, velocity and gripper | Baseline |
| P1 | Remove three velocity values; retain 10-D pose/gripper | Differentiated noise or simulator velocity mismatch hurts learning |
| P2 | Keep raw pose/gripper; causal EMA of velocity at 10 Hz | Motion context helps if derivative noise is reduced |
| P3 | Measured joints/gripper instead of FK state, SO-101 only | Provisional FK/state representation is a bottleneck |
| P4 | Add measured joints to canonical pose | Joint information helps beyond the pose representation |

For P2 screen a small train-selected range such as new-sample EMA weights 0.5 and 0.75. Reset on episode boundaries and implement exactly the same filter during deployment. P3/P4 keep action targets Cartesian, isolating the observation change. A separate joint-action experiment can follow; changing both input and output representations simultaneously would confound the result. Raw SO-101 joints are not directly transferable Panda inputs.

Before full joint smoothing, measure raw 30 Hz stationary variance, encoder quantization, spike rates, velocity/acceleration distributions, power above 5 Hz and image/joint alignment. The existing roughly two-frame image-lag estimate is based on image-motion correlation, not synchronized hardware timestamps. Re-estimate timing on training recordings with visible landmarks; assess causal history/observation-age compensation rather than pairing online observations with unavailable future frames.

Only if those measurements support it, create a separate causal arm-joint low-pass variant before 30→10 Hz decimation. It requires an equivalent online high-rate observation stream and measured filter delay. If that stream is unavailable, use the deployable 10 Hz velocity experiment. Leave raw data intact; log every correction and masking decision. Never filter across episode boundaries, use centred/forward-backward filters in online inputs, or independently smooth rotation-matrix entries.

Preserve gripper closure/release and the follower's loaded-opening plateau. If later testing cleaned arm targets, regenerate FK poses/deltas consistently, retain a raw-reference evaluation, and compare physical tracking. Lower loss against newly smoother labels is not evidence of improvement.

## 4. Small flow-matching head

Flow matching can still generate an action chunk. It changes the generative objective/head; it does not require abandoning chunk length 20 or execution prefix 2. It would be a new policy variant rather than unchanged ACT. Flow matching trains a vector field at sampled noise/time values; integration steps are needed during sampling. [Flow Matching paper](https://arxiv.org/abs/2210.02747). The robotics π0 paper explicitly generates action chunks with flow matching. [π0](https://www.physicalintelligence.company/download/pi0.pdf).

Proposed F0: frozen DINOv2-small features plus a small conditional head, initially two transformer blocks of width 256/eight heads, producing 20 × 7 action values. Compare against a regression head with matched conditioning/capacity and the selected ACT baseline. Remove the ACT VAE/KL objective in the flow branch and use masked flow-vector loss on normalized action chunks. Keep the vision encoder frozen in both small-head controls; cache features only when preprocessing/augmentation is fixed and identical between them.

Train on one sampled interpolation time per example; test four and eight sampling steps. Reuse observation features within each inference call. Use a fixed random-seed protocol and no ground-truth-based best-of-many sample selection. Report end-to-end latency, including vision and integration, and meet the existing two-action/200 ms inference cadence with margin. Memory and latency must be profiled before committing to a long run.

A head of this size is a plausible laptop experiment, not a measured feasibility result yet. The expensive large-VLA configurations are not required. Prioritize it after ACT's data/control comparisons: a new action distribution cannot establish whether jaw calibration is correct.

## 5. Additional experiment most likely to clarify the failure

**H0 — short-history ACT.** Retain one current image and supply three recent proprioceptive observations plus the previous two actually applied commands, with relative times and reset masks. Use the selected proprioceptive representation. Compare with the otherwise identical single-state ACT, retaining chunk 20/prefix 2.

Hypothesis: a single state with a finite-difference velocity does not capture actuator response, observation delay or the transition into contact. Short state/action history adds relatively little compute and tests that hypothesis directly. This remains an ACT extension; use only information available at inference. Historical recorded leader commands may differ from actually sent clipped commands, so use a separately labelled recorded-command history baseline unless sent commands can be verified. Do not claim those inputs are applied actuator commands without evidence. A state-history-only arm is the clean first comparison.

Before a full history run, overfit a small set of reviewed successful training segments to check whether the chosen model/normalization/targets can represent their approach, closure and lift at all. This is a pipeline capacity check, not a generalization or task-success result.

After geometry passes, test a learned SO-101 joint-action policy as a separate route around FK/IK, while retaining a same-input Cartesian-action control. A successful same-embodiment policy identifies a useful baseline before returning to Panda.

## Execution order and decision gates

1. G0 arm/gripper factorial, G1 independent registration, and A0 attention/memory profiling.
2. P1/P2/P3 and H0 state-history-only screens; run the matched A1 backbone comparison with fixed data.
3. G2/G3 constrained physics, while A2–A4 test effective batch 64 and twelve heads offline.
4. Combine only independently supported data/architecture changes and run matched SO-101 evaluations once recorded physical replay passes.
5. Run F0 as a bounded alternative if residual action-distribution errors justify it; return to Panda only after SO-101 reference replay, controller tracking and one learned policy pass their respective gates.

Interpretation must remain conditional: good reference tracking plus failed gripping points to calibration/contact or gripper-command semantics; successful reference gripping plus failed Cartesian tracking points to control; successful physical reference and control with learned-policy failures makes perception/data coverage/model choice a much stronger explanation.
