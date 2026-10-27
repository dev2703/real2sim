# Frozen shared ACT generalization — 27 October 2026

The same frozen SO-trained **semantic ACT** completed **15/20 Panda placements
and 0/20 SO-101 placements** on newly declared paired ball-and-bowl layouts.
Separate lighting, ball-size and friction repetitions reveal a substantial
perception weakness. This checkpoint does not establish reliable completion on
both arms.

All **280 scheduled trials** are retained: 20 layouts × two arms × seven
conditions. The six variations each repeat the same 20 nominal layouts, so this
is **20 unique layouts**, not 280 independent layouts. No checkpoint selection,
training, weight update, controller retuning or failed-layout replacement occurred.

| Condition, changed separately | SO-101 placements | Panda placements | Initial detection failures, SO-101 / Panda |
| --- | ---: | ---: | ---: |
| Nominal | 0/20 | 15/20 | 0 / 0 |
| Lighting 0.6× | 0/20 | 16/20 | 0 / 0 |
| Lighting 1.4× | 0/20 | 0/20 | 20 / 20 |
| Ball radius −5% | 0/20 | 0/20 | 20 / 20 |
| Ball radius +5% | 0/20 | 17/20 | 0 / 0 |
| Sliding friction ⅔× | 0/20 | 14/20 | 0 / 0 |
| Sliding friction 1.5× | 0/20 | 16/20 | 0 / 0 |

The 80 initial-perception failures have zero policy calls and remain scored as
failures, with the unchanged harness's `aborted` classification. They are not
learned grasp failures or renderer crashes. In each of the five conditions with
successful initial perception, Panda achieves held lift in 20/20 trials; its
remaining failures occur during transport. SO-101 achieves no held lifts. Its
approach/grasp failure counts are 1/19 nominal, 0/20 under dim light, 1/19 with
the larger ball, 2/18 at lower friction, and 2/18 at higher friction.

The [saved-image diagnostic](../results/v2/frozen_generalization_frontend_diagnostic_20261027.json)
checks seed 10000 without new rollouts or tuning. Smaller-ball sphere-fit RMS is
2.041/2.161 mm on SO-101/Panda, exceeding the unchanged 1.5 mm gate. Under bright
lighting, SO-101's bowl mask and Panda's ball mask merge with border-touching
background regions. These explain the measured missing detections for that
paired diagnostic image; all other trial outcomes remain separately recorded.

The small differences between Panda's 14–17/20 results do not establish an
improvement or justify choosing a new setting. The paired transitions, complete
failure accounting and per-trial measurements are in the
[curated report](../results/v2/frozen_generalization_20261027.json),
[all-trial CSV](../outputs/v2/frozen_generalization_20261027/trials.csv) and
[full evidence index](../outputs/v2/frozen_generalization_20261027/report.json).
No confidence intervals are reported for these N=20 conditions under the existing
[experiment protocol](experiment_protocol.md).

Frozen ACT was already in use before this test. Here the checkpoint is
`outputs/v2/semantic_act_transfer_20261027_v2/checkpoint`, SHA-256
`9fb2b35ee66ecb4489083f06f2b1697cbe89b55ff696aa56345bbe8e16045523`.
It consumes initial RGB-derived ball/bowl coordinates, live FK grasp-point and
velocity feedback, and causal prior gripper intent/age. **It has no image encoder
and is separate from the uploaded real-demonstration Hugging Face ACT and the
successful native SO-101 RGB/joint ACT.** These results must not be attributed to
either of those checkpoints or to live Panda-image transfer of their weights.

The previous Panda development settings are fixed for both-arm comparison:
XYZ chunk preview index 3, gripper intent index 0, 0.01 rad interior joint margin,
24 mm point-step limit, unchanged robot-specific adapters, original SO-101 start,
and Panda's declared −150 mm base translation and mapped source start. Panda
acquires RGB at its configured parked pose before resetting the identical object
layout to the mapped start. Native bounds, collision/path checks, actuator physics,
12.5 Hz control, 375-tick timeout and one-second released-resting placement scorer
are unchanged. Exact configured static collision geometry remains available to
the kinematic guards. The checkpoint is in evaluation mode with gradients
disabled and inference under `torch.inference_mode()`.

Seeds 10000–10019 were fixed before outcomes. Balls use the existing uniform-area
30 mm start disc; bowls independently use a 30 mm disc about their nominal centre,
with `default_rng(SeedSequence([seed, 20261027]))`. Both arms receive identical
explicit XY pairs in every condition. The
[novelty audit](../results/v2/generalization_layout_novelty_20261027.json) scanned
695 pre-existing JSON artifacts, found no seed overlap and no exact coordinate
pair match among the available explicit/reconstructed references. Missing or
unrecorded historical layouts remain outside that claim. These are nearby
simulation layouts, not broad workspace extrapolation or physical validation.
They have now been used for evaluation and must not be described as untouched in
future tuning.

Lighting scales actual MuJoCo light and headlight ambient/diffuse/specular
components. Ball-size conditions rebuild physical geometry, inertia, spawn height
and scoring with radii 30.72327 and 33.95730 mm, keeping mass at 58 g. **Perception
always assumes the nominal 32.34028 mm radius**; it does not receive the varied
true size. Friction scales the sliding coefficient of every collision-enabled
geom by ⅔ or 1.5, preserving material ratios, torsional/rolling coefficients and
other contact settings. The ball/table coefficient changes from 0.6 to 0.4/0.9;
effective ball/finger values also depend on native material mixing. This is a
scene-wide friction variation, including other robot/static contacts. Actual
per-geom values and sampled ball-contact friction are retained in each report.

The [nominal protocol](../outputs/v2/frozen_generalization_20261027/protocol.json)
was saved before scored execution. A concurrent workspace edit to the JSON
writer stopped the next launches at source verification before inference. The
original writer was recovered and verified against its exact original hash,
then all 143 original dependencies were copied to an isolated runtime. The
[continuation protocol](../outputs/v2/frozen_generalization_20261027_variations/protocol.json)
uses **143/143 identical source hashes and identical scientific settings**;
nominal trials were not rerun. The unrelated workspace edit was preserved.
[Source-isolation evidence](../outputs/v2/frozen_generalization_20261027/source_isolation.json)
and [permanent frozen source copies](../outputs/v2/frozen_generalization_20261027/frozen_dependencies/inventory.json)
retain that distinction. The initial sandbox graphics launch and source-check
launch failures are archived separately; neither executed policy inference or
physics steps. The local macOS renderer required graphics access outside the
sandbox.

Validation: 16 focused tests and Ruff pass. A previously used seed-300 harness
regression reproduces every original state and ACT action exactly on both arms
(375 SO-101 ticks and 91 Panda ticks), separately from the fresh-layout count.
The combined validator checks all 14 complete reports, exact schedules,
checkpoint/source hashes, summaries and paired outcomes. Each trial retains
initial/final images, requested actions, achieved feedback, contact samples and
scoring traces. No selected rollout videos replace the trial denominators.
The [independent review](../results/v2/frozen_generalization_review_20261027.json)
passes with no findings: 280 trial rows, 54,602 ticks in 200 traces, all 80
zero-tick perception failures included, and 600 independently recomputed ACT
action anchors with zero difference. There are no policy-execution aborts.

The earlier requested software diagnostics are already complete: the
[motion-interface reconstruction](v2_motion_interface_20261027.md) placed 25/25
across five orientation/gripper variants, while the
[registration audit](v2_registration_bridge_20261027.md) and
[recorded-demo/HF retest](v2_real_replay_hf_bridge_20261027.md) retain the joint,
camera and visual-input gaps. Physical camera/base/jaw registration remains
provisional. This new generalization test does not resolve that physical mapping
or establish transfer of the uploaded real-data ACT.

For another predeclared evaluation, use fresh seeds and a new novelty audit with
the [runner](../scripts/evaluate_v2_frozen_generalization_20261027.py). The
[combined validator](../scripts/summarize_v2_frozen_generalization_20261027.py)
rechecks the retained two-phase evidence; it does not rerun any trial:

```sh
.venv/bin/python scripts/summarize_v2_frozen_generalization_20261027.py
```
