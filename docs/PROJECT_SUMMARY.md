# Real2Sim-RaC project summary

Consolidated through 30 October 2026. This replaces the dated reports and
duplicated runbooks previously in `docs/`; their full text remains in Git history.
Compact evidence remains under `results/`. Historical output paths inside that
evidence record provenance; cleanup may remove bulk videos, intermediate
checkpoints and superseded runs.

## Current result

The goal is to retain a real-demonstration ACT and learn recovery that executes
through native controllers on both SO-101 and Panda. The working simulation
fallback currently uses RGB geometric planning.

| Controller | SO-101 complete tasks | Panda complete tasks | Scope |
| --- | ---: | ---: | --- |
| Frozen native HF ACT | 0/100 | 0/100 | Reserved nominal seeds 13000–13099 |
| RGB geometric controller | 100/100 | 100/100 | Identical matched resets |
| Frozen ACT followed by RGB recovery | 100/100 | 100/100 | Identical resets; takeover in every trial |
| Frozen ACT + learned phase MLP | 3/10 | 3/10 | Development seeds 12000–12009 |

The [600-trial audit](../results/v2/matched_benchmark_100_20261029.json) checked
reset fingerprints, row/trace hashes and runtime inputs. Raw ACT failed approach;
every assisted trial switched at tick 50 (2 seconds). Mean simulated completion
time for RGB alone was 9.058/10.868 seconds on SO-101/Panda, versus
10.302/13.415 seconds with initial ACT. This provides no evidence that ACT
improves nominal simulated success. Lighting, weights and contact settings stayed
unchanged. The results apply to the tested simulation distribution.

The phase head calls no geometric teacher for evaluation actions. Its first two
development seeds helped compare candidates; eight further fixed seeds scored
2/8 per arm. The combined 3/10 is descriptive development evidence, below the
60% target, rather than an independent reserved benchmark. See the
[20-trial audit](../results/v2/recovery_mlp_tracked1_phase_dev10_20261029.json).
The user's reported 4/5 hardware result is separate evidence; these simulated
failures do not establish that the real checkpoint is poor.

## Code and environment

| Location | Responsibility |
| --- | --- |
| `src/real2sim/data/` | LeRobot loading, audits, splits, resampling and labels |
| `src/real2sim/geometry/` | SO(3), FK, image/planar registration and optical calibration |
| `src/real2sim/sim/` | SO-101 scene/cameras, native kinematics, contacts and follower response |
| `src/real2sim/panda/` | Panda scene, IK, workspace and virtual-source retargeting |
| `src/real2sim/policy/` | ACT/backbone variants, training/validation, native inference and recovery |
| `src/real2sim/evaluation/` | Scoring, video, replay, oracle and domain-gap diagnostics |
| `src/real2sim/recovery/` | Original observable detector, FSM and specialist interface |
| `src/real2sim/experiments/` | Versioned v2 experiment preparation/contracts |
| `scripts/` | Command entry points for evaluation, training, data and calibration |
| `configs/`, `tests/` | Versioned configurations and unit/integration checks |
| `assets/` | Required Menagerie models/meshes, provenance and calibration targets |
| `integrations/lerobot_trace/` | Opt-in collector timing/sent-action trace patch/helper |

Run commands at the repository root. `uv sync --locked` installs Python 3.12
dependencies from `pyproject.toml`/`uv.lock`, including LeRobot 0.6.1 and MuJoCo
3.15. Optional Placo kinematics are intended for Linux; native MuJoCo FK/IK is
the working macOS fallback after a Placo wheel import failure.

Current measured-scalar scenes are
`configs/v2/measured_registration_20261028/sim_{so101,panda}_positive_all_scalars.yaml`.
Physical calibration remains explicitly unaccepted. Native runners default to
the SO-101 source scene and construct the corresponding Panda setup as needed.

```sh
uv run ruff check .
uv run pytest -q tests/unit
# Some integration checks require local data/checkpoints or an offscreen renderer.
uv run pytest -q tests/integration
```

## Data and control contracts

Always load the checkpoint's recorded schema, processors, timing and split.
These pipelines have different contracts:

| Pipeline | Data/timing | Observations and actions |
| --- | --- | --- |
| Original v1 | `bottowork/so101_act_pickplace_v1`, revision `fd65a6600b0cd111a434ece1abfa3d44dc7b4d57`; 40 episodes/47,393 rows at 30 Hz; exclude episodes 25/33/34 with iPhone frame deficits → 28 train/9 validation | iPhone only (`top` faces the ceiling); 10 Hz preparation, RGB `[3,192,256]` letterbox, chunk 20/execute 2 |
| Exploratory v2 | Local `so101_act_pickplace_v2`: 41 episodes/27,736 rows at 25 Hz; reviewed 29/6/6 train/validation/reserved split in `results/v2/split.json` | Wrist key `observation.images.top`, global key `observation.images.iphone`; 12.5 Hz preparation, chunk 25/execute 2 |
| Uploaded native ACT | `bottowork/so101-act-h256-b16`, revision `e696c8a078935432e3501e2c28c1fb2598fbff7e`; control assumes the source's 25 Hz | Wrist first/iPhone second, RGB 640×360; six observed joints and six absolute degree/percent actions; chunk 50/execute 10; saved processors exactly once |

Native `latest/` weights SHA-256:
`5dab91076d1e3b274674a888f522ce5fdddaf69ddd05c1adf3bce1742b99f65c`.
The model uses ResNet18, width 256, four encoder layers and a VAE. Root weights
differ. Saved latest training state records step 30,000, although the experiment
configuration requests 40,000 updates. Its own 35-train/six-validation split
(seed 42) overlaps the exploratory reserved partition, so historical reserved
claims do not transfer to this checkpoint.

Five arm channels use degrees; gripper is percent with 0 closed. Recorded raw
actions are leader/requested targets, not established sent commands or achieved
follower motion. Encoder-lattice checks support degrees without identifying
physical CAD neutral poses. Joint-dependent follower lag has no supported single
global correction; nominal row timestamps do not measure exposure-to-motor latency.
The v1 audit estimated the iPhone image about two frames (67 ms) behind joint
state; this is image-motion evidence, not measured physical synchronization.

The original joint baseline learns recorded actions. Cartesian labels use
achieved next-tick FK motion and leader gripper intent divided by 100. Canonical
observations have 13 values: TCP XYZ, rotation-6D, linear velocity and semantic
gripper opening. Common actions have seven values:
`[dx,dy,dz, rotation_vector_x,y,z, gripper_open]`, in task-frame metres, radians
and 0–1 opening. Task and world axes are aligned in the current scenes. Per-step
translation/rotation caps are 30 mm/0.3 rad, followed by
native workspace, joint, rate, IK, collision and sampled-path checks. Explicit
experimental variants use persistent targets, absolute poses or tool-frame
deltas with their matching adapter/manifest.

Historical architecture decisions fixed train-only statistics, episode-level
splits, letterboxing, achieved Cartesian labels, raw joint labels, bounded
native controllers and separately labelled recovery. The original 29-input
specialist in `recovery/` differs from today's common-action heads. Old 10 Hz
defaults are not the native uploaded ACT contract.

## Measurements and unresolved registration

The operator's [inch measurements](../physical_measurements.md) are present and
applied with estimated ±5% uncertainty. The
[application audit](../results/v2/measurement_application_reaudit_20261028.json)
checks the current source hash and both arms' scenes.

| Quantity | Current measured-scalar interpretation |
| --- | --- |
| Ball | 8 in circumference → radius 32.340 mm; interval 30.723–33.957 mm |
| Base to ball | (+2,+13) in → (50.8,330.2) mm under the working axis convention |
| Ball to bowl | 12 in total distance → 304.8 mm; bearing uses an earlier image-layout prior |
| Bowl | 4.5 in diameter → 114.3 mm, provisionally interpreted as outer rim |
| Mat | 16×12 in → 406.4×304.8 mm |
| Global camera minus ball | (+30.5,+4.5) in horizontal components under the working convention |
| Global camera height | 16 in → 406.4 mm above tabletop |

World uses MuJoCo's Z-up axes. Task origin is the SO-101 base, with +X away
from the robot and +Y left. For the configured identity base transform,
`p_world = p_task + [0,0,table.height + base_z_offset]`; the tabletop is at
task Z `-base_z_offset`. The retained provisional base offset is 12 mm
(historical decision D-014), and Panda shares this task-object frame.
Distances do not identify optical rotations, intrinsics/distortion,
operator-to-CAD axes, joint neutral positions or jaw geometry. Identity arm
signs/offsets remain provisional. Unsupplied bowl depth/base diameter, contact
friction and other physical parameters retain labelled assumptions. The pinned
Menagerie wrist optical camera and 61° FOV are nominal CAD geometry.

In v2's 23,372 train/validation readbacks, 32.94% violate some native joint bound
under the provisional map; shoulder lift accounts for most. Expanding limits,
clipping source poses or applying homing offsets twice would not establish a
physical registration. Known physical link poses and independent verification
are needed.

The 29 October fixed-joint camera audit retains 18 validation anchors with exact
RGB inputs and ACT chunks. Five-arm-joint prefix MAE is 2.858° for recorded views,
11.752° for nominal renders and 9.892° for a training-only yaw hypothesis. The
global-view substitution causes more error than the wrist, with substantial
two-view interactions (6.779° nominal/4.426° yaw). Pose, layout and appearance
change together, so no physical correction is adopted. See
[camera ablation](../results/v2/hf_alignment_camera_ablation_20261028.json) and
[factorial audit](../results/v2/hf_alignment_factorial_20261029.json).

A prepared physical session has no completed calibration captures. Its ruler/
paper route uses 12 individually measured crosses on a rigid target, six tilted
fitting views and two independent verification views per camera. Four corners
estimate verification pose; eight remaining points test it. Independently
measure target-to-base registration, known link/joint poses and jaw pad gaps at
several readbacks. Static mat footage alone cannot resolve unrestricted optics
and base registration. Unknown fields stay null until measured.

## Working RGB control and Panda transfer

`VisualWaypointController` measures RGB ball/bowl targets using supplied camera
and object scale, plans feasible native motion against a static collision map,
then grasps/lifts/carries/releases. The repaired bowl frontend scans fixed
brightness levels and combines duplicate rim observations while retaining
shape/scale/workspace/ambiguity checks. It changes no lighting. Missing ball
targets have no configured-position fallback. A previously observed static bowl
can be remembered during occlusion; recovery reacquires the movable ball.

Panda's old 50 mm X workspace floor excluded valid measured starts. The current
workspace includes the start disc, radius uncertainty and bowl footprint;
deterministic grasp branches still pass native pose/path/contact checks. SO-101
has five arm joints and uses soft carry orientation; Panda keeps its declared
top-down task orientation.

The Panda adapter reconstructs a virtual SO-101 state from measured TCP position,
retaining causal source-command priors for unobservable rotation. This is a lossy
representation. Native feasibility does not ensure source representability.
ACT-dependent endpoints and sampled paths check representation, with a 1 mm
endpoint margin inside the unchanged 3 mm feedback tolerance. Magnitude
backtracking provides no task goal. Out-of-tolerance measured feedback raises
a structured error instead of supplying fictitious source joints.

The [fixed-lighting RGB pilot](../results/v2/fixed_lighting_rgb_robustness_pilot_20261028.json)
completed 42/42 tasks: two arms × seven conditions × three seeds, including wider
layouts, ±5% radius and sliding friction ×⅔/×1.5. The frontend knows the varied
radius; this small scene-aware geometric pilot does not establish tolerance to
unknown size, learned recovery reliability or hardware performance.

## Learned recovery and DAgger

The geometric controller is the fallback/expert; the MLP is a learned controller;
DAgger is the collection/training procedure. A synchronized shadow expert labels
actual learner-visited states, with declared beta choosing expert/learner
execution. Mixed collection success is not learned-only success.

Three causal frames supply 99 observable values: measured TCP pose/velocity and
jaw opening, relative RGB targets, visibility/ages, previous selected action,
embodiment ID and time since a selected jaw change. Frozen ACT supplies its
aligned 256-value decoder feature and common-action proposal. The MLP has hidden
widths 256→128 and seven outputs; no-ACT masks both ACT inputs. A phase head
mixes per-phase actions. Teacher phase, simulator ball pose/held flags and
scoring metrics never enter learned observations.

Initial-reference and tracked-reference cohorts have distinct observation names
and cannot be silently mixed. Tracked RGB refreshes every five ticks, retaining
ages during occlusion. It estimates the on-table ball, not a carried ball in 3D.
Tracked iteration 1 aggregates 6,126 training/2,038 validation rows across both
arms and iterations. Its residual/no-ACT heads scored 0/2 per arm; two retained
Panda residual aborts exposed invalid virtual-source states. Its phase head
produced the 3/10 development result above. Low action error or phase accuracy
does not verify grasping or complete-task success.

The corrected teacher replans on cumulative drift and fresh RGB missed-pickup
evidence even if a free ball stayed at its original goal. Issued native jaw
commands govern continuous close/release dwell; opposing commands reset it.
These fixes postdate existing packs and require new collection; old heads/packs
keep their recorded semantics. The first intervention gate saw only
correction-positive rows and always recovers. It needs genuine ACT-sufficient
negatives to learn when intervention can end.

The [retention audit](../results/v2/recovery_retention_audit_20261029.json) finds
five Panda losses after genuine lift with selected jaws ≤10% open. Ball descent
relative to TCP over 0.4 seconds is 16–21 mm, compatible with slip without
isolating seating, friction or motion as its cause. SO-101 sometimes regresses
to an open pre-grasp command during carry.

`ObservableGraspMonitor` uses issued jaw intent, measured opening/TCP motion and
fresh RGB. A candidate can also mean obstruction; `held_verified` stays false.
Optional `--phase-guard` adds explicitly labelled phase/jaw/motion assistance
and bounded retries without teacher actions or object ground truth. It remains
experimental: final known-failure-seed checks completed one SO-101 task while
Panda missed pickup. Fresh expert-only contract checks completed 2/2 per arm;
no head was retrained. These smoke cases are not reliability estimates. See
[engineering smoke](../results/v2/retention_representation_smoke_20261029.json).

## Recovery integration work, 30 October

`scripts/run_cross_embodiment_recovery.py` connects collection, phase/residual
head training, ACT/no-ACT ablations and evaluation. `--matched-benchmark` runs
all fitted heads alongside raw ACT, RGB and ACT+RGB on identical resets. Both
head artifacts enter the benchmark protocol hash, and serial trials reuse the
frozen ACT model while removing each episode's feature hook. `--plan-only`
prints the exact commands without loading models or writing outputs.

Training arms and evaluation arms can differ: `--train-robots so101 --robots
so101 panda` tests a source-only head on the unseen Panda. Workflow reports
record training and unseen target arms; actual fitted episode/robot IDs are
checked before any development rollout. Reserved seeds cannot enter workflow
training, validation or development selection. Existing compatible collections
can be supplied with `--collections`.

New residual heads bound normalized continuous inputs and select checkpoints
using equal robot/episode validation weights. Binary visibility/robot channels
retain their natural 0/1 scale, including when only one arm appears in training.
Historical artifacts preserve their saved normalization behavior. Phase heads
now support genuine no-ACT fitting/inference, masking both decoder features and
the action prior. Collection loading checks row robot/episode IDs against the
declared manifest. ACT feature extraction follows checkpoint dimensions rather
than a fixed decoder shape; empty packs and no-ACT inputs retain that width.

An analytic MuJoCo grasp-point Jacobian replaces repeated finite differences in
Panda feedback projection, retaining the objective, seeds, limits and 3 mm
acceptance threshold. Five alternating 40-pose batches measured a median
1.729× local speedup, identical acceptance decisions and a maximum position
residual difference of 1.08e-9 m. This is a kinematic microbenchmark, not a
physical latency result. See the [reproducible projection
measurement](../results/v2/virtual_source_projection_performance_20261030.json).

Two corrected-label development heads trained on 2,591 training and 1,863
validation rows with unchanged ACT weights. Equal robot/episode action MAE was
0.07958 with ACT and 0.05255 without ACT. The [training
evidence](../results/v2/recovery_head_training_20261030.json) retains commands,
hashes and phase-transition metrics. On development reset 12010, both heads
completed 0/1 tasks on each arm; RGB and ACT+RGB each completed 1/1 per arm.
Source-only ACT/no-ACT heads also completed 0/1 per arm. No learned trial aborted
or achieved a held lift. These smoke trials establish workflow execution, not
reliable transfer. The [causal trace analysis](../results/v2/recovery_failure_analysis_20261030.json)
identifies phase advancement after missed pickup, repeated native-limit holds
and continuous observation/history shift on the unseen arm. Fixed binary scales
and input clipping do not resolve that shift.

The optional `--recovery-frame source` evaluates SO-101-only heads using a
measured Panda-to-virtual-source observation. Both the ACT action prior and
selected-action history retain the source schema. Source rotational actions
contribute their grasp-point displacement to Panda motion, rather than being
discarded by the fixed top-down orientation. The mapping anchors displacement
on the measured Panda pose, so a zero action never executes the tolerated
projection residual. Native guards and source representability checks still
apply, including for no-ACT heads. This is an explicit experimental adapter;
source rotation remains an unobserved causal prior and target RGB remains a
different input distribution.

The fresh workflow collected 20/20 successful expert rollouts, with 2,915
training and 726 validation rows across both arms. Its new ACT/no-ACT heads had
balanced validation MAE 0.01982/0.006286; both completed 0/1 development tasks
per arm on seed 12011. A further SO-101-only fit with the source-frame adapter
also completed 0/1 per arm on seed 12012. Its Panda ACT-conditioned trial had
zero IK rejections and at most 6.23 µm source projection error, yet failed
approach. This isolates a learned behavior failure even when native commands
and source representation pass their gates.

Across these development cohorts, the 16 learned trials completed no tasks,
achieved no held lifts and had no aborts. The three matched comparisons each
used one reset per arm: raw ACT completed 0/6 tasks, RGB and ACT+RGB each 6/6.
All 30 matched trial/trace hashes, reset fingerprints, unchanged weights and
lighting were verified. These small, different training cohorts do not support
a model ranking or a reliability estimate. The [aggregate engineering
report](../results/v2/cross_embodiment_recovery_20261030.json) records artifacts,
head metrics, trial outcomes and the remaining bottlenecks. Final validation
passed 1,089 unit/selected integration checks plus the new source-frame
integration check; three GPU/optional-payload cases skipped. Ruff and whitespace
checks passed. Reliable learned cross-embodiment completion remains unresolved.

```sh
# Inspect a fresh collection/training/matched comparison workflow
uv run python scripts/run_cross_embodiment_recovery.py \
  --out outputs/v2/recovery_pipeline_NEW --matched-benchmark --plan-only
# Remove --plan-only to execute; add --train-robots so101 for source-only fitting.
```

## Evaluation and reproducibility

Keep raw ACT, RGB, privileged oracle/replay, learned-only and optional-guard
outcomes separately labelled. Count all failures/aborts. Complete pick-and-place
requires held lift, delivery, release and one second of unheld resting placement.
`TrialTracker.success` itself also accepts a ball pushed into a resting bowl;
check `lifted`/`released` for complete-task claims. Failure phases report the
furthest reached: approach, grasp, transport, release or aborted. Approach means
never held and never within the configured 3 cm tool-site-to-ball-centre threshold;
grasp means reached/held without a held lift; transport means lifted without
delivery; release means delivered without a stable unheld success.

Matched benchmark seed roles are training 10000–10023, validation 11000–11005,
development 12000–12009 and reserved 13000–13099. Model selection must not use
reserved outcomes. Match reset fingerprints; older standalone Panda and HF
runners used different bases/starts. Known static collision geometry can guide
native planning; dynamic object poses/held flags belong only to scoring and
labelled diagnostics. Hashes, saved sources/configs and exact inputs record that
distinction.

Changed source/configuration/weights/lighting require a new output directory.
Identical benchmark repetition resumes completed attempts after protocol, seed,
row/trace hash and length checks. Partial attempts remain explicit; retries get
new attempt directories. Historical hashes describe executed historical source,
not the current tree. Shared-host latency does not guarantee physical real-time
execution.

## Cleanup and retained artifacts

The [cleanup manifest](../results/cleanup_manifest.json) records deleted paths,
retained dependencies and final size/check results. Documentation was consolidated
from 65 files into this one summary; 35 historical scripts and 13 superseded
configurations were removed. Reusable source modules remain because they support
tested features and retained workflows.

`outputs/v2/` was reduced from 270 to 28 groups: recent recovery heads/packs and
development evidence, necessary camera/calibration metadata, benchmark summaries,
and small dependencies of legacy training workflows. Six representative benchmark
trials and four phase-head videos remain. Compact result JSONs retain the full
aggregate outcomes and recorded hashes; the complete 600-trial trace collection
is no longer present. Pruned historical benchmark directories cannot pass a full
resume-integrity check; rerun into a new output directory.

Superseded `.cache/v2_*` derived groups were removed. The duplicate v1 Cartesian
image array now uses a relative symlink to the identical joint image array,
preserving existing loader paths. Hugging Face caches, three required v1
checkpoint families, original raw datasets and robot assets remain. Derived data
and ignored outputs can be regenerated with their retained workflows; deleted
historical tracked docs/scripts can be recovered from Git. Cleanup did not create
another bulk archive. Historical experiment-specific training/evaluation must
regenerate its removed data/checkpoints before execution; ignored artifacts are
not recoverable from Git.

Cleanup validation passed 1,007 unit checks with four skips and three graphics
checks deselected after baseline macOS CoreGraphics failures. One skip reflects
an optional historical experiment payload that was pruned; the other three are
backend-specific. Another 54 selected simulation/inference integration checks
and four checkpoint/contract checks passed, including Cartesian inference from
the shared image array. Validation covered selected integration workflows;
the entire integration suite was not executed. Core CLI help, lint and whitespace
checks passed.

## Essential commands

Use new output directories. Replace `so101` with `panda` for the other arm.

```sh
# Working assistance, then unchanged ACT alone
uv run python scripts/evaluate_hf_recovery.py --robot so101 \
  --seeds 500 501 502 503 504 --out outputs/v2/hf_recovery_so101_NEW --video
uv run python scripts/evaluate_hf_act.py --robot so101 \
  --out outputs/v2/hf_only_so101_NEW

# Geometric baseline from its configured start
uv run python scripts/run_visual_waypoint.py --robot so101 --seeds 0 1 2 3 4 \
  --config configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml \
  --out outputs/v2/rgb_so101_NEW --video

# Inspect a matched plan; omit --plan-only to execute
uv run python scripts/benchmark_frozen_act.py --robots so101 \
  --out outputs/v2/benchmark_so101_NEW --plan-only

# Corrective collection, fitted head and learned-only evaluation
uv run python scripts/run_recovery_dagger.py collect --robot so101 \
  --out outputs/v2/recovery_teacher_NEW
uv run python scripts/train_recovery_mlp.py \
  --data <completed-so101-pack> <completed-panda-pack> \
  --use-act --out outputs/v2/recovery_head_NEW
uv run python scripts/run_recovery_dagger.py collect --robot so101 \
  --head <head-directory> --beta .5 --iteration 1 \
  --train-seeds 10012 10013 --val-seeds 11004 --out outputs/v2/dagger_NEW
uv run python scripts/run_recovery_dagger.py evaluate --robot panda \
  --head <head-directory> --seeds 12000 12001 --video --out outputs/v2/recovery_eval_NEW

# Original 10 Hz preparation/training with its matching source
uv run python -m real2sim.data.audit --repo-id bottowork/so101_act_pickplace_v1 \
  --revision fd65a6600b0cd111a434ece1abfa3d44dc7b4d57 \
  --out results/audit --n-val 10 --exclude-episodes 25,33,34
uv run python -m real2sim.data.prepare --mode cartesian --config configs/data.yaml \
  --local-root <v1-dataset> --out-dir .cache/derived_cartesian
uv run python -m real2sim.policy.train --config configs/act_cartesian.yaml \
  --out outputs/act_cartesian_NEW
uv run python -m real2sim.policy.check --checkpoint <checkpoint-directory> \
  --data .cache/derived_cartesian
```

## Condensed history

These results used different checkpoints, starts, data or scenes. Do not pool
them with the matched benchmark.

| Period | Findings retained from earlier reports |
| --- | --- |
| 12–13 October | Audited v1 ceiling view/missing frames; separated leader commands and achieved-motion labels. Original joint/Cartesian learned runs scored 0/10. Workspace-floor/Panda frame fixes did not solve approach. Privileged oracle 10/10 showed feasibility for its own settled start and grasp settings. |
| 16–17 October | Backbone, proprioception, history, flow/regression and objective screens established no learned grasp success. A separate 467-row 25 FPS episode visibly completed the task without measured host timing. Loaded tracking and geometry gaps motivated reference-motion diagnostics. Collector tracing separates requested/sent/subsequently measured actions. |
| 20–24 October | Completed 22 unique exploratory runs with 64,000 examples each and six untouched reserved episodes. Best two-view CLIP validation L1 0.43749 was 4.51% below ResNet, an offline result. Learned approach still failed. Feasible five-joint grasp/carry planning improved privileged diagnostics while preserving acceptance bounds. |
| 24–25 October | Grounding DINO/SAM2 and fixed landmarks supplied conditional image evidence; independent camera confirmation failed. Keypoints/next-follower targets did not solve approach. Three-episode/prefix fits failed capacity gates despite lower average errors. Object interventions showed weak/misdirected dependence. Causal history reduced onset conflicts without revealing future intent. |
| 26 October | RGB controller achieved 4/5 per arm, then 8/10 on fresh layouts and shifted bowls in the earlier scene. Scalars were ingested. Discarded x-only 12-inch bowl offsets caused reach failures; clarification specifies total distance. Height-only camera substitution broke detections and was not accepted as registration. |
| 27 October | Separate simulation-teacher ACT scored SO-101 3/5 and 4/10 development placements. Shared semantic ACT scored Panda 15/20 and SO-101 0/20 on new paired layouts; 280 trials repeat 20 layouts across seven conditions. These differ from real-data HF ACT. Successful-motion replays validated the interface; real-data ACT still failed rendered-scene transfer. |
| 28–29 October | Fixed RGB rim detection, measured Panda workspace and causal source feedback. Completed the 600-trial benchmark, small fixed-lighting RGB pilot, tracked DAgger/phase-head development and retention/representation follow-up above. No physical camera correction or new hardware result was established. |

## Remaining work

1. Separate missed pickup, grasp seating and slip using observable feedback;
   collect learner-visited corrections with the fixed teacher.
2. Cover long failures, stale RGB, repeated jaw transitions and uncertain holding.
   Consuming monitor features requires a versioned contract and new examples.
3. Resolve Panda/source representation and uncertainty without fictitious states
   or relaxed acceptance tolerances.
4. Independently measure optics/base transforms, joint neutrals and pad gaps;
   retest frozen real-data ACT after verification.
5. Collect intervention negatives and matched learned/no-ACT ablations before
   attributing recovery success to ACT features.

Collector tracing is opt-in; maintained instructions live in
[`integrations/lerobot_trace/README.md`](../integrations/lerobot_trace/README.md).
It supplies neither exposure timestamps nor measured hardware overhead without
real traces. This summary describes software/simulation evidence and the
specific physical registration still required.
