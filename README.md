# Real2Sim-RaC: real SO-101 demonstrations → ACT → MuJoCo evaluation

## What this is

Forty teleoperated pick-and-place demonstrations were recorded on a real SO-101 arm (tennis
ball into a bowl) and published at [Hugging Face](https://huggingface.co/datasets/bottowork/so101_act_pickplace_v2). 

The main aim is to train an ACT policy with a recovery MLP head in Mujoco and use the frozen ACT weights for cross embodiment and use trained policy to work on 7DOF panda arm in MujOco


## Data collection

Teleoperation with an SO-101 leader/follower pair through LeRobot (dataset format v3.0,
25 Hz, follower PID P=16 I=0 D=4). Two cameras were recorded; observation.images.top (wrist camera), observation.images.iphone (global top view camera) - both recorded at 1280 * 720, couldn't get iphone to work with 640 * 480 but could be resized easily with pytorch tensor ops

Note every measurement is in inches first number is x axis(length wise) second number y axis (width wise) third z axis (height wise), if single number thats x axis
camera from ball: (30.5, 4.5, 16)
ball & bowl: 12 inches
Ball from robot base: (2,13)
mat: (16,12)
Bowl: 4.5 diameter
Ball: 8 circumference 

Note: there could be +- 5% measurments error.

## Current v2 workflow

The measured scene uses [physical_measurements.md](physical_measurements.md). The
completed matched benchmark scored **100/100 on each arm** with frozen HF ACT
followed by explicit RGB recovery, and 100/100 with RGB recovery alone. ACT alone
scored 0/100 in this simulation. Lighting and HF weights stayed unchanged;
recovery took over in every assisted trial. See the
[matched results and attribution](docs/v2_matched_benchmark_20261028.md) and
[camera alignment and robustness evidence](docs/v2_alignment_and_fixed_lighting_robustness_20261028.md).

A separate [recovery MLP and DAgger experiment](docs/v2_learned_recovery_dagger_20261028.md)
keeps ACT frozen and evaluates learned actions without the geometric teacher.
Those experimental heads have not established reliable complete-task performance
on both arms. The geometric controller remains the working fallback.

The [29 October progress report](docs/v2_progress_20261029.md) records the final
phase-head development result (3/10 per arm), completed checks, and the remaining
grasp-retention, camera-alignment and state-transfer bottlenecks.
The [retention and representation follow-up](docs/v2_retention_and_representation_20261029.md)
adds issued-command feedback, continuous teacher dwell, Panda source-mapping
checks, and an experimental optional phase guard. Retention and physical camera
calibration remain unresolved; the geometric fallback stays the working default.

Run the assisted system with the pinned HF `latest/` checkpoint:

```sh
.venv/bin/python scripts/evaluate_hf_recovery.py --robot so101 --seeds 500 501 502 503 504 \
  --out outputs/v2/hf_recovery_so101_NEXT --video
.venv/bin/python scripts/evaluate_hf_recovery.py --robot panda --seeds 500 501 502 503 504 \
  --out outputs/v2/hf_recovery_panda_NEXT --video
```

Evaluate the unchanged HF policy alone in the measured scene:

```sh
.venv/bin/python scripts/evaluate_hf_act.py --robot so101 --out outputs/v2/hf_only_so101_NEXT
.venv/bin/python scripts/evaluate_hf_act.py --robot panda --out outputs/v2/hf_only_panda_NEXT
```

Run the separate geometric baseline from the configured start:

```sh
.venv/bin/python scripts/run_visual_waypoint.py --robot so101 --seeds 0 1 2 3 4 \
  --config configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml \
  --out outputs/v2/waypoint_so101_NEXT --video
.venv/bin/python scripts/run_visual_waypoint.py --robot panda --seeds 0 1 2 3 4 \
  --config configs/v2/measured_registration_20261028/sim_panda_positive_all_scalars.yaml \
  --out outputs/v2/waypoint_panda_NEXT --video
```

## Install

```
uv sync --locked            # Python 3.12, LeRobot 0.6.1, MuJoCo 3.15, torch 2.11
uv run ruff check .
uv run pytest -q tests/unit
```

## Commands
```
# 1. audit the dataset (uses a local copy if given; downloads otherwise)
uv run python -m real2sim.data.audit --repo-id bottowork/so101_act_pickplace_v1 \
  --revision fd65a6600b0cd111a434ece1abfa3d44dc7b4d57 [--local-root <dir>] \
  --out results/audit --n-val 10 --exclude-episodes 25,33,34

# 2. build the 10 Hz derived datasets (train-only statistics, hashed manifest)
uv run python -m real2sim.data.prepare --mode joint     --config configs/data.yaml --local-root <dir> --out-dir .cache/derived_joint
uv run python -m real2sim.data.prepare --mode cartesian --config configs/data.yaml --local-root <dir> --out-dir .cache/derived_cartesian

# 3. train (smoke test first)
uv run python -m real2sim.policy.train --config configs/act_joint.yaml --out outputs/smoke --smoke-steps 2
uv run python -m real2sim.policy.train --config configs/act_joint.yaml --out outputs/act_joint_so101
uv run python -m real2sim.policy.train --config configs/act_cartesian.yaml --out outputs/act_cartesian

# 4. check a checkpoint against held-out episodes (open-loop, not task success)
uv run python -m real2sim.policy.check --checkpoint outputs/act_joint_so101/checkpoints/last --data .cache/derived_joint

# 5. simulation evaluation (see docs/experiment_protocol.md for the full command set)
uv run python -m real2sim.evaluation.run --robot so101 --checkpoint outputs/act_cartesian/checkpoints/006000 --n-trials 10 --out results/metrics/E2_cart_so101_v2 --video
uv run python -m real2sim.evaluation.oracle --n-trials 10 --out results/metrics/oracle_so101
uv run python scripts/aggregate_trials.py
```

## Models and physics assumptions

- ACT (LeRobot 0.6.1): ResNet-18, `dim_model` 256, chunk 20 at 10 Hz, 2 actions executed per inference or chunking 2s of actions before recalculating, batch 16. Joint baseline learns the recorded leader targets; the Cartesian policy
  learns achieved end-effector motion from forward kinematics.
- Simulation: MuJoCo Menagerie `robotstudio_so101` and `franka_emika_panda`. Ball and bowl are compound primitives.
- The trained model has been published [Hugging Face](https://huggingface.co/bottowork/so101-act-h256-b16)

## Experiments
- Replaced ACT VAE with flow matching head
- Replaced ACT resnet18 backbone with DinoV2 or Clip (fine tuned ViT on our dataset)
- Used end effector delta as action state
- Added a separate Cartesian recovery MLP (hidden widths 256→128, seven action outputs), trained with DAgger corrective labels around frozen ACT.
- On going experiments continued... 
