# Real2Sim-RaC: real SO-101 demonstrations → ACT → MuJoCo evaluation

## What this is

Forty teleoperated pick-and-place demonstrations were recorded on a real SO-101 arm (tennis
ball into a bowl) and published at [Hugging Face](https://huggingface.co/datasets/bottowork/so101_act_pickplace_v2). 

The main aim is to train an ACT policy with a recovery MLP head in Mujoco and use the frozen ACT weights for cross embodiment and use trained policy to work on 7DOF panda arm in MujOco


## Data collection

Teleoperation with an SO-101 leader/follower pair through LeRobot (dataset format v3.0,
25 Hz, follower PID P=16 I=0 D=4). Two cameras were recorded; observation.images.top (wrist camera), observation.images.iphone (global top view camera) - both recorded at 1280 * 720, couldn't get iphone to work with 640 * 480 but could be resized easily with pytorch tensor ops 

## Current v2 workflow

The local dataset has **41 episodes**. Reviewed conditions are 32 baseline, four ball shifts
(6, 8, 9, 10), and five bowl shifts (21–25), with a seeded 29/6/6 episode split.
See the [v2 runbook](docs/v2_implementation_20261020.md) for the 12.5 Hz two-camera pipeline,
CLIP/DINO/flow/kinematics experiments and remaining calibration gates. These experiments are
implemented; v2 learned task success and frozen Panda transfer have not yet been demonstrated.
Current priority is [real/simulation geometry alignment](docs/v2_geometry_alignment_20261021.md),
including the encoder-derived gripper-scale discrepancy and CAD/table collision diagnosis.
The commands below document the retained legacy 10 Hz workflow.

See [cleanup and recovery](docs/cleanup_20261021.md) for removed historical scripts and artifacts.

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

- ACT (LeRobot 0.6.1): ResNet-18, `dim_model` 512, chunk 20 at 10 Hz, 2 actions executed per inference, batch 16. Joint baseline learns the recorded leader targets; the Cartesian policy
  learns achieved end-effector motion from forward kinematics.
- Simulation: MuJoCo Menagerie `robotstudio_so101` and `franka_emika_panda`. Ball and bowl are compound primitives. Physical values are
  provisional standards not precise measurement 

## Experiments
- Replaced ACT VAE with flow matching head
- Replaced ACT resnet18 backbone with DinoV2 or Clip (fine tuned ViT on our dataset)
- Used end effector delta as action state
- Added a recovery MLP head (256->128->512) 
- On going experiments continued... 
