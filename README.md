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

The local dataset has **41 episodes**. Reviewed conditions are 32 baseline, four ball shifts
(6, 8, 9, 10), and five bowl shifts (21–25), with a seeded 29/6/6 episode split.
See the [v2 runbook](docs/v2_implementation_20261020.md) for the 12.5 Hz two-camera pipeline,
CLIP/DINO/flow/kinematics experiments and remaining calibration gates. These experiments are
implemented; v2 learned task success and frozen Panda transfer have not yet been demonstrated.
Current priority is [fixed-base landmark tracking](docs/v2_fixed_landmark_tracking_20261025.md),
followed by independent camera/base registration, joint-zero and gripper verification.
The measured scalars in `configs/task.yaml` remain in use; image tracking does not verify
metric registration. The completed [offline learning screen](docs/v2_learning_screen_20261022.md)
used provisional geometry; physical registration remains unresolved.
The commands below document the retained legacy 10 Hz workflow.

See [documentation index](docs/README.md) for the current contracts and runbooks.
The latest [vision alignment and bottleneck experiments](docs/v2_vision_alignment_and_bottlenecks_20261024.md)
measure randomly selected source frames with local Grounding DINO/SAM2, test fresh
camera confirmation, and compare auxiliary keypoints and feasible follower targets.
The [24 October corrective experiments](docs/v2_corrective_experiments_20261024.md)
add explicit ball-image features and motion-sensitive validation, verify the existing
nominal wrist camera binding, and compare feasible control targets. Independent
physical registration is still required before corrected-label retraining or
reserved checkpoint promotion.
The [matched-object action probes](docs/v2_matched_object_actions_20261025.md) now compare
similar arm states across different ball/bowl layouts, including stationary starts and
localized image swaps. They expose weak action dependence separately from localization.
The [three-episode capacity test](docs/v2_three_episode_motion_20261025.md) checks each
training condition against hold/persistence on movement and stationary-run error before
common development rollouts.
The [executed-prefix comparison](docs/v2_prefix_capacity_20261025.md) tests two-action
supervision with the same model and data. Training errors improve, but the capacity
gate still fails and [matched object probes](docs/v2_prefix_object_actions_20261025.md)
show mixed action dependence. Its [motion-onset audit](docs/v2_motion_onset_audit_20261025.md)
keeps ambiguous examples visible. The [calibration capture sheet](docs/v2_calibration_capture_20261025.md)
includes a blank measurement log and printable board for the independent physical checks.

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
