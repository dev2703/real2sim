# V2 offline learning screen — 22 October 2026

The operator requested all eight experiment groups run before geometry acceptance:
"Run these experiments and will see later". This screen uses the existing provisional
joint mapping and recorded camera images. It does not claim calibrated robot geometry,
simulated placement success or Panda transfer.

## Comparisons and training budget

`results/v2/screen_matrix_v4.json` declares 24 entries with 22 unique training runs.
The duplicate frozen-DINO and task-delta controls reuse their identical trained runs.
Every unique run receives 64,000 sampled training anchors, seed 0, 25-action chunks at
12.5 Hz and the configured effective batch of 16 or 64. Validation/checkpoints occur
every 8,000 anchors. The learning-rate schedule retains its 320,000-anchor horizon.

The queue includes iPhone/wrist/both-camera controls, ResNet18/frozen DINOv2,
batch/LR sweeps, DINOv2 last-two-block fine-tuning, frozen/fine-tuned CLIP, matched
DINO/CLIP regression and flow heads, joint/absolute/task-delta actions, and tool-frame
deltas with/without trajectory-consistency loss.

The Apple M3 uses MPS with physical batch 8 and gradient accumulation 2 or 8.
Short GPU checks fit within the available memory for ResNet and both fine-tuned
transformers. All pretrained weights were already cached; the run uses local compute.

## Data and provenance

The six datasets in `.cache/v2_exploratory_20261022` share the same source rows and
image files. There are 13,878 ticks: 9,617 training, 2,077 validation and 2,184 reserved.
The 29/6/6 episode split is unchanged. Statistics use training episodes only, and all
action variants match a direct reconstruction from the raw recordings bitwise.
See [the preparation audit](../results/v2/exploratory_data_audit_20261022.json).

`configs/v2/task.exploratory_20261022.yaml` and the prepared-data contracts explicitly
mark geometry as provisional. The queue records unmet physical gates as deferred by
the operator. It verifies prepared payload hashes and freezes its source/configuration
inputs; it never fabricates passing calibration evidence.

## Execution and monitoring

The persistent run directory is `outputs/v2/learning_screen_20261022`:

- `STATUS.json`: active stage, child PID, completed jobs, attempts and failures.
- `PLAN.json` and `INPUTS.json`: comparisons, effective budgets and frozen hashes.
- `logs/`: feature-cache and training logs.
- `runs/<arm>/attempt-NNN/`: checkpoints, validation metrics and training reports.
- `feature_caches/`: shared frozen features for the matched head comparisons.

The launch/resume command, from the repository root with local Apple GPU access, is:

```sh
.venv/bin/python scripts/run_v2_learning_screen.py \
  --data-root .cache/v2_exploratory_20261022 \
  --task configs/v2/task.exploratory_20261022.yaml \
  --out outputs/v2/learning_screen_20261022 \
  --device mps --physical-batch 8 --threads 4 --defer-geometry-checks
```

The runner serializes GPU work with the shared v2 training lock. It runs the paired
DINO heads, then paired CLIP heads, then ACT comparisons. Completed runs are reused
when the command resumes. Interrupted runs retain their optimizer/sampler/RNG states;
recovery cannot silently add training examples beyond the declared budget.

Disk space is limited. Runs retain best/last inference weights and metrics. Once a full
screen finishes, its optimizer states are released and the status explicitly marks
`exact_resume_after_screen: false`; further training from completed screens requires
a new optimizer. Feature arrays are released after both matched heads finish, with
their manifests/backbone weights retained. Existing caches and historical runs remain
outside this cleanup. The runner stops if less than 3 GiB remains before another stage.

The detached launch also starts `caffeinate` while the queue process is alive. Keeping
the computer powered and awake is necessary for the local experiments to progress.

## Hardware compatibility fix

The trajectory loss previously used `torch.matrix_exp`, which this MPS build does not
implement. Its SO(3) step now uses an equivalent Rodrigues formula with a stable
small-angle branch. Values and gradients match the CPU matrix exponential, including
zero rotation, and a native MPS forward/backward check passed. Composition still runs
in float32 or higher, preserving the earlier mixed-precision fix.

Compare learning results within their declared camera/backbone/action contracts.
Different action dimensions/units and unresolved FK geometry prevent treating all
normalized validation losses as one task-performance ranking. Physical reference
replay and calibrated closed-loop evaluation remain later work.
