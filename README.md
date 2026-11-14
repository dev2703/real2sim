# Real2Sim-RaC

Real SO-101 pick-and-place demonstrations → frozen ACT → SO-101/Panda MuJoCo
evaluation and recovery. Learned recovery now uses achieved robot motion,
issued jaw commands and RGB target history, with iterative corrective training.
Frozen ACT can hand over to a separately trained MLP.

```sh
uv sync --locked                      # Python 3.12; pinned dependencies in uv.lock
uv run ruff check .
uv run pytest -q tests/unit
```

## Supported CLIs

The following script entry points are supported and should remain listed here:

- `scripts/evaluate_hf_recovery.py`
- `scripts/evaluate_hf_act.py`
- `scripts/benchmark_frozen_act.py`
- `scripts/audit_frozen_act_results.py`
- `scripts/run_recovery_dagger.py`
- `scripts/run_cross_embodiment_recovery.py`
- `scripts/train_recovery_mlp.py`
- `scripts/train_phase_recovery_mlp.py`

An experimental direct weight edit is available separately from frozen ACT:

```sh
# Set SNAPSHOT to your local confirmed v3 export (revision in the notes below).
uv run python -m real2sim.policy.weight_edit_collection --snapshot "$SNAPSHOT" \
  --out outputs/v2/weight_edit_features_NEW \
  --robots so101 panda \
  --source-scene configs/v2/weight_space_20261111/sim_so101.yaml \
  --train-seeds 10090 10091 10092 --val-seeds 11090 11091
uv run python -m real2sim.policy.weight_edit --snapshot "$SNAPSHOT" \
  --data outputs/v2/weight_edit_features_NEW --out outputs/v2/weight_edit_NEW
```

Sources: [demonstrations](https://huggingface.co/datasets/bottowork/so101_act_pickplace_v2)
and [native ACT checkpoint](https://huggingface.co/bottowork/so101-act-h256-b16).


## MLX recovery training

On Apple Silicon, install the optional backend with
`uv pip install --python .venv/bin/python mlx==0.32.4` and add `--backend mlx`
to `scripts/train_recovery_mlp.py`. Gradients and AdamW run in MLX/Metal;
the selected weights export to the existing `head.pt` runtime contract.
ACT stays frozen. Use `scripts/run_recovery_dagger.py evaluate --act-prefix`
for observable stall-triggered, latched handover to that head.

Train aggregated original and corrective packs with optional `--balance-phases`
to balance issued-command stages within each robot/episode. Phase labels affect
training sampling only and are never runtime inputs. Validation selects the
checkpoint; action loss does not establish closed-loop task success.

For learner-visited corrective data, use `run_recovery_dagger.py collect`
with `--head PREVIOUS_HEAD --beta 0.5 --iteration 1 --reference-mode tracked
--act-prefix-until-stall`. This uses the evaluation stall monitor instead of a
fixed ACT prefix. Preserve the head's history, geometry and gripper-clock options.
See [the matched DAgger findings](docs/DAGGER_STAGES_20261115.md).

An optional `--retention-guard` in recovery evaluation (or benchmark controller
option `"retention_guard": true`) adds observable grasp/retain/release sequencing,
loss reacquisition and rejection holds. It requires tracked RGB and native-frame
recovery, and remains off by default. Both matched smoke tasks failed and Panda
lift regressed; this is experimental assistance, not a promoted controller.
See [the retention findings](docs/RETENTION_SEQUENCING_20261115.md).
