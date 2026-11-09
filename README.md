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

The recovery workflow keeps ACT weights frozen, supports phase or residual MLP
heads, and separates training, validation and development episodes. It records
unseen target arms explicitly. Grasp-relative targets, direct ACT-conditioned
actions and a task-event jaw clock are versioned options; their presence does
not establish better task performance. Reliable source-only transfer and robust
learned recovery on both arms remain unresolved.

An experimental direct weight edit is available separately from frozen ACT:

```sh
# Set SNAPSHOT to the confirmed v3 export directory above.
uv run python -m real2sim.policy.weight_edit_collection --snapshot "$SNAPSHOT" \
  --out outputs/v2/weight_edit_features_NEW \
  --train-seeds 10090 10091 10092 --val-seeds 11090 11091
uv run python -m real2sim.policy.weight_edit --snapshot "$SNAPSHOT" \
  --data outputs/v2/weight_edit_features_NEW --out outputs/v2/weight_edit_NEW
```

This fits only ACT's final action layer against successful RGB teacher commands,
selects regularization and edit strength on separate episodes, and exports a
new native checkpoint. Evaluate that export with the matched benchmark before
using it. See [the experiment notes](docs/ACT_WEIGHT_EDIT_20261109.md) for results
and the distinction between action accuracy and completed tasks.

Sources: [demonstrations](https://huggingface.co/datasets/bottowork/so101_act_pickplace_v2)
and [native ACT checkpoint](https://huggingface.co/bottowork/so101-act-h256-b16).
