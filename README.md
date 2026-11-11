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
# Set SNAPSHOT to your local confirmed v3 export (revision in the notes below).
uv run python -m real2sim.policy.weight_edit_collection --snapshot "$SNAPSHOT" \
  --out outputs/v2/weight_edit_features_NEW \
  --robots so101 panda \
  --source-scene configs/v2/weight_space_20261111/sim_so101.yaml \
  --train-seeds 10090 10091 10092 --val-seeds 11090 11091
uv run python -m real2sim.policy.weight_edit --snapshot "$SNAPSHOT" \
  --data outputs/v2/weight_edit_features_NEW --out outputs/v2/weight_edit_NEW
```

This fits only ACT's final action layer against successful RGB teacher commands,
balances the two arms, selects regularization and edit strength on separate
episodes without worsening either arm's validation error, and exports a new
native checkpoint. Panda commands are projected through the existing robot
geometry adapter; this is target-arm supervision. Use `--robots so101` for
source-only fitting. Corrective collection accepts `--rollin-snapshot EDITED`
and `--act-prefix-steps 50`; prefix commands never become teacher labels.
Evaluate each export with the matched benchmark before using it. See the
[initial experiment](docs/ACT_WEIGHT_EDIT_20261109.md) and
[both-arm investigation](docs/ACT_WEIGHT_SPACE_20261111.md) for results and the
distinction between action accuracy and completed tasks. An explicit
`render.near_clip_m: 0.005` prevents scenery extent from choosing a large near
clipping distance; this optional setting applies to both robot scenes.

The decoder editor also fits the existing final decoder feed-forward layers
without adding a runtime network. Collect with `--feature-layer decoder_ffn_input`
and fit with `python -m real2sim.policy.decoder_weight_edit`; see the both-arm
investigation for complete commands. The candidate scene remains exploratory.

Sources: [demonstrations](https://huggingface.co/datasets/bottowork/so101_act_pickplace_v2)
and [native ACT checkpoint](https://huggingface.co/bottowork/so101-act-h256-b16).
