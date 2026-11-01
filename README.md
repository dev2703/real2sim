# Real2Sim-RaC

Real SO-101 pick-and-place demonstrations → frozen ACT → SO-101/Panda MuJoCo
evaluation and recovery. Learned recovery now uses achieved robot motion,
issued jaw commands and RGB target history, with iterative corrective training.
Frozen ACT can hand over to a separately trained MLP. Native-start recovery and
post-ACT recovery use different head artifacts because their starting states
differ. Both arms supply training data; source-only transfer remains unresolved.


```sh
uv sync --locked                      # Python 3.12; pinned dependencies in uv.lock
uv run ruff check .
uv run pytest -q tests/unit
```

Run from the repository root with a new output directory. Replace `so101` with
`panda` for the other arm:

```sh
# Frozen HF ACT followed by RGB geometric recovery
uv run python scripts/evaluate_hf_recovery.py --robot so101 \
  --seeds 500 501 502 503 504 --out outputs/v2/hf_recovery_so101_NEW --video

# Unassisted frozen HF ACT
uv run python scripts/evaluate_hf_act.py --robot so101 \
  --out outputs/v2/hf_only_so101_NEW

# Matched resets: raw ACT, RGB alone, ACT + RGB recovery
uv run python scripts/benchmark_frozen_act.py --robots so101 \
  --out outputs/v2/benchmark_so101_NEW --plan-only

# Verify a completed matched report against its retained trial/trace evidence
uv run python scripts/audit_frozen_act_results.py \
  outputs/v2/benchmark_so101_NEW/report.json

# Frozen ACT followed by a head trained on post-prefix states
uv run python scripts/run_recovery_dagger.py evaluate --robot panda \
  --head outputs/v2/recovery_prefix_adapt_noact_20261030 --act-prefix \
  --seeds 12040 12041 --out outputs/v2/act_mlp_panda_NEW

# Inspect collection, DAgger, training and matched handoff comparison commands
uv run python scripts/run_cross_embodiment_recovery.py \
  --history-mode achieved --action-routing hard --transition-weight .1 \
  --dagger-iterations 1 --dagger-beta .5 --dagger-learner no_act \
  --collection-act-prefix-steps 50 --frozen-prefix --prefix-recovery no_act \
  --out outputs/v2/recovery_pipeline_NEW --matched-benchmark --plan-only
# Remove --plan-only to execute. Add --train-robots so101 to test source-only
# training followed by SO-101/Panda evaluation; defaults train on both arms.
```

The recovery workflow keeps ACT weights frozen, supports phase or residual MLP
heads, and separates training, validation and development episodes. It records
unseen target arms explicitly. Grasp-relative targets, direct ACT-conditioned
actions and a task-event jaw clock are versioned options; their presence does
not establish better task performance. Reliable source-only transfer and robust
learned recovery on both arms remain unresolved.

Sources: [demonstrations](https://huggingface.co/datasets/bottowork/so101_act_pickplace_v2)
and [native ACT checkpoint](https://huggingface.co/bottowork/so101-act-h256-b16).



