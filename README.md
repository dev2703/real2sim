# Real2Sim-RaC

Real SO-101 pick-and-place demonstrations → frozen ACT → SO-101/Panda MuJoCo
evaluation and recovery. The working system adds explicit RGB geometric recovery.
In the matched simulation benchmark, ACT alone completed 0/100 tasks per arm;
RGB alone and ACT followed by RGB recovery each completed 100/100. Recovery took
over in every assisted trial. The experimental learned phase-recovery head
completed 3/10 development tasks per arm.

[Project summary](docs/PROJECT_SUMMARY.md) contains the architecture, data/control
contracts, results, experiment history, remaining problems and commands.
[Physical measurements](physical_measurements.md) retain the operator's inputs.

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

# Collect, fit ACT/no-ACT recovery heads, and compare both arms on matched resets
uv run python scripts/run_cross_embodiment_recovery.py \
  --out outputs/v2/recovery_pipeline_NEW --matched-benchmark --plan-only
# Remove --plan-only to execute. Add --train-robots so101 to test source-only
# training followed by SO-101/Panda evaluation; defaults train on both arms.
# --recovery-frame source also tests the explicit source-frame Panda adapter.
```

The recovery workflow keeps ACT weights frozen, supports phase or residual MLP
heads, and separates training, validation and development episodes. It records
unseen target arms explicitly. These are experimental learned heads; reliable
learned cross-embodiment task completion remains unresolved.

Sources: [demonstrations](https://huggingface.co/datasets/bottowork/so101_act_pickplace_v2)
and [native ACT checkpoint](https://huggingface.co/bottowork/so101-act-h256-b16).
The native runner uses the pinned `latest/` checkpoint and its saved processors.
Simulation results do not establish physical robot performance or calibrated
camera registration.
