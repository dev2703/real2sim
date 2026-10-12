# AGENTS.md — operating rules for coding agents in this repository

1. Read `plan.md` first. It is the specification. `docs/implementation_plan.md` is the status
   board; `docs/decisions.md` records what is approved and what is pending.
2. Human approval protocol (plan §0.1): architecture, system design, milestone design, skill
   documents and material technical decisions need an explicit user choice between presented
   options. Silence is not approval. Keep working on independent, already-approved tasks while
   waiting; mark blocked branches `BLOCKED — AWAITING DECISION`.
3. Never command physical robot motion, change firmware, PID gains, motor registers, torque
   limits, `max_relative_target` or calibration. Hardware is operator-controlled.
4. Never fabricate episodes, metadata, metrics, images, checkpoints, calibration values or
   success. Report `0/N` when that is the result. Replay or scripted motion is captioned as such
   and is never an ACT rollout.
5. The primary policy gets no privileged simulator object poses. Oracle inputs exist only for
   explicitly labelled diagnostics with identical feature schemas.
6. Do not change camera keys, action semantics, frames, frequency, normalisation or splits
   silently; every change goes through config + tests + `docs/decisions.md`.
7. Commands: `uv sync --locked`, `uv run ruff check .`, `uv run pytest -q tests/unit`,
   `uv run pytest -q tests/integration`, `uv run python -m real2sim.<module> --help`.
   Tests must really run; no placeholder passes.
8. After each milestone, append to `docs/progress.md` (commit, commands, verified outcomes,
   blockers, next action) and report in chat using the plan §9 block.
9. Ownership: integrator owns data/policy/evaluation/docs/merges; the simulation worktree owns
   `src/real2sim/sim/**`, `assets/**` and sim integration tests. No concurrent edits to
   `docs/data_contract.md`, `configs/data.yaml`, `pyproject.toml`.
10. No `.agents/skills/` or `SKILL.md` before gate G1, and then only with user approval.
