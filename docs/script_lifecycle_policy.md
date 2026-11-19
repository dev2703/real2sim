# Script lifecycle policy

To keep experiment logic testable and avoid script sprawl, follow these rules:

1. **Helper-first logic**: new experiment logic belongs in `src/real2sim/`
   modules, with unit tests under `tests/unit/`. Scripts in `scripts/` should
   stay as thin CLI orchestration wrappers.
2. **One-off diagnostics**: temporary diagnostics must be date-stamped (for
   example, `diag_v2_example_20261103.py`) and archived or removed after use.
   Any restore-manifest discipline used for cleanup/restore must be preserved
   when files are archived.
3. **Supported entry points**: supported CLIs must stay listed in the root
   `README.md` under `## Supported CLIs`.

## Guardrails

- Keep the tracked script count under **120** Python files in `scripts/`.
- Keep `diag_*.py` tracked drivers at or below **32**.

If a change requires exceeding these guardrails, clean up/retire scripts in the
same change or explicitly revise this policy and rationale.

The V2 wrist-zero and camera-uncertainty drivers were retired in `a041d2a`.
The maintained registration tests cover fit and verification separation, the
free-landmark wrist-roll ambiguity, and unclipped joint-limit reporting. The
retired drivers' native-base-to-mat transform and conditional four-corner fit
with shifted principal point, unequal focal lengths, and radial distortion
were specific to those diagnostics. The maintained planar initializer assumes
centered, square-pixel, distortion-free intrinsics.
