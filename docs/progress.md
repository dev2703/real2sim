# Current project progress

The current offline screen is documented in
[the learning runbook](v2_learning_screen_20261022.md). Its live status is
`outputs/v2/learning_screen_20261022/STATUS.json`; do not infer completion
from a dated prose update.

The screen covers camera selection, backbones, batch sizes and learning rates,
fine-tuning, matched flow/regression heads and alternative action targets.
Physical geometry remains provisional: see
[joint-zero and camera findings](v2_geometry_alignment_20261021.md).
Offline prediction errors do not establish physical task success or Panda transfer.

After the queue finishes or fails, inspect its reports and compare executed-prefix
errors, phase errors and persistence baselines before selecting models. Preserve
the original frozen runs when fixing problems found during review.

The previous chronological work log was removed during the 22 October cleanup.
Contracts, decisions, calibration evidence and experiment reports remain in their
dedicated documents. Use [the documentation index](README.md) to find them.
