# Operator-anchored geometry snapshot, 13 October 2026

`camera_measured.json` is the existing output of
`scripts/diag_camera_fit_measured.py`, copied from `outputs/tmp_appearance/` for
provenance. It has not been regenerated during commit preparation. The script
uses hand-read robot/mat pixels and operator-supplied dimensions; it is a
constrained estimate, not independently validated calibration.

The parent directory's PnP/backprojection reports use the superseded geometry.
Primary hold, oracle and learned-policy tables also require regeneration under
`configs/task.yaml`. Preserve their historical configurations when comparing.
