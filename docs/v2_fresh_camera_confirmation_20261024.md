# Fresh camera confirmation with frozen vision rules — 24 October 2026

Two new source frames independently confirm that the current shared camera-to-mat
alignment is not ready for adoption. Their corner RMS errors are **14.54 px** and
**30.57 px**, exceeding the predeclared 3 px working scale. The detector prompts,
segmentation settings, edge rules and original camera K/T remained unchanged.
Nothing was refitted or substituted after these results.

This check addresses the development-data limitation in the earlier
[model-assisted camera investigation](v2_model_camera_geometry_20261024.md): earlier
verification scores had been inspected while debugging. The new frames were chosen
only after freezing that investigation's final pipeline.

The [confirmation report](../results/v2/fresh_model_camera_confirmation_20261024.json)
records the full frozen plan, input hashes, observed/predicted corner pixels and
all errors. The persisted plan at
`outputs/v2/fresh_camera_confirmation_20261024/plan.json` precedes image decoding
and model inference. Seed **2026102402** selects one unused early training episode
from `[2,3,6]`, one unused late training episode, and a uniform random source frame
in each. All eight original development episodes are excluded. Both new images
are assigned the role `confirmation`; reserved policy episodes remain untouched.

| New source frame | Regime | Corner RMS | Corner p95 | Fixed 3 px result |
|---|---|---:|---:|---|
| Episode 2 / frame 82 | Early | 14.544 px | 18.790 px | Fail |
| Episode 15 / frame 19 | Late | 30.574 px | 42.050 px | Fail |

Grounding DINO and SAM2 run locally using the same pinned weights, text/box prompts
and thresholds as the development investigation. The exact previously executed v4
edge extractor runs from its retained source snapshot. Both frames produce usable
edge measurements; neither is excluded or replaced. The original early/late K/T
projects the same assumed 0.305 by 0.406 m mat corners, with no new optimization.
The scorer checks every frozen source-file hash before computing results.

The failure applies to the **shared camera/mat-pose hypothesis**. Segmentation and
edge error, a moved/rotated mat, camera changes, physical dimensions and restrictive
optical assumptions remain possible contributors. These two images do not uniquely
separate those causes, identify hidden physical parameters, verify named CAD
identities, or validate joint zeros. The conditional focal/height estimates from
the development report remain exploratory; this confirmation provides no basis
to adopt them or regenerate policies as physically corrected labels.

Artifacts remain under `outputs/v2/fresh_camera_confirmation_20261024`,
`fresh_camera_confirmation_models_20261024`, and
`fresh_camera_confirmation_edges_20261024`. The reproducible workflow is:

```sh
.venv/bin/python scripts/confirm_v2_model_camera.py prepare \
  --out outputs/v2/fresh_camera_confirmation_NEXT
.venv/bin/python scripts/measure_v2_vision_geometry.py \
  --selection outputs/v2/fresh_camera_confirmation_NEXT/selection.json \
  --out outputs/v2/fresh_camera_confirmation_models_NEXT --camera iphone
.venv/bin/python outputs/v2/model_mat_edges_20261024_v4/executed_extractor.py \
  --model outputs/v2/fresh_camera_confirmation_models_NEXT/report.json \
  --out outputs/v2/fresh_camera_confirmation_edges_NEXT
.venv/bin/python scripts/confirm_v2_model_camera.py score \
  --root outputs/v2/fresh_camera_confirmation_NEXT \
  --model outputs/v2/fresh_camera_confirmation_models_NEXT/report.json \
  --annotations outputs/v2/fresh_camera_confirmation_edges_NEXT/annotations.json \
  --out results/v2/fresh_camera_confirmation_NEXT.json
```

Replaying that seed reproduces the same confirmation pair; it is not an additional
independent sample. A later adaptive experiment needs its own declared sampling and
new confirmation protocol. The actual two-frame workflow and all frozen hash checks
completed successfully; Ruff passes for the confirmation script. No geometry,
joint/collision limit, contact option, policy checkpoint or reserved evaluation
was changed.
