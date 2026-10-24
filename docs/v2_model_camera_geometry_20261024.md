# Model-assisted camera geometry from random frames — 24 October 2026

New Grounding DINO detections and SAM2 masks provide source-image evidence for the
randomly selected frames. They produce useful **conditional** camera estimates;
they do not identify every geometry parameter. Neither shared camera/mat-pose
model passes every whole-frame verification at the declared 3 px scale. No task,
joint mapping, collision model or checkpoint is changed by this diagnostic.

The [complete report](../results/v2/model_camera_geometry_20261024.json) retains
the fitted cameras, uncertainty scenarios, held-out errors, all six previous base
yaw basins and input hashes. Raw source images, model overlays and edge overlays
remain under `outputs/v2/random_vision_geometry_20261024`,
`grounded_geometry_20261024`, and `model_mat_edges_20261024_v4`.

## Selection and image measurements

Seed 20261024 selects three distinct training episodes in each predeclared early
(episodes below 7) and late camera regime, with one uniform random source frame
per episode. The first two frames in each regime fit the camera; the third
verifies it. Two additional random validation episodes supply verification frames.
Selection, source PNG/video hashes and whole-frame roles were fixed before model
inference. Reserved policy episodes are excluded.

The selected iPhone frames are early fit episode 4/frame 179 and 0/frame 98;
early verification 5/405; late fit 27/541 and 26/150; late verification 28/252,
31/299 and 13/696. All eight remain in the final diagnostic, without substitution.

The model's visible mat mask excludes paper, bowl, cable and robot occlusions.
Orientation-constrained straight-segment consensus proposes each outer edge.
The source grayscale image then refines dark-to-light transitions within 6 px.
Rules retain contrast at least 12 grayscale levels, 3 px model-consensus support,
2.5 px refined-line support and at least 80 px visible span. Intersecting visible
straight lines gives virtual corners rather than rounded-corner pixel centers.

Two concrete extraction errors were found in **training fit frames**: a convex
hull's supposed top-right corner could be a paper occlusion, and the longest
horizontal mask cut could follow a white gripper instead of the mat's outer top.
The final rules use line consensus and the top mask's upper envelope. Those rules
are frozen before the final verification projection scores; intermediate failed
extraction versions are retained locally. Earlier exploratory verification scores
were inspected during debugging, so verification here means exclusion from numerical
camera fitting, not a pristine untouched acceptance set. The declared 3 px working uncertainty
is an assumption, not the segmenter's confidence or a calibrated error interval.

## Conditional camera estimates

Mat side lengths are the existing operator/config values, 0.305 by 0.406 m, with
the same physical-axis assignment hypothesis used previously. The baseline assumes
square pixels, source-center principal point and zero distortion. It shares one
camera-to-mat pose within each regime and fits only training-frame corner pixels.
The mat frame is **not** an independently registered robot base frame.

| Regime | Focal length at 1280×720 | Camera distance normal to mat | Vertical FOV |
|---|---:|---:|---:|
| Early | 1399.17 px | 0.53837 m | 28.86° |
| Late | 1354.48 px | 0.53573 m | 29.77° |

The baseline distances are compatible with the existing approximate 21-inch
operator camera-height value under these assumptions. This agreement does not
independently verify camera calibration or applicability of the older measurement.

| Whole source frame | Role | Corner RMS under the shared regime camera/mat pose |
|---|---|---:|
| 4/179 | Early fit | 5.71 px |
| 0/98 | Early fit | 6.04 px |
| 5/405 | Early verification | 6.15 px |
| 27/541 | Late fit | 0.53 px |
| 26/150 | Late fit | 0.69 px |
| 28/252 | Late verification | 9.55 px |
| 31/299 | Late verification | 0.64 px |
| 13/696 | Late verification | 30.74 px |

A low residual on the **pooled** corners (1.45 px early and 0.46 px late) hides
frame disagreement. The report therefore keeps every frame's error. The late
episode-13 mat corners differ substantially from the fitted placement; segmentation,
occlusion, a moved/rotated mat, camera changes and intrinsic assumptions remain
possible causes. These observations do not uniquely distinguish them. The shared
pose is consequently a diagnostic hypothesis, not an accepted alignment.

The camera is locally full rank, 7/7, only under the baseline optical restrictions.
Freeing fx/fy/cx/cy exposes rank 8/10, and additionally freeing two radial-distortion
parameters gives rank 8/12. Repeated views of the same plane and pose do not remove
those null directions. The deterministic principal-point, pixel-aspect, distortion,
mat-size and corner perturbations are scenario envelopes, not confidence intervals.

## Fixed-foot CAD cross-check

The SAM foot mask comes from a recorded, VLM-chosen box prompt. It covers a visible
front pad and may include sidewall, orange clamp or occluded pixels. It is not a
named physical CAD correspondence. All six earlier repeated-hole XY/yaw basins
are projected through the new mat cameras, without refitting a base pose.

The top-surface-only comparison favors a different basin from the solid-foot
comparison. This is a useful warning about image-model target semantics. Including
the CAD lower-foot sidewalls gives basin 0 symmetric contour RMS 9.59–16.71 px,
with mask IoU 0.494–0.729 across eight frames. The other candidate basins and both
directions' uncapped contour errors remain in the report. None is promoted from
model confidence, nearest repeated holes or a selectively chosen median.

These errors combine possible camera/base placement mismatch, printed-part/CAD
differences, segmentation error and unmatched occlusion semantics. They do not
justify accepting a base transform or fitting physical joint zeros against it.
Unique static identities or additional independent geometry remain needed to
separate those causes.

Reproduce with fresh output directories:

```sh
.venv/bin/python scripts/extract_v2_model_mat_edges.py \
  --model outputs/v2/grounded_geometry_20261024/report.json \
  --out outputs/v2/model_mat_edges_NEXT
.venv/bin/python scripts/diag_v2_multiframe_camera.py \
  --annotations outputs/v2/model_mat_edges_NEXT/annotations.json \
  --out outputs/v2/model_multiframe_camera_NEXT
.venv/bin/python scripts/diag_v2_model_base_silhouette.py \
  --masks outputs/v2/model_mat_edges_NEXT/foot_masks.json \
  --camera outputs/v2/model_multiframe_camera_NEXT/report.json \
  --out outputs/v2/model_base_silhouette_NEXT
```

Nine focused tests pass, including whole-frame split preservation, source checksums,
free-intrinsic null directions, verification independence, occluded-gripper edge
rejection and missing raw-contrast refusal. Ruff passes. The contour rasterizer
operates on copied CAD arrays and needs no graphics context or simulator mutation.
