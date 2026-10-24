# Fixed-base tracking and mat-boundary audit — 25 October 2026

The corrected frozen run tracks **115 source frames**, with two supported fixed regions in **112**. The withheld upper-base region agrees with the fitted image similarity: its maximum per-frame p95 residual on supported frames is **2.09 px**. This establishes useful local image-coordinate consistency, not physical camera/base calibration.

The [summary and per-ray range audit](../results/v2/fixed_landmark_tracking_20261025.json) consume the completed [tracker report](../outputs/v2/fixed_landmark_tracks_20261025/report.json) and its frozen plan. The audit does not refit tracks, change thresholds or select replacement frames.

## Corrected reference and selection

The 24 October prototype placed a cabinet-handle ROI over the wrist in its arm-clear reference. Those points were moving-arm features; RANSAC rejected them in that completed run. The corrected run removes the ROI and uses only the fixed foot and lower table clamp. The upper fixed-base holder, including tentative E/I image anchors, stays outside the fit. The original run and executed source are retained.

Seed **2026102501** selected one frame from each third of each of the 35 training/validation episodes. All **189** frames in the supplied prior tracking, first-frame stability and model-measurement inventories were excluded before decoding. This yields **105** newly selected frames plus **10** earlier model-measured frames scored separately. Freshness is relative to those inventories; it is not a claim about every historical inspection. Reserved episodes are excluded. Re-decoded earlier model frames were checked for exact pixel identity.

The limited-support frames are 9/494, 25/29, 27/682. They remain in the record. The summary below excludes their relative-edge values from the supported, uncensored distribution.

## Image motion and withheld base checks

| Capture group | Supported frames | Median image scale | Median per-frame static displacement |
|---|---:|---:|---:|
| Episodes below 7 | 22 | 1.00180 | 0.64 px |
| Episodes from 7 | 90 | 0.97222 | 11.78 px |

These are changes in capture coordinates. The similarity can combine camera pose, cropping, stabilization, focus/zoom and parallax. Both fitting regions lie on the left of the image; there is no independently tracked distant background. Agreement at the withheld holder does not quantify extrapolation error across the mat.

| Tentative image anchor | Accepted / attempted | Median / maximum similarity residual |
|---|---:|---:|
| candidate_E_upper_opening | 74 / 115 | 0.41 / 1.28 px |
| candidate_I_side_hole | 95 / 115 | 0.27 / 1.01 px |

E/I names remain unverified CAD identities. Their successful image tracks do not validate metric points, physical joint zeros or jaw registration.

## Raw mat edges and search-range limits

Raw signed-contrast scans use the frozen corridors; no segmentation mask enters their extraction. This audit reconstructs each retained ray's offset from the saved scan geometry. It flags a target edge if any retained ray lies within **2 px of the ±32 px search boundary**. A flag indicates possible truncation; it does not prove that the true edge is outside the corridor. Search windows are not widened after inspecting these results.

| Edge | Measured target frames | Relative measurements | Range-limited targets | Supported, uncensored relative frames |
|---|---:|---:|---:|---:|
| left | 115 | 115 | 43 | 70 |
| right | 30 | 30 | 0 | 29 |
| top | 69 | 69 | 0 | 69 |
| bottom | 32 | 0 | 3 | 0 |

The late left-boundary measurements often approach the search limit and must not be treated as complete uncensored displacement estimates. The reference bottom edge fails the predeclared line-consensus threshold, so **no relative bottom measurement is available**, even where a target bottom is visible. All available scan corridors, including the contrast-sampling offsets, remain inside the source images.

The following descriptive ranges use only supported static fits and measured reference/target edges without a flagged retained ray. They are per-frame signed normal displacement in reference pixels, not metric mat movement or confidence intervals. Frames from the same episode are correlated.

| Edge | Minimum / median / maximum frame-median relative displacement |
|---|---:|
| left | -26.85 / 0.04 / 8.54 px |
| right | -26.77 / -8.68 / 9.44 px |
| top | -3.17 / 2.05 / 10.08 px |
| bottom | Unavailable |

## Model boundaries versus direct image boundaries

The ten previously model-measured frames are compared without changing their masks, corners or raw-edge rules. On shared measured edges, the model's virtual line and the raw contrast boundary can be compared directly. Each cell below gives median / p95 absolute point-to-line distance; † marks a search-range-limited raw edge.

| Episode / frame | Left | Right | Top | Bottom |
|---|---:|---:|---:|---:|
| 0 / 98 | 0.54 / 1.48 | 1.20 / 2.03 | — | 1.14 / 2.19 |
| 2 / 82 | 0.50 / 1.45 | 1.48 / 2.10 | — | — |
| 4 / 179 | 0.44 / 1.12 | — | 1.68 / 2.51 | — |
| 5 / 405 | 0.42 / 1.10 | — | 1.74 / 2.43 | — |
| 13 / 696 | 0.39 / 1.20 | — | — | 0.90 / 2.21 |
| 15 / 19 | 0.44 / 1.31 | — | — | 1.00 / 2.11 |
| 26 / 150 | 0.39 / 1.02† | — | — | 8.08 / 10.43 |
| 27 / 541 | 0.54 / 1.23† | — | 1.59 / 1.95 | — |
| 28 / 252 | 0.54 / 1.42† | — | 1.12 / 2.62 | — |
| 31 / 299 | 0.54 / 1.58† | — | 1.71 / 2.67 | — |

Most shared lines agree within a few pixels; episode 26/frame 150 has an approximately 8.08 px median bottom disagreement. This is a boundary disagreement, not a pure segmentation-error measurement: visible contrast, stitching/shadow, occlusions and virtual edge definitions differ. The left measurements marked † retain their range limitation. Earlier shared camera/mat-pose failures therefore cannot be attributed solely to segmentation or solely to physical camera movement.

## Physical registration gate

The stationary upper opening, side hole and foot offer nominal CAD heights of 69.6, 45.7 and 15.1 mm, but the [landmark review](v2_fixed_base_landmark_review_20261024.md) records zero verified physical identities. Repeated foot apertures, uncertain printed-part revision and unmeasured intrinsics/distortion still prevent an accepted multi-height camera/base fit. No joint-zero/gripper correction or corrected-label retraining follows from this diagnostic.

The next useful capture remains a short calibration-board sequence with independently measured board-to-base registration, arm-clear views of named base features, and a measured loaded hold/release. A board on the mat alone does not register the robot base.

Reproduce the summary without rerunning tracking:

```sh
.venv/bin/python scripts/summarize_v2_fixed_landmark_tracks.py \
  --run outputs/v2/fixed_landmark_tracks_20261025 \
  --out results/v2/fixed_landmark_tracking_20261025.json \
  --doc docs/v2_fixed_landmark_tracking_20261025.md
```
