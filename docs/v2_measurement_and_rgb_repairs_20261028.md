# Supplied measurements and RGB frontend repairs — 28 October 2026

`physical_measurements.md` exists and is usable. It supplies dimensions and
relative positions, with an estimated ±5% error. Calling these measurements
“missing” was inaccurate. The current measured-scalar scenes apply all reported
scalar quantities within those bounds. The
[application audit](../results/v2/measurement_application_reaudit_20261028.json)
verifies both arms against the current source-file SHA-256 and checks that their
lighting configurations equal the original waypoint configurations.

The applied ball radius is 32.340 mm, base-to-ball components are (50.8, 330.2)
mm under the current axis convention, ball-to-bowl total distance is 304.8 mm,
mat dimensions are 406.4 × 304.8 mm, and the camera height is 406.4 mm above
the tabletop. Bowl diameter 114.3 mm is provisionally interpreted as the outer
diameter. Existing operator clarifications and the prior image-derived bowl
bearing are retained. The text supplies neither an optical orientation/focal
calibration nor robot joint neutral offsets relative to the CAD model. Those
specific transforms remain estimates; they do not prevent running the simulator.

The setup generator previously hardcoded the first camera height and uncertainty
(16 inches and 5%) even if the measurement record changed. It now reads both from
the record, recalculates ball radius from the current circumference, and records
all applied fields and the current source hash. The preparation report also reads
the ball–bowl distance and its interval from that record.

## RGB bowl detection repaired without changing lighting

The original bowl detector used one neutral-colour mask at brightness ≥80. This
merged the bright rim with dim sidewall, shadow or nearby robot pixels. Its
external contour then failed the existing convexity/circularity checks even
though the rim was visible.

The repaired frontend observes complete contours at fixed brightness levels.
It preserves the original metric radius, circle residual, convexity, workspace
and independent-object ambiguity checks. Repeated measurements of one spatial
rim are combined before ambiguity ranking. Missing targets still have no
configured-position fallback. This changes image processing; it does not change
scene lighting, materials, contact physics, or task-success scoring.

Five common development seeds (300–304) were run for the full configured horizon
in the measured-scalar scenes, before any Panda workspace correction:

| Controller | SO-101 pick/place | Panda pick/place | Bowl acquired |
| --- | ---: | ---: | --- |
| Original RGB frontend | 0/5 | 0/5 | 0/5 on each arm |
| Repaired RGB frontend | 5/5 | 2/5 | 5/5 on each arm |

Full reports:
[SO-101 baseline](../outputs/v2/rgb_so101_measured_reaudit_20261028_render/report.json),
[Panda baseline](../outputs/v2/rgb_panda_measured_reaudit_20261028_render/report.json),
[SO-101 repair](../outputs/v2/rgb_so101_measured_multilevel_20261028/report.json),
[Panda repair](../outputs/v2/rgb_panda_measured_multilevel_20261028/report.json).
All completed without aborts, native-command rejections, or runtime source
changes. These are RGB geometric controller results, not frozen ACT successes.

The remaining three Panda failures occur before planning: the measured ball X
positions (approximately 24.5, 40.5 and 35.5 mm) fall below the former 50 mm
workspace minimum. The full ball is detectable without that inconsistent
eligibility bound. Native reach and path validation, rather than this arbitrary
image-target filter, must determine whether the corresponding grasp can execute.

Focused tests cover dimensional/source provenance, a complete rim connected to
dim pixels, duplicate threshold observations, and two genuinely independent
matching bowls. Ambiguous scenes, incomplete ball silhouettes and wrong object
scales continue to be rejected.

Reproduce the measurement application audit with:

```sh
.venv/bin/python scripts/audit_v2_measurement_application.py \
  --out results/v2/measurement_application_NEW.json
```
