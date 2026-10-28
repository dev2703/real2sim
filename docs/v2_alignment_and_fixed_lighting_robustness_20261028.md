# Camera sensitivity and fixed-lighting robustness — 28 October 2026

The next transfer diagnostic holds the frozen latest HF model and real joint
inputs fixed while changing individual image streams. The global iPhone render
produces the larger prediction mismatch in these anchors. A training-only frame
yaw hypothesis improves the aggregate but does not resolve transfer.

The [curated evidence](../results/v2/hf_alignment_camera_ablation_20261028.json)
links the full report, image panels and source hashes. Eighteen anchors use all
six checkpoint validation episodes: the first pose that satisfies native reset
constraints in both scenes, followed by offsets 0, 25 and 50 source frames. These
validation episodes score the hypothesis; they never fit its yaw. All anchors
were usable. Every identical-input repetition was exactly equal.

| Image condition; same real joint state throughout | Five-arm-joint prefix MAE |
| --- | ---: |
| Both recorded cameras | 2.858° |
| Both nominal measured-scene renders | 11.752° |
| Both training-yaw renders | 9.892° |
| Nominal rendered wrist + recorded global | 4.400° |
| Recorded wrist + nominal rendered global | 8.567° |
| Training-yaw rendered wrist + recorded global | 3.579° |
| Recorded wrist + training-yaw rendered global | 8.153° |

The measured-scene yaw hypothesis reduces the two-render mean by about 16%.
The remaining error is still much larger than the recorded-image reference.
This points toward global view/layout/appearance alignment as a useful next
target. The test jointly changes optical pose, objects, background and rendered
appearance; it does not identify a unique physical camera correction. No new
camera/joint offset is adopted from a poor fit. Lighting and HF weights remain
unchanged, and the completed report records no runtime source changes.

The previous training-only yaw comes from the first downward 45% gripper crossing
in 35 source training episodes under the current provisional FK/grasp offset.
That is an explicit geometric hypothesis, not independently measured camera or
joint registration. Previous raw rollouts with this hypothesis reduced closest
approach but still completed 0/2 tasks; its images above do not establish task
success. Larger previous joint/camera landmark fits had substantial held-out
pixel residuals and remain unadopted.

Reproduce:

```sh
.venv/bin/python scripts/diagnose_hf_alignment_inputs.py \
  --out outputs/v2/hf_alignment_inputs_NEW
```

## Broader fixed-lighting evaluation

`scripts/prepare_fixed_lighting_robustness.py` generates isolated task copies and
a shared explicit layout protocol. It never edits the nominal source scene.
The prepared local plan is
`outputs/v2/fixed_lighting_robustness_20261028/protocol.json` with seeds 1000–1099.
The matched benchmark can consume one condition's source scene, layout scenarios
and optional model effects.

| Condition | Ball disc | Bowl disc | Other change |
| --- | ---: | ---: | --- |
| Nominal | 3 cm | Fixed | None |
| Bowl jitter | 3 cm | 3 cm | None |
| Extended layout | 5 cm | 3 cm | Feasibility stress test |
| Ball small / large | 3 cm | 3 cm | Radius −5% / +5%, constant mass |
| Friction low / high | 3 cm | 3 cm | Collision sliding coefficient ×⅔ / ×1.5 |

Ball and bowl draws use separate deterministic random streams. All arms and
controllers receive the same explicit XY pair for a seed. Radius variants
rebuild sphere geometry, inertia, spawn height and success scoring. A runner
must disclose whether its RGB frontend receives the true varied radius or a
fixed nominal assumption; this plan alone does not freeze that perception input.

Friction variants apply `apply_model_condition` once after model construction.
They scale only the sliding coefficient of all collision-enabled geometries,
including robot/static contacts, preserving torsional/rolling coefficients and
all lighting arrays. Scaling only the ball material would not consistently
change effective contacts because contact parameters use geom priorities and,
at equal priority, elementwise maximum friction. [MuJoCo documentation](https://mujoco.readthedocs.io/en/3.3.3/modeling.html#contact-parameters)

The plan predeclares tests; it does not claim any robustness success before
rollouts. Infeasible starts, missed RGB detections, aborts and failures must stay
in the denominator. No lighting perturbations are included.

```sh
.venv/bin/python scripts/prepare_fixed_lighting_robustness.py \
  --out outputs/v2/fixed_lighting_robustness_NEW
```

Three focused robustness-generation tests and the existing training-only yaw
regression pass; Ruff passes for both new diagnostic tools and their tests.

The completed [RGB-only robustness pilot](../results/v2/fixed_lighting_rgb_robustness_pilot_20261028.json)
uses three separate seeds, 14000–14002, in each of the seven conditions above.
Both arms completed all 21 tasks each (42/42 total), with no aborts or lighting
changes. This is a small geometric-controller pilot, rather than learned-head
or ACT-only evidence. The RGB frontend receives the actual varied sphere radius;
the size conditions therefore test scene-aware operation, not tolerance to an
unknown radius. Mid-trajectory occlusion and independently measured camera
registration remain untested by this pilot.
