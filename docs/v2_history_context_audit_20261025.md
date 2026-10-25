# Causal history and hold-transition residuals — 25 October 2026

The prefix-only policy's residual stationary movement is concentrated near transitions. A single four-observation history experiment is justified by added causal information; globally forcing low-velocity observations to hold could suppress legitimate initiation.

The [reproducible audit](../scripts/audit_v2_history_context.py) scores only the original training episodes 11,8,23 using their completed prefix2 checkpoint. The [report](../results/v2/history_context_audit_20261025.json) retains source hashes, selection, rules and every repeated-history conflict. No labels, inputs or sampling weights are changed by this descriptive audit.

## Causal-history informativeness

History vectors concatenate current and previous canonical observations, oldest first. Missing initial slots repeat the first observation within the same episode. Label-conflict counts compare only real valid targets. Ages/validity flags are excluded from this equality diagnostic, so tiny timestamp differences cannot manufacture unique states.

| Observations in history | Span at12.5Hz | Exact-history groups with differing next translation labels |
|---|---:|---:|
| 1 | 0.00 s | 12 |
| 2 | 0.08 s | 8 |
| 4 | 0.24 s | 4 |
| 8 | 0.56 s | 3 |

Four observations reduce conflict groups from12 to4; eight reduce them only to3. The remaining four-observation conflicts still include next-label differences near3.95mm in episode8 and3.92mm in episode23. A causal history therefore adds information but does not reveal every operator's future onset timing.

## Prefix2 residuals on the same valid training actions

| Stationary stratum | Actions | Mean predicted translation step | Mean target-vector error |
|---|---:|---:|---:|
| All stationary | 201 | 0.2986 mm | 0.2898 mm |
| Exact-zero translation labels | 175 | 0.2017 mm | 0.2017 mm |
| Four causal low-motion observations | 106 | 0.0471 mm | 0.0490 mm |
| Other causal history | 95 | 0.5792 mm | 0.5586 mm |
| No moving label within four past/future ticks | 89 | 0.0254 mm | 0.0254 mm |
| Moving label within four future ticks | 91 | 0.5318 mm | 0.5164 mm |
| Moving label within four past ticks | 85 | 0.6442 mm | 0.6212 mm |

Stationary uses the unchanged translation≤0.4mm and rotation≤0.01rad thresholds. Causal low-motion history means four observations with tool speed≤0.005m/s and arm span≤0.5°. The remaining strata inspect recorded labels descriptively; their future labels are never available to the proposed policy or used to filter training. Strata overlap and do not form independent samples.

The long stationary stratum is already comparatively quiet. Global hold weighting could trade away motion initiation without addressing transition context. Four past/current observations are the smaller next intervention; an eight-observation history provides little extra exact-conflict reduction here.

## Controlled next experiment

Use the same frozen ResNet backbone, three episodes, original action labels/action statistics/images, seed, optimizer horizon, two-action supervision and unchanged movement/stationary gates. Add only causal canonical history4, including per-slot age/valid flags, giving60 state features. Tile the exact original canonical-state normalization across slots and keep age/valid units unchanged.

Offline and live transforms must reset at episode/environment boundaries and consume every control tick, including queued-action ticks without inference. The same observation must not update history twice. First slots have explicit validity; later observations cannot affect earlier feature vectors. The existing streaming transform already implements these requirements and must be checked against the prepared data.

The input projection becomes wider, so this is an observation-context experiment with the same backbone rather than an identical-architecture causal attribution. A matched-width control would be required to isolate history content from added input capacity. Do not combine class balancing, a hold gate or relabeling in this run. Score motion onsets separately, retain the existing object-swap checks, and proceed to common development rollouts only after the unchanged training gate passes.

No success, calibration correction or physical actuator equivalence is established by this audit. The source observations/labels and all Cartesian metrics remain provisional.

```sh
.venv/bin/python scripts/audit_v2_history_context.py \
  --out outputs/v2/history_context_audit_NEXT
```
