# Panda feedback and measured workspace fixes — 28 October 2026

The latest Hugging Face ACT remains unchanged, with SHA-256
`5dab91076d1e3b274674a888f522ce5fdddaf69ddd05c1adf3bce1742b99f65c`.
Its live Panda evaluation completes **0/5** tasks in development seeds 300–304.
The separate RGB geometric controller now completes **5/5** full held-lift,
release and resting-placement tasks in the measured scene. Initial-object oracle
feasibility independently completes **5/5**. Lighting remains unchanged.

The [curated evidence](../results/v2/panda_hf_measured_workspace_20261028.json)
records reports, hashes, limits, changes and their scope. Five reused development
layouts establish working engineering examples; they do not establish broad
physical or simulation reliability.

The final unassisted comparison was repeated after workspace registration with
the same **1.2%** initial source gripper as the assisted evaluation: again **0/5**,
all 3,750 IK requests accepted, no abort, unchanged weights and runtime hashes.
The additional fully closed-start comparison also yielded 0/5. All ten final
unassisted videos and all five RGB success videos fully decode with FFmpeg.

Two concrete adapter/configuration bugs were fixed:

* Panda's position-only inverse reconstructs five source joints from three tool
  coordinates. It previously regularized toward the preceding observed source
  state, discarding commanded source rotation. A perfectly achieved command
  could return a wrist-roll observation **22°** away and a shoulder-pan observation
  **4.53°** away, despite a 0.00016 mm position residual. The inverse now preserves
  the most recent causal command in its unobservable coordinates and corrects
  its observable position using actual Panda feedback. These remain virtual
  source joints, not a complete measured representation of Panda.
* The historical Panda planning box began at **x = 50 mm**. The measured ball
  centre is **x = 50.8 mm**, with a configured **30 mm** start disc. Three of the
  five development layouts were excluded before IK ran. The task box now includes
  the measured start disc, upper ball-radius uncertainty and bowl footprint.
  Its resulting lower x bound is **−13.1573 mm**. Native joint limits, contact
  checks, path checks, z bounds and IK acceptance tolerances remain unchanged.

Panda grasp planning additionally explores nine deterministic robot-joint seeds
and chooses the valid branch requiring least joint travel. Every branch passes
the original native pose and path gates. Workspace registration and branch
search were tested together; their individual success contributions were not
isolated. The HF Panda setup now propagates the registered measured Panda
configuration, rather than loading the historical task box unconditionally.

All 3,750 commands in the first five-trial latest-HF evaluation passed Panda IK.
No run aborted, no actuator saturation occurred, and the weights/runtime hashes
remained unchanged. The closest tool-to-ball distances were 345–378 mm. This is
an approach failure despite executable commands, not evidence that the user's
real-world policy is poor. Rendered wrist/global observations, incomplete physical
optical registration and the lossy cross-embodiment state/action representation
remain the main transfer bottlenecks. A geometric recovery system must be reported
as assisted control; its successes cannot be credited to unassisted frozen ACT.

Reproduce the RGB result:

```sh
.venv/bin/python scripts/run_visual_waypoint.py --robot panda \
  --config configs/v2/measured_registration_20261028/sim_panda_positive_all_scalars.yaml \
  --seeds 300 301 302 303 304 --video --out outputs/v2/panda_measured_rgb_NEW
```

The projection/workspace tests cover achieved-command consistency, correction
from measured tool error, native grasp/path feasibility for every development
layout, deterministic workspace derivation and no controller-side privileged
object-metric access. Existing Panda camera rigidity and SO101 RGB controller
tests also pass.
