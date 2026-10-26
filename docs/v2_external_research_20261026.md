# External references and implementation priorities — 26 October 2026

The closest useful comparisons support fixing action/control semantics and proving a
grasp in the actual scene before changing the visual backbone again. The results
below are reported by their authors; none was reproduced in this workspace. Cube
results do not validate this tennis-ball/bowl task or physical SO-101 calibration.
GitHub sources were read through the connected GitHub tool; papers and project
documentation were read directly on the web.

## Comparable repositories

| Source inspected | What the source establishes | Application here |
| --- | --- | --- |
| [rocPAI-Forge/so101-simstudio, Home20 protocol](https://github.com/rocPAI-Forge/so101-simstudio/blob/c8d048285a705d70712de474bfe229e8f0727d92/labs/lab01_pnp_agent/home20.md) | Reports ACT **53/60** across 20 fixed cube poses and three seeds, trained on 140 simulation episodes, using six absolute joint targets, 50K updates and `n_action_steps=50`. Evaluation excludes expert takeover. | A native-joint, matching-scene ACT baseline is credible. Stage counts for approach, closure, lift, transfer and released placement expose where failure occurs. Its longer execution horizon also demonstrates that shorter horizons are not a universal improvement. |
| [isaka1022/so101-sim2real, limitations](https://isaka1022.github.io/so101-sim2real/limitations/) | Identifies two different failure mechanisms: position-only IK reaches a tilted branch, and convex-hull jaw collisions make sparse contacts. Reports 36/200 scripted placements without added jaw pads versus 200/200 with them. The pads protrude beyond the mesh and are explicitly **unmeasured surrogate geometry**. | Inspect the actual between-jaw TCP, approach axis and contact points. An offset or constraint can fix a reachable grasp; friction sweeps cannot fix the wrong contact surface. Added collision pads would be a separately labelled simulation experiment. |
| [Yangjingyuan-HK/so101-real2sim](https://github.com/Yangjingyuan-HK/so101-real2sim/blob/main/README.md) | Its MuJoCo replay section openly reports unreliable carton retention despite increased friction, grip force and reduced gravity. It separately reports a successful Isaac Sim hold. | A replay that resembles the demonstration is insufficient evidence of a stable loaded grasp. This repository is useful as a failure comparison, rather than a proven MuJoCo solution to copy. |
| [EE5108-DigitalTwins scripted controller](https://github.com/EE5108-DigitalTwins/lerobot_mujoco_sim/blob/912f6f15d9b0052db9719a86c872b659191be716/controllers/scripted_fsm_controller.py) | Implements staged approach/descend/close/lift/place, Mink IK, pose tolerances and lift-failure retries. | Its sequence and failure handling are useful for a simulator baseline. Its object-specific offsets and joint/geometry conventions are not calibration values for this project. |

## Action meaning and timing

[LeRobot v0.6.1 follower control](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/so_follower.py)
reads `Present_Position` for observations and sends `Goal_Position` for actions.
`max_relative_target` can change a requested command before sending; the return value
is the sent command. It does not establish when the motor applied or achieved it.
Consequently, a recorded next follower pose and a recorded servo target are distinct
supervision choices. Changing one into the other requires a different controller
contract, not only a different normalizer.

The [original ACT loader](https://github.com/tonyzhaozh/act/blob/main/utils.py)
uses different observation/action alignment for real and simulated data: its real
data path starts the action slice one tick earlier. This is direct evidence that
alignment matters, **not** evidence that this dataset needs that particular shift.
Estimate lag from applicable time traces or compare candidate offsets on training
and development trajectories while retaining provenance; do not fit a joint-zero
offset to a dynamic delay.

[Diffusion Policy's real robot evaluator](https://github.com/real-stanford/diffusion_policy/blob/main/eval_real_robot.py)
anchors predicted action times to the latest observation timestamp, measures
observation/inference latency, and removes targets whose scheduled time has expired.
The transferable principle is to preserve observation time and desired execution
time through the queue. A simulator that advances only while inference is idle
does not measure real acquisition-to-action latency.

## ACT changes that are meaningful to test

The [original ACT repository](https://github.com/tonyzhaozh/act/blob/main/README.md)
provides scripted demonstrations, joint-space replay, training and closed-loop
evaluation. It notes that smoothness/success can improve after loss plateaus.
That supports longer training after action labels and executability are established;
it does not establish that another long run repairs a wrong scene or controller.

[LeRobot's tagged configuration](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/configuration_act.py)
separates prediction chunk length from the number of queued actions executed.
Temporal ensembling requires `n_action_steps=1`. In the
[tagged implementation](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/modeling_act.py),
the exponential weight starts at the **oldest** prediction: positive `0.01` favours
older predictions slightly; zero is uniform; negative values favour newer ones.

For this workspace, a prefix-two supervised checkpoint has an unsupervised future
tail. Averaging all 25 future positions would use outputs that were not trained.
Use a fully supervised checkpoint for the normal ensembler, or explicitly restrict
the ensemble to the trained prefix. Compare queue-two, queue-one and one declared
ensemble setting on the same development starts, counting rejected targets and
achieved grasp stages. This is an inference experiment, not a new architecture.

The local `ChunkExecutor` was also inspected: it calls `predict_action_chunk`
directly and consumes a queue, bypassing upstream `select_action` and its temporal
ensembler. Consequently, accepting `temporal_ensemble_coeff` in an ACT config did
not by itself enable ensembling in that executor. The standalone
[`BoundedTemporalEnsembler`](../src/real2sim/policy/temporal_ensemble.py) now provides
explicit tick alignment, prefix capping, finite-input checks and reset behavior
for executor integration. Its 26 focused tests pass; this establishes software
behavior, not an improvement in learned pick-and-place success.

## Panda transfer and sparse correction

[MimicGen's custom-environment tutorial](https://mimicgen.github.io/docs/tutorials/datagen_custom.html)
requires explicit conversions between action commands and controller target poses,
object poses, and subtask boundaries. It composes object-relative segments and
supports changes in robot embodiment. This supports a task-space transfer interface;
it does not demonstrate zero-shot SO-101-to-Panda transfer of a frozen visual ACT.

[Robosuite's controller documentation](https://robosuite.ai/docs/modules/controllers.html)
distinguishes joint targets, operational-space targets and IK targets. It also
documents different rotation frames for its OSC and IK implementations. A shared
seven-element vector therefore does not establish a shared action meaning.

**Proposed application here (inference from these sources):** retain object-relative
approach/grasp/transport/release waypoints and an explicitly defined grasp frame.
Use each robot's adapter to turn these into executable targets. Reuse a seed or
last accepted joint target between waypoints; invoke bounded IK correction when
tracking error exceeds a declared threshold or a phase changes, and count every
correction. For SO-101's five arm joints, prioritise position plus feasible approach
axis rather than demanding arbitrary six-DOF poses. For Panda, verify its TCP,
opening-to-gap mapping, workspace and orientation convention independently.

Report pure ACT and ACT with correction separately, including correction frequency,
maximum displacement and fallback/rejection counts. Successful scripted trajectories
establish scene/control feasibility. Successful assisted trajectories establish the
combined system. Only unassisted rollouts establish success of the frozen policy.

## Concrete priority order

1. Apply source-backed scalar measurements, retaining uncertainty and unresolved
   coordinate signs. Establish a loaded grasp/release and requested → sent → achieved
   execution trace in the current MuJoCo scene.
2. Run a native-joint command-target baseline and a task-space achieved-motion
   baseline under their own controllers. Score reach, bilateral grasp, sustained
   lift, bowl transfer and settled release. Reproduce failure before sweeping gains.
3. Once execution is feasible, train against the chosen measured data/control
   contract, then compare a small declared execution-horizon/ensemble experiment.
4. Verify the same task-space waypoints on Panda and then test the frozen policy
   through the validated adapter. Keep any corrective controller visible in results.

No outside repository was installed, no external weights were promoted and no
outside result was substituted for a local test.
