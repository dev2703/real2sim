"""Inference-time adapters between the frozen ACT checkpoint and the simulated SO-101 (plan M4.2-3).

* :class:`ObservationAdapter` assembles the policy batch from ``env.observation()`` only. It is
  bound to the checkpoint's audited contract (camera key, image shape, state names) and refuses
  anything else instead of adapting silently (plan section 7: feature meaning, order, units and
  train-time presence must match). It never reads ``info['privileged_metrics']``.
* :class:`JointActionAdapter` applies a 6-D joint target in dataset units (degrees, gripper %).
* :class:`CartesianActionAdapter` applies the canonical 7-D task-frame delta relative to the
  *current observed* EE pose: clamp -> IK (seeded from the current native qpos) -> native step;
  rejected IK targets hold the current pose and are counted, never silently applied.
  The explicit ``target_reference='accepted_target'`` experiment instead integrates deltas
  into a persistent accepted pose, with feedback bounded by the same per-tick limits.
* :class:`ChunkExecutor` runs one ``predict_action_chunk`` per ``n_action_steps`` ticks and drops
  the queue on reset. It advances the observation adapter's EE-position memory on *every* tick so
  the canonical proprio velocity is the one-tick finite difference ``(p_k - p_{k-1}) * hz`` of
  ``real2sim.data.labels`` even on ticks without an inference call.
* :class:`PandaObservationAdapter` / :class:`PandaActionAdapter` (plan M5) serve the SAME
  cartesian contract on the Panda: the proprio is the Panda env's canonical 13 floats with the
  velocity slot taken from the adapter's own ``advance_tick`` memory (checked against the env's
  value, so the semantics cannot drift apart), and the 7-D unnormalised action goes straight to
  ``real2sim.panda.adapter.PandaCartesianActionAdapter.apply``. A joint checkpoint is refused.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from real2sim.data.labels import (
    ABSOLUTE_ACTION_NAMES,
    ACTION_NAMES,
    PROPRIO_NAMES,
    TOOL_ACTION_NAMES,
)
from real2sim.geometry.so3 import apply_delta, delta_action, is_rotation, make_T, matrix_from_rot6d
from real2sim.geometry.so101_fk import JointMapping

IPHONE_KEY = "observation.images.iphone"
STATE_KEY = "observation.state"
IMAGE_SHAPE_CHW = (3, 192, 256)
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_STATE_NAMES = [f"{j}.pos" for j in JOINT_NAMES]
# Observation keys that may exist in ``env.observation()`` / ``info`` but must never reach the
# policy: simulator diagnostics (``sim.*``) and ground truth (``privileged*``).
FORBIDDEN_PREFIXES = ("privileged", "sim.")
POLICY_KEYS = frozenset({IPHONE_KEY, STATE_KEY})


class ContractError(ValueError):
    """The checkpoint contract cannot be served by the simulated observation stream."""


def normalize_image_chw(normalizer, chw_float01: np.ndarray) -> np.ndarray:
    """``[3,H,W]`` float32 in [0,1] (env image) -> the normalizer's ImageNet normalisation.

    Equivalent to ``normalizer.image(hwc_uint8)`` applied to the uint8 image the env letterboxed;
    the env already divided by 255 so only the mean/std step remains.
    """
    chw = np.asarray(chw_float01, np.float32)
    if chw.shape != IMAGE_SHAPE_CHW:
        raise ContractError(f"expected image {IMAGE_SHAPE_CHW} float in [0,1], got {chw.shape}")
    # NaN compares False against both bounds, so check finiteness explicitly.
    if not np.all(np.isfinite(chw)) or chw.min() < 0.0 or chw.max() > 1.0:
        raise ContractError("image must be finite and within [0, 1]")
    return (chw - normalizer.img_mean) / normalizer.img_std


def contract_kind(contract: dict) -> str:
    """``'joint'`` or ``'cartesian'``; raises :class:`ContractError` for anything else."""
    cams = list(contract.get("camera_keys", []))
    if not cams or len(cams) != len(set(cams)) or any(not k.startswith("observation.images.") for k in cams):
        raise ContractError("contract camera_keys must be unique ordered image keys")
    shape = tuple(int(x) for x in contract.get("image_shape_chw", []))
    if shape != IMAGE_SHAPE_CHW:
        raise ContractError(f"contract image_shape_chw {shape} != {IMAGE_SHAPE_CHW}; refusing to adapt")
    names = list(contract.get("state_names", []))
    if "visual_grounding" in contract:
        from real2sim.policy.visual_grounding import BallImageFeatures, grounded_state_names

        spec = BallImageFeatures.from_dict(contract["visual_grounding"])
        if (names != grounded_state_names() or spec.camera_key not in cams
                or "observation_variant" in contract):
            raise ContractError("visual grounding state names/camera differ from its transform")
        kind = "cartesian"
    elif "observation_variant" in contract:
        from real2sim.policy.observation_variants import ObservationVariant, state_names

        variant = ObservationVariant.from_dict(contract["observation_variant"])
        if names != state_names(variant):
            raise ContractError("observation variant state names differ from its saved transform")
        if variant.policy_hz != float(contract.get("policy_hz", 10)):
            raise ContractError("observation variant rate differs from the policy contract")
        kind = "cartesian"
    elif names == list(PROPRIO_NAMES):
        kind = "cartesian"
    elif names == JOINT_STATE_NAMES or names == list(JOINT_NAMES):
        kind = "joint"
    else:
        raise ContractError(
            f"contract state_names {names} match neither the joint contract {JOINT_STATE_NAMES} "
            f"nor the canonical proprio {list(PROPRIO_NAMES)}; no new feature slots are served")
    actions = list(contract.get("action_names", []))
    if kind == "cartesian" and actions == list(ABSOLUTE_ACTION_NAMES):
        kind = "cartesian_absolute"
    elif kind == "cartesian" and actions == JOINT_STATE_NAMES:
        kind = "canonical_joint"
    if kind == "cartesian" and actions == list(TOOL_ACTION_NAMES):
        kind = "cartesian_tool"
    expected = list(TOOL_ACTION_NAMES) if kind == "cartesian_tool" else list(ACTION_NAMES) if kind == "cartesian" else list(ABSOLUTE_ACTION_NAMES) if kind == "cartesian_absolute" else JOINT_STATE_NAMES
    if actions != expected and not (kind == "joint" and actions == list(JOINT_NAMES)):
        raise ContractError(f"contract action_names {actions} != {expected}")
    return kind


class ObservationAdapter:
    """Build the policy batch from ``env.observation()`` under the checkpoint contract."""

    def __init__(self, contract: dict, normalizer, *, device: str = "cpu"):
        self.contract = contract
        self.kind = contract_kind(contract)
        self.normalizer = normalizer
        self.device = device
        self.state_dim = len(contract["state_names"])
        self.camera_keys = list(contract["camera_keys"])
        if len(normalizer.state_mean) != self.state_dim:
            raise ContractError("normalizer state statistics do not match contract state_names")
        # EE position observed at the previous control tick (None before the first tick, so the
        # velocity slot is zero exactly as ``labels.episode_labels`` zeroes vel[0]).
        self._prev_ee_pos: np.ndarray | None = None
        self._transform = None
        self._visual_grounding = None
        if "visual_grounding" in contract:
            from real2sim.policy.visual_grounding import BallImageFeatures

            self._visual_grounding = BallImageFeatures.from_dict(contract["visual_grounding"])
        if "observation_variant" in contract:
            from real2sim.policy.observation_variants import StreamingObservationTransform

            self._transform = StreamingObservationTransform(contract["observation_variant"])
        self._tick = 0
        self._cached_tick = None
        self._cached_state = None

    def reset(self) -> None:
        self._prev_ee_pos = None
        self._tick = 0
        self._cached_tick = None
        self._cached_state = None
        if self._transform is not None:
            self._transform.reset()

    def advance_tick(self, env) -> None:
        """Record the current EE position as ``p_{k-1}`` for the next tick's velocity.

        Must be called once per control tick *before* the action is applied (i.e. on the
        observation the policy would see at this tick), whether or not inference ran.
        """
        if self.kind != "joint":
            # Filters/history observe every tick, including queued actions without inference.
            if self._transform is not None:
                self.state(env)
            self._prev_ee_pos = env.ee_pose()[:3, 3].copy()
        self._tick += 1

    def _canonical_state(self, env) -> np.ndarray:
        return np.asarray(env.canonical_proprio(self._prev_ee_pos), np.float32)

    def state(self, env, images: dict | None = None) -> np.ndarray:
        """Observable state in contract units; never simulator object poses.

        Pure read: the velocity uses the position stored by :meth:`advance_tick` at the previous
        tick, so calling ``state`` more than once per tick does not change the feature.
        """
        if self.kind == "joint":
            x = np.asarray(env.real_state(), np.float32)
        else:
            if self._transform is not None and self._cached_tick == self._tick:
                return self._cached_state.copy()
            x = self._canonical_state(env)
            if self._transform is not None:
                joints = env.real_state() if self._transform.config.requires_joints else None
                x = self._transform.update(x, joints, self._tick / self._transform.config.policy_hz)
                self._cached_state, self._cached_tick = x.copy(), self._tick
        if self._visual_grounding is not None:
            from real2sim.policy.visual_grounding import append_ball_features, rgb_from_env

            if images is None or self._visual_grounding.camera_key not in images:
                raise ContractError("visual grounding requires the current observed image")
            image = rgb_from_env(images[self._visual_grounding.camera_key], self._visual_grounding)
            x = append_ball_features(x, image, self._visual_grounding)
        if x.shape != (self.state_dim,):
            raise ContractError(f"state length {x.shape} != contract {self.state_dim}")
        return x

    def batch(self, obs: dict, env) -> dict[str, torch.Tensor]:
        # Only the contract's two inputs are copied; ``sim.*`` diagnostics and any privileged
        # metrics in ``info`` never reach the policy.
        if any(k.startswith("privileged") for k in obs):
            raise ContractError("observation carries privileged keys")
        missing = [key for key in self.camera_keys if key not in obs]
        if missing:
            raise ContractError(f"observation lacks {missing}")
        state = self.normalizer.state(self.state(env, obs)).astype(np.float32)
        if not np.all(np.isfinite(state)):
            raise ContractError("normalised state is not finite")
        batch = {
            STATE_KEY: torch.from_numpy(state)[None].to(self.device),
        }
        for key in self.camera_keys:
            img = normalize_image_chw(self.normalizer, obs[key])
            batch[key] = torch.from_numpy(np.ascontiguousarray(img))[None].to(self.device)
        assert_policy_batch(batch, self.camera_keys)
        return batch


def assert_policy_batch(batch: dict, camera_keys: list[str] | None = None) -> None:
    """Runtime guard (plan rule 5): a policy batch holds exactly the contract inputs, never GT."""
    expected = POLICY_KEYS if camera_keys is None else {STATE_KEY, *camera_keys}
    bad = [k for k in batch if k.startswith(FORBIDDEN_PREFIXES)]
    if bad or set(batch) != expected:
        raise ContractError(f"policy batch keys {sorted(batch)} != {sorted(expected)}; forbidden: {bad}")


@dataclass
class ApplyResult:
    obs: dict
    info: dict
    truncated: bool
    saturation: float  # fraction of action components clamped by the adapter / env limits
    ik_rejected: bool = False
    detail: dict = field(default_factory=dict)


class JointActionAdapter:
    """6-D unnormalised joint target (deg, gripper %) -> ``env.step_real_joint_targets``."""

    dim = 6

    def __init__(self, env):
        self.env = env

    def apply(self, action: np.ndarray) -> ApplyResult:
        a = np.asarray(action, np.float64).reshape(-1)
        if a.shape != (6,) or not np.all(np.isfinite(a)):
            raise ValueError("joint action must be six finite values")
        obs, _, _, truncated, info = self.env.step_real_joint_targets(a)
        clipped = info["control"]["limit_clipped"]
        return ApplyResult(obs, info, truncated, float(np.mean(clipped)),
                           detail={"limit_clipped": clipped,
                                   "rate_limited": info["control"]["rate_limited"]})


class CartesianActionAdapter:
    """7-D task-frame delta -> bounded pose target -> IK -> native step.

    ``current_pose`` preserves the checkpoint's existing runtime contract. The opt-in
    ``accepted_target`` mode integrates each clamped delta into the last accepted Cartesian
    target instead of repeatedly following the displaced measured pose. Its total feedback
    from the measured pose is clamped by the SAME translation/rotation limits; any excess is
    discarded, preventing unbounded accumulation when the native servo cannot keep up. Only
    accepted IK followed by a successful native step advances the reference. IK acceptance
    does not imply exact tracking: native joint/rate limits and residual checks still apply.

    Call ``reset()`` with every environment reset (``ChunkExecutor`` does this), or explicitly
    ``rebase_target()`` when changing control ownership. Both discard the old reference; no
    automatic rebase conceals drift or repeated rejection. On rejection the current native
    pose is held, the gripper command is honoured, and the previous reference is retained.
    This changes action execution semantics and must be evaluated before policy deployment.
    """

    dim = 7

    def __init__(self, env, kinematics, *, max_step_m: float = 0.03, max_step_rad: float = 0.3,
                 target_reference: str = "current_pose"):
        self.env = env
        self.ik = kinematics
        self.max_step_m = float(max_step_m)
        self.max_step_rad = float(max_step_rad)
        if any(not np.isfinite(v) or v <= 0 for v in (self.max_step_m, self.max_step_rad)):
            raise ValueError("Cartesian step limits must be finite and positive")
        if target_reference not in ("current_pose", "accepted_target"):
            raise ValueError("target_reference must be 'current_pose' or 'accepted_target'")
        self.target_reference = target_reference
        self.mapping: JointMapping = env.mapping
        self._accepted_target: np.ndarray | None = None
        self.ik_rejections = 0
        self.residuals: list[dict] = []

    def reset(self) -> None:
        """Clear episode state; the first action anchors to the next measured pose."""
        self._accepted_target = None
        self.ik_rejections = 0
        self.residuals.clear()

    @property
    def accepted_target(self) -> np.ndarray | None:
        """A copy of the persistent target, never a mutable view of controller state."""
        return None if self._accepted_target is None else self._accepted_target.copy()

    def _current_pose(self) -> np.ndarray:
        pose = np.asarray(self.env.ee_pose(), dtype=np.float64)
        if (pose.shape != (4, 4) or not np.all(np.isfinite(pose))
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not is_rotation(pose[:3, :3])):
            raise RuntimeError("current EE pose must be a finite proper homogeneous transform")
        return pose.copy()

    def rebase_target(self) -> np.ndarray:
        """Explicitly discard accumulated error and anchor to the current measured pose.

        The measured pose is an initial reference, not an IK-certified command. Subsequent
        actions still require bounded feedback and accepted IK before any target is applied.
        This does not clear rejection statistics or invoke the simulator.
        """
        if self.target_reference != "accepted_target":
            raise ValueError("rebasing requires target_reference='accepted_target'")
        self._accepted_target = self._current_pose()
        return self._accepted_target.copy()

    def apply(self, action: np.ndarray) -> ApplyResult:
        a = np.asarray(action, np.float64).reshape(-1)
        if a.shape != (7,) or not np.all(np.isfinite(a)):
            raise ValueError("cartesian action must be seven finite values")
        clamped = np.zeros(7, bool)
        dp, dr, g = a[:3].copy(), a[3:6].copy(), float(a[6])
        if (n := np.linalg.norm(dp)) > self.max_step_m:
            dp *= self.max_step_m / n
            clamped[:3] = True
        if (n := np.linalg.norm(dr)) > self.max_step_rad:
            dr *= self.max_step_rad / n
            clamped[3:6] = True
        if not 0.0 <= g <= 1.0:
            g = float(np.clip(g, 0.0, 1.0))
            clamped[6] = True
        T_cur = self._current_pose()
        reference = T_cur
        if self.target_reference == "accepted_target":
            if self._accepted_target is None:
                self._accepted_target = T_cur.copy()
            reference = self._accepted_target
        requested_target = apply_delta(reference, np.concatenate([dp, dr]))
        target = requested_target
        feedback_clamped = np.zeros(6, bool)
        requested_feedback = delta_action(T_cur, requested_target)
        bounded_feedback = requested_feedback.copy()
        if self.target_reference == "accepted_target":
            for part, limit in ((slice(0, 3), self.max_step_m), (slice(3, 6), self.max_step_rad)):
                if (n := np.linalg.norm(bounded_feedback[part])) > limit:
                    bounded_feedback[part] *= limit / n
                    feedback_clamped[part] = True
            # Do not retain the unbounded request for later catch-up. Zero actions retain a
            # stable target until this explicit feedback cap is reached by a large disturbance.
            target = apply_delta(T_cur, bounded_feedback)
        # MuJoCo joint limits are soft: measured qpos can overshoot jnt_range by ~1e-3 rad under
        # contact, and NativeKinematics.solve refuses out-of-range seeds. Clip like replay.py does;
        # the clip is diagnostic-only (the seed), never a change to the upstream model.
        lo_all, hi_all = self.env.limits[:, 0], self.env.limits[:, 1]
        seed = np.clip(self.env.sim_state(), lo_all, hi_all)
        lo, hi = self.env.limits[5]
        requested_grip = float(self.mapping.gripper_real_to_sim_rad(g * 100.0))
        grip = float(np.clip(requested_grip, lo, hi))
        native_gripper_clipped = grip != requested_grip
        result = self.ik.solve(target, seed, gripper=grip)
        self.residuals.append({"position_error_m": result.position_error_m,
                               "orientation_error_rad": result.orientation_error_rad,
                               "accepted": result.accepted, "reason": result.reason,
                               "iterations": result.iterations})
        if result.accepted:
            q = result.qpos
        else:
            self.ik_rejections += 1
            q = seed.copy()  # hold the current arm pose; gripper command still honoured
            q[5] = grip
        obs, _, _, truncated, info = self.env.step(q)
        if self.target_reference == "accepted_target" and result.accepted:
            self._accepted_target = target.copy()
        all_clamped = clamped.copy()
        all_clamped[:6] |= feedback_clamped
        all_clamped[6] |= native_gripper_clipped
        saturation = float(np.mean(all_clamped))
        return ApplyResult(obs, info, truncated, saturation, ik_rejected=not result.accepted,
                           detail={"clamped": all_clamped.tolist(), "ik_reason": result.reason,
                                   "target_task": target[:3, 3].tolist(),
                                   "target_reference": self.target_reference,
                                   "action_clamped": clamped.tolist(),
                                   "requested_gripper_rad": requested_grip,
                                   "bounded_gripper_rad": grip,
                                   "native_gripper_clipped": native_gripper_clipped,
                                   "feedback_clamped": feedback_clamped.tolist(),
                                   "reference_pose": reference.tolist(),
                                   "requested_target_pose": requested_target.tolist(),
                                   "bounded_target_pose": target.tolist(),
                                   "requested_feedback": requested_feedback.tolist(),
                                   "bounded_feedback": bounded_feedback.tolist(),
                                   "accepted_target_pose": (None if self._accepted_target is None
                                                            else self._accepted_target.tolist()),
                                   "target_advanced": bool(self.target_reference == "accepted_target"
                                                           and result.accepted),
                                   "position_error_m": result.position_error_m,
                                   "orientation_error_rad": result.orientation_error_rad})


class AbsolutePoseActionAdapter:
    """Ten-dimensional absolute tool pose -> the existing bounded delta controller.

    Each prediction specifies a task-frame pose, not a delta to integrate. The wrapped
    adapter must reference current_pose so previously accepted targets are not added twice.
    All IK, workspace, contact, gripper and rate constraints remain on the wrapped path.
    """

    dim = 10
    needs_obs = True

    def __init__(self, delta_adapter):
        controller = getattr(delta_adapter, "panda", delta_adapter)
        if getattr(controller, "target_reference", "current_pose") != "current_pose":
            raise ValueError("absolute poses require a current-pose bounded controller")
        self.delta_adapter = delta_adapter
        self.env = delta_adapter.env

    def reset(self):
        self.delta_adapter.reset()

    @property
    def stats(self):
        return getattr(self.delta_adapter, "stats", None)

    @property
    def ik_rejections(self):
        return self.delta_adapter.ik_rejections

    def apply(self, action, obs=None):
        a = np.asarray(action, dtype=float)
        if a.shape != (10,) or not np.isfinite(a).all():
            raise ValueError("absolute pose action must contain ten finite values")
        target = make_T(matrix_from_rot6d(a[3:9]), a[:3])
        delta = np.r_[delta_action(self.env.ee_pose(), target), a[9]]
        if getattr(self.delta_adapter, "needs_obs", False):
            result = self.delta_adapter.apply(delta, obs)
        else:
            result = self.delta_adapter.apply(delta)
        result.detail.update(action_representation="cartesian_absolute", absolute_target_pose=target.tolist())
        return result


class ChunkExecutor:
    """Open-loop execution of ``n_action_steps`` actions per policy call; re-observes afterwards."""

    def __init__(self, policy, normalizer, obs_adapter: ObservationAdapter, action_adapter,
                 n_action_steps: int | None = None):
        self.policy = policy
        self.normalizer = normalizer
        self.obs_adapter = obs_adapter
        self.action_adapter = action_adapter
        # ``is None``, not ``or``: an explicit 0 is invalid and must not fall back to the default.
        n = int(policy.config.n_action_steps) if n_action_steps is None else int(n_action_steps)
        if n < 1 or n > int(policy.config.chunk_size):
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        self.n_action_steps = n
        self.queue: list[np.ndarray] = []
        self.latencies_ms: list[float] = []
        self.n_policy_calls = 0

    def reset(self) -> None:
        self.policy.reset()
        self.obs_adapter.reset()
        if hasattr(self.action_adapter, "reset"):
            self.action_adapter.reset()
        self.queue.clear()
        self.latencies_ms.clear()
        self.n_policy_calls = 0

    def _infer(self, obs: dict, env) -> list[np.ndarray]:
        t0 = time.perf_counter()
        batch = self.obs_adapter.batch(obs, env)
        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(batch)
        raw = self.normalizer.unaction(chunk).detach().cpu().numpy()
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        self.n_policy_calls += 1
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != self.action_adapter.dim:
            raise ContractError(f"policy chunk shape {raw.shape} does not match adapter dim")
        return [raw[0, i].astype(np.float64) for i in range(self.n_action_steps)]

    def step(self, obs: dict, env) -> ApplyResult:
        if not self.queue:
            self.queue = self._infer(obs, env)
        # p_k of this tick becomes p_{k-1} of the next one, regardless of whether we inferred.
        self.obs_adapter.advance_tick(env)
        action = self.queue.pop(0)
        if getattr(self.action_adapter, "needs_obs", False):
            result = self.action_adapter.apply(action, obs)
        else:
            result = self.action_adapter.apply(action)
        result.detail["action"] = action.tolist()
        return result


# ---------------------------------------------------------------------------- Panda (M5)
VELOCITY_SLICE = slice(9, 12)  # ee_vx, ee_vy, ee_vz in PROPRIO_NAMES


class PandaObservationAdapter(ObservationAdapter):
    """Cartesian contract on ``PandaPickPlaceEnv``; refuses the joint contract.

    ``state`` is the env's canonical proprio with the velocity slot recomputed as the one-tick
    finite difference ``(p_k - p_{k-1}) * control_hz`` from :meth:`advance_tick`'s memory, exactly
    as the SO-101 path does. The env's own finite difference (``_ee_velocity``) must agree; a
    disagreement means the two implementations diverged and is a contract error, not something
    to paper over.
    """

    VELOCITY_ATOL = 1e-4  # m/s; float32 rounding of a one-tick finite difference

    def __init__(self, contract: dict, normalizer, *, device: str = "cpu"):
        super().__init__(contract, normalizer, device=device)
        if self.kind not in ("cartesian", "cartesian_absolute", "cartesian_tool"):
            raise ContractError(
                "a 6-D SO-101 joint-space checkpoint has no meaning on the 7-joint Panda; only the "
                "canonical 13-float / 7-D cartesian contract transfers (plan E3)")
        if self._transform is not None and self._transform.config.requires_joints:
            raise ContractError("SO-101 measured joint observations cannot be served on Panda")

    def _canonical_state(self, env) -> np.ndarray:
        x = np.asarray(env.canonical_proprio(), np.float32)
        if x.shape != (13,):
            raise ContractError(f"canonical state length {x.shape} != (13,)")
        p = env.ee_pose()[:3, 3]
        if self._prev_ee_pos is None:
            vel = np.zeros(3)
        else:
            vel = (p - self._prev_ee_pos) * float(env.control_hz)
        if not np.allclose(x[VELOCITY_SLICE], vel, atol=self.VELOCITY_ATOL):
            raise ContractError(
                f"Panda env velocity {x[VELOCITY_SLICE].tolist()} != one-tick finite difference "
                f"{vel.tolist()}; velocity semantics diverged from the SO-101 path")
        x[VELOCITY_SLICE] = vel.astype(np.float32)
        return x


class PandaActionAdapter:
    """Thin ``ChunkExecutor`` face over ``PandaCartesianActionAdapter`` (7-D, needs the obs)."""

    dim = 7
    needs_obs = True

    def __init__(self, panda_adapter):
        self.panda = panda_adapter
        self.env = panda_adapter.env
        self.ik_rejections = 0

    def reset(self) -> None:
        self.panda.reset_stats()
        self.ik_rejections = 0

    @property
    def stats(self) -> dict:
        return self.panda.stats

    def apply(self, action: np.ndarray, obs: dict) -> ApplyResult:
        a = np.asarray(action, np.float64).reshape(-1)
        if a.shape != (7,) or not np.all(np.isfinite(a)):
            raise ValueError("cartesian action must be seven finite values")
        res = self.panda.apply(a, obs)
        g_clamped = not 0.0 <= float(a[6]) <= 1.0  # PandaCartesianActionAdapter clips silently
        clamped = [res.translation_clamped] * 3 + [res.rotation_clamped] * 3 + [g_clamped]
        self.ik_rejections += int(res.rejected)
        return ApplyResult(res.observation, res.info, res.truncated, float(np.mean(clamped)),
                           ik_rejected=res.rejected,
                           detail={"clamped": clamped, "ik_reason": res.reason,
                                   "target_task": res.target_pose[:3, 3].tolist(),
                                   "position_error_m": res.residual_position_m,
                                   "orientation_error_rad": res.residual_orientation_rad,
                                   "ik_iterations": res.ik_iterations})
