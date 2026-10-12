"""Inference-time adapters between the frozen ACT checkpoint and the simulated SO-101 (plan M4.2-3).

* :class:`ObservationAdapter` assembles the policy batch from ``env.observation()`` only. It is
  bound to the checkpoint's audited contract (camera key, image shape, state names) and refuses
  anything else instead of adapting silently (plan section 7: feature meaning, order, units and
  train-time presence must match). It never reads ``info['privileged_metrics']``.
* :class:`JointActionAdapter` applies a 6-D joint target in dataset units (degrees, gripper %).
* :class:`CartesianActionAdapter` applies the canonical 7-D task-frame delta relative to the
  *current observed* EE pose: clamp -> IK (seeded from the current native qpos) -> native step;
  rejected IK targets hold the current pose and are counted, never silently applied.
* :class:`ChunkExecutor` runs one ``predict_action_chunk`` per ``n_action_steps`` ticks and drops
  the queue on reset. It advances the observation adapter's EE-position memory on *every* tick so
  the canonical proprio velocity is the one-tick finite difference ``(p_k - p_{k-1}) * hz`` of
  ``real2sim.data.labels`` even on ticks without an inference call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from real2sim.data.labels import ACTION_NAMES, PROPRIO_NAMES
from real2sim.geometry.so3 import apply_delta, is_rotation
from real2sim.geometry.so101_fk import JointMapping

IPHONE_KEY = "observation.images.iphone"
STATE_KEY = "observation.state"
IMAGE_SHAPE_CHW = (3, 192, 256)
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_STATE_NAMES = [f"{j}.pos" for j in JOINT_NAMES]
FORBIDDEN_PREFIXES = ("privileged", "sim.")


class ContractError(ValueError):
    """The checkpoint contract cannot be served by the simulated observation stream."""


def normalize_image_chw(normalizer, chw_float01: np.ndarray) -> np.ndarray:
    """``[3,H,W]`` float32 in [0,1] (env image) -> the normalizer's ImageNet normalisation.

    Equivalent to ``normalizer.image(hwc_uint8)`` applied to the uint8 image the env letterboxed;
    the env already divided by 255 so only the mean/std step remains.
    """
    chw = np.asarray(chw_float01, np.float32)
    if chw.shape != IMAGE_SHAPE_CHW or chw.min() < 0.0 or chw.max() > 1.0:
        raise ContractError(f"expected image {IMAGE_SHAPE_CHW} float in [0,1], got {chw.shape}")
    return (chw - normalizer.img_mean) / normalizer.img_std


def contract_kind(contract: dict) -> str:
    """``'joint'`` or ``'cartesian'``; raises :class:`ContractError` for anything else."""
    cams = list(contract.get("camera_keys", []))
    if cams != [IPHONE_KEY]:
        raise ContractError(f"contract camera_keys {cams} != [{IPHONE_KEY!r}]; refusing to adapt")
    shape = tuple(int(x) for x in contract.get("image_shape_chw", []))
    if shape != IMAGE_SHAPE_CHW:
        raise ContractError(f"contract image_shape_chw {shape} != {IMAGE_SHAPE_CHW}; refusing to adapt")
    names = list(contract.get("state_names", []))
    if names == list(PROPRIO_NAMES):
        kind = "cartesian"
    elif names == JOINT_STATE_NAMES or names == list(JOINT_NAMES):
        kind = "joint"
    else:
        raise ContractError(
            f"contract state_names {names} match neither the joint contract {JOINT_STATE_NAMES} "
            f"nor the canonical proprio {list(PROPRIO_NAMES)}; no new feature slots are served")
    actions = list(contract.get("action_names", []))
    expected = list(ACTION_NAMES) if kind == "cartesian" else JOINT_STATE_NAMES
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
        if len(normalizer.state_mean) != self.state_dim:
            raise ContractError("normalizer state statistics do not match contract state_names")
        # EE position observed at the previous control tick (None before the first tick, so the
        # velocity slot is zero exactly as ``labels.episode_labels`` zeroes vel[0]).
        self._prev_ee_pos: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_ee_pos = None

    def advance_tick(self, env) -> None:
        """Record the current EE position as ``p_{k-1}`` for the next tick's velocity.

        Must be called once per control tick *before* the action is applied (i.e. on the
        observation the policy would see at this tick), whether or not inference ran.
        """
        if self.kind == "cartesian":
            self._prev_ee_pos = env.ee_pose()[:3, 3].copy()

    def state(self, env) -> np.ndarray:
        """Observable state in contract units; never simulator object poses.

        Pure read: the velocity uses the position stored by :meth:`advance_tick` at the previous
        tick, so calling ``state`` more than once per tick does not change the feature.
        """
        if self.kind == "joint":
            x = np.asarray(env.real_state(), np.float32)
        else:
            x = np.asarray(env.canonical_proprio(self._prev_ee_pos), np.float32)
        if x.shape != (self.state_dim,):
            raise ContractError(f"state length {x.shape} != contract {self.state_dim}")
        return x

    def batch(self, obs: dict, env) -> dict[str, torch.Tensor]:
        # Only the contract's two inputs are copied; ``sim.*`` diagnostics and any privileged
        # metrics in ``info`` never reach the policy.
        if "privileged_metrics" in obs or any(k.startswith("privileged") for k in obs):
            raise ContractError("observation carries privileged keys")
        if IPHONE_KEY not in obs:
            raise ContractError(f"observation lacks {IPHONE_KEY}")
        img = normalize_image_chw(self.normalizer, obs[IPHONE_KEY])
        state = self.normalizer.state(self.state(env)).astype(np.float32)
        return {
            IPHONE_KEY: torch.from_numpy(np.ascontiguousarray(img))[None].to(self.device),
            STATE_KEY: torch.from_numpy(state)[None].to(self.device),
        }


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
    """7-D ``[dx,dy,dz,drx,dry,drz,g]`` relative to the current observed EE pose -> IK -> step."""

    dim = 7

    def __init__(self, env, kinematics, *, max_step_m: float = 0.03, max_step_rad: float = 0.3):
        self.env = env
        self.ik = kinematics
        self.max_step_m = float(max_step_m)
        self.max_step_rad = float(max_step_rad)
        self.mapping: JointMapping = env.mapping
        self.ik_rejections = 0
        self.residuals: list[dict] = []

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
        T_cur = self.env.ee_pose()
        if not is_rotation(T_cur[:3, :3]):
            raise RuntimeError("current EE pose is not a proper rotation")
        target = apply_delta(T_cur, np.concatenate([dp, dr]))
        # MuJoCo joint limits are soft: measured qpos can overshoot jnt_range by ~1e-3 rad under
        # contact, and NativeKinematics.solve refuses out-of-range seeds. Clip like replay.py does;
        # the clip is diagnostic-only (the seed), never a change to the upstream model.
        lo_all, hi_all = self.env.limits[:, 0], self.env.limits[:, 1]
        seed = np.clip(self.env.sim_state(), lo_all, hi_all)
        lo, hi = self.env.limits[5]
        grip = float(np.clip(self.mapping.gripper_real_to_sim_rad(g * 100.0), lo, hi))
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
        saturation = float(np.mean(clamped))
        return ApplyResult(obs, info, truncated, saturation, ik_rejected=not result.accepted,
                           detail={"clamped": clamped.tolist(), "ik_reason": result.reason,
                                   "target_task": target[:3, 3].tolist(),
                                   "position_error_m": result.position_error_m,
                                   "orientation_error_rad": result.orientation_error_rad})


class ChunkExecutor:
    """Open-loop execution of ``n_action_steps`` actions per policy call; re-observes afterwards."""

    def __init__(self, policy, normalizer, obs_adapter: ObservationAdapter, action_adapter,
                 n_action_steps: int | None = None):
        self.policy = policy
        self.normalizer = normalizer
        self.obs_adapter = obs_adapter
        self.action_adapter = action_adapter
        n = n_action_steps or int(policy.config.n_action_steps)
        if n < 1 or n > int(policy.config.chunk_size):
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        self.n_action_steps = n
        self.queue: list[np.ndarray] = []
        self.latencies_ms: list[float] = []
        self.n_policy_calls = 0

    def reset(self) -> None:
        self.policy.reset()
        self.obs_adapter.reset()
        self.queue.clear()
        self.latencies_ms.clear()
        self.n_policy_calls = 0

    def _infer(self, obs: dict, env) -> list[np.ndarray]:
        batch = self.obs_adapter.batch(obs, env)
        t0 = time.perf_counter()
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
        result = self.action_adapter.apply(action)
        result.detail["action"] = action.tolist()
        return result
