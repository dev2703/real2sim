"""Reproducible episode-level appearance, camera, layout and physics variation.

The model is restored before every application. Randomization never changes
actuator gains, joint limits, object geometry, or the success scorer. Call
``apply(seed)`` before reset and pass its explicit ``scenario`` to the reset.
These variations are training inputs, not a claim of robust task completion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from numbers import Real
from pathlib import Path

import mujoco
import numpy as np

SCHEMA = "episode_domain_randomization_v1"
_STREAM_NAMESPACE = 20261112


def _finite_scalar(value, name, *, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return float(value)


def _positive_interval(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value positive interval")
    low, high = (_finite_scalar(part, name) for part in value)
    if not 0 < low <= high:
        raise ValueError(f"{name} must be an ordered positive interval")
    return low, high


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Declared bounds; translation offsets are in each camera's mounting frame."""

    ball_position_jitter_m: float = .025
    bowl_position_jitter_m: float = .015
    camera_position_jitter_m: float = .008
    camera_rotation_jitter_deg: float = 3.
    camera_fovy_scale: tuple[float, float] = (.95, 1.05)
    light_scale: tuple[float, float] = (.7, 1.3)
    color_gain: tuple[float, float] = (.9, 1.1)
    color_offset: float = .025
    friction_scale: tuple[float, float] = (.8, 1.2)
    ball_mass_scale: tuple[float, float] = (.85, 1.15)

    def __post_init__(self):
        for name in ("ball_position_jitter_m", "bowl_position_jitter_m", "camera_position_jitter_m"):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        object.__setattr__(self, "camera_rotation_jitter_deg", _finite_scalar(
            self.camera_rotation_jitter_deg, "camera_rotation_jitter_deg", maximum=180.))
        object.__setattr__(self, "color_offset", _finite_scalar(self.color_offset, "color_offset", maximum=1.))
        for name in ("camera_fovy_scale", "light_scale", "color_gain", "friction_scale", "ball_mass_scale"):
            object.__setattr__(self, name, _positive_interval(getattr(self, name), name))

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, Mapping):
            raise ValueError("Domain randomization config must be an object")
        if unknown := payload.keys() - {field.name for field in fields(cls)}:
            raise ValueError(f"Unknown domain randomization fields: {sorted(unknown)}")
        return cls(**payload)

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        # JSON-compatible lists keep configuration checksums independent of
        # whether an equivalent profile came from Python or a JSON document.
        return json.loads(json.dumps(asdict(self), allow_nan=False))


def _digest_json(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _disc(rng, radius):
    angle = rng.uniform(0., 2 * np.pi)
    distance = radius * np.sqrt(rng.uniform())
    return distance * np.array([np.cos(angle), np.sin(angle)])


def _quaternion_product(left, right):
    """MuJoCo wxyz product, retaining a unit camera-local rotation."""
    w1, v1, w2, v2 = left[0], left[1:], right[0], right[1:]
    result = np.r_[w1 * w2 - v1 @ v2, w1 * v2 + w2 * v1 + np.cross(v1, v2)]
    return result / np.linalg.norm(result)


class DomainRandomizer:
    """Restore an immutable model baseline, then apply one recorded sample.

    Layout and physics use separate random streams from per-geom appearance, so
    two arms with different model sizes receive identical ball/bowl positions
    and mass/friction factors for a shared seed. Sampling never mutates state.
    """

    _MODEL_ARRAYS = (
        "body_pos", "body_mass", "body_inertia", "geom_friction", "geom_rgba", "mat_rgba",
        "cam_pos", "cam_quat", "cam_fovy", "cam_intrinsic", "light_ambient", "light_diffuse",
        "light_specular",
    )
    _HEADLIGHT_ARRAYS = ("ambient", "diffuse", "specular")

    def __init__(self, env, config=None):
        if config is None:
            config = DomainRandomizationConfig()
        elif isinstance(config, (str, Path)):
            config = DomainRandomizationConfig.from_json(config)
        elif not isinstance(config, DomainRandomizationConfig):
            config = DomainRandomizationConfig.from_dict(config)
        self.env, self.config = env, config
        model = env.model
        self._baseline = {name: np.array(getattr(model, name), copy=True) for name in self._MODEL_ARRAYS}
        self._headlight = {name: np.array(getattr(model.vis.headlight, name), copy=True)
                           for name in self._HEADLIGHT_ARRAYS}
        self._ball_center = np.asarray(env.task["ball"]["start_region"]["center_task_xy"], float).copy()
        self._bowl_center = np.asarray(env.task["bowl"]["center_task_xy"], float).copy()
        if any(value.shape != (2,) or not np.isfinite(value).all()
               for value in (self._ball_center, self._bowl_center)):
            raise ValueError("Domain randomization needs finite task-frame ball/bowl centers")
        self._ball_body = int(model.body("ball").id)
        self._collision_geoms = np.flatnonzero((model.geom_contype | model.geom_conaffinity) != 0)
        if (self._baseline["body_mass"][self._ball_body] <= 0
                or np.any(self._baseline["body_inertia"][self._ball_body] <= 0)):
            raise ValueError("Domain randomization requires a positive ball mass and inertia")
        fov = self._baseline["cam_fovy"]
        if (np.any(fov * config.camera_fovy_scale[0] <= 0)
                or np.any(fov * config.camera_fovy_scale[1] >= 180)):
            raise ValueError("Randomized camera FOV must remain strictly between 0 and 180 degrees")
        hasher = hashlib.sha256()
        for name, values in sorted({**self._baseline, **{
                f"headlight.{key}": value for key, value in self._headlight.items()},
                "ball_center": self._ball_center, "bowl_center": self._bowl_center}.items()):
            hasher.update(name.encode())
            hasher.update(str((values.dtype.str, values.shape)).encode())
            hasher.update(values.tobytes())
            values.setflags(write=False)
        self.baseline_sha256 = hasher.hexdigest()
        self.profile = {"schema": SCHEMA, "config": config.to_dict(),
                        "friction_scope": "sliding coefficient of all collision-enabled geoms",
                        "camera_translation_scope": "componentwise local mounting-frame offsets",
                        "camera_rotation_scope": "camera-local rotation vector, bounded norm",
                        "geometry_and_scorer_changed": False,
                        "actuator_gains_and_joint_limits_changed": False}
        self.profile_sha256 = _digest_json(self.profile)

    def _rng(self, seed, stream):
        return np.random.default_rng(np.random.SeedSequence([seed, _STREAM_NAMESPACE, stream]))

    def sample(self, seed):
        """Return a JSON-compatible deterministic descriptor without modifying the model."""
        if type(seed) is not int or seed < 0:
            raise ValueError("Randomization seed must be a nonnegative integer")
        cfg = self.config
        layout_rng, camera_rng, light_rng, color_rng, physics_rng = (
            self._rng(seed, stream) for stream in range(5))
        scenario = {
            "ball_xy": (self._ball_center + _disc(layout_rng, cfg.ball_position_jitter_m)).tolist(),
            "bowl_xy": (self._bowl_center + _disc(layout_rng, cfg.bowl_position_jitter_m)).tolist(),
        }
        cameras = len(self._baseline["cam_fovy"])
        camera_position = camera_rng.uniform(-cfg.camera_position_jitter_m, cfg.camera_position_jitter_m,
                                             size=(cameras, 3))
        axes = camera_rng.normal(size=(cameras, 3))
        axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
        rotation = axes * camera_rng.uniform(0., np.deg2rad(cfg.camera_rotation_jitter_deg), (cameras, 1))
        parameters = {
            "camera_position_delta_m": camera_position.tolist(),
            "camera_rotation_vector_rad": rotation.tolist(),
            "camera_fovy_scale": camera_rng.uniform(*cfg.camera_fovy_scale, size=cameras).tolist(),
            "light_scale": float(light_rng.uniform(*cfg.light_scale)),
            "color_gain": color_rng.uniform(*cfg.color_gain, size=3).tolist(),
            "geom_color_offset": color_rng.uniform(-cfg.color_offset, cfg.color_offset,
                size=(len(self._baseline["geom_rgba"]), 3)).tolist(),
            "material_color_offset": color_rng.uniform(-cfg.color_offset, cfg.color_offset,
                size=(len(self._baseline["mat_rgba"]), 3)).tolist(),
            "friction_scale": float(physics_rng.uniform(*cfg.friction_scale)),
            "ball_mass_scale": float(physics_rng.uniform(*cfg.ball_mass_scale)),
        }
        descriptor = {"schema": SCHEMA, "seed": seed,
                      "profile_sha256": self.profile_sha256, "baseline_sha256": self.baseline_sha256,
                      "scenario": scenario, "parameters": parameters}
        descriptor["sha256"] = _digest_json(descriptor)
        return descriptor

    def restore(self):
        """Restore original parameters and recompute MuJoCo's inertial constants."""
        model = self.env.model
        for name, value in self._baseline.items():
            getattr(model, name)[:] = value
        for name, value in self._headlight.items():
            getattr(model.vis.headlight, name)[:] = value
        mujoco.mj_setConst(model, self.env.data)

    def apply(self, seed):
        """Restore, randomize and return the scenario to supply to the next reset."""
        descriptor = self.sample(seed)
        self.restore()
        model, params = self.env.model, descriptor["parameters"]
        model.cam_pos[:] += np.asarray(params["camera_position_delta_m"]).reshape(model.ncam, 3)
        for index, vector in enumerate(params["camera_rotation_vector_rad"]):
            vector = np.asarray(vector)
            angle = float(np.linalg.norm(vector))
            delta = np.r_[np.cos(angle / 2), vector * (np.sin(angle / 2) / angle if angle else .5)]
            model.cam_quat[index] = _quaternion_product(self._baseline["cam_quat"][index], delta)
        new_fovy = self._baseline["cam_fovy"] * np.asarray(params["camera_fovy_scale"])
        model.cam_fovy[:] = new_fovy
        # FOV cameras use cam_fovy; sensor/intrinsics cameras use focal length.
        # Update both representations to keep the actual optics and reported K
        # consistent, including non-square or off-center calibrated cameras.
        focal_scale = np.tan(np.deg2rad(self._baseline["cam_fovy"]) / 2) / np.tan(np.deg2rad(new_fovy) / 2)
        model.cam_intrinsic[:, :2] *= focal_scale[:, None]
        lighting_scale = params["light_scale"]
        for name in ("light_ambient", "light_diffuse", "light_specular"):
            getattr(model, name)[:] = np.clip(self._baseline[name] * lighting_scale, 0., 1.)
        for name in self._HEADLIGHT_ARRAYS:
            getattr(model.vis.headlight, name)[:] = np.clip(self._headlight[name] * lighting_scale, 0., 1.)
        gain = np.asarray(params["color_gain"])
        for name, offset_name in (("geom_rgba", "geom_color_offset"), ("mat_rgba", "material_color_offset")):
            baseline = self._baseline[name]
            offsets = np.asarray(params[offset_name]).reshape(len(baseline), 3)
            getattr(model, name)[:, :3] = np.clip(baseline[:, :3] * gain + offsets, 0., 1.)
        model.geom_friction[self._collision_geoms, 0] *= params["friction_scale"]
        model.body_mass[self._ball_body] *= params["ball_mass_scale"]
        model.body_inertia[self._ball_body] *= params["ball_mass_scale"]
        mujoco.mj_setConst(model, self.env.data)
        return descriptor
