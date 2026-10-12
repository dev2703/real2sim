"""Recovery specialist MLP (D-005 option A): 29 -> 256 -> 256 -> 128 -> 7.

The module stores its input/output normalisation statistics and the ordered feature/action
name lists in the checkpoint. :meth:`RecoveryMLP.predict` and :meth:`RecoveryMLP.load` refuse a
feature list that differs from the one the weights were trained with (plan §7: meaning, order
and normalisation must match, not just shape).

Normalisation is per column ``(v - mean) / std`` with two guards (:meth:`set_normalisation`):
a column whose training std is ~0 (constant: e.g. ``gripper_contact`` or ``bowl_conf`` in
approach-phase corrections) is *not* scaled (std := 1, mean kept, flagged in ``x_const`` /
``y_const``), and every other std is floored at ``std_floor``. A bare ``1e-6`` floor turned a
0.01 deviation on a never-varying feature into z = 1e4, which swamped the input LayerNorm and,
on the target side, the Huber loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from real2sim.data.labels import ACTION_NAMES
from real2sim.recovery.detector import FEATURE_NAMES

CHECKPOINT_FORMAT = "real2sim.recovery_mlp.v1"
STD_FLOOR = 1e-3  # floor for a varying column's std (raw units)
CONST_TOL = 1e-6  # std at or below this -> column treated as constant (unscaled)


def _guard_std(std: torch.Tensor, std_floor: float, const_tol: float) -> tuple[torch.Tensor, torch.Tensor]:
    if std_floor <= const_tol:
        raise ValueError(f"std_floor ({std_floor}) must exceed const_tol ({const_tol})")
    if not torch.all(torch.isfinite(std)) or torch.any(std < 0):
        raise ValueError("std must be finite and non-negative")
    const = std <= const_tol
    guarded = torch.where(const, torch.ones_like(std), std.clamp_min(std_floor))
    return guarded, const


@dataclass(frozen=True)
class MLPSpec:
    input_dim: int = 29
    hidden: tuple[int, ...] = (256, 256, 128)
    output_dim: int = 7
    activation: str = "relu"
    input_layernorm: bool = True
    dropout: float = 0.1
    translation_m: float = 0.03
    rotation_rad: float = 0.3
    gripper: tuple[float, float] = (0.0, 1.0)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MLPSpec:
        m = config.get("mlp", config)
        b = m.get("action_bounds", {})
        return cls(
            input_dim=int(config.get("features", {}).get("input_dim", 29)),
            hidden=tuple(int(h) for h in m.get("hidden", (256, 256, 128))),
            output_dim=int(m.get("output_dim", 7)),
            activation=str(m.get("activation", "relu")),
            input_layernorm=bool(m.get("input_layernorm", True)),
            dropout=float(m.get("dropout", 0.1)),
            translation_m=float(b.get("translation_m", 0.03)),
            rotation_rad=float(b.get("rotation_rad", 0.3)),
            gripper=tuple(float(g) for g in b.get("gripper", (0.0, 1.0))),
        )

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        lo = torch.tensor([-self.translation_m] * 3 + [-self.rotation_rad] * 3 + [self.gripper[0]])
        hi = torch.tensor([self.translation_m] * 3 + [self.rotation_rad] * 3 + [self.gripper[1]])
        return lo, hi


_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class RecoveryMLP(nn.Module):
    def __init__(
        self,
        spec: MLPSpec | None = None,
        feature_names: list[str] | None = None,
        action_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec or MLPSpec()
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.action_names = list(action_names or ACTION_NAMES)
        if len(self.feature_names) != self.spec.input_dim:
            raise ValueError(f"{len(self.feature_names)} feature names for input_dim {self.spec.input_dim}")
        if len(self.action_names) != self.spec.output_dim:
            raise ValueError(f"{len(self.action_names)} action names for output_dim {self.spec.output_dim}")
        if self.spec.output_dim != 7:
            raise ValueError("action bounds are defined for the 7-D canonical action only")

        act = _ACTIVATIONS[self.spec.activation]
        layers: list[nn.Module] = []
        if self.spec.input_layernorm:
            layers.append(nn.LayerNorm(self.spec.input_dim))
        d = self.spec.input_dim
        for h in self.spec.hidden:
            layers += [nn.Linear(d, h), act(), nn.Dropout(self.spec.dropout)]
            d = h
        layers.append(nn.Linear(d, self.spec.output_dim))
        self.net = nn.Sequential(*layers)

        n_in, n_out = self.spec.input_dim, self.spec.output_dim
        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))
        self.register_buffer("y_mean", torch.zeros(n_out))
        self.register_buffer("y_std", torch.ones(n_out))
        self.register_buffer("x_const", torch.zeros(n_in, dtype=torch.bool))
        self.register_buffer("y_const", torch.zeros(n_out, dtype=torch.bool))
        lo, hi = self.spec.bounds()
        self.register_buffer("bound_lo", lo)
        self.register_buffer("bound_hi", hi)

    # ------------------------------------------------------------ normalisation
    def set_normalisation(
        self,
        x_mean,
        x_std,
        y_mean,
        y_std,
        std_floor: float = STD_FLOOR,
        const_tol: float = CONST_TOL,
    ) -> None:
        """Store training-set statistics; constant columns are passed through unscaled.

        ``std <= const_tol`` marks a column constant: its std is set to 1 so a deviation at
        inference enters at its raw scale (order 1 for a flag or a confidence) instead of being
        multiplied by ``1/eps``. Other stds are floored at ``std_floor`` (same units as the raw
        column; 1e-3 is well below any real spread of a metre/radian/confidence feature).
        """
        xs, xc = _guard_std(torch.as_tensor(x_std, dtype=torch.float32), std_floor, const_tol)
        ys, yc = _guard_std(torch.as_tensor(y_std, dtype=torch.float32), std_floor, const_tol)
        self.x_mean.copy_(torch.as_tensor(x_mean, dtype=torch.float32))
        self.x_std.copy_(xs)
        self.x_const.copy_(xc)
        self.y_mean.copy_(torch.as_tensor(y_mean, dtype=torch.float32))
        self.y_std.copy_(ys)
        self.y_const.copy_(yc)

    def constant_columns(self) -> dict[str, list[str]]:
        """Names of the feature / action columns that were constant in the training rows."""
        return {
            "features": [n for n, c in zip(self.feature_names, self.x_const.tolist(), strict=True) if c],
            "actions": [n for n, c in zip(self.action_names, self.y_const.tolist(), strict=True) if c],
        }

    def normalise_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def forward_normalised(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Unclamped output in normalised units (training loss, Huber delta in these units)."""
        if x_raw.shape[-1] != self.spec.input_dim:
            raise ValueError(f"expected input dim {self.spec.input_dim}, got {x_raw.shape[-1]}")
        return self.net((x_raw - self.x_mean) / self.x_std)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Raw 7-D action, de-normalised and clamped exactly like the ACT action adapters.

        Translation and rotation are clamped by vector *norm* (direction preserved), as
        ``CartesianActionAdapter`` / ``PandaCartesianActionAdapter`` do; a per-component box would
        let a diagonal step reach sqrt(3) x the limit and change its direction. Gripper in [lo, hi].
        """
        y = self.forward_normalised(x_raw) * self.y_std + self.y_mean
        dp, dr, g = y[..., :3], y[..., 3:6], y[..., 6:7]
        dp = dp * torch.clamp(self.spec.translation_m / dp.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1.0)
        dr = dr * torch.clamp(self.spec.rotation_rad / dr.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1.0)
        g = g.clamp(self.spec.gripper[0], self.spec.gripper[1])
        return torch.cat([dp, dr, g], dim=-1)

    def check_feature_names(self, feature_names: list[str]) -> None:
        if list(feature_names) != self.feature_names:
            raise ValueError(
                "feature name list does not match the checkpoint: "
                f"got {list(feature_names)[:5]}..., expected {self.feature_names[:5]}..."
            )

    @torch.no_grad()
    def predict(self, x_raw, feature_names: list[str]) -> torch.Tensor:
        """Inference entry point; refuses a mismatching feature list."""
        self.check_feature_names(feature_names)
        was_training = self.training
        self.eval()
        try:
            return self.forward(torch.as_tensor(x_raw, dtype=torch.float32))
        finally:
            self.train(was_training)

    # ----------------------------------------------------------------- save/load
    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "spec": asdict(self.spec),
                "feature_names": self.feature_names,
                "action_names": self.action_names,
                "state_dict": self.state_dict(),
                "extra": extra or {},
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, feature_names: list[str] | None = None, map_location="cpu") -> RecoveryMLP:
        ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
        spec_d = dict(ckpt["spec"])
        spec_d["hidden"] = tuple(spec_d["hidden"])
        spec_d["gripper"] = tuple(spec_d["gripper"])
        model = cls(MLPSpec(**spec_d), ckpt["feature_names"], ckpt["action_names"])
        model.load_state_dict(ckpt["state_dict"])
        if feature_names is not None:
            model.check_feature_names(feature_names)
        model.eval()
        return model
