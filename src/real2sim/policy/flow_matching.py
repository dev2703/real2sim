"""Small frozen-vision action-chunk heads for the bounded F0 experiment.

The flow and regression arms have exactly the same conditioning and parameter
schema. Flow uses x(t)=(1-t)*noise+t*action and learns action-noise; regression
uses zero action/time inputs. Both return normalized Cartesian chunks for the
existing ChunkExecutor. This is deliberately separate from ACT and its VAE.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from real2sim.policy.dataset import Normalizer

FEATURE_KEY = "observation.dino_features"


@dataclass
class FlowHeadConfig:
    objective: str = "flow"
    state_dim: int = 13
    action_dim: int = 7
    chunk_size: int = 20
    n_action_steps: int = 2
    feature_dim: int = 384
    feature_tokens: int = 266
    dim_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    dim_feedforward: int = 1024
    sampling_steps: int = 4
    sampling_seed: int = 0
    camera_key: str = "observation.images.iphone"
    dinov2_config: dict = field(default_factory=dict)
    dinov2_model_id: str = "facebook/dinov2-small"
    dinov2_revision: str = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    type: str = "small_chunk_head"

    def __post_init__(self):
        if self.objective not in {"flow", "regression"}:
            raise ValueError("objective must be flow or regression")
        for name in ("state_dim", "action_dim", "chunk_size", "n_action_steps", "feature_dim",
                     "feature_tokens", "dim_model", "n_heads", "n_layers", "dim_feedforward"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim_model % self.n_heads or self.dim_model % 2:
            raise ValueError("dim_model must be even and divisible by n_heads")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps exceeds chunk_size")
        if self.sampling_steps not in (4, 8):
            raise ValueError("F0 sampling_steps must be 4 or 8")
        if self.type != "small_chunk_head":
            raise ValueError("unexpected head checkpoint type")


def seeded_noise(shape, *, device, dtype, seed: int | list[int]) -> torch.Tensor:
    """Generate on CPU so the seed protocol also works on MPS.

    Per-example seeds make offline predictions independent of validation batch
    boundaries. No target-dependent sampling or best-of-many selection is used.
    """
    if isinstance(seed, int):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    else:
        if len(seed) != shape[0]:
            raise ValueError("one sampling seed is required per example")
        noise = torch.stack([
            torch.randn(shape[1:], generator=torch.Generator().manual_seed(int(s))) for s in seed
        ])
    return noise.to(device=device, dtype=dtype)


class ChunkHead(nn.Module):
    """Two cross-attending transformer blocks; vision projected once per chunk."""

    def __init__(self, config: FlowHeadConfig):
        super().__init__()
        self.config = config
        d = config.dim_model
        self.feature_projection = nn.Linear(config.feature_dim, d)
        self.state_projection = nn.Linear(config.state_dim, d)
        self.action_projection = nn.Linear(config.action_dim, d)
        self.time_projection = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.action_position = nn.Parameter(torch.randn(config.chunk_size, d) * .02)
        self.feature_position = nn.Parameter(torch.randn(config.feature_tokens, d) * .02)
        layer = nn.TransformerDecoderLayer(
            d, config.n_heads, config.dim_feedforward, dropout=0., activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerDecoder(layer, config.n_layers, norm=nn.LayerNorm(d))
        self.output = nn.Linear(d, config.action_dim)
        self.register_buffer("time_frequencies", torch.exp(
            -math.log(10000) * torch.arange(d // 2, dtype=torch.float32) / max(d // 2 - 1, 1)
        ), persistent=False)

    def condition(self, state: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        c = self.config
        if state.ndim != 2 or state.shape[1] != c.state_dim:
            raise ValueError("state shape does not match head configuration")
        if features.shape != (len(state), c.feature_tokens, c.feature_dim):
            raise ValueError("DINO feature shape does not match head configuration")
        image = self.feature_projection(features) + self.feature_position
        return torch.cat([self.state_projection(state).unsqueeze(1), image], dim=1)

    def field(self, x: torch.Tensor, t: torch.Tensor, memory: torch.Tensor,
              action_is_pad: torch.Tensor | None = None) -> torch.Tensor:
        phase = t.reshape(-1, 1).to(x.dtype) * self.time_frequencies.to(x.dtype).unsqueeze(0)
        time = self.time_projection(torch.cat([phase.sin(), phase.cos()], dim=-1))
        query = self.action_projection(x) + self.action_position + time.unsqueeze(1)
        hidden = self.blocks(query, memory, tgt_key_padding_mask=action_is_pad)
        return self.output(hidden)

    def training_loss(self, batch: dict, *, generator: torch.Generator | None = None,
                      noise: torch.Tensor | None = None, time: torch.Tensor | None = None):
        """Masked MSE; padded/extrapolated targets never enter valid query inputs."""
        action = batch["action"]
        pad = batch["action_is_pad"].bool()
        if action.shape[1:] != (self.config.chunk_size, self.config.action_dim):
            raise ValueError("action shape does not match head configuration")
        if pad.shape != action.shape[:2]:
            raise ValueError("action padding shape does not match actions")
        rows = (~pad).any(dim=1)
        valid_components = int((~pad).sum()) * self.config.action_dim
        if not rows.any():
            # Avoid all-masked attention softmax, retain a differentiable zero.
            return sum(p.sum() * 0 for p in self.parameters()), {"valid_components": 0}
        action = action[rows]
        pad = pad[rows]
        action = action.masked_fill(pad.unsqueeze(-1), 0)
        memory = self.condition(batch["observation.state"][rows], batch[FEATURE_KEY][rows])
        if self.config.objective == "flow":
            if noise is None:
                noise = torch.randn(batch["action"].shape, generator=generator, device="cpu")
            if time is None:
                time = torch.rand(len(batch["action"]), generator=generator, device="cpu")
            noise = noise.to(device=action.device, dtype=action.dtype)[rows]
            noise = noise.masked_fill(pad.unsqueeze(-1), 0)
            time = time.to(device=action.device, dtype=action.dtype)[rows]
            if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
                raise ValueError("interpolation times must be in [0, 1]")
            x = noise + time[:, None, None] * (action - noise)
            target = action - noise
        else:
            x = torch.zeros_like(action)
            time = action.new_zeros(len(action))
            target = action
        prediction = self.field(x, time, memory, pad)
        error = (prediction - target).square().masked_fill(pad.unsqueeze(-1), 0)
        loss = error.sum() / valid_components
        return loss, {"valid_components": valid_components}

    @torch.no_grad()
    def sample(self, state: torch.Tensor, features: torch.Tensor, *, steps: int | None = None,
               seed: int | list[int] = 0) -> torch.Tensor:
        memory = self.condition(state, features)
        shape = (len(state), self.config.chunk_size, self.config.action_dim)
        if self.config.objective == "regression":
            return self.field(state.new_zeros(shape), state.new_zeros(len(state)), memory)
        steps = self.config.sampling_steps if steps is None else steps
        if steps not in (4, 8):
            raise ValueError("F0 sampling requires four or eight Euler steps")
        x = seeded_noise(shape, device=state.device, dtype=state.dtype, seed=seed)
        for step in range(steps):
            t = state.new_full((len(state),), step / steps)
            x = x + self.field(x, t, memory) / steps
        return x


class FlowHeadPolicy(nn.Module):
    """Frozen DINO + trainable head, or head alone while reading cached features."""

    def __init__(self, config: FlowHeadConfig, *, backbone: nn.Module | None = None):
        super().__init__()
        self.config = config
        self.head = ChunkHead(config)
        self.backbone = backbone
        if backbone is not None:
            backbone.requires_grad_(False).eval()
        self.reset()

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def reset(self, seed: int | None = None):
        self._seed = self.config.sampling_seed if seed is None else int(seed)
        self._calls = 0
        self._queue = deque()

    def features(self, batch: dict) -> torch.Tensor:
        if FEATURE_KEY in batch:
            return batch[FEATURE_KEY]
        if self.backbone is None:
            raise ValueError("raw-image inference requires an attached frozen DINO backbone")
        with torch.no_grad():
            features = self.backbone(batch[self.config.camera_key])["feature_map"]
        return features.flatten(2).transpose(1, 2)

    def forward(self, batch: dict, **kwargs):
        return self.head.training_loss({**batch, FEATURE_KEY: self.features(batch)}, **kwargs)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, *, sampling_seed: int | list[int] | None = None,
                             sampling_steps: int | None = None):
        if sampling_seed is None:
            sampling_seed = self._seed + self._calls
            self._calls += 1
        return self.head.sample(batch["observation.state"], self.features(batch),
                                steps=sampling_steps, seed=sampling_seed)

    @torch.no_grad()
    def select_action(self, batch: dict):
        if not self._queue:
            chunk = self.predict_action_chunk(batch)
            self._queue.extend(chunk[:, :self.config.n_action_steps].transpose(0, 1))
        return self._queue.popleft()


def build_frozen_backbone(config: FlowHeadConfig, *, pretrained_path: str | None = None):
    from real2sim.policy.dinov2_act import DinoSpatialBackbone

    if not config.dinov2_config:
        raise ValueError("DINO architecture must be embedded in the checkpoint")
    cfg = SimpleNamespace(dinov2_config=config.dinov2_config, dinov2_trainable_blocks=0,
                          dinov2_pretrained_path=pretrained_path or "")
    return DinoSpatialBackbone(cfg, initialize_pretrained=pretrained_path is not None).eval()


def save_flow_checkpoint(path: Path, policy: FlowHeadPolicy, stats: dict, contract: dict,
                         run_info: dict, *, backbone_state: dict | None = None):
    """Save a portable inference checkpoint; never rely on an external HF cache."""
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in policy.state_dict().items()}
    if policy.backbone is None:
        if backbone_state is None:
            raise ValueError("a portable checkpoint requires frozen backbone weights")
        state.update({f"backbone.{k}": v.detach().cpu().contiguous() for k, v in backbone_state.items()})
    save_file(state, path / "model.safetensors")
    for name, value in (("config.json", asdict(policy.config)), ("stats.json", stats),
                        ("real2sim_contract.json", contract), ("run_info.json", run_info)):
        (path / name).write_text(json.dumps(value, indent=2) + "\n")


def load_flow_checkpoint(path: str | Path, device: str = "cpu"):
    from safetensors.torch import load_file

    path = Path(path)
    config = FlowHeadConfig(**json.loads((path / "config.json").read_text()))
    policy = FlowHeadPolicy(config, backbone=build_frozen_backbone(config))
    policy.load_state_dict(load_file(path / "model.safetensors"), strict=True)
    policy.to(device).eval()
    contract = json.loads((path / "real2sim_contract.json").read_text())
    return policy, Normalizer.from_json(path / "stats.json"), contract
