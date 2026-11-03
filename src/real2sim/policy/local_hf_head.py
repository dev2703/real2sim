"""Local residual action head over the user's frozen native ACT decoder.

The original image encoder, transformer and action head are preserved. Only a
small residual head is fitted to simulation command supervision; native saved
pre/postprocessing and the 25 Hz action contract remain in force.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn

from real2sim.artifacts import sha256_file
from real2sim.policy.native_lerobot_act import NativeLeRobotACT


class ResidualActionHead(nn.Module):
    def __init__(self, original, mean, std, kind="mlp"):
        super().__init__()
        self.original = copy.deepcopy(original).requires_grad_(False)
        self.register_buffer("feature_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(std, dtype=torch.float32).clamp_min(.01))
        self.kind = kind
        self.residual = (nn.Linear(256, 6) if kind == "linear" else nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 6)))
        final = self.residual if kind == "linear" else self.residual[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features):
        normalized = (features - self.feature_mean) / self.feature_std
        return self.original(features) + self.residual(normalized)


class LocalHeadACT(NativeLeRobotACT):
    """Run the original HF visual ACT with an explicitly identified local head."""

    def __init__(self, path, head_dir, *, device="cpu"):
        super().__init__(path, device=device)
        directory = Path(head_dir)
        self.head_metadata = json.loads((directory / "head.json").read_text())
        if self.weights_sha256 != self.head_metadata["base_checkpoint_sha256"]:
            raise ValueError("Local head was trained for a different HF checkpoint")
        checkpoint = directory / "head.pt"
        if sha256_file(checkpoint) != self.head_metadata["head_sha256"]:
            raise ValueError("Local head checksum mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        head = ResidualActionHead(self.policy.model.action_head, payload["feature_mean"],
                                  payload["feature_std"], self.head_metadata["kind"])
        head.load_state_dict(payload, strict=True)
        self.policy.model.action_head = head.to(device)
        self.policy.eval().requires_grad_(False)
        self.local_head_sha256 = self.head_metadata["head_sha256"]
