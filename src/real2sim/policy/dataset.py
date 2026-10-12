"""Torch dataset over a derived (``real2sim.data.prepare``) directory.

Each item is one policy tick: the image(s) and state at that tick plus the next ``chunk_size``
actions of the same episode with an ``action_is_pad`` mask. Normalisation uses the derived
dataset's train-only statistics; images use ImageNet mean/std (pretrained ResNet-18).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from real2sim.data.resample import chunk_indices


class Normalizer:
    """Mean/std normaliser for state and action, serialisable with the checkpoint."""

    def __init__(self, stats: dict):
        self.stats = stats
        self.state_mean = np.asarray(stats["observation.state"]["mean"], np.float32)
        self.state_std = np.asarray(stats["observation.state"]["std"], np.float32)
        self.action_mean = np.asarray(stats["action"]["mean"], np.float32)
        self.action_std = np.asarray(stats["action"]["std"], np.float32)
        img = stats["image_normalization"]
        self.img_mean = np.asarray(img["mean"], np.float32).reshape(3, 1, 1)
        self.img_std = np.asarray(img["std"], np.float32).reshape(3, 1, 1)

    @classmethod
    def from_json(cls, path: str | Path) -> Normalizer:
        return cls(json.loads(Path(path).read_text()))

    def state(self, x: np.ndarray) -> np.ndarray:
        return (x - self.state_mean) / self.state_std

    def action(self, x: np.ndarray) -> np.ndarray:
        return (x - self.action_mean) / self.action_std

    def unaction(self, x):
        if isinstance(x, torch.Tensor):
            return x * torch.as_tensor(self.action_std, device=x.device) + torch.as_tensor(self.action_mean, device=x.device)
        return x * self.action_std + self.action_mean

    def image(self, hwc_uint8: np.ndarray) -> np.ndarray:
        chw = hwc_uint8.transpose(2, 0, 1).astype(np.float32) / 255.0
        return (chw - self.img_mean) / self.img_std


class DerivedTickDataset(Dataset):
    def __init__(self, root: str | Path, episodes: list[int], chunk_size: int, normalizer: Normalizer | None = None):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.normalizer = normalizer or Normalizer.from_json(self.root / "stats.json")
        a = np.load(self.root / "arrays.npz")
        self.arrays = {k: a[k] for k in a.files}
        self.camera_keys: list[str] = self.manifest["camera_keys"]
        self._images: dict | None = None  # opened lazily per process (memmaps must not be pickled)
        ep = self.arrays["episode_index"]
        self.indices = np.flatnonzero(np.isin(ep, np.asarray(episodes)))
        if len(self.indices) == 0:
            raise ValueError(f"no ticks for episodes {episodes}")
        self.chunk_size = chunk_size

    @property
    def images(self) -> dict:
        if self._images is None:
            self._images = {
                k: np.load(self.root / self.manifest["image_files"][k], mmap_mode="r") for k in self.camera_keys
            }
        return self._images

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_images"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        t = int(self.indices[i])
        idx, pad = chunk_indices(self.arrays["ep_start"], self.arrays["ep_len"], t, self.chunk_size)
        n = self.normalizer
        item = {
            "observation.state": torch.from_numpy(n.state(self.arrays["state"][t]).astype(np.float32)),
            "action": torch.from_numpy(n.action(self.arrays["action"][idx]).astype(np.float32)),
            "action_is_pad": torch.from_numpy(pad),
            "episode_index": int(self.arrays["episode_index"][t]),
            "tick": t,
        }
        for k in self.camera_keys:
            item[k] = torch.from_numpy(np.ascontiguousarray(n.image(np.asarray(self.images[k][t]))))
        return item
