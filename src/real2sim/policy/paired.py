"""Training-only replacement of eligible real images by provenance-checked renders."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class PairedViewDataset(Dataset):
    def __init__(self, base, root, probability=0.5, *, evaluation=False):
        self.base, self.root = base, Path(root)
        self.probability = float(probability)
        if not 0 <= self.probability <= 1:
            raise ValueError("replacement probability must be in [0,1]")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if not evaluation and (self.manifest.get("usage") == "diagnostic_only" or self.manifest.get("counterfactual")):
            raise ValueError("counterfactual renders cannot train against unchanged real actions")
        for key in ("source", "split", "files_sha256"):
            if self.manifest["derived_manifest"][key] != base.manifest[key]:
                raise ValueError(f"paired dataset {key} differs from real data")
        expected_split = "val" if evaluation else "train"
        if self.manifest["split_name"] != expected_split:
            raise ValueError("paired split is not permitted for this use")
        if not set(base.arrays["episode_index"][base.indices]) <= set(base.manifest["split"][expected_split]):
            raise ValueError("base dataset crosses the permitted episode split")
        self.image_files = self.manifest.get("image_files") or {self.manifest["camera_key"]: "images.npy"}
        if list(self.image_files) != base.camera_keys:
            raise ValueError("paired camera order differs from real data")
        if any(Path(name).name != name for name in self.image_files.values()):
            raise ValueError("paired image filenames must be local basenames")
        checksums = self.manifest.get("sha256", {})
        if set(checksums) != {"ticks.npy", *self.image_files.values()}:
            raise ValueError("paired manifest must checksum exactly ticks and all camera images")
        for filename, expected in checksums.items():
            if file_sha256(self.root / filename) != expected:
                raise ValueError(f"paired data checksum mismatch: {filename}")
        self.ticks = np.load(self.root / "ticks.npy", allow_pickle=False)
        if self.ticks.ndim != 1 or self.ticks.dtype.kind not in "iu" or len(np.unique(self.ticks)) != len(self.ticks):
            raise ValueError("paired ticks must be unique integer indices")
        if not len(self.ticks) or not np.isin(self.ticks, base.indices).all():
            raise ValueError("paired ticks must belong to the selected split")
        self.lookup = {int(t): i for i, t in enumerate(self.ticks)}
        self._images = None
        for images in self.image_views.values():
            if images.dtype != np.uint8 or images.shape != (len(self.ticks), *base.manifest["image_shape_hwc"]):
                raise ValueError("paired images have wrong shape or dtype")
        self.normalizer = base.normalizer

    @property
    def image_views(self):
        if self._images is None:
            self._images = {key: np.load(self.root / name, mmap_mode="r", allow_pickle=False)
                            for key, name in self.image_files.items()}
        return self._images

    @property
    def images(self):
        """Legacy first-camera view; multi-camera callers use image_views."""
        return self.image_views[next(iter(self.image_files))]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_images"] = None
        return state

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        paired = self.lookup.get(item["tick"])
        if paired is not None and torch.rand(()).item() < self.probability:
            for key, images in self.image_views.items():
                item[key] = torch.from_numpy(np.ascontiguousarray(self.normalizer.image(images[paired])))
        return item
