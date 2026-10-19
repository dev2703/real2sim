"""Checkpointable training batches, without worker prefetch changing resume order."""

import random

import numpy as np
import torch
from torch.utils.data import default_collate


class ResumableBatches:
    def __init__(self, dataset, batch_size: int, seed: int):
        if batch_size < 1 or len(dataset) < batch_size:
            raise ValueError("a complete positive batch is required")
        self.dataset, self.batch_size = dataset, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(dataset), generator=self.generator)
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position + self.batch_size > len(self.order):
            self.order = torch.randperm(len(self.dataset), generator=self.generator)
            self.position = 0
        indices = self.order[self.position:self.position + self.batch_size].tolist()
        self.position += self.batch_size
        return default_collate([self.dataset[i] for i in indices])

    def state_dict(self):
        return {"generator": self.generator.get_state(), "order": self.order.clone(),
                "position": self.position, "batch_size": self.batch_size, "length": len(self.dataset)}

    def load_state_dict(self, state):
        if state["length"] != len(self.dataset) or state["batch_size"] != self.batch_size:
            raise ValueError("resume dataset length or physical batch differs")
        order = state["order"].cpu()
        if not torch.equal(order.sort().values, torch.arange(len(self.dataset))) or not 0 <= state["position"] <= len(order):
            raise ValueError("invalid resumed sampler")
        self.order, self.position = order, int(state["position"])
        self.generator.set_state(state["generator"].cpu())


def rng_state():
    value = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        value["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        value["mps"] = torch.mps.get_rng_state()
    return value


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if "cuda" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in value["cuda"]])
    if "mps" in value and torch.backends.mps.is_available():
        torch.mps.set_rng_state(value["mps"].cpu())
