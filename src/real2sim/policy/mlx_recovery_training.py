"""Optional Metal training backend; export into the existing frozen-ACT runtime.

MLX is imported only when explicitly selected. Torch is used for artifact
serialization and validation, never for this backend's gradients or optimizer.
"""
from __future__ import annotations

import numpy as np


class MLXRecoveryTrainer:
    def __init__(self, model, learning_rate):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        self.mx, self.model = mx, model
        self.optim_module = optim
        state = model.state_dict()
        self.constants = {name: mx.array(state[name].numpy()) for name in (
            "observation_mean", "observation_std", "feature_mean", "feature_std", "action_scales")}
        layers = []
        self.indices = []
        for index, layer in enumerate(model.network):
            if hasattr(layer, "weight"):
                layers.append(nn.Linear(layer.in_features, layer.out_features))
                self.indices.append(index)
            else:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)
        self.network.load_weights([(f"layers.{i}.{name}", mx.array(state[f"network.{i}.{name}"].numpy()))
                                   for i in self.indices for name in ("weight", "bias")])
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        self.value_and_grad = nn.value_and_grad(self.network, self.loss)

    def forward(self, network, batch):
        mx, c = self.mx, self.constants
        obs = (batch["observations"] - c["observation_mean"]) / c["observation_std"]
        features = (batch["features"] - c["feature_mean"]) / c["feature_std"]
        if self.model.normalization_clip is not None:
            limit = self.model.normalization_clip
            obs, features = mx.clip(obs, -limit, limit), mx.clip(features, -limit, limit)
        prior = batch["base_actions"] / c["action_scales"]
        if not self.model.use_act:
            prior, features = mx.zeros_like(prior), mx.zeros_like(features)
        return (prior + network(mx.concatenate([obs, prior, features], axis=-1))) * c["action_scales"]

    def loss(self, network, batch):
        mx = self.mx
        error = mx.abs((self.forward(network, batch) - batch["target_actions"])
                       / self.constants["action_scales"])
        return mx.mean(mx.where(error < .1, .5 * error ** 2 / .1, error - .05))

    def step(self, batch):
        mx = self.mx
        batch = {name: mx.array(value.numpy()) for name, value in batch.items()}
        loss, gradients = self.value_and_grad(self.network, batch)
        gradients, _ = self.optim_module.clip_grad_norm(gradients, 1.)
        self.optimizer.update(self.network, gradients)
        mx.eval(self.network.parameters(), self.optimizer.state, loss)
        return float(loss)

    def export(self):
        import torch
        from mlx.utils import tree_flatten

        parameters = dict(tree_flatten(self.network.parameters()))
        with torch.no_grad():
            for i in self.indices:
                for name in ("weight", "bias"):
                    getattr(self.model.network[i], name).copy_(
                        torch.from_numpy(np.array(parameters[f"layers.{i}.{name}"])))
