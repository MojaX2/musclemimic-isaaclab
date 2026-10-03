"""Frozen babbling forward-model features for developmental policy transfer."""

import torch

from run_infant_babbling import virtual_lengths
from train_infant_world_model import WorldModel


class WorldFeatureEncoder:
    def __init__(self, checkpoint, device, random_features=False):
        if checkpoint["mode"] not in ("touch", "no_touch"):
            raise ValueError("Transfer requires a touch or no-touch forward model")
        self.mode = checkpoint["mode"]
        self.random_features = random_features
        self.model = WorldModel(checkpoint["input_size"], checkpoint["output_size"]).to(device)
        if not random_features:
            self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.statistics = {name: tuple(value.to(device) for value in pair)
                           for name, pair in checkpoint["statistics"].items()}
        self.body_weight = None

    @classmethod
    def from_path(cls, path, device, random_features=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls(checkpoint, device, random_features)

    @classmethod
    def restore_snapshot(cls, snapshot, device):
        restored = cls(snapshot, device)
        restored.random_features = snapshot.get("random_features", False)
        return restored

    def encode(self, env, measure, previous_action):
        if self.body_weight is None:
            self.body_weight = float(env.source.body_mass.sum()) * 9.81
        current = torch.cat((env.velocity, virtual_lengths(env.muscles, env.position),
                             env.muscles.activity), dim=1)
        touch = torch.log1p(measure["touch"].clamp_min(0) / self.body_weight)
        if self.mode == "no_touch":
            touch = torch.zeros_like(touch)
        values = []
        for name, tensor in (("current", current), ("action", previous_action),
                             ("tactile", touch)):
            mean, deviation = self.statistics[name]
            values.append((tensor - mean) / deviation)
        observation = torch.cat(values, dim=1)
        if observation.shape[1] != self.model.network[0].in_features:
            raise ValueError("Infant transfer feature shape differs from babbling checkpoint")
        with torch.no_grad():
            return self.model.network[:-1](observation).detach()

    def snapshot(self):
        return {
            "mode": self.mode, "random_features": self.random_features,
            "input_size": self.model.network[0].in_features,
            "output_size": self.model.network[-1].out_features,
            "state_dict": {name: value.detach().cpu() for name, value in self.model.state_dict().items()},
            "statistics": {name: tuple(value.detach().cpu() for value in pair)
                           for name, pair in self.statistics.items()},
        }
