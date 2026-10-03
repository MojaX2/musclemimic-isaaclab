"""Frozen next-contact prediction features for infant policy transfer."""

import torch

from run_infant_babbling import virtual_lengths
from train_infant_tactile_model import TactileModel


class TactileFeatureEncoder:
    def __init__(self, checkpoint, device, random_features=False, normalize_features=False,
                 representation="hidden", contact_blend=0.5):
        if representation not in ("hidden", "predicted_contact", "blended_contact"):
            raise ValueError("Unsupported tactile representation")
        if not 0 <= contact_blend <= 1:
            raise ValueError("Contact blend must be between zero and one")
        self.random_features = random_features
        self.normalize_features = normalize_features
        self.representation = representation
        self.contact_blend = contact_blend
        self.prediction_horizon_control_steps = checkpoint.get(
            "prediction_horizon_control_steps", 1)
        self.shuffle_actions = False
        self.feature_size = (128 if representation == "hidden" else checkpoint["output_size"])
        self.model = TactileModel(checkpoint["input_size"], checkpoint["output_size"]).to(device)
        if not random_features:
            self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval().requires_grad_(False)
        self.statistics = {name: tuple(value.to(device) for value in pair)
                           for name, pair in checkpoint["statistics"].items()}
        self.active_regions = list(checkpoint["active_regions"])
        self.body_weight = None

    @classmethod
    def from_path(cls, path, device, random_features=False, normalize_features=False,
                  representation="hidden", contact_blend=0.5):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls(checkpoint, device, random_features, normalize_features,
                   representation, contact_blend)

    @classmethod
    def restore_snapshot(cls, snapshot, device):
        restored = cls(snapshot, device, normalize_features=snapshot.get("normalize_features", False),
                       representation=snapshot.get("representation", "hidden"),
                       contact_blend=snapshot.get("contact_blend", 0.5))
        restored.random_features = snapshot.get("random_features", False)
        return restored

    def normalized_input(self, env, measure, muscle_action):
        if self.body_weight is None:
            self.body_weight = float(env.source.body_mass.sum()) * 9.81
        current = torch.cat((env.velocity, virtual_lengths(env.muscles, env.position),
                             env.muscles.activity), dim=1)
        touch = torch.log1p(measure["touch"].clamp_min(0) / self.body_weight)
        values = []
        for name, tensor in (("qpos", env.position), ("current", current),
                             ("tactile", touch), ("action", muscle_action)):
            mean, deviation = self.statistics[name]
            values.append((tensor - mean) / deviation)
        observation = torch.cat(values, dim=1)
        if observation.shape[1] != self.model.network[0].in_features:
            raise ValueError("Infant tactile feature shape differs from babbling checkpoint")
        return observation

    def encode(self, env, measure, muscle_action):
        with torch.no_grad():
            feature_action = (muscle_action.roll(1, dims=0)
                              if self.shuffle_actions else muscle_action)
            normalized = self.normalized_input(env, measure, feature_action)
            if self.representation in ("predicted_contact", "blended_contact"):
                features = self.model(normalized).sigmoid()
                if self.representation == "blended_contact":
                    current = (torch.log1p(measure["touch"][:, self.active_regions].clamp_min(0) /
                                           self.body_weight) > 1e-4).float()
                    features = current + self.contact_blend * (features - current)
            else:
                features = self.model.network[:-1](normalized)
            if self.normalize_features and self.representation == "hidden":
                features = torch.nn.functional.layer_norm(features, (self.feature_size,))
            return features.detach()

    def predict(self, env, measure, muscle_action):
        with torch.no_grad():
            return self.model(self.normalized_input(env, measure,
                                                    muscle_action)).sigmoid()

    def snapshot(self):
        return {
            "kind": "tactile_next_contact_v1", "random_features": self.random_features,
            "normalize_features": self.normalize_features,
            "representation": self.representation,
            "contact_blend": self.contact_blend,
            "prediction_horizon_control_steps": self.prediction_horizon_control_steps,
            "input_size": self.model.network[0].in_features,
            "output_size": self.model.network[-1].out_features,
            "active_regions": self.active_regions,
            "state_dict": {name: value.detach().cpu() for name, value in self.model.state_dict().items()},
            "statistics": {name: tuple(value.detach().cpu() for value in pair)
                           for name, pair in self.statistics.items()},
        }
