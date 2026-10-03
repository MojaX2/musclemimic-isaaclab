"""Learn action-conditioned next-contact prediction from infant motor babbling."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from developmental_skin import REGIONS


def load_transitions(path, reset_interval, prediction_horizon=1):
    if prediction_horizon < 1:
        raise ValueError("Prediction horizon must be positive")
    if reset_interval and prediction_horizon >= reset_interval:
        raise ValueError("Prediction horizon must be shorter than the reset interval")
    with np.load(path) as trajectory:
        if prediction_horizon >= len(trajectory["current"]):
            raise ValueError("Prediction horizon must be shorter than the trajectory")
        if "qpos_before" in trajectory:
            if prediction_horizon != 1 and reset_interval:
                raise ValueError("Multistep explicit records need an uninterrupted trajectory")
            steps = np.arange(len(trajectory["current"]) - prediction_horizon + 1)
            records = {"qpos": trajectory["qpos_before"][steps],
                       "current": trajectory["current"][steps],
                       "tactile": trajectory["tactile"][steps],
                       "action": trajectory["action"][steps],
                       "target": trajectory["next_tactile"][steps + prediction_horizon - 1]}
        else:
            steps = np.arange(1, len(trajectory["current"]) - prediction_horizon)
            if reset_interval:
                valid = np.ones(len(steps), dtype=bool)
                for offset in range(prediction_horizon + 1):
                    valid &= (steps + offset) % reset_interval != 0
                steps = steps[valid]
            records = {"qpos": trajectory["qpos"][steps - 1],
                       "current": trajectory["current"][steps],
                       "tactile": trajectory["tactile"][steps],
                       "action": trajectory["action"][steps],
                       "target": trajectory["tactile"][steps + prediction_horizon]}
        records["step"] = np.broadcast_to(steps[:, None],
                                          trajectory["current"][steps].shape[:2])
    return {name: value.reshape(-1, value.shape[-1]) if name != "step" else value.reshape(-1)
            for name, value in records.items()}


def subset(records, selector):
    return {name: value[selector] for name, value in records.items()}


def combine(groups):
    return {name: np.concatenate([group[name] for group in groups]) for name in groups[0]}


def trajectory_split(paths, heldout, reset_interval, validation_fraction=0.2,
                     prediction_horizon=1):
    training, validation, testing = [], [], []
    for path in paths:
        records = load_transitions(path, reset_interval, prediction_horizon)
        if path.resolve() == heldout.resolve():
            testing.append(records)
        else:
            boundary = int((records["step"].max() + 1) * (1 - validation_fraction))
            training.append(subset(records,
                                   records["step"] + prediction_horizon < boundary))
            validation.append(subset(records, records["step"] >= boundary))
    if len(testing) != 1 or not training or not validation:
        raise ValueError("Provide training trajectories and one distinct held-out trajectory")
    return combine(training), combine(validation), testing[0]


def prepare_features(splits, device, region_names=None):
    training = splits[0]
    statistics = {}
    for name in ("qpos", "current", "tactile", "action"):
        statistics[name] = (training[name].mean(0), np.maximum(training[name].std(0), 0.01))
    rate = (training["target"] > 1e-4).mean(0)
    active = (np.asarray([REGIONS.index(name) for name in region_names], dtype=np.int64)
              if region_names is not None else np.flatnonzero((rate > 0.02) & (rate < 0.98)))
    if not len(active):
        raise ValueError("No varying tactile regions in motor-babbling trajectories")
    if len(set(active.tolist())) != len(active):
        raise ValueError("Tactile region selection must not contain duplicates")
    prepared = []
    for split in splits:
        tensors = {name: torch.as_tensor((split[name] - statistics[name][0]) /
                                          statistics[name][1], dtype=torch.float32, device=device)
                   for name in statistics}
        tensors["label"] = torch.as_tensor(split["target"][:, active] > 1e-4,
                                            dtype=torch.float32, device=device)
        tensors["current_contact"] = torch.as_tensor(split["tactile"][:, active] > 1e-4,
                                                      dtype=torch.float32, device=device)
        prepared.append(tensors)
    return prepared, statistics, active


def model_input(split, mode, permutation=None):
    action = split["action"]
    if mode == "no_action":
        action = torch.zeros_like(action)
    elif mode == "shuffled_action":
        action = action[permutation]
    return torch.cat((split["qpos"], split["current"], split["tactile"], action), dim=1)


class TactileModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, 128), nn.GELU(),
                                     nn.Linear(128, 128), nn.GELU(),
                                     nn.Linear(128, output_size))

    def forward(self, observation):
        return self.network(observation)


def remap_input_normalization(checkpoint, statistics):
    names = ("qpos", "current", "tactile", "action")
    previous_mean = torch.cat([checkpoint["statistics"][name][0] for name in names])
    previous_deviation = torch.cat([checkpoint["statistics"][name][1] for name in names])
    updated_mean = torch.cat([torch.as_tensor(statistics[name][0]).float()
                              for name in names])
    updated_deviation = torch.cat([torch.as_tensor(statistics[name][1]).float()
                                   for name in names])
    if (previous_mean.shape != updated_mean.shape or
            checkpoint["input_size"] != len(updated_mean)):
        raise ValueError("Pretrained tactile input dimensions do not match")
    state = {name: value.clone() for name, value in checkpoint["state_dict"].items()}
    original_weight = state["network.0.weight"].clone()
    state["network.0.weight"] = original_weight * (updated_deviation /
                                                      previous_deviation).unsqueeze(0)
    state["network.0.bias"] += original_weight @ ((updated_mean - previous_mean) /
                                                    previous_deviation)
    return state


def train_condition(splits, mode, seed, epochs=100, patience=15, initial_state=None):
    training, validation, testing = splits
    torch.manual_seed(seed)
    generator = torch.Generator(device=training["label"].device).manual_seed(seed + 1000)
    features = []
    for split in splits:
        permutation = torch.randperm(len(split["label"]), generator=generator,
                                     device=split["label"].device)
        features.append(model_input(split, mode, permutation))
    model = TactileModel(features[0].shape[1], training["label"].shape[1]).to(features[0].device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best_validation = float("inf")
    best_epoch = 0
    best_parameters = None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(training["label"]), generator=generator,
                               device=features[0].device)
        for indices in order.split(256):
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(features[0][indices]), training["label"][indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(nn.functional.binary_cross_entropy_with_logits(
                model(features[1]), validation["label"]))
        if validation_loss < best_validation - 1e-5:
            best_validation = validation_loss
            best_epoch = epoch + 1
            best_parameters = {name: value.detach().clone()
                               for name, value in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= patience:
            break
    model.load_state_dict(best_parameters)
    with torch.no_grad():
        probability = model(features[2]).sigmoid()
        label = testing["label"]
        transitions = label != testing["current_contact"]
        squared_error = (probability - label).square()
        predicted = probability >= 0.5
        true_positive = (predicted & (label > 0.5)).sum(0).float()
        precision_denominator = predicted.sum(0).float()
        recall_denominator = label.sum(0)
        f1 = 2 * true_positive / (precision_denominator + recall_denominator).clamp_min(1)
        metrics = {
            "best_epoch": best_epoch,
            "validation_bce": best_validation,
            "heldout_bce": float(nn.functional.binary_cross_entropy_with_logits(
                model(features[2]), label)),
            "heldout_brier": float(squared_error.mean()),
            "heldout_transition_brier": float(squared_error[transitions].mean()),
            "heldout_macro_contact_f1": float(f1.mean()),
        }
    return metrics, {name: value.detach().cpu() for name, value in best_parameters.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", nargs="+", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reset-interval", type=int, default=20)
    parser.add_argument("--prediction-horizon", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--regions", nargs="+", choices=REGIONS)
    parser.add_argument("--initialize-from", type=Path)
    args = parser.parse_args()
    checkpoint_path = args.output.with_suffix(".pt")
    if args.output.exists() or checkpoint_path.exists():
        parser.error("Output or tactile checkpoint already exists")
    if args.reset_interval < 0 or min(args.epochs, args.patience,
                                      args.prediction_horizon) < 1:
        parser.error("Reset interval must be nonnegative and training lengths positive")
    splits = trajectory_split(args.trajectories, args.heldout, args.reset_interval,
                              prediction_horizon=args.prediction_horizon)
    prepared, statistics, active = prepare_features(splits, args.device, args.regions)
    initial_state = None
    if args.initialize_from is not None:
        initial_checkpoint = torch.load(args.initialize_from, map_location="cpu",
                                        weights_only=True)
        if (initial_checkpoint["output_size"] != len(active) or
                initial_checkpoint["active_regions"] != active.tolist()):
            parser.error("Pretrained tactile regions do not match")
        initial_state = remap_input_normalization(initial_checkpoint, statistics)
    records = []
    selected = None
    for seed in args.seeds:
        record = {"model_seed": seed}
        for mode in ("action", "no_action", "shuffled_action"):
            metrics, weights = train_condition(prepared, mode, seed, args.epochs,
                                               args.patience, initial_state)
            record[mode] = metrics
            if mode == "action" and (selected is None or
                    metrics["validation_bce"] < selected[0]):
                selected = (metrics["validation_bce"], seed, weights)
        records.append(record)
        print(json.dumps(record), flush=True)
    testing = prepared[2]
    persistence = testing["current_contact"]
    report = {
        "training_trajectories": [str(path.resolve()) for path in args.trajectories
                                  if path.resolve() != args.heldout.resolve()],
        "heldout_trajectory": str(args.heldout.resolve()),
        "initialize_from": (str(args.initialize_from.resolve())
                            if args.initialize_from else None),
        "reset_interval": args.reset_interval,
        "prediction_horizon_control_steps": args.prediction_horizon,
        "samples": dict(zip(("train", "validation", "heldout"),
                            [len(split["label"]) for split in prepared])),
        "active_regions": [REGIONS[index] for index in active],
        "heldout_transition_fraction": float((testing["label"] != persistence).float().mean()),
        "heldout_persistence_brier": float((persistence - testing["label"]).square().mean()),
        "results": records,
        "scope": "Next-step contact prediction from motor babbling, not acquired body schema",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": selected[2], "model_seed": selected[1],
                "input_size": sum(prepared[0][name].shape[1]
                                  for name in ("qpos", "current", "tactile", "action")),
                "output_size": len(active), "active_regions": active.tolist(),
                "prediction_horizon_control_steps": args.prediction_horizon,
                "source_checkpoint": (str(args.initialize_from.resolve())
                                      if args.initialize_from else None),
                "statistics": {name: tuple(torch.as_tensor(value).float() for value in pair)
                               for name, pair in statistics.items()}}, checkpoint_path)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
