"""Test whether infant babbling touch improves held-out body-state prediction."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def trajectory_split(paths, heldout, validation_fraction=0.2):
    training, validation, testing = [], [], []
    for path in paths:
        with np.load(path) as trajectory:
            records = {key: trajectory[key].reshape(-1, trajectory[key].shape[-1])
                       for key in ("current", "action", "tactile", "target")}
            count = len(trajectory["current"])
            boundary = int(count * (1 - validation_fraction)) * trajectory["current"].shape[1]
        if path.resolve() == heldout.resolve():
            testing.append(records)
        else:
            training.append({key: value[:boundary] for key, value in records.items()})
            validation.append({key: value[boundary:] for key, value in records.items()})
    if len(testing) != 1 or not training or not validation:
        raise ValueError("Provide at least two training trajectories and one distinct held-out trajectory")
    return tuple({key: np.concatenate([record[key] for record in group])
                  for key in ("current", "action", "tactile", "target")}
                 for group in (training, validation, testing))


def normalize(training, validation, testing, device):
    state_width = training["target"].shape[1]
    statistics = {}
    for name in ("current", "action", "tactile"):
        values = training[name]
        statistics[name] = (values.mean(0), np.maximum(values.std(0), 0.01))
    residual = training["target"] - training["current"][:, :state_width]
    statistics["residual"] = (residual.mean(0), np.maximum(residual.std(0), 0.01))
    result = []
    for split in (training, validation, testing):
        converted = {}
        for name in ("current", "action", "tactile"):
            mean, std = statistics[name]
            converted[name] = torch.as_tensor((split[name] - mean) / std,
                                               dtype=torch.float32, device=device)
        mean, std = statistics["residual"]
        delta = split["target"] - split["current"][:, :state_width]
        converted["target"] = torch.as_tensor((delta - mean) / std,
                                               dtype=torch.float32, device=device)
        result.append(converted)
    return result, statistics


def inputs(split, mode, permutation=None):
    touch = split["tactile"]
    action = split["action"]
    if mode in ("no_touch", "shuffled_action"):
        touch = torch.zeros_like(touch)
    elif mode == "shuffled_touch":
        touch = touch[permutation]
    if mode == "shuffled_action":
        action = action[permutation]
    return torch.cat((split["current"], action, touch), dim=1)


class WorldModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, 256), nn.GELU(),
                                     nn.Linear(256, 256), nn.GELU(),
                                     nn.Linear(256, output_size))

    def forward(self, observation):
        return self.network(observation)


def train_condition(splits, statistics, mode, seed, epochs=100, patience=15):
    train, validation, test = splits
    torch.manual_seed(seed)
    generator = torch.Generator(device=train["current"].device).manual_seed(seed + 1000)
    prepared = []
    for split in splits:
        permutation = torch.randperm(len(split["current"]), generator=generator,
                                     device=split["current"].device)
        prepared.append(inputs(split, mode, permutation))
    model = WorldModel(prepared[0].shape[1], train["target"].shape[1]).to(prepared[0].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best_validation = float("inf")
    best_parameters = None
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train["target"]), generator=generator,
                               device=prepared[0].device)
        for indices in order.split(256):
            prediction = model(prepared[0][indices])
            loss = (prediction - train["target"][indices]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_error = float((model(prepared[1]) - validation["target"]).square().mean())
        if validation_error < best_validation - 1e-5:
            best_validation = validation_error
            best_epoch = epoch + 1
            best_parameters = {name: value.clone() for name, value in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= patience:
            break
    model.load_state_dict(best_parameters)
    with torch.no_grad():
        normalized_error = (model(prepared[2]) - test["target"]).square()
    residual_std = torch.as_tensor(statistics["residual"][1],
                                   dtype=torch.float32, device=prepared[0].device)
    physical_error = normalized_error * residual_std.square()
    velocity_width = 2 * test["target"].shape[1] - test["current"].shape[1]
    if not 0 < velocity_width < test["target"].shape[1]:
        raise ValueError("Cannot infer velocity width from infant state and target")
    metrics = {"best_epoch": best_epoch,
               "validation_normalized_mse": best_validation,
               "heldout_normalized_mse": float(normalized_error.mean()),
               "heldout_velocity_mse": float(physical_error[:, :velocity_width].mean()),
               "heldout_muscle_length_mse": float(physical_error[:, velocity_width:].mean())}
    checkpoint = {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "input_size": prepared[0].shape[1], "output_size": train["target"].shape[1],
        "mode": mode, "model_seed": seed,
        "statistics": {name: tuple(torch.as_tensor(value).float() for value in pair)
                       for name, pair in statistics.items()},
    }
    return metrics, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", nargs="+", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    checkpoint_paths = {mode: args.output.with_name(f"{args.output.stem}_{mode}.pt")
                        for mode in ("touch", "no_touch")}
    if args.output.exists() or any(path.exists() for path in checkpoint_paths.values()):
        parser.error("Output or transferable checkpoint already exists")
    splits = trajectory_split(args.trajectories, args.heldout)
    normalized, statistics = normalize(*splits, args.device)
    records = []
    best_checkpoints = {}
    for seed in args.seeds:
        result = {"model_seed": seed}
        for mode in ("touch", "no_touch", "shuffled_touch", "shuffled_action"):
            metrics, checkpoint = train_condition(normalized, statistics, mode, seed,
                                                  args.epochs, args.patience)
            result[mode] = metrics
            if mode in checkpoint_paths and (mode not in best_checkpoints or
                    metrics["validation_normalized_mse"] < best_checkpoints[mode][0]):
                best_checkpoints[mode] = (metrics["validation_normalized_mse"], checkpoint)
        records.append(result)
        print(json.dumps(result), flush=True)
    heldout = splits[2]
    velocity_width = 2 * heldout["target"].shape[1] - heldout["current"].shape[1]
    persistence_error = (heldout["target"] - heldout["current"][:, :heldout["target"].shape[1]]) ** 2
    report = {"training_trajectories": [str(path.resolve()) for path in args.trajectories
                                        if path.resolve() != args.heldout.resolve()],
              "heldout_trajectory": str(args.heldout.resolve()),
              "samples": {name: len(split["current"])
                          for name, split in zip(("train", "validation", "heldout"), splits)},
              "model_initialization_seeds": args.seeds,
              "transfer_checkpoints": {mode: str(path.resolve())
                                       for mode, path in checkpoint_paths.items()},
              "heldout_persistence_velocity_mse": float(persistence_error[:, :velocity_width].mean()),
              "heldout_persistence_muscle_length_mse": float(persistence_error[:, velocity_width:].mean()),
              "results": records,
              "scope": "Held-out next-state prediction; not proof of acquired body schema"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode, path in checkpoint_paths.items():
        torch.save(best_checkpoints[mode][1], path)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
