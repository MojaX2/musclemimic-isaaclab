"""Audit contact-persistence blending and action pairing on held-out babbling."""

import argparse
import json
from pathlib import Path

import torch

from train_infant_tactile_model import TactileModel, load_transitions


def blend_brier(current, target, predicted, blend):
    if current.shape != target.shape or predicted.shape != target.shape:
        raise ValueError("Contact arrays must have matching shapes")
    if not 0 <= blend <= 1:
        raise ValueError("Contact blend must be between zero and one")
    error = (current + blend * (predicted - current) - target).square()
    changed = current != target
    return {"brier": float(error.mean()),
            "transition_brier": float(error[changed].mean()) if changed.any() else None,
            "unchanged_brier": float(error[~changed].mean()) if (~changed).any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reset-interval", type=int, default=20)
    parser.add_argument("--blends", nargs="+", type=float,
                        default=[0., 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.])
    parser.add_argument("--shuffle-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if (args.reset_interval < 0 or args.shuffle_trials < 1 or not args.blends or
            any(not 0 <= blend <= 1 for blend in args.blends)):
        parser.error("Reset interval, shuffle trials, or blend values are invalid")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    horizon = checkpoint.get("prediction_horizon_control_steps", 1)
    records = load_transitions(args.trajectory, args.reset_interval, horizon)
    values = {name: torch.as_tensor(records[name], dtype=torch.float32)
              for name in ("qpos", "current", "tactile", "action", "target")}
    normalized = {name: (values[name] - checkpoint["statistics"][name][0]) /
                        checkpoint["statistics"][name][1]
                  for name in ("qpos", "current", "tactile", "action")}
    fixed = torch.cat([normalized[name] for name in ("qpos", "current", "tactile")], dim=1)
    action = normalized["action"]
    model = TactileModel(checkpoint["input_size"], checkpoint["output_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    active = checkpoint["active_regions"]
    current = (values["tactile"][:, active] > 1e-4).float()
    target = (values["target"][:, active] > 1e-4).float()
    generator = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        predicted = model(torch.cat((fixed, action), dim=1)).sigmoid()
        shuffled = [model(torch.cat((fixed, action[torch.randperm(
            len(action), generator=generator)]), dim=1)).sigmoid()
                    for _ in range(args.shuffle_trials)]
    rows = []
    for blend in args.blends:
        true = blend_brier(current, target, predicted, blend)
        shuffled_brier = [blend_brier(current, target, guess, blend)["brier"]
                          for guess in shuffled]
        rows.append({"blend": blend, **true,
                     "shuffled_action_brier_mean": sum(shuffled_brier) / len(shuffled_brier),
                     "shuffled_action_brier_min": min(shuffled_brier),
                     "shuffled_action_brier_max": max(shuffled_brier)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint.resolve()),
                                       "trajectory": str(args.trajectory.resolve()),
                                       "samples": len(target),
                                       "active_regions": active,
                                       "prediction_horizon_control_steps": horizon,
                                       "transition_fraction": float((current != target).float().mean()),
                                       "shuffle_trials": args.shuffle_trials,
                                       "shuffle_seed": args.seed,
                                       "results": rows,
                                       "scope": "Held-out future-contact calibration, not "
                                                "a body schema or locomotion test"},
                                      indent=2) + "\n")


if __name__ == "__main__":
    main()
