"""Screen full-range antagonist excitation for infant standing balance."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from infant_crawl_env import InfantCrawlEnv
from infant_stand_resets import align_stand_feet
from sweep_infant_stand_feedback import evaluate
from train_infant_crawl_ppo import jittered_positions
from train_infant_stand_ppo import load_bias


EXCITATIONS = ((0.2, 0.2), (0.4, 0.3), (0.4, 0.4), (0.5, 0.5))
POSITION_GAINS = (1.5, 3.0, 6.0)
VELOCITY_GAINS = (0.5, 1.0, 2.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--heldout-seeds", nargs="+", type=int, default=[53, 59, 61])
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.duration_s <= 0 or args.top < 1 or not args.heldout_seeds:
        parser.error("Duration, top count, and held-out seeds must be positive")
    position_grid, velocity_grid = np.meshgrid(POSITION_GAINS, VELOCITY_GAINS,
                                               indexing="ij")
    position_gains = torch.as_tensor(position_grid.ravel(), device="cuda:0",
                                     dtype=torch.float32)
    velocity_gains = torch.as_tensor(velocity_grid.ravel(), device="cuda:0",
                                     dtype=torch.float32)
    nominal_env = InfantCrawlEnv(args.scene, args.pose, len(position_gains))
    bias = load_bias(args.bias, nominal_env.device)
    screened = []
    for baseline, amplitude in EXCITATIONS:
        outcome = evaluate(nominal_env, bias, position_gains, velocity_gains, 1,
                           args.duration_s, baseline, amplitude)
        for index in range(nominal_env.worlds):
            screened.append({"baseline": baseline, "amplitude": amplitude,
                             "position_gain": float(position_gains[index]),
                             "velocity_gain": float(velocity_gains[index]),
                             "standing_fraction": outcome["standing_fraction"][index],
                             "final_standing": outcome["final_standing"][index],
                             "final_chest_height_m": outcome["final_chest_height_m"][index]})
        print(json.dumps({"baseline": baseline, "amplitude": amplitude,
                          "best_standing_fraction": max(outcome["standing_fraction"])}),
              flush=True)
    selected = sorted(screened, key=lambda row: (row["final_standing"],
                                                   row["standing_fraction"]),
                      reverse=True)[:args.top]
    validation_env = InfantCrawlEnv(args.scene, args.pose, 8)
    validation_bias = load_bias(args.bias, validation_env.device)
    heldout = []
    for seed in args.heldout_seeds:
        torch.manual_seed(seed)
        raw = jittered_positions(validation_env, 0.05, 0.01)
        positions, alignment = align_stand_feet(validation_env.source, raw,
                                                validation_env.initial[0],
                                                validation_env.root_address)
        for configuration in selected:
            position_gain = torch.full((8,), configuration["position_gain"],
                                       device=validation_env.device)
            velocity_gain = torch.full((8,), configuration["velocity_gain"],
                                       device=validation_env.device)
            outcome = evaluate(validation_env, validation_bias, position_gain,
                               velocity_gain, 1, args.duration_s,
                               configuration["baseline"], configuration["amplitude"],
                               positions)
            heldout.append({"seed": seed, "configuration": configuration,
                            "mean_standing_fraction": float(np.mean(outcome["standing_fraction"])),
                            "final_standing_count": int(np.sum(outcome["final_standing"])),
                            "final_chest_height_m": outcome["final_chest_height_m"]})
        print(json.dumps({"heldout_seed": seed, "aligned_count": len(alignment)}),
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()),
                                       "bias": str(args.bias.resolve()),
                                       "duration_s": args.duration_s,
                                       "selection": "Nominal pose only; held-out seeds not used for selection",
                                       "nominal_screen": screened, "selected": selected,
                                       "heldout": heldout,
                                       "scope": "Muscle drive range and PD gain diagnostic; no walking policy"},
                                      indent=2) + "\n")


if __name__ == "__main__":
    main()
