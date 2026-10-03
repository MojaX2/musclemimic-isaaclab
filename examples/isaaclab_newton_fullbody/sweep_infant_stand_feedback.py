"""Screen feedback gains and control intervals for dynamic infant standing."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from developmental_skin import REGIONS
from infant_crawl_env import InfantCrawlEnv
from train_infant_crawl_ppo import jittered_positions
from train_infant_stand_ppo import load_bias


FOOT_IDS = (REGIONS.index("right_foot"), REGIONS.index("left_foot"))


def evaluate(env, bias, position_gains, velocity_gains, physics_per_control,
             duration_seconds, baseline, amplitude, initial_positions=None):
    env.reset(initial_positions)
    control_steps = round(duration_seconds / (float(env.source.opt.timestep) * physics_per_control))
    supported = torch.zeros(env.worlds, device=env.device)
    last = env.measure()
    for _ in range(control_steps):
        drives = env.feedback_drives(bias.expand(env.worlds, -1),
                                     position_gains[:, None], velocity_gains[:, None])
        last = env.step(env.action_from_drives(drives, baseline, amplitude), physics_per_control)
        supported += ((last["chest_height"] > 0.45) &
                      (last["ground_touch"][:, FOOT_IDS] > 2).all(-1)).float()
    return {"standing_fraction": (supported / control_steps).cpu().tolist(),
            "final_standing": ((last["chest_height"] > 0.45) &
                               (last["ground_touch"][:, FOOT_IDS] > 2).all(-1)).cpu().tolist(),
            "final_chest_height_m": last["chest_height"].cpu().tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=2.0)
    parser.add_argument("--screen-results", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.screen_results:
        screened = json.loads(args.screen_results.read_text())
        if (screened["scene"] != str(args.scene.resolve()) or
                screened["pose"] != str(args.pose.resolve()) or
                screened["bias"] != str(args.bias.resolve()) or
                screened["duration_seconds"] != args.duration_seconds):
            parser.error("Screen results do not match this evaluation")
        rows = screened["rows"]
    else:
        position_grid, velocity_grid = np.meshgrid([1.5, 3.0, 6.0, 12.0],
                                                   [0.25, 0.5, 1.0, 2.0], indexing="ij")
        position_gains = torch.as_tensor(position_grid.ravel(), device="cuda:0", dtype=torch.float32)
        velocity_gains = torch.as_tensor(velocity_grid.ravel(), device="cuda:0", dtype=torch.float32)
        env = InfantCrawlEnv(args.scene, args.pose, position_gains.numel())
        bias = load_bias(args.bias, env.device)
        rows = []
        for physics_per_control in (1, 2, 5, 10):
            for baseline, amplitude in ((0.2, 0.3), (0.4, 0.3)):
                result = evaluate(env, bias, position_gains, velocity_gains,
                                  physics_per_control, args.duration_seconds, baseline, amplitude)
                for index in range(env.worlds):
                    rows.append({"physics_per_control": physics_per_control,
                                 "position_gain": float(position_gains[index]),
                                 "velocity_gain": float(velocity_gains[index]),
                                 "baseline": baseline, "amplitude": amplitude,
                                 "standing_fraction": result["standing_fraction"][index],
                                 "final_standing": result["final_standing"][index],
                                 "final_chest_height_m": result["final_chest_height_m"][index]})
            print(json.dumps({"physics_per_control": physics_per_control,
                              "best": max(rows, key=lambda row: row["standing_fraction"])}), flush=True)
    selected = sorted(rows, key=lambda row: row["standing_fraction"], reverse=True)[:4]
    selected.append({"physics_per_control": 10, "position_gain": 3.0,
                     "velocity_gain": 1.0, "baseline": 0.2, "amplitude": 0.3})
    validation = InfantCrawlEnv(args.scene, args.pose, 8)
    validation_bias = load_bias(args.bias, validation.device)
    heldout = []
    for seed in (41, 43, 47):
        torch.manual_seed(seed)
        positions = jittered_positions(validation, 0.05, 0.01)
        for configuration in selected:
            position_gain = torch.full((validation.worlds,), configuration["position_gain"],
                                       device=validation.device)
            velocity_gain = torch.full((validation.worlds,), configuration["velocity_gain"],
                                       device=validation.device)
            result = evaluate(validation, validation_bias, position_gain, velocity_gain,
                              configuration["physics_per_control"], args.duration_seconds,
                              configuration["baseline"], configuration["amplitude"], positions)
            heldout.append({"seed": seed, "configuration": configuration,
                            "mean_standing_fraction": float(np.mean(result["standing_fraction"])),
                            "final_standing_fraction": float(np.mean(result["final_standing"])),
                            "final_chest_height_m": result["final_chest_height_m"]})
        print(json.dumps({"heldout_seed": seed}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()),
                                       "bias": str(args.bias.resolve()),
                                       "duration_seconds": args.duration_seconds,
                                       "rows": rows, "heldout": heldout}, indent=2) + "\n")


if __name__ == "__main__":
    main()
