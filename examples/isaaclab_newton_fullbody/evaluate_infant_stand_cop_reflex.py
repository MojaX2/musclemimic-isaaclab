"""Evaluate evolved infant standing pressure reflex beyond its training horizon."""

import argparse
import json
from pathlib import Path

import torch

from infant_crawl_env import InfantCrawlEnv
from infant_stand_resets import align_stand_feet
from sweep_infant_cop_reflex import evaluate
from train_infant_crawl_ppo import jittered_positions
from train_infant_stand_ppo import load_bias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-steps", type=int, default=800)
    parser.add_argument("--seeds", nargs="+", type=int, default=[109, 113, 127])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.control_steps < 1 or not args.seeds:
        parser.error("Control steps and seeds must be positive")
    saved = json.loads(args.evolution.read_text())
    selected = saved["selected"]["gains"]
    if len(selected) != 2:
        parser.error("Evolution report must contain two selected gains")
    env = InfantCrawlEnv(args.scene, args.pose, 8)
    bias = load_bias(args.bias, env.device)
    records = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        raw = jittered_positions(env, 0.05, 0.01)
        positions, _ = align_stand_feet(env.source, raw, env.initial[0],
                                        env.root_address)
        for label, gains in (("evolved", selected), ("previous", [-20., -2.]),
                             ("no_reflex", [0., 0.])):
            gains_tensor = torch.tensor(gains, device=env.device).expand(8, -1)
            fraction, final, chest = evaluate(env, bias, gains_tensor, positions,
                                              args.control_steps)
            records.append({"seed": seed, "condition": label, "gains": gains,
                            "standing_fraction": float(fraction.mean()),
                            "final_standing_count": int(final.sum()),
                            "per_world_standing_fraction": fraction.cpu().tolist(),
                            "final_chest_height_m": chest.cpu().tolist()})
        print(json.dumps({"seed": seed}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()),
                                       "bias": str(args.bias.resolve()),
                                       "evolution": str(args.evolution.resolve()),
                                       "duration_s": args.control_steps *
                                                     float(env.source.opt.timestep),
                                       "seeds": args.seeds, "records": records,
                                       "scope": "Longer-horizon standing, not stepping or walking"},
                                      indent=2) + "\n")


if __name__ == "__main__":
    main()
