"""Compare static-torque standing bias with the evolved infant bias."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import mujoco
import torch

from evaluate_infant_newton_crawl import newton_crawl_environment
from evolve_infant_stand_cop_reflex import aligned_positions
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from sweep_infant_cop_reflex import evaluate
from train_infant_stand_ppo import load_bias


def interpolate_bias(previous, equilibrium, fraction):
    if previous.shape != equilibrium.shape or not 0 <= fraction <= 1:
        raise ValueError("Standing bias interpolation needs matching shapes and a fraction in [0, 1]")
    return previous + fraction * (equilibrium - previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--previous-bias", type=Path, required=True)
    parser.add_argument("--equilibrium-bias", type=Path, required=True)
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument("--backend", choices=("direct", "newton"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=800)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7001, 7003, 7007])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0., 0.5, 1.])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if min(args.worlds, args.control_steps) < 1 or not args.seeds or not args.fractions:
        parser.error("Worlds, control steps, seeds, and fractions must be nonempty and positive")
    if any(not 0 <= fraction <= 1 for fraction in args.fractions):
        parser.error("Bias fractions must be in [0, 1]")
    gain_values = json.loads(args.evolution.read_text())["selected"]["gains"]
    if len(gain_values) != 6:
        parser.error("Selected standing pressure reflex needs six gains")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    context = (newton_crawl_environment(source, args.scene, args.pose, args.worlds,
                                       controlled_joint_names=CRAWL_JOINTS)
               if args.backend == "newton" else
               nullcontext(InfantCrawlEnv(args.scene, args.pose, args.worlds)))
    with context as env:
        previous = load_bias(args.previous_bias, env.device)
        equilibrium = load_bias(args.equilibrium_bias, env.device)
        gains = torch.as_tensor(gain_values, device=env.device).expand(env.worlds, -1)
        records = []
        for seed in args.seeds:
            positions = aligned_positions(env, seed, env.worlds)
            pose_hash = hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest()
            for fraction in args.fractions:
                bias = interpolate_bias(previous, equilibrium, fraction)
                standing, final, chest, late = evaluate(
                    env, bias, gains, positions, args.control_steps, return_late=True)
                row = {"seed": seed, "bias_fraction": fraction,
                       "pose_sha256": pose_hash,
                       "standing_fraction": float(standing.mean()),
                       "late_standing_fraction": float(late.mean()),
                       "final_standing_count": int(final.sum()),
                       "per_world_standing_fraction": standing.cpu().tolist(),
                       "final_chest_height_m": chest.cpu().tolist()}
                records.append(row)
                print(json.dumps({key: row[key] for key in
                                  ("seed", "bias_fraction", "standing_fraction",
                                   "late_standing_fraction", "final_standing_count")}),
                      flush=True)
        report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
                  "previous_bias": str(args.previous_bias.resolve()),
                  "equilibrium_bias": str(args.equilibrium_bias.resolve()),
                  "evolution": str(args.evolution.resolve()), "backend": args.backend,
                  "worlds": args.worlds, "seeds": args.seeds, "fractions": args.fractions,
                  "duration_s": args.control_steps * float(env.source.opt.timestep),
                  "solver_limit_steps": getattr(env, "solver_limit_steps", None),
                  "records": records,
                  "scope": "Fixed-bias standing audit; not learned walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
