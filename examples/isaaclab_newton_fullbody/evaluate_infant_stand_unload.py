"""Measure how partial body-weight support changes infant standing survival."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import mujoco
import torch

from evaluate_infant_newton_crawl import newton_crawl_environment
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
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
    parser.add_argument("--backend", choices=("direct", "newton"), required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=800)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3101, 3103, 3107])
    parser.add_argument("--unload-fractions", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--planar-stiffness", type=float, default=0.0)
    parser.add_argument("--planar-damping", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.worlds < 1 or args.control_steps < 1 or not args.seeds:
        parser.error("Worlds, control steps, and seeds must be positive")
    if not args.unload_fractions or any(not 0 <= value <= 1
                                        for value in args.unload_fractions):
        parser.error("Unload fractions must be between zero and one")
    if (not math.isfinite(args.planar_stiffness) or
            not math.isfinite(args.planar_damping) or
            args.planar_stiffness < 0 or args.planar_damping < 0):
        parser.error("Planar tether gains must be nonnegative")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    selected = json.loads(args.evolution.read_text())["selected"]["gains"]
    if len(selected) != 6:
        parser.error("Expected a six-gain standing pressure reflex")
    context = (newton_crawl_environment(source, args.scene, args.pose, args.worlds,
                                        controlled_joint_names=CRAWL_JOINTS)
               if args.backend == "newton" else
               nullcontext(InfantCrawlEnv(args.scene, args.pose, args.worlds,
                                          controlled_joint_names=CRAWL_JOINTS)))
    records = []
    with context as environment:
        bias = load_bias(args.bias, environment.device)
        gains = torch.tensor(selected, device=environment.device).expand(args.worlds, -1)
        for seed in args.seeds:
            torch.manual_seed(seed)
            raw = jittered_positions(environment, 0.05, 0.01)
            positions, alignment = align_stand_feet(environment.source, raw,
                                                    environment.initial[0],
                                                    environment.root_address)
            pose_hash = hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest()
            for unload in args.unload_fractions:
                standing, final, chest, late = evaluate(
                    environment, bias, gains, positions, args.control_steps,
                    return_late=True, root_vertical_unload=unload,
                    root_planar_stiffness=args.planar_stiffness,
                    root_planar_damping=args.planar_damping)
                feet = environment.measure()["foot_force"]
                records.append({"seed": seed, "root_vertical_unload": unload,
                                "initial_pose_sha256": pose_hash,
                                "initial_foot_distance_m": [row["foot_distance_m"]
                                                            for row in alignment],
                                "standing_fraction": float(standing.mean()),
                                "late_standing_fraction": float(late.mean()),
                                "final_standing_count": int(final.sum()),
                                "per_world_standing_fraction": standing.cpu().tolist(),
                                "per_world_late_standing_fraction": late.cpu().tolist(),
                                "final_chest_height_m": chest.cpu().tolist(),
                                "final_foot_force_n": feet.cpu().tolist(),
                                "solver_limit_steps": getattr(environment,
                                                              "solver_limit_steps", None)})
                print(json.dumps({"seed": seed, "unload": unload,
                                  "standing": float(standing.mean()),
                                  "late": float(late.mean()),
                                  "final": int(final.sum())}), flush=True)
        solver_limit_steps = (sum(row["solver_limit_steps"] for row in records)
                              if args.backend == "newton" else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()),
                                       "backend": args.backend,
                                       "bias": str(args.bias.resolve()),
                                       "evolution": str(args.evolution.resolve()),
        "worlds": args.worlds, "seeds": args.seeds,
                                       "planar_stiffness_n_per_m": args.planar_stiffness,
                                       "planar_damping_ns_per_m": args.planar_damping,
                                       "duration_s": args.control_steps *
                                                     float(source.opt.timestep),
                                       "body_weight_n": float(source.body_mass.sum() *
                                                              -source.opt.gravity[2]),
                                       "records": records,
                                       "solver_limit_steps": solver_limit_steps,
                                       "scope": "Externally supported standing feasibility; "
                                                "not self-supported stance or walking"},
                                      indent=2) + "\n")


if __name__ == "__main__":
    main()
