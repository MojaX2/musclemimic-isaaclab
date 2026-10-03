"""Measure how long the MIMo infant stands with no control or tonic muscle tone."""

import argparse
import json
from pathlib import Path

import torch

from developmental_skin import REGIONS
from infant_crawl_env import InfantCrawlEnv


def rollout(env, muscle_tone, control_steps, physics_per_control):
    env.reset()
    action = torch.full((env.worlds, 2 * env.source.nu), muscle_tone, device=env.device)
    chest, head, feet, trunk, pelvis = [], [], [], [], []
    foot_ids = [REGIONS.index("right_foot"), REGIONS.index("left_foot")]
    for _ in range(control_steps):
        measure = env.step(action, physics_per_control)
        chest.append(measure["chest_height"].clone())
        head.append(measure["head_height"].clone())
        feet.append(measure["ground_touch"][:, foot_ids].clone())
        trunk.append(measure["ground_touch"][:, REGIONS.index("trunk")].clone())
        pelvis.append(measure["ground_touch"][:, REGIONS.index("pelvis")].clone())
    chest = torch.stack(chest)
    head = torch.stack(head)
    feet = torch.stack(feet)
    trunk = torch.stack(trunk)
    pelvis = torch.stack(pelvis)
    return {"muscle_tone": muscle_tone,
            "final_chest_height_m": chest[-1].cpu().tolist(),
            "final_head_height_m": head[-1].cpu().tolist(),
            "fraction_chest_above_0p45_m": float((chest > 0.45).float().mean()),
            "fraction_both_feet_supported": float((feet > 2).all(-1).float().mean()),
            "fraction_trunk_ground_contact": float((trunk > 2).float().mean()),
            "fraction_pelvis_ground_contact": float((pelvis > 2).float().mean()),
            "final_foot_forces_n": feet[-1].cpu().tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--control-steps", type=int, default=20)
    parser.add_argument("--physics-per-control", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    env = InfantCrawlEnv(args.scene, args.pose, args.worlds)
    result = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "worlds": args.worlds,
              "duration_s": args.control_steps * args.physics_per_control * env.source.opt.timestep,
              "unactuated": rollout(env, 0.0, args.control_steps, args.physics_per_control),
              "tonic_0p2": rollout(env, 0.2, args.control_steps, args.physics_per_control),
              "scope": "Static standing start and open-loop fall test; not balance or walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: result[name] for name in ("unactuated", "tonic_0p2")}, indent=2))


if __name__ == "__main__":
    main()
