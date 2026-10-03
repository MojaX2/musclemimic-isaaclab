"""Sweep stance-leg pitch drives for ground-fixed infant foot advancement."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import torch

from evolve_infant_stand_cop_reflex import aligned_positions
from evolve_infant_stand_step import PARAMETERS, evaluate
from infant_crawl_env import InfantCrawlEnv
from train_infant_stand_ppo import load_bias


def candidate_grid(hip_values=(-1., 0., 1.), ankle_values=(-1., 0., 1.)):
    return [(1., 1., -1., 2., 2., 0.5, hip, ankle)
            for hip, ankle in itertools.product(hip_values, ankle_values)]


def summarize_group(result, start, end):
    summary = {}
    for name, value in result.items():
        if isinstance(value, torch.Tensor):
            part = value[start:end]
            summary[name] = (int(part.sum()) if part.dtype == torch.bool
                             else float(part.mean()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds-per-candidate", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=400)
    parser.add_argument("--stance-preload", action="store_true")
    parser.add_argument("--swing-excitation", type=float, default=0.0)
    parser.add_argument("--hip-values", nargs="+", type=float,
                        default=[-1., 0., 1.])
    parser.add_argument("--ankle-values", nargs="+", type=float,
                        default=[-1., 0., 1.])
    parser.add_argument("--seeds", nargs="+", type=int, default=[8401, 8403])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if min(args.worlds_per_candidate, args.control_steps) < 1 or not args.seeds:
        parser.error("Positive evaluation sizes and at least one seed are required")
    if not 0 <= args.swing_excitation <= 1:
        parser.error("Swing excitation must be in [0, 1]")
    candidates = candidate_grid(args.hip_values, args.ankle_values)
    env = InfantCrawlEnv(args.scene, args.pose,
                         len(candidates) * args.worlds_per_candidate)
    foot_id = mujoco.mj_name2id(env.source, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    bias = load_bias(args.bias, env.device)
    gain_values = json.loads(args.evolution.read_text())["selected"]["gains"]
    gains = torch.tensor(gain_values, device=env.device).expand(env.worlds, -1)
    parameters = torch.tensor(candidates, device=env.device).repeat_interleave(
        args.worlds_per_candidate, dim=0)
    records = []
    for seed in args.seeds:
        positions = aligned_positions(env, seed, args.worlds_per_candidate)
        pose_hash = hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest()
        result = evaluate(env, positions.repeat(len(candidates), 1), bias, gains,
                          parameters, args.control_steps, foot_id,
                          shift_end=0.4, swing_start=0.2, swing_end=0.65,
                          stance_preload=args.stance_preload,
                          swing_excitation=args.swing_excitation)
        for index, candidate in enumerate(candidates):
            start = index * args.worlds_per_candidate
            row = {"seed": seed, "candidate": candidate,
                   "pose_sha256": pose_hash,
                   **summarize_group(result, start, start + args.worlds_per_candidate)}
            records.append(row)
            print(json.dumps({key: row[key] for key in
                              ("seed", "candidate", "world_forward_step",
                               "relative_placement", "combined_step",
                               "max_supported_world_advance_m",
                               "standing_fraction")}), flush=True)
    report = {"scene": str(args.scene.resolve()),
              "pose": str(args.pose.resolve()), "bias": str(args.bias.resolve()),
              "evolution": str(args.evolution.resolve()),
              "parameter_names": PARAMETERS, "candidates": candidates,
              "seeds": args.seeds, "worlds_per_candidate": args.worlds_per_candidate,
              "duration_s": args.control_steps * float(env.source.opt.timestep),
              "stance_preload": args.stance_preload,
              "swing_excitation": args.swing_excitation,
              "records": records,
              "scope": "Stance pitch sweep for world-frame stepping, not walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
