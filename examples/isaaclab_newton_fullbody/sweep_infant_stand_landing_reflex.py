"""Sweep tactile-triggered bilateral pitch recovery after infant foot landing."""

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
from sweep_infant_stand_stance_pitch import summarize_group
from train_infant_stand_ppo import load_bias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameters", nargs=len(PARAMETERS), type=float,
                        default=[1., 1., -1., 2., 2., 0.5, 0.75, 0.5])
    parser.add_argument("--hip-values", nargs="+", type=float,
                        default=[-0.5, 0., 0.5])
    parser.add_argument("--ankle-values", nargs="+", type=float,
                        default=[-0.5, 0., 0.5])
    parser.add_argument("--landing-duration", type=float, default=0.5)
    parser.add_argument("--swing-end", type=float, default=0.85)
    parser.add_argument("--swing-excitation", type=float, default=0.5)
    parser.add_argument("--worlds-per-candidate", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=400)
    parser.add_argument("--seeds", nargs="+", type=int, default=[9301, 9303])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if (min(args.worlds_per_candidate, args.control_steps) < 1 or
            not args.seeds or not args.hip_values or not args.ankle_values or
            args.landing_duration <= 0 or args.swing_end <= 0.2 or
            not 0 <= args.swing_excitation <= 1):
        parser.error("Invalid landing-reflex sweep settings")
    candidates = list(itertools.product(args.hip_values, args.ankle_values))
    env = InfantCrawlEnv(args.scene, args.pose,
                         len(candidates) * args.worlds_per_candidate)
    foot_id = mujoco.mj_name2id(env.source, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    bias = load_bias(args.bias, env.device)
    gain_values = json.loads(args.evolution.read_text())["selected"]["gains"]
    gains = torch.tensor(gain_values, device=env.device).expand(env.worlds, -1)
    parameters = torch.tensor(args.parameters, device=env.device).expand(env.worlds, -1)
    landing_parameters = torch.tensor(candidates, device=env.device).repeat_interleave(
        args.worlds_per_candidate, dim=0)
    records = []
    for seed in args.seeds:
        positions = aligned_positions(env, seed, args.worlds_per_candidate)
        pose_hash = hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest()
        result = evaluate(env, positions.repeat(len(candidates), 1), bias, gains,
                          parameters, args.control_steps, foot_id,
                          shift_end=0.4, swing_start=0.2, swing_end=args.swing_end,
                          stance_preload=True, swing_excitation=args.swing_excitation,
                          fitness_mode="gait", landing_parameters=landing_parameters,
                          landing_duration=args.landing_duration)
        for index, candidate in enumerate(candidates):
            start = index * args.worlds_per_candidate
            row = {"seed": seed, "candidate": candidate,
                   "pose_sha256": pose_hash,
                   **summarize_group(result, start, start + args.worlds_per_candidate)}
            records.append(row)
            print(json.dumps({key: row[key] for key in
                              ("seed", "candidate", "gait_precursor",
                               "post_step_support_0p5s", "standing_fraction",
                               "longest_post_step_support_s")}), flush=True)
    report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "bias": str(args.bias.resolve()),
              "evolution": str(args.evolution.resolve()),
              "step_parameters": args.parameters,
              "landing_parameters": candidates,
              "landing_duration_s": args.landing_duration,
              "swing_end_s": args.swing_end,
              "swing_excitation": args.swing_excitation,
              "seeds": args.seeds, "worlds_per_candidate": args.worlds_per_candidate,
              "duration_s": args.control_steps * float(env.source.opt.timestep),
              "records": records,
              "scope": "Tactile landing recovery sweep; not sustained walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
