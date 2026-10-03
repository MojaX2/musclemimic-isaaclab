"""Validate a fixed tactile-triggered infant step in direct Warp or Isaac Lab/Newton."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import mujoco
import torch

from evaluate_infant_newton_crawl import newton_body_id, newton_crawl_environment
from evolve_infant_stand_cop_reflex import aligned_positions
from evolve_infant_stand_step import PARAMETERS, evaluate, summarize
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from train_infant_stand_ppo import load_bias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument("--backend", choices=("direct", "newton"), required=True)
    parser.add_argument("--parameters", nargs=len(PARAMETERS), type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=400)
    parser.add_argument("--shift-end", type=float, default=0.4)
    parser.add_argument("--swing-start", type=float, default=0.2)
    parser.add_argument("--swing-end", type=float, default=0.95)
    parser.add_argument("--trace-every", type=int, default=0)
    parser.add_argument("--stance-preload", action="store_true")
    parser.add_argument("--swing-excitation", type=float, default=0.0)
    parser.add_argument("--fitness-mode", choices=("world", "gait"), default="world")
    parser.add_argument("--landing-hip-drive", type=float, default=0.0)
    parser.add_argument("--landing-ankle-drive", type=float, default=0.0)
    parser.add_argument("--landing-duration", type=float, default=0.5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7901, 7903, 7907])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if min(args.worlds, args.control_steps) < 1 or not args.seeds:
        parser.error("Positive world and step counts and at least one seed are required")
    if args.trace_every < 0:
        parser.error("Trace interval must be nonnegative")
    if not (0 < args.shift_end and 0 <= args.swing_start < args.swing_end):
        parser.error("Invalid weight-shift or swing timing")
    if not 0 <= args.swing_excitation <= 1:
        parser.error("Swing excitation must be in [0, 1]")
    if args.landing_duration <= 0:
        parser.error("Landing reflex duration must be positive")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    context = (newton_crawl_environment(source, args.scene, args.pose, args.worlds,
                                        controlled_joint_names=CRAWL_JOINTS)
               if args.backend == "newton" else
               nullcontext(InfantCrawlEnv(args.scene, args.pose, args.worlds)))
    with context as env:
        foot_id = (newton_body_id(env.collision_model, "right_foot")
                   if args.backend == "newton" else
                   mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_BODY, "right_foot"))
        bias = load_bias(args.bias, env.device)
        gain_values = json.loads(args.evolution.read_text())["selected"]["gains"]
        gains = torch.tensor(gain_values, device=env.device).expand(env.worlds, -1)
        candidate = torch.tensor(args.parameters, device=env.device)
        landing_parameters = torch.tensor([args.landing_hip_drive,
                                           args.landing_ankle_drive],
                                          device=env.device).expand(env.worlds, -1)
        records = []
        for seed in args.seeds:
            positions = aligned_positions(env, seed, env.worlds)
            pose_hash = hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest()
            for label, parameters in (("control", torch.zeros_like(candidate)),
                                      ("candidate", candidate)):
                result = evaluate(env, positions, bias, gains,
                                  parameters.expand(env.worlds, -1),
                                  args.control_steps, foot_id,
                                  args.shift_end, args.swing_start, args.swing_end,
                                  args.trace_every if label == "candidate" else 0,
                                  args.stance_preload, args.swing_excitation,
                                  args.fitness_mode,
                                  landing_parameters if label == "candidate" else None,
                                  args.landing_duration)
                row = {"seed": seed, "condition": label, "pose_sha256": pose_hash,
                       **summarize(result)}
                if args.trace_every and label == "candidate":
                    row["completed_worlds"] = torch.nonzero(
                        result["world_forward_step"]).flatten().cpu().tolist()
                    row["relative_placement_worlds"] = torch.nonzero(
                        result["relative_placement"]).flatten().cpu().tolist()
                    row["combined_step_worlds"] = torch.nonzero(
                        result["combined_step"]).flatten().cpu().tolist()
                    row["gait_precursor_worlds"] = torch.nonzero(
                        result["gait_precursor"]).flatten().cpu().tolist()
                    row["sustained_gait_precursor_worlds"] = torch.nonzero(
                        result["sustained_gait_precursor_0p5s"]).flatten().cpu().tolist()
                    row["post_step_support_worlds"] = torch.nonzero(
                        result["post_step_support_0p5s"]).flatten().cpu().tolist()
                    row["trace"] = result["trace"]
                records.append(row)
                print(json.dumps({key: value for key, value in row.items()
                                  if key != "trace"}), flush=True)
        report = {"scene": str(args.scene.resolve()),
                  "pose": str(args.pose.resolve()), "bias": str(args.bias.resolve()),
                  "evolution": str(args.evolution.resolve()),
                  "backend": args.backend, "worlds": args.worlds,
                  "duration_s": args.control_steps * float(source.opt.timestep),
                  "trace_every": args.trace_every,
                  "shift_end_s": args.shift_end,
                  "swing_start_s": args.swing_start,
                  "swing_end_s": args.swing_end,
                  "stance_preload": args.stance_preload,
                  "swing_excitation": args.swing_excitation,
                  "fitness_mode": args.fitness_mode,
                  "landing_hip_drive": args.landing_hip_drive,
                  "landing_ankle_drive": args.landing_ankle_drive,
                  "landing_duration_s": args.landing_duration,
                  "seeds": args.seeds, "parameter_names": PARAMETERS,
                  "candidate_parameters": args.parameters, "records": records,
                  "scope": "Fixed single-step control audit; not sustained walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
