"""Audit an evolved infant crawl oscillator on unseen initial poses."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evolve_infant_crawl_cpg import evaluate, summarize
from infant_crawl_env import PALM_JOINTS, PALM_SENSOR_VERSION, InfantCrawlEnv
from train_infant_crawl_ppo import SupportPolicy, jittered_positions


def load_parameters(path, width, device):
    with np.load(path) as state:
        parameters = state["parameters"].copy()
    if width in (7, 8, 9) and parameters.ndim == 1 and 6 <= parameters.size < width:
        parameters = np.pad(parameters, (0, width - parameters.size))
    if parameters.shape != (width,):
        raise ValueError("Oscillator parameter count does not match the selected model")
    return torch.as_tensor(parameters, device=device, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oscillator", type=Path, required=True)
    parser.add_argument("--baseline-oscillator", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=40)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--height-gate", action="store_true")
    parser.add_argument("--shin-force-gate", action="store_true")
    parser.add_argument("--shin-leg-gate", action="store_true")
    parser.add_argument("--contact-guard", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if min(args.worlds, args.control_steps, args.physics_per_control) < 1:
        parser.error("Evaluation dimensions must be positive")
    if sum((args.height_gate, args.shin_force_gate, args.shin_leg_gate)) > 1:
        parser.error("Select only one oscillator amplitude gate")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (tuple(checkpoint["joint_names"]) != PALM_JOINTS or
            checkpoint.get("palm_sensor_version") != PALM_SENSOR_VERSION or
            checkpoint.get("world_features") is not None):
        raise ValueError("CPG evaluation requires a compatible palm-control policy")
    env = InfantCrawlEnv(args.scene, args.pose, args.worlds,
                         controlled_joint_names=PALM_JOINTS)
    policy = SupportPolicy(checkpoint["observation_size"], len(PALM_JOINTS)).to(env.device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    with np.load(args.oscillator) as state:
        width = int(state["parameters"].size)
    selected = load_parameters(args.oscillator, width, env.device)
    baseline = (load_parameters(args.baseline_oscillator, width, env.device)
                if args.baseline_oscillator else
                torch.tensor([1.5] + [0.] * (width - 1), device=env.device))
    torch.manual_seed(args.seed)
    positions = jittered_positions(env, 0.05, 0.01)
    baseline_result = evaluate(env, policy, baseline.expand(args.worlds, -1), positions,
                               args.control_steps, args.physics_per_control,
                               checkpoint["use_touch"], record_trace=args.trace)
    learned_result = evaluate(env, policy, selected.expand(args.worlds, -1), positions,
                              args.control_steps, args.physics_per_control,
                              checkpoint["use_touch"], record_trace=args.trace,
                              height_gate=args.height_gate,
                              shin_force_gate=args.shin_force_gate,
                              shin_leg_gate=args.shin_leg_gate,
                              contact_guard=args.contact_guard)
    report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "checkpoint": str(args.checkpoint.resolve()),
              "oscillator": str(args.oscillator.resolve()),
              "baseline_oscillator": str(args.baseline_oscillator.resolve())
              if args.baseline_oscillator else None,
              "seed": args.seed, "worlds": args.worlds,
              "initial_pose_sha256": hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest(),
              "duration_s": args.control_steps * args.physics_per_control *
                            float(env.source.opt.timestep),
              "trace": args.trace,
              "height_gate": args.height_gate,
              "shin_force_gate": args.shin_force_gate,
              "shin_leg_gate": args.shin_leg_gate,
              "contact_guard": args.contact_guard,
              "baseline_parameters": baseline.cpu().tolist(),
              "selected_parameters": selected.cpu().tolist(),
              "baseline": summarize(baseline_result), "learned": summarize(learned_result),
              "scope": "Held-out CPG locomotion criterion; not independent training-seed replication"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"seed": args.seed,
                      "baseline_success": sum(report["baseline"]["crawl_success"]),
                      "learned_success": sum(report["learned"]["crawl_success"])}), flush=True)


if __name__ == "__main__":
    main()
