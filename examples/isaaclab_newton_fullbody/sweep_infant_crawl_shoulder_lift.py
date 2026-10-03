"""Measure the isolated effect of alternating shoulder lift on a fixed crawl gait."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from evolve_infant_crawl_cpg import evaluate
from infant_crawl_env import PALM_JOINTS, PALM_SENSOR_VERSION, InfantCrawlEnv
from train_infant_crawl_ppo import SupportPolicy, jittered_positions


def lift_parameter_grid(base_parameters, amplitudes, worlds, phase_offset=None):
    if base_parameters.shape != (7,) or worlds < 1:
        raise ValueError("Shoulder-lift sweep requires seven base parameters and positive worlds")
    if any(amplitude < 0 or amplitude > 0.8 for amplitude in amplitudes):
        raise ValueError("Shoulder-lift amplitudes must be in [0, 0.8]")
    if phase_offset is not None and not -math.pi <= phase_offset <= math.pi:
        raise ValueError("Shoulder-lift phase offset must be in [-pi, pi]")
    variants = [torch.cat((base_parameters, base_parameters.new_tensor(
                    [amplitude] if phase_offset is None else [amplitude, phase_offset])))
                for amplitude in amplitudes]
    return torch.repeat_interleave(torch.stack(variants), worlds, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oscillator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0, 0.15, 0.3, 0.45, 0.6])
    parser.add_argument("--phase-offset-rad", type=float)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=80)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--seed", type=int, default=571)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if min(args.worlds, args.control_steps, args.physics_per_control) < 1:
        parser.error("Evaluation dimensions must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (tuple(checkpoint["joint_names"]) != PALM_JOINTS or
            checkpoint.get("palm_sensor_version") != PALM_SENSOR_VERSION or
            checkpoint.get("world_features") is not None):
        raise ValueError("Shoulder-lift sweep requires a compatible palm-control policy")
    with np.load(args.oscillator) as state:
        base_parameters = torch.as_tensor(state["parameters"].copy(), dtype=torch.float32)
    parameters = lift_parameter_grid(base_parameters, args.amplitudes, args.worlds,
                                     args.phase_offset_rad)
    env = InfantCrawlEnv(args.scene, args.pose, len(args.amplitudes) * args.worlds,
                         controlled_joint_names=PALM_JOINTS)
    policy = SupportPolicy(checkpoint["observation_size"], len(PALM_JOINTS)).to(env.device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    torch.manual_seed(args.seed)
    positions = jittered_positions(env, 0.05, 0.01)
    positions = positions[:args.worlds].repeat(len(args.amplitudes), 1)
    result = evaluate(env, policy, parameters.to(env.device), positions, args.control_steps,
                      args.physics_per_control, checkpoint["use_touch"], gait_objective=True)
    records = []
    for index, amplitude in enumerate(args.amplitudes):
        segment = slice(index * args.worlds, (index + 1) * args.worlds)
        records.append({"amplitude": amplitude,
                        "mean": {name: float(values[segment].float().mean())
                                 for name, values in result.items()},
                        "crawl_success_count": int(result["crawl_success"][segment].sum())})
    report = {"seed": args.seed, "worlds_per_amplitude": args.worlds,
              "duration_s": args.control_steps * args.physics_per_control *
                            float(env.source.opt.timestep),
              "initial_pose_sha256": hashlib.sha256(
                  positions[:args.worlds].cpu().numpy().tobytes()).hexdigest(),
              "oscillator": str(args.oscillator.resolve()),
              "shoulder_lift_phase_offset_rad": args.phase_offset_rad,
              "checkpoint": str(args.checkpoint.resolve()),
              "records": records,
              "scope": "Isolated shoulder-lift amplitude sweep on a fixed gait; not gait learning"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
