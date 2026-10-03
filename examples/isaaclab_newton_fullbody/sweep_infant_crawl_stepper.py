"""Compare contact-triggered stance/swing stepping commands on matched infant poses."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evolve_infant_crawl_cpg import evaluate
from infant_crawl_env import PALM_JOINTS, PALM_SENSOR_VERSION, InfantCrawlEnv
from infant_crawl_stepper import TactileStepper
from train_infant_crawl_ppo import SupportPolicy, jittered_positions


DEFAULT_PARAMETERS = ((0., 0., 0., 0., 0., 0.),
                      (0.5, 0.3, 0.4, 0.2, 0., 0.),
                      (0.8, 0.4, 0.6, 0.3, 0., 0.),
                      (0.5, 0.3, 0.4, 0.2, 0.3, -0.3),
                      (0.5, 0.3, 0.4, 0.2, -0.3, 0.3))


class BlendedSupportPolicy:
    def __init__(self, primary, secondary, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError("Policy blend fraction must be between zero and one")
        self.primary = primary
        self.secondary = secondary
        self.fraction = fraction

    def actor(self, observation):
        return ((1 - self.fraction) * self.primary.actor(observation) +
                self.fraction * self.secondary.actor(observation))


def parameter_grid(variants, worlds, device):
    parameters = torch.as_tensor(variants, device=device, dtype=torch.float32)
    if parameters.ndim != 2 or parameters.shape[1] != 6 or worlds < 1:
        raise ValueError("Stepper sweep requires six parameters and positive worlds")
    if not ((parameters[:, :4] >= 0).all() and (parameters[:, :4] <= 1).all() and
            (parameters[:, 4:].abs() <= 1).all()):
        raise ValueError("Stepper parameters exceed the supported drive range")
    return torch.repeat_interleave(parameters, worlds, dim=0)


def checkpoint_variants(path):
    with np.load(path) as state:
        selected = state["parameters"].copy()
    if selected.shape != (6,):
        raise ValueError("Stepper checkpoint must contain six parameters")
    return ((0., 0., 0., 0., 0., 0.), tuple(selected.tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter-set", nargs=6, type=float, action="append")
    parser.add_argument("--parameters-file", type=Path)
    parser.add_argument("--absolute-targets", action="store_true")
    parser.add_argument("--forward-reach", action="store_true")
    parser.add_argument("--strong-plant", action="store_true")
    parser.add_argument("--leg-push", action="store_true")
    parser.add_argument("--stance-loss-grace-steps", type=int, default=0)
    parser.add_argument("--max-plant-steps", type=int, default=8)
    parser.add_argument("--stance-brace-strength", type=float, default=0.0)
    parser.add_argument("--alternate-on-abort", action="store_true")
    parser.add_argument("--mask-policy-touch", action="store_true")
    parser.add_argument("--blend-checkpoint", type=Path)
    parser.add_argument("--blend-fraction", type=float, default=0.0)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=80)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--seed", type=int, default=787)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if min(args.worlds, args.control_steps, args.physics_per_control) < 1:
        parser.error("Evaluation dimensions must be positive")
    if args.stance_loss_grace_steps < 0:
        parser.error("Stance loss grace steps must be nonnegative")
    if args.max_plant_steps < 2:
        parser.error("Maximum plant steps must be at least two")
    if not 0 <= args.stance_brace_strength <= 1:
        parser.error("Stance brace strength must be between zero and one")
    if not 0 <= args.blend_fraction <= 1 or (args.blend_fraction > 0 and
                                            args.blend_checkpoint is None):
        parser.error("A positive policy blend requires a checkpoint and fraction in [0, 1]")
    if args.parameters_file is not None and args.parameter_set is not None:
        parser.error("Select a parameter file or explicit parameter sets")
    variants = (checkpoint_variants(args.parameters_file) if args.parameters_file is not None
                else args.parameter_set or DEFAULT_PARAMETERS)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (tuple(checkpoint["joint_names"]) != PALM_JOINTS or
            checkpoint.get("palm_sensor_version") != PALM_SENSOR_VERSION or
            checkpoint.get("world_features") is not None):
        raise ValueError("Stepper sweep requires a compatible palm-control policy")
    env = InfantCrawlEnv(args.scene, args.pose, len(variants) * args.worlds,
                         controlled_joint_names=PALM_JOINTS)
    policy = SupportPolicy(checkpoint["observation_size"], len(PALM_JOINTS)).to(env.device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    if args.blend_checkpoint is not None:
        secondary_checkpoint = torch.load(args.blend_checkpoint, map_location="cpu",
                                          weights_only=True)
        if any(secondary_checkpoint.get(key) != checkpoint.get(key) for key in
               ("observation_size", "joint_names", "use_touch", "palm_sensor_version")):
            raise ValueError("Blended policy checkpoint does not match the primary policy")
        secondary = SupportPolicy(checkpoint["observation_size"], len(PALM_JOINTS)).to(env.device)
        secondary.load_state_dict(secondary_checkpoint["policy"])
        secondary.eval()
        policy = BlendedSupportPolicy(policy, secondary, args.blend_fraction)
    controller = TactileStepper(parameter_grid(variants, args.worlds, env.device),
                                max_plant_steps=args.max_plant_steps,
                                absolute_targets=args.absolute_targets,
                                forward_reach=args.forward_reach,
                                strong_plant=args.strong_plant,
                                leg_push=args.leg_push,
                                stance_loss_grace_steps=args.stance_loss_grace_steps,
                                stance_brace_strength=args.stance_brace_strength,
                                alternate_on_abort=args.alternate_on_abort)
    torch.manual_seed(args.seed)
    positions = jittered_positions(env, 0.05, 0.01)
    positions = positions[:args.worlds].repeat(len(variants), 1)
    result = evaluate(env, policy, None, positions, args.control_steps,
                      args.physics_per_control, checkpoint["use_touch"] and
                      not args.mask_policy_touch,
                      gait_objective=True, drive_controller=controller)
    records = []
    for index, values in enumerate(variants):
        segment = slice(index * args.worlds, (index + 1) * args.worlds)
        records.append({"parameters": list(values),
                        "mean": {name: float(tensor[segment].float().mean())
                                 for name, tensor in result.items()},
                        "crawl_success_count": int(result["crawl_success"][segment].sum())})
    report = {"seed": args.seed, "worlds_per_variant": args.worlds,
              "duration_s": args.control_steps * args.physics_per_control *
                            float(env.source.opt.timestep),
              "initial_pose_sha256": hashlib.sha256(
                  positions[:args.worlds].cpu().numpy().tobytes()).hexdigest(),
              "checkpoint": str(args.checkpoint.resolve()),
              "parameters_file": str(args.parameters_file.resolve()) if args.parameters_file else None,
              "absolute_targets": args.absolute_targets,
              "forward_reach": args.forward_reach,
              "strong_plant": args.strong_plant,
              "leg_push": args.leg_push,
              "stance_loss_grace_steps": args.stance_loss_grace_steps,
              "max_plant_steps": args.max_plant_steps,
              "stance_brace_strength": args.stance_brace_strength,
              "alternate_on_abort": args.alternate_on_abort,
              "mask_policy_touch": args.mask_policy_touch,
              "blend_checkpoint": str(args.blend_checkpoint.resolve()) if args.blend_checkpoint else None,
              "blend_fraction": args.blend_fraction,
              "records": records,
              "scope": "Hand-designed contact-event stance/swing exploration, not learned gait"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
