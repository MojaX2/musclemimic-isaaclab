"""Fine-tune an existing infant palm-support policy in Isaac Lab/Newton."""

import argparse
import json
from pathlib import Path

import mujoco
import torch

from evaluate_infant_newton_crawl import newton_crawl_environment
from infant_crawl_env import PALM_JOINTS, PALM_SENSOR_VERSION
from train_infant_crawl_ppo import SupportPolicy, observe, rollout, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=40)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--joint-jitter", type=float, default=0.05)
    parser.add_argument("--forward-reward-scale", type=float, default=0.0)
    parser.add_argument("--switch-reward-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1171)
    args = parser.parse_args()
    checkpoint_path = args.output.with_suffix(".pt")
    if args.output.exists() or checkpoint_path.exists():
        parser.error("Output or checkpoint already exists")
    if min(args.updates, args.worlds, args.control_steps, args.physics_per_control) < 1:
        parser.error("Rollout dimensions must be positive")
    if args.joint_jitter < 0:
        parser.error("Joint jitter must be nonnegative")
    if args.forward_reward_scale < 0 or args.switch_reward_scale < 0:
        parser.error("Reward scales must be nonnegative")
    starting = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
    if (tuple(starting["joint_names"]) != PALM_JOINTS or
            starting.get("palm_sensor_version") != PALM_SENSOR_VERSION or
            not starting["use_touch"] or starting.get("world_features") is not None or
            starting.get("oscillator_parameters") is not None):
        raise ValueError("Expected a compatible tactile palm-support checkpoint")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    if source.eq_active0.any():
        raise ValueError("Newton bridge requires inactive equalities")
    torch.manual_seed(args.seed)
    with newton_crawl_environment(source, args.scene, args.pose, args.worlds) as environment:
        observation_size = observe(environment, environment.measure(), True).shape[-1]
        if observation_size != starting["observation_size"]:
            raise ValueError("Checkpoint observation size does not match Newton environment")
        policy = SupportPolicy(observation_size, len(PALM_JOINTS)).to(environment.device)
        policy.load_state_dict(starting["policy"])
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        history = []
        for iteration in range(args.updates):
            batch = rollout(environment, policy, args.control_steps,
                            args.physics_per_control, True, args.joint_jitter,
                            forward_reward_scale=args.forward_reward_scale,
                            switch_reward_scale=args.switch_reward_scale)
            update(policy, optimizer, batch)
            record = {"update": iteration + 1,
                      "mean_reward": float(batch[3].mean()),
                      "four_support_fraction": float(batch[5].mean()),
                      "mean_forward_displacement_m": float((
                          environment.position[:, environment.root_address] -
                          environment.initial[:, environment.root_address]).mean()),
                      "solver_limit_steps": environment.solver_limit_steps}
            history.append(record)
            print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "observation_size": observation_size,
                "joint_names": PALM_JOINTS, "use_touch": True,
                "palm_sensor_version": PALM_SENSOR_VERSION,
                "world_features": None, "oscillator_parameters": None,
                "source_checkpoint": str(args.initialize_from.resolve()),
                "training_backend": "isaaclab_newton"}, checkpoint_path)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()),
                                       "initialize_from": str(args.initialize_from.resolve()),
                                       "checkpoint": str(checkpoint_path.resolve()),
                                       "updates": args.updates, "worlds": args.worlds,
                                       "control_steps": args.control_steps,
                                       "physics_per_control": args.physics_per_control,
                                       "joint_jitter": args.joint_jitter,
                                       "forward_reward_scale": args.forward_reward_scale,
                                       "switch_reward_scale": args.switch_reward_scale,
                                       "seed": args.seed, "history": history,
                                       "scope": "Newton support fine-tuning; held-out crawl evaluation required"},
                                      indent=2) + "\n")


if __name__ == "__main__":
    main()
