"""Train tactile-feedback residual control around evolved infant standing tone."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from developmental_skin import REGIONS
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from infant_stand_resets import align_stand_feet
from train_infant_crawl_ppo import SupportPolicy, jittered_positions, observe, update
from train_infant_stand_support import stand_score


FOOT_IDS = (REGIONS.index("right_foot"), REGIONS.index("left_foot"))


def load_bias(path, device):
    with np.load(path) as saved:
        bias = saved["bias"]
        joint_names = saved["joint_names"].tolist()
    if joint_names != list(CRAWL_JOINTS) or bias.shape != (len(CRAWL_JOINTS),):
        raise ValueError("Standing bias checkpoint does not match the infant action order")
    return torch.as_tensor(bias, device=device, dtype=torch.float32)


def configure_stand_exploration(policy, log_std, zero_actor_head=False):
    if not -3 <= log_std <= 0:
        raise ValueError("Exploration log standard deviation must be between -3 and zero")
    with torch.no_grad():
        if zero_actor_head:
            policy.actor[-1].weight.zero_()
            policy.actor[-1].bias.zero_()
        policy.log_std.fill_(log_std)


def step_stand_control(env, measure, bias, residual, residual_scale,
                       physics_per_control, cop_gains=None,
                       muscle_baseline=0.2, muscle_amplitude=0.3):
    if cop_gains is not None:
        from sweep_infant_cop_reflex import cop_reflex_drives
    updates = physics_per_control if cop_gains is not None else 1
    for _ in range(updates):
        drives = env.feedback_drives(bias.expand(env.worlds, -1),
                                     1.5 if cop_gains is not None else 3.0, 1.0)
        if cop_gains is not None:
            drives = cop_reflex_drives(env, measure, drives, cop_gains)
        if residual is not None:
            drives = (drives + residual_scale * residual.clamp(-1, 1)).clamp(-1, 1)
        measure = env.step(env.action_from_drives(drives, muscle_baseline,
                                                  muscle_amplitude),
                           1 if cop_gains is not None else physics_per_control)
    return measure


def rollout(env, policy, bias, control_steps, physics_per_control, residual_scale,
            use_touch, joint_jitter, balanced_resets=False, cop_gains=None,
            muscle_baseline=0.2, muscle_amplitude=0.3, late_standing_scale=0.0):
    positions = jittered_positions(env, joint_jitter, 0.01)
    if balanced_resets:
        positions, _ = align_stand_feet(env.source, positions, env.initial[0],
                                        env.root_address)
    env.reset(positions)
    measure = env.measure()
    observations, actions, log_probs, rewards, values, standing = [], [], [], [], [], []
    for _ in range(control_steps):
        observation = observe(env, measure, use_touch)
        with torch.no_grad():
            residual, log_prob, value = policy.sample(observation)
            measure = step_stand_control(env, measure, bias, residual, residual_scale,
                                         physics_per_control, cop_gains,
                                         muscle_baseline, muscle_amplitude)
            reward = stand_score(env, measure) - 0.002 * residual.square().mean(-1)
            upright = ((measure["chest_height"] > 0.45) &
                       (measure["ground_touch"][:, FOOT_IDS] > 2).all(-1)).float()
            if len(rewards) >= control_steps - max(1, control_steps // 4):
                reward += late_standing_scale * upright
        observations.append(observation)
        actions.append(residual)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        standing.append(upright)
    return tuple(torch.stack(items) for items in
                 (observations, actions, log_probs, rewards, values, standing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--control-steps", type=int, default=20)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--joint-jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-touch", action="store_true")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--balanced-resets", action="store_true")
    parser.add_argument("--cop-evolution", type=Path)
    parser.add_argument("--late-standing-scale", type=float, default=0.0)
    parser.add_argument("--zero-actor-head", action="store_true")
    parser.add_argument("--exploration-log-std", type=float, default=-0.7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    checkpoint = args.output.with_suffix(".pt")
    if args.output.exists() or checkpoint.exists():
        parser.error("Output or checkpoint already exists")
    if min(args.updates, args.worlds, args.control_steps, args.physics_per_control) < 1:
        parser.error("Training dimensions must be positive")
    if args.late_standing_scale < 0:
        parser.error("Late standing scale must be nonnegative")
    if not -3 <= args.exploration_log_std <= 0 or not 0 < args.learning_rate <= 1e-2:
        parser.error("Exploration and learning rate must be in valid ranges")
    if args.zero_actor_head and args.initialize_from is not None:
        parser.error("Zero actor head is only valid without an initial policy")
    torch.manual_seed(args.seed)
    env = InfantCrawlEnv(args.scene, args.pose, args.worlds)
    env.reset()
    bias = load_bias(args.bias, env.device)
    cop_values = (json.loads(args.cop_evolution.read_text())["selected"]["gains"]
                  if args.cop_evolution is not None else None)
    if cop_values is not None and len(cop_values) != 6:
        parser.error("Standing COP evolution must contain six gains")
    cop_gains = (torch.tensor(cop_values, device=env.device,
                              dtype=torch.float32).expand(env.worlds, -1)
                 if cop_values is not None else None)
    muscle_baseline = 0.4 if cop_gains is not None else 0.2
    observation_size = observe(env, env.measure(), not args.no_touch).shape[-1]
    policy = SupportPolicy(observation_size, len(CRAWL_JOINTS)).to(env.device)
    if args.initialize_from is not None:
        previous = torch.load(args.initialize_from, map_location=env.device, weights_only=True)
        if (previous["observation_size"] != observation_size or
                list(previous["joint_names"]) != list(CRAWL_JOINTS) or
                previous["use_touch"] != (not args.no_touch) or
                previous["residual_scale"] != args.residual_scale or
                previous.get("cop_gains") != cop_values or
                (cop_values is not None and previous.get("cop_update_physics_steps", 1) != 1) or
                previous.get("muscle_baseline", 0.2) != muscle_baseline or
                not torch.allclose(previous["bias"], bias)):
            raise ValueError("Standing curriculum checkpoint does not match the current task")
        policy.load_state_dict(previous["policy"])
    if args.initialize_from is None:
        configure_stand_exploration(policy, args.exploration_log_std,
                                    args.zero_actor_head)
    else:
        with torch.no_grad():
            policy.log_std.clamp_(max=args.exploration_log_std)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    history = []
    for iteration in range(args.updates):
        batch = rollout(env, policy, bias, args.control_steps, args.physics_per_control,
                        args.residual_scale, not args.no_touch, args.joint_jitter,
                        args.balanced_resets, cop_gains, muscle_baseline, 0.3,
                        args.late_standing_scale)
        update(policy, optimizer, batch)
        with torch.no_grad():
            policy.log_std.clamp_(max=args.exploration_log_std)
        record = {"update": iteration + 1, "mean_reward": float(batch[3].mean()),
                  "standing_fraction": float(batch[5].mean()),
                  "final_standing_fraction": float(batch[5][-1].mean()),
                  "mean_exploration_std": float(policy.log_std.exp().mean())}
        history.append(record)
        print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "observation_size": observation_size,
                "bias": bias.cpu(), "joint_names": CRAWL_JOINTS,
                "residual_scale": args.residual_scale, "use_touch": not args.no_touch,
                "balanced_resets": args.balanced_resets,
                "cop_gains": cop_values, "muscle_baseline": muscle_baseline,
                "muscle_amplitude": 0.3,
                "zero_actor_head": args.zero_actor_head,
                "exploration_log_std": args.exploration_log_std,
                "learning_rate": args.learning_rate,
                "cop_update_physics_steps": 1 if cop_values is not None else None}, checkpoint)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()),
                                       "bias": str(args.bias.resolve()),
                                       "initialize_from": str(args.initialize_from.resolve())
                                       if args.initialize_from else None,
                                       "updates": args.updates, "worlds": args.worlds,
                                       "seed": args.seed, "use_touch": not args.no_touch,
                                       "balanced_resets": args.balanced_resets,
                                       "cop_evolution": str(args.cop_evolution.resolve())
                                       if args.cop_evolution else None,
                                       "late_standing_scale": args.late_standing_scale,
                                       "zero_actor_head": args.zero_actor_head,
                                       "exploration_log_std": args.exploration_log_std,
                                       "learning_rate": args.learning_rate,
                                       "duration_s": (args.control_steps *
                                                      args.physics_per_control *
                                                      float(env.source.opt.timestep)),
                                       "history": history,
                                       "scope": "Standing residual PPO; not stepping or walking"},
                                      indent=2) + "\n")


if __name__ == "__main__":
    main()
