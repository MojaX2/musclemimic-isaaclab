"""Train a tactile-feedback antagonist policy for infant posture support."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from infant_crawl_env import CRAWL_JOINTS, PALM_JOINTS, PALM_SENSOR_VERSION, InfantCrawlEnv
from infant_crawl_oscillator import (OSCILLATOR_IDS, SHOULDER_LIFT_IDS, cpg_drives,
                                    exclusive_contact_state, fixed_limb_drives, oscillator_phase)
from infant_tactile_features import TactileFeatureEncoder
from infant_world_features import WorldFeatureEncoder
from train_infant_crawl_support import support_score


class SupportPolicy(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(observation_size, 256), nn.Tanh(),
                                   nn.Linear(256, 256), nn.Tanh(),
                                   nn.Linear(256, action_size))
        self.critic = nn.Sequential(nn.Linear(observation_size, 256), nn.Tanh(),
                                    nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))
        self.log_std = nn.Parameter(torch.full((action_size,), -0.7))

    def distribution(self, observation):
        mean = self.actor(observation)
        return torch.distributions.Normal(mean, self.log_std.clamp(-3, 0).exp())

    def sample(self, observation):
        distribution = self.distribution(observation)
        raw_action = distribution.sample()
        return raw_action, distribution.log_prob(raw_action).sum(-1), self.critic(observation).squeeze(-1)


def observe(env, measure, use_touch=True, latent_features=None, phase_features=None):
    root = env.root_address
    position_error = env.initial[:, env.crawl_qpos_ids] - env.position[:, env.crawl_qpos_ids]
    joint_velocity = env.velocity[:, env.crawl_qvel_ids]
    root_velocity = env.velocity[:, root:root + 6]
    root_pose = env.position[:, root + 2:root + 7] - env.initial[:, root + 2:root + 7]
    touch = torch.log1p(measure["ground_touch"].clamp(min=0)) / 8
    if not use_touch:
        touch = torch.zeros_like(touch)
    base = torch.cat((position_error, joint_velocity * 0.1, root_velocity * 0.1,
                      root_pose, measure["chest_height"][:, None],
                      measure["head_height"][:, None], touch,
                      env.muscles.activity[:, env.crawl_actuators],
                      env.muscles.activity[:, env.crawl_actuators + env.source.nu]), dim=-1)
    if getattr(env, "controlled_joint_names", CRAWL_JOINTS) == PALM_JOINTS:
        palm_touch = torch.log1p(measure["palm_shin_force"].clamp_min(0)) / 8
        if not use_touch:
            palm_touch = torch.zeros_like(palm_touch)
        base = torch.cat((base, palm_touch), dim=-1)
    if phase_features is not None:
        base = torch.cat((base, phase_features), dim=-1)
    if latent_features is None:
        return base
    return torch.cat((base, latent_features), dim=1)


def jittered_positions(env, joint_jitter, height_jitter):
    positions = env.initial.clone()
    positions[:, env.root_address + 2] += (2 * torch.rand(env.worlds, device=env.device) - 1) * height_jitter
    for index, address in enumerate(env.crawl_qpos_ids.tolist()):
        joint_id = env.source.actuator_trnid[int(env.crawl_actuators[index]), 0]
        lower, upper = env.source.jnt_range[joint_id]
        perturbation = (2 * torch.rand(env.worlds, device=env.device) - 1) * joint_jitter
        positions[:, address] = (positions[:, address] + perturbation).clamp(
            float(lower) + 1e-4, float(upper) - 1e-4)
    return positions


def forward_crawl_reward(previous, current, duration, scale):
    if duration <= 0 or scale < 0:
        raise ValueError("Forward reward duration must be positive and scale nonnegative")
    speed = ((current["forward_displacement"] - previous["forward_displacement"]) /
             duration).clamp(-1, 1)
    supports = current["palm_shin_force"] > 2
    grounded = supports[:, :2].any(-1) & supports[:, 2:].any(-1)
    elevated = ((current["chest_height"] > 0.16) &
                (current["head_height"] > 0.14))
    return scale * torch.where(speed < 0, speed, speed * (grounded & elevated).float())


def hand_switch_reward(previous_side, palm_shin_force, chest_height, head_height, scale):
    current_side = exclusive_contact_state(palm_shin_force[:, :2])
    shin_supported = (palm_shin_force[:, 2:] > 2).any(-1)
    elevated = (chest_height > 0.16) & (head_height > 0.14)
    changed = ((previous_side != 0) & (current_side != 0) &
               (previous_side != current_side) & shin_supported & elevated)
    remembered_side = torch.where(current_side != 0, current_side, previous_side)
    return scale * changed.float(), remembered_side


def rollout(env, policy, control_steps, physics_per_control, use_touch, joint_jitter,
            world_features=None, forward_reward_scale=0.0, oscillator_parameters=None,
            switch_reward_scale=0.0, reference_policy=None, action_mask=None):
    env.reset(jittered_positions(env, joint_jitter, 0.01))
    measure = env.measure()
    latent = (torch.zeros((env.worlds, getattr(world_features, "feature_size", 256)), device=env.device)
              if world_features is not None else None)
    observations, actions, old_log_probs, rewards, values = [], [], [], [], []
    support_counts = []
    last_hand_side = (exclusive_contact_state(measure["palm_shin_force"][:, :2])
                      if switch_reward_scale > 0 else None)
    interval = (physics_per_control * float(env.source.opt.timestep)
                if oscillator_parameters is not None else 0.0)
    for step in range(control_steps):
        phase = (oscillator_phase(oscillator_parameters, step * interval)
                 if oscillator_parameters is not None else None)
        observation = observe(env, measure, use_touch, latent, phase)
        with torch.no_grad():
            action, log_prob, value = policy.sample(observation)
            drives = action.clamp(-1, 1)
            if oscillator_parameters is not None:
                if reference_policy is not None:
                    reference = reference_policy.actor(observation[:, :-2]).clamp(-1, 1)
                    drives = fixed_limb_drives(drives, reference, oscillator_parameters,
                                               step * interval)
                    log_prob = policy.distribution(observation).log_prob(action)[:, action_mask].sum(-1)
                else:
                    drives = cpg_drives(drives, oscillator_parameters, step * interval)
            muscle_action = env.action_from_drives(drives)
            if world_features is not None:
                latent = world_features.encode(env, measure, muscle_action)
            previous_measure = measure
            measure = env.step(muscle_action, physics_per_control)
            require_palms = getattr(env, "controlled_joint_names", CRAWL_JOINTS) == PALM_JOINTS
            reward = support_score(measure, require_palms) - 0.005 * action.square().mean(-1)
            if forward_reward_scale > 0:
                reward += forward_crawl_reward(previous_measure, measure,
                                               physics_per_control * float(env.source.opt.timestep),
                                               forward_reward_scale)
            if switch_reward_scale > 0:
                switch_reward, last_hand_side = hand_switch_reward(
                    last_hand_side, measure["palm_shin_force"], measure["chest_height"],
                    measure["head_height"], switch_reward_scale)
                reward += switch_reward
        observations.append(observation)
        actions.append(action)
        old_log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        support_counts.append(((measure["palm_shin_force" if require_palms else "support_force"] > 2).all(-1) &
                               (measure["chest_height"] > 0.16) &
                               (measure["head_height"] > 0.14)).float())
    return (torch.stack(observations), torch.stack(actions), torch.stack(old_log_probs),
            torch.stack(rewards), torch.stack(values), torch.stack(support_counts))


def initialize_policy_from_checkpoint(policy, checkpoint, extend_phase=False,
                                      extra_observation_size=0):
    state = checkpoint["policy"]
    extension = (2 if extend_phase else 0) + extra_observation_size
    if extension:
        state = {name: value.clone() for name, value in state.items()}
        for name in ("actor.0.weight", "critic.0.weight"):
            width = state[name].shape[0]
            state[name] = torch.cat((state[name], state[name].new_zeros((width, extension))), dim=1)
    policy.load_state_dict(state)


def gae(rewards, values, gamma=0.99, decay=0.95):
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(values[0])
    for step in reversed(range(rewards.shape[0])):
        delta = rewards[step] + gamma * next_value - values[step]
        running = delta + gamma * decay * running
        advantages[step] = running
        next_value = values[step]
    return advantages, advantages + values


def update(policy, optimizer, batch, epochs=4, minibatch_size=256, action_mask=None,
           reference_policy=None, reference_strength=0.0):
    if reference_strength < 0 or (reference_strength > 0 and reference_policy is None):
        raise ValueError("Positive reference strength requires a reference policy")
    if len(batch) not in (6, 7):
        raise ValueError("PPO batch must contain six or seven tensors")
    observations, actions, old_log_probs, rewards, values, _ = batch[:6]
    action_weights = batch[6].flatten(0, 1) if len(batch) == 7 else None
    advantages, targets = gae(rewards, values)
    observations = observations.flatten(0, 1)
    actions = actions.flatten(0, 1)
    old_log_probs = old_log_probs.flatten()
    advantages = advantages.flatten()
    targets = targets.flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    for _ in range(epochs):
        for indices in torch.randperm(len(observations), device=observations.device).split(minibatch_size):
            distribution = policy.distribution(observations[indices])
            action_log_prob = distribution.log_prob(actions[indices])
            entropy = distribution.entropy()
            if action_mask is not None:
                action_log_prob = action_log_prob[:, action_mask]
                entropy = entropy[:, action_mask]
            if action_weights is not None:
                weights = action_weights[indices]
                if action_mask is not None:
                    weights = weights[:, action_mask]
                action_log_prob = action_log_prob * weights
                entropy = entropy * weights
            log_prob = action_log_prob.sum(-1)
            ratio = (log_prob - old_log_probs[indices]).exp()
            surrogate = torch.minimum(ratio * advantages[indices],
                                      ratio.clamp(0.8, 1.2) * advantages[indices])
            value_loss = (policy.critic(observations[indices]).squeeze(-1) - targets[indices]).square().mean()
            loss = -surrogate.mean() + 0.5 * value_loss - 0.001 * entropy.sum(-1).mean()
            if reference_strength > 0:
                with torch.no_grad():
                    reference_action = reference_policy.actor(observations[indices])
                loss += reference_strength * (distribution.mean - reference_action).square().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--control-steps", type=int, default=20)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--joint-jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-touch", action="store_true")
    parser.add_argument("--palm-control", action="store_true")
    parser.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument("--random-world-model-features", action="store_true")
    parser.add_argument("--tactile-model-checkpoint", type=Path)
    parser.add_argument("--random-tactile-features", action="store_true")
    parser.add_argument("--normalize-tactile-features", action="store_true")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--oscillator", type=Path)
    parser.add_argument("--fixed-oscillator-limbs", action="store_true")
    parser.add_argument("--forward-reward-scale", type=float, default=0.0)
    parser.add_argument("--switch-reward-scale", type=float, default=0.0)
    args = parser.parse_args()
    checkpoint = args.output.with_suffix(".pt")
    if args.output.exists() or checkpoint.exists():
        parser.error("Output or checkpoint already exists")
    if min(args.updates, args.worlds, args.control_steps, args.physics_per_control) < 1:
        parser.error("All rollout dimensions must be positive")
    if args.forward_reward_scale < 0 or args.switch_reward_scale < 0:
        parser.error("Reward scales must be nonnegative")
    if args.switch_reward_scale > 0 and not args.palm_control:
        parser.error("Hand switch reward requires palm control")
    if args.random_world_model_features and args.world_model_checkpoint is None:
        parser.error("Random feature control requires a world-model checkpoint for matching statistics")
    if args.random_tactile_features and args.tactile_model_checkpoint is None:
        parser.error("Random tactile features require a tactile model checkpoint")
    if args.normalize_tactile_features and args.tactile_model_checkpoint is None:
        parser.error("Tactile feature normalization requires a tactile model checkpoint")
    if args.world_model_checkpoint is not None and args.tactile_model_checkpoint is not None:
        parser.error("Select one babbling feature model")
    if args.oscillator is not None and (not args.palm_control or args.world_model_checkpoint is not None):
        parser.error("Oscillator PPO requires palm control without world-model features")
    if args.fixed_oscillator_limbs and (args.oscillator is None or args.initialize_from is None):
        parser.error("Fixed oscillator limbs require an oscillator and a pretrained support checkpoint")
    if args.fixed_oscillator_limbs and args.tactile_model_checkpoint is not None:
        parser.error("Fixed oscillator limbs do not support tactile-model features")
    torch.manual_seed(args.seed)
    joint_names = PALM_JOINTS if args.palm_control else CRAWL_JOINTS
    env = InfantCrawlEnv(args.scene, args.pose, args.worlds,
                         controlled_joint_names=joint_names)
    env.reset()
    world_features = None
    if args.world_model_checkpoint is not None:
        world_features = WorldFeatureEncoder.from_path(args.world_model_checkpoint, env.device,
                                                        args.random_world_model_features)
    if args.tactile_model_checkpoint is not None:
        world_features = TactileFeatureEncoder.from_path(args.tactile_model_checkpoint, env.device,
                                                          args.random_tactile_features,
                                                          args.normalize_tactile_features)
    oscillator_values = None
    oscillator_parameters = None
    if args.oscillator is not None:
        with np.load(args.oscillator) as oscillator:
            oscillator_values = oscillator["parameters"].tolist()
        if len(oscillator_values) not in (7, 8, 9):
            raise ValueError("PPO oscillator must have seven to nine parameters")
        oscillator_parameters = torch.tensor(oscillator_values, device=env.device).expand(args.worlds, -1)
    initial_latent = (torch.zeros((env.worlds, getattr(world_features, "feature_size", 256)),
                                  device=env.device)
                      if world_features is not None else None)
    initial_phase = (oscillator_phase(oscillator_parameters, 0.)
                     if oscillator_parameters is not None else None)
    observation_size = observe(env, env.measure(), not args.no_touch, initial_latent,
                               initial_phase).shape[-1]
    policy = SupportPolicy(observation_size, len(joint_names)).to(env.device)
    reference_policy = None
    action_mask = None
    frozen_reference = None
    if args.initialize_from is not None:
        previous = torch.load(args.initialize_from, map_location=env.device, weights_only=True)
        previous_oscillator = previous.get("oscillator_parameters")
        extend_phase = previous_oscillator is None and oscillator_values is not None
        extend_tactile = (previous.get("world_features") is None and
                          args.tactile_model_checkpoint is not None)
        extra_width = (2 if extend_phase else 0) + (128 if extend_tactile else 0)
        if (previous["observation_size"] != observation_size - extra_width or
                tuple(previous["joint_names"]) != joint_names or
                previous["use_touch"] != (not args.no_touch) or
                previous.get("palm_sensor_version") !=
                (PALM_SENSOR_VERSION if args.palm_control else None) or
                previous.get("world_features") is not None or
                (world_features is not None and not extend_tactile) or
                (previous_oscillator is not None and previous_oscillator != oscillator_values)):
            raise ValueError("Crawl curriculum checkpoint does not match the current task")
        initialize_policy_from_checkpoint(policy, previous, extend_phase,
                                          extra_observation_size=128 if extend_tactile else 0)
        if args.fixed_oscillator_limbs:
            if not extend_phase:
                raise ValueError("Fixed oscillator limbs require a non-oscillating support checkpoint")
            reference_policy = SupportPolicy(previous["observation_size"], len(joint_names)).to(env.device)
            initialize_policy_from_checkpoint(reference_policy, previous)
            reference_policy.eval().requires_grad_(False)
            frozen_reference = {"observation_size": previous["observation_size"],
                                "policy": {name: value.detach().cpu()
                                           for name, value in previous["policy"].items()}}
            action_mask = torch.ones(len(joint_names), dtype=torch.bool, device=env.device)
            oscillated_ids = OSCILLATOR_IDS + (SHOULDER_LIFT_IDS if len(oscillator_values) >= 8 else ())
            action_mask[list(oscillated_ids)] = False
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    history = []
    for iteration in range(args.updates):
        batch = rollout(env, policy, args.control_steps, args.physics_per_control,
                        not args.no_touch, args.joint_jitter, world_features,
                        args.forward_reward_scale, oscillator_parameters,
                        args.switch_reward_scale, reference_policy, action_mask)
        update(policy, optimizer, batch, action_mask=action_mask)
        record = {"update": iteration + 1, "mean_reward": float(batch[3].mean()),
                  "elevated_four_support_fraction": float(batch[5].mean()),
                  "mean_forward_displacement_m": float((env.position[:, env.root_address] -
                                                         env.initial[:, env.root_address]).mean())}
        history.append(record)
        print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "observation_size": observation_size,
                "joint_names": joint_names, "use_touch": not args.no_touch,
                "palm_sensor_version": PALM_SENSOR_VERSION if args.palm_control else None,
                "forward_reward_scale": args.forward_reward_scale,
                "switch_reward_scale": args.switch_reward_scale,
                "oscillator_parameters": oscillator_values,
                "phase_observation_version": "sin_cos_v1" if oscillator_values is not None else None,
                "frozen_limb_reference": frozen_reference,
                "feature_timing": "current_state_current_action_to_next_observation",
                "world_features": world_features.snapshot() if world_features else None}, checkpoint)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()), "seed": args.seed,
                                       "worlds": args.worlds, "updates": args.updates,
                                       "use_touch": not args.no_touch,
                                       "palm_control": args.palm_control,
                                       "palm_sensor_version": PALM_SENSOR_VERSION if args.palm_control else None,
                                       "forward_reward_scale": args.forward_reward_scale,
                                       "switch_reward_scale": args.switch_reward_scale,
                                       "oscillator": str(args.oscillator.resolve()) if args.oscillator else None,
                                       "phase_observation_version": "sin_cos_v1" if args.oscillator else None,
                                       "fixed_oscillator_limbs": args.fixed_oscillator_limbs,
                                       "initialize_from": str(args.initialize_from.resolve())
                                       if args.initialize_from else None,
                                       "world_model_checkpoint": str(args.world_model_checkpoint.resolve())
                                       if args.world_model_checkpoint else None,
                                       "random_world_model_features": args.random_world_model_features,
                                       "tactile_model_checkpoint": str(args.tactile_model_checkpoint.resolve())
                                       if args.tactile_model_checkpoint else None,
                                       "random_tactile_features": args.random_tactile_features,
                                       "normalize_tactile_features": args.normalize_tactile_features,
                                       "joint_jitter_rad": args.joint_jitter,
                                       "history": history,
                                       "scope": "PPO with optional CPG; strict crawl gait requires held-out evaluation"},
                                      indent=2) + "\n")


if __name__ == "__main__":
    main()
