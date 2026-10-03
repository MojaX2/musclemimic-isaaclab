"""Adapt the infant support policy around a fixed tactile stance/swing controller."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from infant_crawl_env import PALM_JOINTS, PALM_SENSOR_VERSION, InfantCrawlEnv
from infant_crawl_oscillator import exclusive_contact_state
from infant_crawl_stepper import TactileStepper
from train_infant_crawl_ppo import (SupportPolicy, forward_crawl_reward,
                                    hand_switch_reward, jittered_positions, observe, update)
from train_infant_crawl_support import support_score


def sustained_support_reward(measure, step, control_steps, support_scale, late_scale):
    contact = measure["palm_shin_force"] > 2
    elevated = (measure["chest_height"] > 0.16) & (measure["head_height"] > 0.14)
    supported = elevated & contact[:, :2].any(-1) & contact[:, 2:].any(-1)
    late = step >= control_steps - max(1, control_steps // 4)
    return supported.float() * (support_scale + late_scale * float(late)), supported


def future_supported_forward_reward(forward_speed, supported, window_steps, scale):
    if forward_speed.shape != supported.shape or forward_speed.ndim != 2:
        raise ValueError("Forward speed and support must share time and world axes")
    if window_steps < 1 or scale < 0:
        raise ValueError("Future support window and scale must be valid")
    reward = torch.zeros_like(forward_speed)
    if window_steps >= len(supported) or scale == 0:
        return reward
    cumulative = torch.cat((torch.zeros_like(supported[:1]),
                            supported.float().cumsum(0)), dim=0)
    future_fraction = (cumulative[window_steps + 1:] -
                       cumulative[1:-window_steps]) / window_steps
    reward[:-window_steps] = (scale * forward_speed[:-window_steps].clamp_min(0) *
                             supported[:-window_steps].float() * future_fraction)
    return reward


def supported_hand_reach_reward(previous, current, swing_side, active, scale):
    if scale < 0:
        raise ValueError("Hand reach reward scale must be nonnegative")
    if scale == 0:
        return torch.zeros_like(active, dtype=torch.float32)
    before = previous["palm_relative_x"]
    after = current["palm_relative_x"]
    contact = current["palm_shin_force"] > 2
    if before.shape != after.shape or before.shape != contact[:, :2].shape:
        raise ValueError("Hand reach needs matching two-palm measurements")
    moving = (after - before).gather(1, swing_side[:, None]).squeeze(1)
    stance_contact = contact[:, :2].gather(1, (1 - swing_side)[:, None]).squeeze(1)
    supported = (active & stance_contact & contact[:, 2:].any(-1) &
                 (current["chest_height"] > 0.16) & (current["head_height"] > 0.14))
    return scale * moving.clamp(-0.05, 0.05) * supported.float()


def rollout(env, policy, controller, control_steps, physics_per_control,
            joint_jitter, use_touch, forward_scale, switch_scale,
            completion_scale=0.0, abort_penalty=0.0,
            mask_overridden_actions=False, stepper_observation=False,
            world_features=None, minimum_support_scale=0.0,
            late_support_scale=0.0, future_forward_scale=0.0,
            future_support_window_steps=20, stepper_reference_policy=None,
            hand_reach_reward_scale=0.0):
    env.reset(jittered_positions(env, joint_jitter, 0.01))
    controller.reset()
    measure = env.measure()
    latent = (torch.zeros((env.worlds, world_features.feature_size), device=env.device)
              if world_features is not None else None)
    last_hand_side = exclusive_contact_state(measure["palm_shin_force"][:, :2])
    interval = physics_per_control * float(env.source.opt.timestep)
    observations, actions, log_probs, rewards, values = [], [], [], [], []
    action_weights = []
    supported_steps = torch.zeros(env.worlds, device=env.device)
    support_history = []
    forward_speed_history = []
    for step in range(control_steps):
        observation = observe(env, measure, use_touch, latent_features=latent)
        if stepper_observation:
            observation = torch.cat((observation, controller.observation()), dim=-1)
        with torch.no_grad():
            action, log_prob, value = policy.sample(observation)
            previous_completed = controller.completed.clone()
            previous_aborted = controller.aborted.clone()
            reference_drives = (stepper_reference_policy.actor(observation).clamp(-1, 1)
                                if stepper_reference_policy is not None else None)
            drive = controller(action.clamp(-1, 1), measure, reference_drives)
            if mask_overridden_actions:
                weights = (~controller.override_mask).float().clone()
                log_prob = (policy.distribution(observation).log_prob(action) *
                            weights).sum(-1)
                action_weights.append(weights)
            previous = measure
            muscle_action = env.action_from_drives(drive)
            if world_features is not None:
                latent = world_features.encode(env, measure, muscle_action)
            measure = env.step(muscle_action, physics_per_control)
            reward = support_score(measure, require_palms=True)
            reward += forward_crawl_reward(previous, measure, interval, forward_scale)
            switch_reward, last_hand_side = hand_switch_reward(
                last_hand_side, measure["palm_shin_force"], measure["chest_height"],
                measure["head_height"], switch_scale)
            reward += switch_reward - 0.005 * action.square().mean(-1)
            elevated = (measure["chest_height"] > 0.16) & (measure["head_height"] > 0.14)
            support_reward, minimum_supported = sustained_support_reward(
                measure, step, control_steps, minimum_support_scale, late_support_scale)
            reward += support_reward
            reward += supported_hand_reach_reward(
                previous, measure, controller.swing_side, controller.stage != 0,
                hand_reach_reward_scale)
            if future_forward_scale > 0:
                forward_speed_history.append(((measure["forward_displacement"] -
                                               previous["forward_displacement"]) /
                                              interval).clamp(-1, 1))
                support_history.append(minimum_supported)
            reward += completion_scale * (controller.completed - previous_completed) * elevated.float()
            reward -= abort_penalty * (controller.aborted - previous_aborted)
            supported_steps += minimum_supported.float()
        observations.append(observation)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
    reward_tensor = torch.stack(rewards)
    if future_forward_scale > 0:
        reward_tensor += future_supported_forward_reward(
            torch.stack(forward_speed_history), torch.stack(support_history),
            future_support_window_steps, future_forward_scale)
    batch = (torch.stack(observations), torch.stack(actions), torch.stack(log_probs),
             reward_tensor, torch.stack(values), supported_steps / control_steps)
    if mask_overridden_actions:
        batch += (torch.stack(action_weights),)
    summary = {"mean_reward": float(batch[3].mean()),
               "mean_forward_m": float(measure["forward_displacement"].mean()),
               "mean_minimum_support_fraction": float(batch[5].mean()),
               "mean_step_completions": float(controller.completed.mean()),
               "mean_forward_qualified_completions": float(
                   controller.forward_qualified_completions.mean()),
               "mean_completed_reach_m": float(
                   (controller.completed_reach_m /
                    controller.completed.clamp_min(1)).mean()),
               "mean_swing_active_fraction": float(controller.active_steps.mean() /
                                                   control_steps),
               "mean_trainable_action_fraction": (float(batch[6].mean())
                                                  if mask_overridden_actions else 1.0)}
    return batch, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--stepper-parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--control-steps", type=int, default=160)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--joint-jitter", type=float, default=0.02)
    parser.add_argument("--forward-reward-scale", type=float, default=2.0)
    parser.add_argument("--switch-reward-scale", type=float, default=0.1)
    parser.add_argument("--stance-loss-grace-steps", type=int, default=2)
    parser.add_argument("--policy-anchor", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1031)
    args = parser.parse_args()
    weights_path = args.output.with_suffix(".pt")
    if args.output.exists() or weights_path.exists():
        parser.error("Output or checkpoint already exists")
    if min(args.updates, args.worlds, args.control_steps, args.physics_per_control) < 1:
        parser.error("Training dimensions must be positive")
    if min(args.joint_jitter, args.forward_reward_scale, args.switch_reward_scale,
           args.stance_loss_grace_steps, args.policy_anchor) < 0:
        parser.error("Jitter, reward scales, grace steps, and anchor must be nonnegative")
    torch.manual_seed(args.seed)
    env = InfantCrawlEnv(args.scene, args.pose, args.worlds,
                         controlled_joint_names=PALM_JOINTS)
    checkpoint = torch.load(args.support_checkpoint, map_location=env.device,
                            weights_only=True)
    observation_size = observe(env, env.measure(), checkpoint["use_touch"]).shape[-1]
    if (tuple(checkpoint["joint_names"]) != PALM_JOINTS or
            checkpoint["observation_size"] != observation_size or
            checkpoint.get("palm_sensor_version") != PALM_SENSOR_VERSION or
            checkpoint.get("world_features") is not None or
            checkpoint.get("oscillator_parameters") is not None):
        raise ValueError("Support checkpoint does not match the infant stepper task")
    with np.load(args.stepper_parameters) as state:
        parameters = np.asarray(state["parameters"], dtype=np.float32)
    if parameters.shape != (6,) or not np.isfinite(parameters).all():
        raise ValueError("Stepper parameters must be six finite values")
    controller = TactileStepper(torch.as_tensor(parameters, device=env.device).expand(args.worlds, -1),
                                absolute_targets=True, forward_reach=True,
                                strong_plant=True, leg_push=True,
                                stance_loss_grace_steps=args.stance_loss_grace_steps)
    policy = SupportPolicy(observation_size, len(PALM_JOINTS)).to(env.device)
    policy.load_state_dict(checkpoint["policy"])
    reference_policy = None
    if args.policy_anchor > 0:
        reference_policy = copy.deepcopy(policy)
        reference_policy.eval().requires_grad_(False)
    with torch.no_grad():
        policy.log_std.clamp_(max=-1.5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    history = []
    for iteration in range(args.updates):
        batch, summary = rollout(env, policy, controller, args.control_steps,
                                 args.physics_per_control, args.joint_jitter,
                                 checkpoint["use_touch"], args.forward_reward_scale,
                                 args.switch_reward_scale)
        update(policy, optimizer, batch, reference_policy=reference_policy,
               reference_strength=args.policy_anchor)
        summary["update"] = iteration + 1
        history.append(summary)
        print(json.dumps(summary), flush=True)
    result = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "support_checkpoint": str(args.support_checkpoint.resolve()),
              "stepper_parameters": str(args.stepper_parameters.resolve()),
              "seed": args.seed, "worlds": args.worlds, "updates": args.updates,
              "duration_s": args.control_steps * args.physics_per_control *
                            float(env.source.opt.timestep),
              "forward_reward_scale": args.forward_reward_scale,
              "switch_reward_scale": args.switch_reward_scale,
              "stance_loss_grace_steps": args.stance_loss_grace_steps,
              "policy_anchor": args.policy_anchor,
              "history": history,
              "scope": "PPO adaptation around a fixed tactile controller; held-out crawl gait unverified"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "observation_size": observation_size,
                "joint_names": PALM_JOINTS, "use_touch": checkpoint["use_touch"],
                "palm_sensor_version": PALM_SENSOR_VERSION, "world_features": None,
                "oscillator_parameters": None,
                "stepper_parameters": parameters.tolist(),
                "stance_loss_grace_steps": args.stance_loss_grace_steps,
                "policy_anchor": args.policy_anchor}, weights_path)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
