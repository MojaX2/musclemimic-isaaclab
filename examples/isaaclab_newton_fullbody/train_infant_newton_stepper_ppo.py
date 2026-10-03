"""Adapt infant palm PPO around a tactile stepper in Isaac Lab/Newton."""

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from evaluate_infant_newton_crawl import newton_crawl_environment
from infant_crawl_env import PALM_JOINTS, PALM_SENSOR_VERSION
from infant_crawl_stepper import TactileStepper
from infant_tactile_features import TactileFeatureEncoder
from train_infant_crawl_ppo import (SupportPolicy, initialize_policy_from_checkpoint,
                                    observe, update)
from train_infant_crawl_stepper_ppo import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--continue-from-policy", type=Path)
    parser.add_argument("--stepper-parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=160)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--joint-jitter", type=float, default=0.02)
    parser.add_argument("--forward-reward-scale", type=float, default=2.0)
    parser.add_argument("--switch-reward-scale", type=float, default=0.1)
    parser.add_argument("--completion-scale", type=float, default=0.5)
    parser.add_argument("--abort-penalty", type=float, default=0.1)
    parser.add_argument("--policy-anchor", type=float, default=0.25)
    parser.add_argument("--exploration-log-std", type=float, default=-1.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-support-scale", type=float, default=0.0)
    parser.add_argument("--late-support-scale", type=float, default=0.0)
    parser.add_argument("--future-forward-scale", type=float, default=0.0)
    parser.add_argument("--hand-reach-reward-scale", type=float, default=0.0)
    parser.add_argument("--future-support-window-steps", type=int, default=20)
    parser.add_argument("--stance-loss-grace-steps", type=int, default=2)
    parser.add_argument("--stance-brace-strength", type=float, default=0.0)
    parser.add_argument("--stepper-residual-strength", type=float, default=0.0)
    parser.add_argument("--stepper-min-reach-m", type=float, default=0.0)
    parser.add_argument("--stepper-observation", action="store_true")
    parser.add_argument("--tactile-model-checkpoint", type=Path)
    parser.add_argument("--random-tactile-features", action="store_true")
    parser.add_argument("--tactile-representation", choices=("hidden", "predicted_contact",
                                                          "blended_contact"),
                        default="hidden")
    parser.add_argument("--contact-blend", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1297)
    args = parser.parse_args()
    checkpoint_path = args.output.with_suffix(".pt")
    if args.output.exists() or checkpoint_path.exists():
        parser.error("Output or checkpoint already exists")
    if min(args.updates, args.worlds, args.control_steps, args.physics_per_control) < 1:
        parser.error("Rollout dimensions must be positive")
    if args.checkpoint_every < 0:
        parser.error("Checkpoint interval must be nonnegative")
    snapshot_updates = (range(args.checkpoint_every, args.updates, args.checkpoint_every)
                        if args.checkpoint_every else ())
    snapshot_paths = {update_count: checkpoint_path.with_name(
        f"{checkpoint_path.stem}_update{update_count:03d}.pt")
        for update_count in snapshot_updates}
    if any(path.exists() for path in snapshot_paths.values()):
        parser.error("An intermediate checkpoint already exists")
    if min(args.joint_jitter, args.forward_reward_scale, args.switch_reward_scale,
           args.completion_scale, args.abort_penalty, args.policy_anchor,
           args.stance_loss_grace_steps, args.minimum_support_scale,
           args.late_support_scale) < 0:
        parser.error("Reward, jitter, anchor, and grace values must be nonnegative")
    if (args.future_forward_scale < 0 or args.hand_reach_reward_scale < 0 or
            args.future_support_window_steps < 1 or
            (args.future_forward_scale > 0 and
             args.future_support_window_steps >= args.control_steps)):
        parser.error("Future support reward needs a positive window shorter than the rollout")
    if not 0 <= args.stepper_residual_strength <= 1:
        parser.error("Stepper residual strength must be between zero and one")
    if not 0 <= args.stepper_min_reach_m <= 0.2:
        parser.error("Minimum hand reach must be between zero and 0.2 m")
    if not 0 <= args.stance_brace_strength <= 1:
        parser.error("Stance brace strength must be between zero and one")
    if not -3 <= args.exploration_log_std <= 0:
        parser.error("Exploration log standard deviation must be between -3 and zero")
    if not 0 < args.learning_rate <= 1e-2:
        parser.error("Learning rate must be positive and at most 0.01")
    if not 0 <= args.contact_blend <= 1:
        parser.error("Contact blend must be between zero and one")
    if args.random_tactile_features and args.tactile_model_checkpoint is None:
        parser.error("Random tactile features require a tactile model checkpoint")
    if args.tactile_representation != "hidden" and args.tactile_model_checkpoint is None:
        if args.continue_from_policy is None:
            parser.error("Tactile representation requires a tactile model checkpoint")
    if (args.continue_from_policy is not None and
            (args.tactile_model_checkpoint is not None or args.random_tactile_features)):
        parser.error("Continued policy restores its own frozen tactile encoder")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    if source.eq_active0.any():
        raise ValueError("Newton bridge requires inactive equalities")
    checkpoint = torch.load(args.support_checkpoint, map_location="cpu", weights_only=True)
    continuation = (torch.load(args.continue_from_policy, map_location="cpu", weights_only=True)
                    if args.continue_from_policy else None)
    if (tuple(checkpoint["joint_names"]) != PALM_JOINTS or
            checkpoint.get("palm_sensor_version") != PALM_SENSOR_VERSION or
            not checkpoint["use_touch"] or checkpoint.get("world_features") is not None or
            checkpoint.get("oscillator_parameters") is not None):
        raise ValueError("Expected a compatible tactile palm-support checkpoint")
    with np.load(args.stepper_parameters) as saved:
        parameters = np.asarray(saved["parameters"], dtype=np.float32)
    if parameters.shape != (6,) or not np.isfinite(parameters).all():
        raise ValueError("Stepper parameters must be six finite values")
    torch.manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with newton_crawl_environment(source, args.scene, args.pose, args.worlds) as environment:
        observation_size = observe(environment, environment.measure(), True).shape[-1]
        if observation_size != checkpoint["observation_size"]:
            raise ValueError("Checkpoint observation size does not match the infant model")
        controller = TactileStepper(torch.as_tensor(parameters, device=environment.device).expand(
            args.worlds, -1), absolute_targets=True, forward_reach=True,
            strong_plant=True, leg_push=True,
            stance_loss_grace_steps=args.stance_loss_grace_steps,
            stance_brace_strength=args.stance_brace_strength,
            residual_strength=args.stepper_residual_strength,
            min_reach_m=args.stepper_min_reach_m)
        world_features = (TactileFeatureEncoder.restore_snapshot(
            continuation["world_features"], environment.device)
            if continuation is not None and continuation.get("world_features") is not None else
            TactileFeatureEncoder.from_path(
                args.tactile_model_checkpoint, environment.device,
                random_features=args.random_tactile_features, normalize_features=True,
                representation=args.tactile_representation,
                contact_blend=args.contact_blend)
            if args.tactile_model_checkpoint is not None else None)
        feature_size = (controller.observation_size if args.stepper_observation else 0)
        feature_size += world_features.feature_size if world_features is not None else 0
        policy = SupportPolicy(observation_size + feature_size,
                               len(PALM_JOINTS)).to(environment.device)
        initialize_policy_from_checkpoint(policy, checkpoint,
                                          extra_observation_size=feature_size)
        if continuation is not None:
            if (continuation["observation_size"] != observation_size + feature_size or
                    tuple(continuation["joint_names"]) != PALM_JOINTS or
                    continuation.get("palm_sensor_version") != PALM_SENSOR_VERSION or
                    continuation.get("stepper_observation_version") !=
                    (1 if args.stepper_observation else None) or
                    not np.allclose(continuation["stepper_parameters"], parameters,
                                    rtol=0, atol=1e-6)):
                raise ValueError("Continued policy does not match the Newton stepper task")
            if (continuation.get("stepper_residual_strength", 0.0) > 0 and
                    continuation["stepper_residual_strength"] != args.stepper_residual_strength):
                raise ValueError("Continued residual policy has a different stepper strength")
            policy.load_state_dict(continuation["policy"])
        stepper_reference = None
        if args.stepper_residual_strength > 0:
            stepper_reference = copy.deepcopy(policy)
            if continuation is not None and continuation.get("stepper_reference_policy") is not None:
                stepper_reference.load_state_dict(continuation["stepper_reference_policy"])
            stepper_reference.eval().requires_grad_(False)
        reference = None
        if args.policy_anchor > 0:
            reference = copy.deepcopy(policy).eval().requires_grad_(False)
        with torch.no_grad():
            policy.log_std.clamp_(max=args.exploration_log_std)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        history = []
        def checkpoint_contents(update_count):
            return {"policy": policy.state_dict(),
                    "observation_size": observation_size + feature_size,
                    "joint_names": PALM_JOINTS, "use_touch": True,
                    "palm_sensor_version": PALM_SENSOR_VERSION,
                    "world_features": world_features.snapshot() if world_features else None,
                    "oscillator_parameters": None, "training_backend": "isaaclab_newton",
                    "stepper_parameters": parameters.tolist(),
                    "stance_loss_grace_steps": args.stance_loss_grace_steps,
                    "stance_brace_strength": args.stance_brace_strength,
                    "exploration_log_std": args.exploration_log_std,
                    "learning_rate": args.learning_rate,
                    "stepper_residual_strength": args.stepper_residual_strength,
                    "stepper_min_reach_m": args.stepper_min_reach_m,
                    "hand_reach_reward_scale": args.hand_reach_reward_scale,
                    "stepper_reference_policy": (
                        {name: value.detach().cpu() for name, value in
                         stepper_reference.state_dict().items()}
                        if stepper_reference is not None else None),
                    "stepper_observation_version": (1 if args.stepper_observation else None),
                    "source_checkpoint": str(args.support_checkpoint.resolve()),
                    "continued_from": (str(args.continue_from_policy.resolve())
                                       if args.continue_from_policy else None),
                    "training_updates": update_count}
        for iteration in range(args.updates):
            batch, summary = rollout(environment, policy, controller, args.control_steps,
                                     args.physics_per_control, args.joint_jitter,
                                     True, args.forward_reward_scale,
                                     args.switch_reward_scale,
                                     completion_scale=args.completion_scale,
                                     abort_penalty=args.abort_penalty,
                                     mask_overridden_actions=True,
                                     stepper_observation=args.stepper_observation,
                                     world_features=world_features,
                                     minimum_support_scale=args.minimum_support_scale,
                                     late_support_scale=args.late_support_scale,
                                     future_forward_scale=args.future_forward_scale,
                                     future_support_window_steps=args.future_support_window_steps,
                                     stepper_reference_policy=stepper_reference,
                                     hand_reach_reward_scale=args.hand_reach_reward_scale)
            update(policy, optimizer, batch, reference_policy=reference,
                   reference_strength=args.policy_anchor)
            with torch.no_grad():
                policy.log_std.clamp_(max=args.exploration_log_std)
            summary["update"] = iteration + 1
            summary["mean_exploration_std"] = float(policy.log_std.exp().mean())
            summary["mean_aborts"] = float(controller.aborted.mean())
            summary["solver_limit_steps"] = environment.solver_limit_steps
            history.append(summary)
            print(json.dumps(summary), flush=True)
            if iteration + 1 in snapshot_paths:
                torch.save(checkpoint_contents(iteration + 1),
                           snapshot_paths[iteration + 1])
    torch.save(checkpoint_contents(args.updates), checkpoint_path)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()),
                                       "support_checkpoint": str(args.support_checkpoint.resolve()),
                                       "continue_from_policy": (str(args.continue_from_policy.resolve())
                                                                if args.continue_from_policy else None),
                                       "stepper_parameters": str(args.stepper_parameters.resolve()),
                                       "checkpoint": str(checkpoint_path.resolve()),
                                       "updates": args.updates, "worlds": args.worlds,
                                       "rollout_duration_s": (args.control_steps *
                                                              args.physics_per_control *
                                                              float(environment.source.opt.timestep)),
                                       "checkpoint_every": args.checkpoint_every,
                                       "control_steps": args.control_steps,
                                       "physics_per_control": args.physics_per_control,
                                       "joint_jitter": args.joint_jitter,
                                       "forward_reward_scale": args.forward_reward_scale,
                                       "switch_reward_scale": args.switch_reward_scale,
                                       "completion_scale": args.completion_scale,
                                       "abort_penalty": args.abort_penalty,
                                       "policy_anchor": args.policy_anchor,
                                       "minimum_support_scale": args.minimum_support_scale,
                                       "late_support_scale": args.late_support_scale,
                                       "future_forward_scale": args.future_forward_scale,
                                       "hand_reach_reward_scale": args.hand_reach_reward_scale,
                                       "future_support_window_steps": args.future_support_window_steps,
                                       "stance_loss_grace_steps": args.stance_loss_grace_steps,
                                       "stance_brace_strength": args.stance_brace_strength,
                                       "exploration_log_std": args.exploration_log_std,
                                       "learning_rate": args.learning_rate,
                                       "stepper_residual_strength": args.stepper_residual_strength,
                                       "stepper_min_reach_m": args.stepper_min_reach_m,
                                       "previous_stepper_min_reach_m": (
                                           continuation.get("stepper_min_reach_m", 0.0)
                                           if continuation is not None else None),
                                       "stepper_observation_version": (1 if args.stepper_observation else None),
                                       "tactile_model_checkpoint": (str(args.tactile_model_checkpoint.resolve())
                                                                    if args.tactile_model_checkpoint else None),
                                       "random_tactile_features": (world_features.random_features
                                                                   if world_features else False),
                                       "tactile_representation": (world_features.representation
                                                                  if world_features else None),
                                       "contact_blend": (world_features.contact_blend
                                                         if world_features else None),
                                       "tactile_prediction_horizon_control_steps": (
                                           world_features.prediction_horizon_control_steps
                                           if world_features else None),
                                       "masked_overridden_actions": True,
                                       "seed": args.seed, "history": history,
                                       "scope": "Stepper-assisted PPO; held-out crawl evaluation required"},
                                      indent=2) + "\n")


if __name__ == "__main__":
    main()
