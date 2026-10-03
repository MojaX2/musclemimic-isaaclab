"""Evolve tactile stepper muscle targets around a fixed Newton crawl policy."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from evaluate_infant_newton_crawl import newton_crawl_environment
from evolve_infant_crawl_cpg import evaluate
from infant_crawl_env import PALM_JOINTS, PALM_SENSOR_VERSION
from infant_crawl_stepper import TactileStepper
from infant_tactile_features import TactileFeatureEncoder
from train_infant_crawl_ppo import SupportPolicy, jittered_positions, observe


PARAMETER_NAMES = ("lift_shoulder_horizontal", "lift_shoulder_ad_ab",
                   "reach_shoulder_horizontal", "wrist_pitch",
                   "diagonal_hip_swing", "diagonal_knee_swing")


def sustained_crawl_score(result, parameters, forward_priority=1.0):
    if forward_priority <= 0:
        raise ValueError("Forward priority must be positive")
    forward = result["forward_displacement_m"]
    support = result["minimum_support_fraction"]
    late = result["final_quarter_support_fraction"]
    switches = result["hand_switches"].clamp(max=2)
    completed = result.get("step_completions", torch.zeros_like(switches)).clamp(max=2)
    aborted = result.get("step_aborts", torch.zeros_like(switches)).clamp(max=4)
    progress = forward.clamp(min=0, max=0.3) * support * late
    return (forward_priority * 30 * progress + 1.5 * support + late +
            0.4 * switches * late + completed * late - 0.25 * aborted -
            forward_priority * 5 * (-forward).clamp(min=0, max=0.2) -
            0.01 * parameters.square().mean(-1))


def evaluate_candidates(environment, policy, encoder, parameters, positions,
                        control_steps, physics_per_control, stance_loss_grace_steps,
                        residual_strength=0.0, reference_policy=None,
                        forward_priority=1.0, stance_brace_strength=0.0):
    controller = TactileStepper(parameters, absolute_targets=True,
                                forward_reach=True, strong_plant=True, leg_push=True,
                                stance_loss_grace_steps=stance_loss_grace_steps,
                                stance_brace_strength=stance_brace_strength,
                                residual_strength=residual_strength)
    result = evaluate(
        environment, policy, None, positions, control_steps,
        physics_per_control, True, gait_objective=True,
        drive_controller=controller,
        observation_augment=lambda observation, stepper: torch.cat(
            (observation, stepper.observation()), dim=-1),
        world_features=encoder, reference_policy=reference_policy)
    result["selection_score"] = sustained_crawl_score(result, parameters,
                                                       forward_priority)
    return result


def load_policy(checkpoint_path, environment):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (tuple(checkpoint["joint_names"]) != PALM_JOINTS or
            checkpoint.get("palm_sensor_version") != PALM_SENSOR_VERSION or
            not checkpoint["use_touch"] or
            checkpoint.get("stepper_observation_version") != 1 or
            checkpoint.get("world_features", {}).get("kind") !=
            "tactile_next_contact_v1"):
        raise ValueError("Expected a tactile Newton stepper policy with stage observations")
    encoder = TactileFeatureEncoder.restore_snapshot(checkpoint["world_features"],
                                                       environment.device)
    expected = observe(environment, environment.measure(), True).shape[-1]
    expected += encoder.feature_size + TactileStepper.observation_size
    if checkpoint["observation_size"] != expected:
        raise ValueError("Checkpoint observation size does not match the infant")
    policy = SupportPolicy(expected, len(PALM_JOINTS)).to(environment.device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    reference_policy = None
    if checkpoint.get("stepper_residual_strength", 0.0) > 0:
        reference_state = checkpoint.get("stepper_reference_policy")
        if reference_state is None:
            raise ValueError("Residual policy is missing its frozen reference")
        reference_policy = SupportPolicy(expected, len(PALM_JOINTS)).to(environment.device)
        reference_policy.load_state_dict(reference_state)
        reference_policy.eval().requires_grad_(False)
    return checkpoint, policy, encoder, reference_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stepper-parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--control-steps", type=int, default=320)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--forward-priority", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if (args.population < 4 or min(args.repeats, args.generations,
                                   args.control_steps, args.physics_per_control) < 1):
        parser.error("Search dimensions are invalid")
    if args.forward_priority <= 0:
        parser.error("Forward priority must be positive")
    with np.load(args.stepper_parameters) as saved:
        initial = np.asarray(saved["parameters"], dtype=np.float32)
    if initial.shape != (6,) or not np.isfinite(initial).all():
        parser.error("Initial stepper needs six finite parameters")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    if source.eq_active0.any():
        parser.error("Newton bridge needs inactive equality constraints")
    torch.manual_seed(args.seed)
    worlds = args.population * args.repeats
    lower = torch.tensor([0., 0., 0., 0., -1., -1.], device="cuda:0")
    upper = torch.tensor([1., 1., 1., 1., 1., 1.], device="cuda:0")
    mean = torch.as_tensor(initial, device="cuda:0").clone()
    deviation = torch.tensor([0.15, 0.15, 0.15, 0.12, 0.25, 0.25], device="cuda:0")
    history = []
    pool = []
    with newton_crawl_environment(source, args.scene, args.pose, worlds) as environment:
        checkpoint, policy, encoder, reference_policy = load_policy(
            args.checkpoint, environment)
        if not np.allclose(checkpoint["stepper_parameters"], initial, atol=1e-6):
            raise ValueError("Initial stepper parameters do not match the policy")
        grace = checkpoint.get("stance_loss_grace_steps", 2)
        for generation in range(args.generations):
            candidates = (mean + torch.randn((args.population, 6), device=environment.device) *
                          deviation).clamp(lower, upper)
            candidates[0] = mean.clamp(lower, upper)
            candidates[1] = torch.as_tensor(initial, device=environment.device)
            parameters = candidates.repeat_interleave(args.repeats, dim=0)
            scores = []
            metrics = []
            for seed_offset in (0, 1):
                torch.manual_seed(args.seed + 100 * generation + seed_offset)
                starts = jittered_positions(environment, 0.05, 0.01)[:args.repeats]
                positions = starts.repeat(args.population, 1)
                result = evaluate_candidates(
                    environment, policy, encoder, parameters, positions,
                    args.control_steps, args.physics_per_control, grace,
                    checkpoint.get("stepper_residual_strength", 0.0), reference_policy,
                    args.forward_priority, checkpoint.get("stance_brace_strength", 0.0))
                scores.append(result["selection_score"].reshape(args.population,
                                                                    args.repeats).mean(-1))
                metrics.append({name: result[name].float().reshape(args.population,
                                                                    args.repeats).mean(-1)
                                for name in ("forward_displacement_m",
                                             "minimum_support_fraction",
                                             "final_quarter_support_fraction",
                                             "hand_switches", "step_completions",
                                             "step_aborts", "crawl_success")})
            score = torch.stack(scores).mean(0)
            elite = candidates[score.topk(max(2, args.population // 3)).indices]
            mean = (0.5 * mean + 0.5 * elite.mean(0)).clamp(lower, upper)
            deviation = (0.6 * deviation + 0.4 * elite.std(0, unbiased=False)).clamp(0.04, 0.5)
            for candidate_id in range(args.population):
                pool.append((float(score[candidate_id]), candidates[candidate_id].clone()))
            winner = int(score.argmax())
            record = {"generation": generation + 1, "best_score": float(score[winner]),
                      "best_parameters": candidates[winner].cpu().tolist()}
            for name in metrics[0]:
                record[name] = float(torch.stack([item[name][winner]
                                                  for item in metrics]).mean())
            history.append(record)
            print(json.dumps(record), flush=True)
    contenders = sorted(pool, key=lambda item: item[0], reverse=True)[:4]
    contenders.append((None, torch.as_tensor(initial, device="cuda:0")))
    contenders.append((None, mean.clone()))
    validation = []
    with newton_crawl_environment(source, args.scene, args.pose, 8) as environment:
        checkpoint, policy, encoder, reference_policy = load_policy(
            args.checkpoint, environment)
        torch.manual_seed(args.seed + 10000)
        positions = jittered_positions(environment, 0.05, 0.01)
        for training_score, candidate in contenders:
            result = evaluate_candidates(
                environment, policy, encoder, candidate.expand(8, -1), positions,
                args.control_steps, args.physics_per_control,
                checkpoint.get("stance_loss_grace_steps", 2),
                checkpoint.get("stepper_residual_strength", 0.0), reference_policy,
                args.forward_priority, checkpoint.get("stance_brace_strength", 0.0))
            validation.append({"training_score": training_score,
                               "parameters": candidate.cpu().tolist(),
                               "selection_score": float(result["selection_score"].mean()),
                               "forward_m": float(result["forward_displacement_m"].mean()),
                               "support_fraction": float(result["minimum_support_fraction"].mean()),
                               "late_support_fraction": float(result["final_quarter_support_fraction"].mean()),
                               "hand_switches": float(result["hand_switches"].mean()),
                               "step_completions": float(result["step_completions"].mean()),
                               "step_aborts": float(result["step_aborts"].mean()),
                               "crawl_success_count": int(result["crawl_success"].sum())})
    selected = max(validation, key=lambda item: item["selection_score"])
    report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "checkpoint": str(args.checkpoint.resolve()),
              "stepper_parameters": str(args.stepper_parameters.resolve()),
              "seed": args.seed, "population": args.population, "repeats": args.repeats,
              "generations": args.generations,
              "duration_s": args.control_steps * args.physics_per_control *
                            float(source.opt.timestep),
              "parameter_names": PARAMETER_NAMES,
              "forward_priority": args.forward_priority,
              "stepper_residual_strength": checkpoint.get("stepper_residual_strength", 0.0),
              "stance_brace_strength": checkpoint.get("stance_brace_strength", 0.0),
              "history": history, "validation": validation, "selected": selected,
              "scope": "Newton GA search; held-out crawl evaluation required"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
